from typing import Any
import logging, copy
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn
from plyfile import PlyData, PlyElement
import numpy as np
from torch.cuda.amp import autocast
from scipy.optimize import linear_sum_assignment

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.datasets.multiscale_data import MultiScaleBatch
from torch_geometric.data import Data
from torch_geometric.nn import knn
from torch_points3d.core.data_transform import GridSampling3D
from torch_points3d.models.change_detection.position_embedding import PositionEmbeddingCoordsSine

from torch_points3d.datasets.change_detection.pair import PairBatch, PairMultiScaleBatch

log = logging.getLogger(__name__)

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule

def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

dice_loss_jit = torch.jit.script(dice_loss)  # type: torch.jit.ScriptModule

def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    return loss.mean(1).sum() / num_masks

sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class BaseFactoryPSI:
    def __init__(self, module_name_down_1, module_name_down_2, module_name_change_up, module_name_sem_up, modules_lib):
        self.module_name_down_1 = module_name_down_1
        self.module_name_down_2 = module_name_down_2
        self.module_name_change_up = module_name_change_up
        self.module_name_sem_up = module_name_sem_up
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == "UP_CHANGE":
            return getattr(self.modules_lib, self.module_name_change_up, None)
        elif flow.upper() == "UP_SEM":
            return getattr(self.modules_lib, self.module_name_sem_up, None)
        elif "1" in flow:
            return getattr(self.modules_lib, self.module_name_down_1, None)
        else:
            return getattr(self.modules_lib, self.module_name_down_2, None)


####################SIAMESE KP CONV UNSHARED (PSEUDO SIAMESE)############################
class SiameseKPConvUnshared(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self._change_classes = dataset.change_classes
        self._sem_classes = dataset.sem_classes
        self._weight_change_classes = dataset.weight_classes
        if self._weight_change_classes is not None:
            if len(self._weight_change_classes) != self._change_classes:
                self._weight_change_classes = None
        try:
            self._ignore_label = dataset.ignore_label
        except:
            self._ignore_label = None
        self._use_category = getattr(option, "use_category", False)
        if self._use_category:
            if not dataset.class_to_segments:
                raise ValueError(
                    "The dataset needs to specify a class_to_segments property when using category information for segmentation"
                )
            self._class_to_seg = dataset.class_to_segments
            self._num_categories = len(self._class_to_seg)
            log.info("Using category information for the predictions with %i categories", self._num_categories)
        else:
            self._num_categories = 0

        # Assemble encoder / decoder
        opt = copy.deepcopy(option)
        super(UnwrappedUnetBasedModel, self).__init__(opt)
        self._spatial_ops_dict = {"neighbour_finder": [], "sampler": [], "upsample_op": []}

        self._init_from_compact_format(opt, model_type, dataset, modules)

        # Unshared weight :  2 down modules
        # Build final MLP
        last_mlp_opt = option.mlp_cls
        if self._use_category:
            self.FC_layer = MultiHeadClassifier(
                last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=last_mlp_opt.dropout,
                bn_momentum=last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = last_mlp_opt.nn[0] + self._num_categories
            self.FC_layer = Sequential()
            for i in range(1, len(last_mlp_opt.nn)):
                self.FC_layer.add_module(
                    str(i),
                    Sequential(
                        *[
                            Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                            FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                            nn.LeakyReLU(0.2),
                        ]
                    ),
                )
                in_feat = last_mlp_opt.nn[i]

            if last_mlp_opt.dropout:
                self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

            self.FC_layer.add_module("Class", Lin(in_feat, self._change_classes, bias=False))
            self.FC_layer.add_module("Softmax", nn.LogSoftmax(-1))

        #### INSTANCE DECODER ####
        self.mask_dim = 64
        self.num_queries = 160
        hidden_dim = last_mlp_opt.nn[0] + self._num_categories
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.class_embed_head = nn.Linear(hidden_dim, self._sem_classes)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=hidden_dim,
            gauss_scale=1.0,
            normalize=True,
        )
        self.mask_features_head = Lin(hidden_dim, hidden_dim, bias=False)

        self.num_decoders = 4
        self.shared_decoder = True
        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()
        dim_list = [2048, 512, 256, 128, 64]
        self.sample_sizes = [200, 800, 3200, 12800, 51200]
        self.max_sample_size = False
        self.is_eval = True
        self.use_level_embed = False
        self.hlevels = [0, 1, 2, 3, 4]
        self.num_heads = 8
        pre_norm = False
        dropout = 0.0

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    )
                )

                tmp_squeeze_attention.append(
                    nn.Linear(dim_list[hlevel], self.mask_dim)
                )

                tmp_self_attention.append(
                    SelfAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    )
                )

                tmp_ffn_attention.append(
                    FFNLayer(
                        d_model=self.mask_dim,
                        dim_feedforward=hidden_dim,
                        dropout=dropout,
                        normalize_before=pre_norm,
                    )
                )

            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)

        ##########################

        self.eos_coef = 0.1
        self.empty_weight = torch.ones(self._sem_classes + 1)
        self.empty_weight[-1] = self.eos_coef

        self.loss_names = ["loss_cd", "offset_norm_loss", "offset_dir_loss", "semantic_loss", "score_loss"]

        self.lambda_reg = self.get_from_opt(option, ["loss_weights", "lambda_reg"])
        if self.lambda_reg:
            self.loss_names += ["loss_reg"]

        self.lambda_internal_losses = self.get_from_opt(option, ["loss_weights", "lambda_internal_losses"])

        self.visual_names = ["data_visual"]


    def _init_from_compact_format(self, opt, model_type, dataset, modules_lib):
        """Create a unetbasedmodel from the compact options format - where the
        same convolution is given for each layer, and arguments are given
        in lists
        """
        self.down_modules_1 = nn.ModuleList()
        self.down_modules_2 = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_change_modules = nn.ModuleList()
        self.up_sem_modules = nn.ModuleList()

        self.save_sampling_id_1 = opt.down_conv_1.get('save_sampling_id')
        self.save_sampling_id_2 = opt.down_conv_2.get('save_sampling_id')

        # Factory for creating up and down modules
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name_1 = opt.down_conv_1.module_name
        down_conv_cls_name_2 = opt.down_conv_2.module_name
        up_conv_change_name = opt.up_change.module_name if opt.get('up_change') is not None else None
        up_conv_sem_name = opt.up_sem.module_name if opt.get('up_sem') is not None else None

        self._factory_module = factory_module_cls(
            down_conv_cls_name_1, down_conv_cls_name_2, up_conv_change_name, up_conv_sem_name, modules_lib
        )  # Create the factory object

        # Loal module
        contains_global = hasattr(opt, "innermost") and opt.innermost is not None
        if contains_global:
            inners = self._create_inner_modules(opt.innermost, modules_lib)
            for inner in inners:
                self.inner_modules.append(inner)
        else:
            self.inner_modules.append(Identity())

        # Down modules
        for i in range(len(opt.down_conv_1.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_1, i, "DOWN_1")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_1.append(down_module)
        for i in range(len(opt.down_conv_2.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv_2, i, "DOWN_2")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules_2.append(down_module)

        # Up modules
        if up_conv_change_name:
            for i in range(len(opt.up_change.up_conv_nn)):
                args = self._fetch_arguments(opt.up_change, i, "UP_CHANGE")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_change_modules.append(up_module)
        if up_conv_sem_name:
            for i in range(len(opt.up_sem.up_conv_nn)):
                args = self._fetch_arguments(opt.up_sem, i, "UP_SEM")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_sem_modules.append(up_module)

        self.metric_loss_module, self.miner_module = BaseModel.get_metric_loss_and_miner(
            getattr(opt, "metric_loss", None), getattr(opt, "miner", None)
        )

    def _get_factory(self, model_name, modules_lib) -> BaseFactoryPSI:
        factory_module_cls = getattr(modules_lib, "{}Factory".format(model_name), None)
        if factory_module_cls is None:
            factory_module_cls = BaseFactoryPSI
        return factory_module_cls

    def set_input(self, data, device):
        data = data.to(device)
        data.x = add_ones(data.pos, data.x, True)
        self.batch_idx = data.batch
        if isinstance(data, PairMultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
        else:
            self.pre_computed = None
            self.upsample = None
        if getattr(data, "pos_target", None) is not None:
            data.x_target = add_ones(data.pos_target, data.x_target, True)
            if isinstance(data, PairMultiScaleBatch):
                self.pre_computed_target = data.multiscale_target
                self.upsample_target = data.upsample_target
                del data.multiscale_target
                del data.upsample_target
            else:
                self.pre_computed_target = None
                self.upsample_target = None

            self.input0, self.input1 = data.to_data()
            self.batch_idx_target = data.batch_target
            self.change_labels_1 = data.y_target[:, -1].to(device)
            self.sem_labels_0, self.inst_labels_0 = self.get_instance_masks(data.batch, data.y[:, -3].to(device),
                                                                            data.y[:, -2].to(device))
            self.sem_labels_1, self.inst_labels_1 = self.get_instance_masks(data.batch_target,
                                                                            data.y_target[:, -3].to(device),
                                                                            data.y_target[:, -2].to(device))
            self.labels = [self.change_labels_1, self.sem_labels_0, self.inst_labels_0, self.sem_labels_1,
                           self.inst_labels_1]
        else:
            self.input = data
            self.change_labels_1 = None
            self.sem_labels_0 = None
            self.sem_labels_1 = None
            self.inst_labels_0 = None
            self.inst_labels_1 = None
            self.labels = None

    def dim_correct(self, knn_idx, layer_index):
        try:
            knn_idx = knn_idx.reshape(knn_idx[0], -1, self.k_n[layer_index])
        except:
            print("wrong dim")
        return knn_idx

    def feature_backbone(self):
        stack_down = []
        enc_f_0, enc_f_1 = [], []
        dec_f_l_0, dec_f_l_1 = [], []
        sampler_list1, sampler_list2 = [], []
        pos_list_0, pos_list_1 = [], []
        batch_list_0, batch_list_1 = [], []

        data0 = self.input0
        data1 = self.input1

        for i in range(len(self.down_modules_1) - 1):
            data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)
            nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
            diff = data1.clone()
            diff.x = data1.x - data0.x[nn_list[1,:],:]

            stack_down.append(diff)
            enc_f_0.append(data0)
            enc_f_1.append(data1)
            pos_list_0.append(data0.pos)
            pos_list_1.append(data1.pos)
            batch_list_0.append(data0.batch)
            batch_list_1.append(data1.batch)
            for sampler in self.down_modules_1[i].sampler:
                if sampler is not None:
                    sampler_list1.append(sampler)
            for sampler in self.down_modules_2[i].sampler:
                if sampler is not None:
                    sampler_list2.append(sampler)
        #1024
        data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)

        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data = data1.clone()
        data.x = data1.x - data0.x[nn_list[1,:],:]
        innermost = False

        dec_f_l_0.append(data0)
        dec_f_l_1.append(data1)
        pos_list_0.append(data0.pos)
        pos_list_1.append(data1.pos)
        batch_list_0.append(data0.batch)
        batch_list_1.append(data1.batch)
        for sampler in self.down_modules_1[-1].sampler:
            if sampler is not None:
                sampler_list1.append(sampler)
        for sampler in self.down_modules_2[-1].sampler:
            if sampler is not None:
                sampler_list2.append(sampler)

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data1)
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_change_modules)):
            if i == 0 and innermost:
                data = self.up_change_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_change_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)

            data0 = self.up_sem_modules[i]((data0, enc_f_0.pop()), precomputed=self.upsample_target)
            data1 = self.up_sem_modules[i]((data1, enc_f_1.pop()), precomputed=self.upsample_target)

            dec_f_l_0.append(data0)
            dec_f_l_1.append(data1)

        change_feature = data.x
        if self._use_category:
            self.change_output = self.FC_layer(change_feature, self.category)
        else:
            self.change_output = self.FC_layer(change_feature)

        output1 = [data0, dec_f_l_0, pos_list_0, sampler_list1, batch_list_0]
        output2 = [data1, dec_f_l_1, pos_list_1, sampler_list2, batch_list_1]
        return output1, output2

    def feature_encoder(self):
        sampler_list1, sampler_list2 = [], []
        pos_list_0, pos_list_1 = [], []
        batch_list_0, batch_list_1 = [], []
        enc_f_0, enc_f_1 = [], []

        stack_down = []

        data0 = self.input0
        data1 = self.input1

        for i in range(len(self.down_modules_1) - 1):
            data0 = self.down_modules_1[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules_2[i](data1, precomputed=self.pre_computed_target)

            enc_f_0.append(data0)
            enc_f_1.append(data1)
            pos_list_0.append(data0.pos)
            pos_list_1.append(data1.pos)
            batch_list_0.append(data0.batch)
            batch_list_1.append(data1.batch)
            for sampler in self.down_modules_1[i].sampler:
                if sampler is not None:
                    sampler_list1.append(sampler)
            for sampler in self.down_modules_2[i].sampler:
                if sampler is not None:
                    sampler_list2.append(sampler)

            nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
            diff = data1.clone()
            diff.x = data1.x - data0.x[nn_list[1, :], :]
            stack_down.append(diff)
        # 1024
        data0 = self.down_modules_1[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules_2[-1](data1, precomputed=self.pre_computed_target)

        pos_list_0.append(data0.pos)
        pos_list_1.append(data1.pos)
        batch_list_0.append(data0.batch)
        batch_list_1.append(data1.batch)
        for sampler in self.down_modules_1[-1].sampler:
            if sampler is not None:
                sampler_list1.append(sampler)
        for sampler in self.down_modules_2[-1].sampler:
            if sampler is not None:
                sampler_list2.append(sampler)

        nn_list = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)
        data = data1.clone()
        data.x = data1.x - data0.x[nn_list[1, :], :]

        return data0, data1, data, stack_down, sampler_list1, sampler_list2, pos_list_0, pos_list_1, batch_list_0, batch_list_1, enc_f_0, enc_f_1

    def feature_decoder(self, i_f_0, i_f_1, enc_f_0, enc_f_1):
        dec_f_l_0, dec_f_l_1 = [], []
        dec_f_l_0.append(i_f_0)
        dec_f_l_1.append(i_f_1)
        for i in range(len(self.up_sem_modules)):
            last_f_0, last_f_1 = enc_f_0.pop(), enc_f_1.pop()
            i_f_0 = self.up_sem_modules[i]((i_f_0, last_f_0), precomputed=self.upsample_target)
            i_f_1 = self.up_sem_modules[i]((i_f_1, last_f_1), precomputed=self.upsample_target)
            dec_f_l_0.append(i_f_0)
            dec_f_l_1.append(i_f_1)

        return i_f_0, i_f_1, dec_f_l_0, dec_f_l_1

    def change_decoder(self, data, data1, stack_down):
        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data1)
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_change_modules)):
            if i == 0 and innermost:
                data = self.up_change_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_change_modules[i]((data, stack_down.pop()), precomputed=self.upsample_target)
        change_feature = data.x
        if self._use_category:
            change_output = self.FC_layer(change_feature, self.category)
        else:
            change_output = self.FC_layer(change_feature)
        return change_output

    def instance_decoder(self, pcd_features, dec_f, pos_list, sampler_list, batch_list):
        batch_size = torch.unique(batch_list[0]).shape[0]
        # pos_list.reverse()
        pos_encodings_pcd, b_l = self.get_pos_encs(pos_list, batch_list)
        mask_features = self.mask_features_head(pcd_features.x)
        queries, query_pos = self.query_init(batch_size, pcd_features.x.device)

        predictions_class = []
        predictions_mask = []

        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            for i, hlevel in enumerate(self.hlevels):
                output_class, outputs_mask, attn_mask = self.mask_module(
                    queries,
                    mask_features,
                    batch_list,
                    sampler_list,
                    pcd_features,
                    len(dec_f) - hlevel - 1,
                    ret_attn_mask=True,
                )
                decomposed_aux = dec_f[hlevel].x
                decomposed_attn = attn_mask
                curr_sample_size = max(
                    [pcd.shape[0] for pcd in decomposed_aux]
                )

                if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                    raise RuntimeError(
                        "only a single point gives nans in cross-attention"
                    )

                if not (self.max_sample_size or self.is_eval):
                    curr_sample_size = min(
                        curr_sample_size, self.sample_sizes[hlevel]
                    )

                rand_idx = []
                mask_idx = []
                decomposed_aux_l, decomposed_attn_l = [], []
                for b_id in range(len(decomposed_aux)):# for b_id in b_l[-1-hlevel]:
                    pcd_size = decomposed_aux[b_id].shape[0]
                    decomposed_aux_l.append(decomposed_aux[b_id])
                    decomposed_attn_l.append(decomposed_attn[b_id])
                    if pcd_size <= curr_sample_size:
                        # we do not need to sample
                        # take all points and pad the rest with zeroes and mask it
                        idx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.long,
                            device=queries.device,
                        )

                        midx = torch.ones(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )

                        idx[:pcd_size] = torch.arange(
                            pcd_size, device=queries.device
                        )

                        midx[:pcd_size] = False  # attend to first points
                    else:
                        # we have more points in pcd as we like to sample
                        # take a subset (no padding or masking needed)
                        idx = torch.randperm(
                            decomposed_aux[b_id].shape[0], device=queries.device
                        )[:curr_sample_size]
                        midx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

                decomposed_aux = decomposed_aux_l
                decomposed_attn = decomposed_attn_l

                batched_aux = torch.stack(
                    [
                        decomposed_aux[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn = torch.stack(
                    [
                        decomposed_attn[k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_pos_enc = torch.stack(
                    [
                        pos_encodings_pcd[-1-hlevel][0][k][rand_idx[k], :]
                        for k in range(len(rand_idx))
                    ]
                )

                batched_attn.permute((0, 2, 1))[
                    batched_attn.sum(1) == rand_idx[0].shape[0]
                ] = False

                m = torch.stack(mask_idx)
                batched_attn = torch.logical_or(batched_attn, m[..., None])
                src_pcd = self.lin_squeeze[decoder_counter][i](
                    batched_aux.permute((1, 0, 2))
                )
                if self.use_level_embed:
                    src_pcd += self.level_embed.weight[i]

                output = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(
                        self.num_heads, dim=0
                    ).permute((0, 2, 1)),
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )
                output = self.self_attention[decoder_counter][i](
                    output,
                    tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_pos,
                )
                # FFN
                queries = self.ffn_attention[decoder_counter][i](
                    output
                ).permute((1, 0, 2))

                predictions_class.append(output_class)
                predictions_mask.append(outputs_mask)

        output_class, outputs_mask = self.mask_module(
            queries,
            mask_features,
            batch_list,
            sampler_list,
            pcd_features,
            0,
            ret_attn_mask=False,
        )

        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)
        aux_outputs = self._set_aux_loss(predictions_class, predictions_mask)

        return {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": aux_outputs
        }

    def query_init(self, batch_size, device):
        query_pos = (
            torch.rand(
                batch_size,
                self.mask_dim,
                self.num_queries,
                device=device,
            )
            - 0.5
        )

        queries = torch.zeros_like(query_pos).permute((0, 2, 1))
        query_pos = query_pos.permute((2, 0, 1))
        return queries, query_pos

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

        # data0, data1, change_f, \
        # stack_down, sampler_list1, sampler_list2, \
        # pos_list_0, pos_list_1, \
        # batch_list_0, batch_list_1, \
        # enc_f_0, enc_f_1 = self.feature_encoder()
        # pcd_features_0, pcd_features_1, dec_f_0, dec_f_1 = self.feature_decoder(data0, data1, enc_f_0, enc_f_1)
        # self.change_output = self.change_decoder(change_f, data1, stack_down)
        # self.inst_output_0 = self.instance_decoder(pcd_features_0, dec_f_0, pos_list_0, sampler_list1, batch_list_0)
        # self.inst_output_1 = self.instance_decoder(pcd_features_1, dec_f_1, pos_list_1, sampler_list2, batch_list_1)

        output1, output2 = self.feature_backbone()
        self.inst_output_0 = self.instance_decoder(output1[0], output1[1], output1[2], output1[3], output1[4])
        self.inst_output_1 = self.instance_decoder(output2[0], output2[1], output2[2], output2[3], output2[4])

        if self.labels is not None:
            self.compute_loss()

        self.data_visual0 = self.input0
        self.data_visual0.inst_pred = self.inst_output_0
        self.data_visual1 = self.input1
        self.data_visual1.change_pred = torch.max(self.change_output, -1)[1]
        self.data_visual1.inst_pred = self.inst_output_1

        self.output = [self.change_output, self.inst_output_0, self.inst_output_1]

        return self.output

    def compute_loss(self):
        if self._weight_change_classes is not None:
            self._weight_change_classes = self._weight_change_classes.to(self.change_output.device)
        self.loss = 0

        # Get regularization on weights
        if self.lambda_reg:
            self.loss_reg = self.get_regularization_loss(regularizer_type="l2", lambda_reg=self.lambda_reg)
            self.loss += self.loss_reg

        # Collect internal losses and set them with self and them to self for later tracking
        if self.lambda_internal_losses:
            print('lambda_internal_losses')
            self.loss += self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)

        ####### Change Loss #######
        # Final cross entrop loss
        if self._ignore_label is not None:
            self.loss_seg = F.nll_loss(self.change_output, self.change_labels_1.long(), weight=self._weight_change_classes, ignore_index=self._ignore_label)
        else:
            self.loss_seg = F.nll_loss(self.change_output, self.change_labels_1.long(), weight=self._weight_change_classes)

        if torch.isnan(self.loss_seg).sum() == 1:
            print(self.loss_seg)
        self.loss += self.loss_seg

        # ####### Instance Loss #######
        # loss0 = self.instance_loss(self.inst_output_0["pred_logits"], self.inst_output_0["pred_masks"], self.inst_output_0["aux_outputs"], self.sem_labels_0, self.inst_labels_0, self.input0.batch)
        # loss1 = self.instance_loss(self.inst_output_1["pred_logits"], self.inst_output_1["pred_masks"], self.inst_output_1["aux_outputs"], self.sem_labels_1, self.inst_labels_1, self.input1.batch)
        # self.loss += loss0
        # self.loss += loss1

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G

    def mask_module(
            self,
            query_feat,
            mask_features,
            batch_list,
            sampler_list,
            pcd_features,
            num_pooling_steps,
            ret_attn_mask=True,
    ):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []
        batch_size = torch.unique(batch_list[0])
        for _b in batch_size:
            index = torch.argwhere(batch_list[0] == _b).squeeze(-1)
            output_masks.append(
                mask_features[index] @ mask_embed[_b].T
            )
        output_masks = torch.cat(output_masks)

        if ret_attn_mask:
            attn_mask = pcd_features.clone()
            attn_mask.x = output_masks
            for i in range(num_pooling_steps):
                sampler = sampler_list[i]
                attn_mask = sampler(attn_mask.clone())
            attn_mask = attn_mask.x
            attn_mask = torch.where(attn_mask.detach().sigmoid() < 0.5, False, True)
            return (
                outputs_class,
                output_masks,
                attn_mask,
            )
        return outputs_class, output_masks

    def get_pos_encs(self, coords, batch_list):
        pos_encodings_pcd, b_l = [],[]
        for i, sample_b in enumerate(batch_list):
            _b = torch.unique(sample_b)
            pos_encodings_pcd.append([[]])
            b_l.append([])
            for bid in _b:
                index = torch.argwhere(sample_b == bid).squeeze(-1)
                b_l[-1].append(index)
                scene_min = coords[i][index].min(dim=0)[0][None, ...]
                scene_max = coords[i][index].max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(
                        coords[i][index][None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )

                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))

        return pos_encodings_pcd, b_l

    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def get_instance_masks(self,
        batch_list,
        sem_labels,
        inst_labels,
        ignore_class_threshold=100,
        filter_out_classes=[],
    ):
        masks, sems = [], []
        _b = torch.unique(batch_list)

        for bid in _b:
            b_index = torch.argwhere(batch_list == bid).squeeze(-1)
            mask_b, label_ids = [], []
            instance_ids = inst_labels[b_index].unique()
            b_sem = sem_labels[b_index]
            b_inst = inst_labels[b_index]
            for instance_id in instance_ids:
                if instance_id == -1:
                    continue

                # TODO is it possible that a ignore class (255) is an instance???
                # instance == -1 ???

                tmp = b_inst[
                    instance_id == b_inst
                ]
                label_id = b_sem[
                    instance_id == b_inst
                ][0]

                if label_id == 0:
                    print("aaaaa")

                if (
                    label_id in filter_out_classes
                ):  # floor, wall, undefined==255 is not included
                    continue

                if (
                    255 in filter_out_classes
                    and label_id.item() == 255
                    and tmp.shape[0] < ignore_class_threshold
                ):
                    continue
                # Mask
                mask_b.append(instance_id == b_inst)
                label_ids.append(label_id)
            if len(mask_b) != 0:
                mask_b = torch.stack(mask_b)
            if len(label_ids) != 0:
                label_ids = torch.stack(label_ids)
            masks.append(mask_b)
            sems.append(label_ids)

        return sems, masks

    def hungarian_matcher(self, sem_output, sem_target, inst_output, inst_target, batch_list, _cost_mask=5.0, _cost_class=5.0, _cost_dice=5.0):

        """More memory-friendly matching"""
        bs, num_queries = sem_output.shape[:2]

        indices = []
        ignore_label = 0
        ignore_bid = []

        # Iterate through batch size
        for b in range(bs):
            if isinstance(sem_target[b], List):
                ignore_bid.append(b)
                continue
            b_bid = torch.argwhere(b == batch_list).squeeze(-1)
            out_prob = sem_output[b].softmax(
                -1
            )  # [num_queries, num_classes]
            tgt_ids = sem_target[b].clone()

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            filter_ignore = tgt_ids == 253
            tgt_ids[filter_ignore] = 0
            cost_class = -out_prob[:, tgt_ids]
            cost_class[
            :, filter_ignore
            ] = (
                -1.0
            )  # for ignore classes pretend perfect match ;) TODO better worst class match?

            out_mask = inst_output[
                b_bid
            ].T  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = inst_target[b].to(out_mask)

            point_idx = torch.arange(
                tgt_mask.shape[1], device=tgt_mask.device
            )

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(
                    out_mask[:, point_idx], tgt_mask[:, point_idx]
                )

            C = (
                _cost_mask * cost_mask
                + _cost_class * cost_class
                + _cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu().detach().numpy()

            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ], ignore_bid

    def correct_batch(self, sem_output, sem_target, inst_output, inst_target, ignore_batch, batch_list):
        sem_output_cor, sem_target_cor, inst_output_cor, inst_target_cor = [], [], [], []
        bs, num_queries = sem_output.shape[:2]
        for b in range(bs):
            if b in ignore_batch:
                continue
            b_bid = torch.argwhere(b == batch_list).squeeze(-1)
            sem_output_cor.append(sem_output[b])
            sem_target_cor.append(sem_target[b])
            inst_output_cor.append(inst_output[b_bid])
            inst_target_cor.append(inst_target[b])

        sem_output = torch.stack(sem_output_cor)
        sem_target = sem_target_cor
        inst_output = torch.concat(inst_output_cor)
        inst_target = inst_target_cor

        return sem_output, sem_target, inst_output, inst_target

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, ignore_batch):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs.float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t[J] for t, (_, J) in zip(targets, indices)]
        )
        # target_classes = torch.full(
        #     src_logits.shape[:2],
        #     self._sem_classes - 1,
        #     dtype=torch.int64,
        #     device=src_logits.device,
        # )
        target_classes = torch.full(
            src_logits.shape[:2],
            0,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes
        )
        return loss_ce

    def loss_masks(self, outputs, targets, indices, ignore_batch, batch_list):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        loss_masks = []
        loss_dices = []
        _b = torch.unique(batch_list)
        flag = 0
        for batch_id in _b:
            if batch_id in ignore_batch:
                flag -= 1
                continue
            map_id, target_id = indices[batch_id + flag][0], indices[batch_id + flag][1]
            b_bid = torch.argwhere(batch_id == batch_list).squeeze(-1)
            map = outputs[b_bid][:, map_id].T
            target_mask = targets[batch_id][target_id]

            point_idx = torch.arange(
                target_mask.shape[1], device=target_mask.device
            )

            num_masks = target_mask.shape[0]
            map = map[:, point_idx]
            target_mask = target_mask[:, point_idx].float()

            loss_mask = sigmoid_ce_loss_jit(map, target_mask, num_masks)
            loss_dice = dice_loss_jit(map, target_mask, num_masks)
            loss_masks.append(loss_mask)
            loss_dices.append(loss_dice)
        return torch.sum(torch.stack(loss_masks)), torch.sum(torch.stack(loss_dices))

    def instance_loss(self, sem_output, inst_output, aux_output, sem_target, inst_target, batch_list):
        loss = 0.0
        #### Output Header ####
        indices, ignore_batch = self.hungarian_matcher(sem_output, sem_target, inst_output, inst_target, batch_list)
        sem_output_c, sem_target_c, _, _ = self.correct_batch(sem_output, sem_target, inst_output, inst_target, ignore_batch, batch_list)
        loss_ce = self.loss_labels(sem_output_c, sem_target_c, indices, ignore_batch)
        loss_mask, loss_dices = self.loss_masks(inst_output, inst_target, indices, ignore_batch, batch_list)
        loss += loss_ce
        loss += loss_mask
        loss += loss_dices
        #### Aux Output ####
        for aux in aux_output:
            sem_output, inst_output = aux["pred_logits"], aux["pred_masks"]
            indices, ignore_batch = self.hungarian_matcher(sem_output, sem_target, inst_output, inst_target, batch_list)
            sem_output_c, sem_target_c, _, _ = self.correct_batch(sem_output, sem_target, inst_output, inst_target, ignore_batch, batch_list)
            loss_ce = self.loss_labels(sem_output_c, sem_target_c, indices, ignore_batch)
            loss_mask, loss_dices = self.loss_masks(inst_output, inst_target, indices, ignore_batch, batch_list)
            loss += loss_ce
            loss += loss_mask
            loss += loss_dices
        return loss

class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")