import logging
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
from torch import nn
import numpy as np
import copy

from torch.cuda.amp import autocast
from torch_points3d.modules.KPConv import *
from torch_points3d.core.base_conv.partial_dense import *
from torch_points3d.core.common_modules import MultiHeadClassifier
from torch_points3d.models.base_model import BaseModel
from torch_points3d.models.base_architectures.unet import UnwrappedUnetBasedModel
from torch_points3d.third_party.pointnet2.pointnet2_utils import furthest_point_sample
from torch_points3d.models.change_detection.inst_dir.matcher import HungarianMatcher
from torch_points3d.models.change_detection.inst_dir.criterion import SetCriterion

from torch_geometric.nn import knn
from torch_points3d.datasets.change_detection.pair import PairMultiScaleBatch
from torch_points3d.models.change_detection.position_embedding import PositionEmbeddingCoordsSine

log = logging.getLogger(__name__)

class BaseFactoryPSI:
    def __init__(self, module_name_down, module_name_sem_up, module_name_change_up, modules_lib):
        self.module_name_down = module_name_down
        self.module_name_sem_up = module_name_sem_up
        self.module_name_change_up = module_name_change_up
        self.modules_lib = modules_lib

    def get_module(self, flow):
        if flow.upper() == "UP_SEM":
            return getattr(self.modules_lib, self.module_name_sem_up, None)
        elif "DOWN" in flow:
            return getattr(self.modules_lib, self.module_name_down, None)
        else:
            return getattr(self.modules_lib, self.module_name_change_up, None)


####################SIAMESE ENCODER FUSION KP CONV ############################
class MSENet(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self.change_num_classes = dataset.change_classes
        self.sem_num_classes = dataset.sem_classes
        self.change_weight_classes = dataset.change_weight_classes
        self.inst_weight_classes = dataset.inst_weight_classes
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
            self.change_FC_layer = MultiHeadClassifier(
                last_mlp_opt.nn[0],
                self._class_to_seg,
                dropout_proba=last_mlp_opt.dropout,
                bn_momentum=last_mlp_opt.bn_momentum,
            )
        else:
            in_feat = last_mlp_opt.nn[0] + self._num_categories
            self.change_FC_layer = Sequential()
            # self.sem_FC_layer = Sequential()
            for i in range(1, len(last_mlp_opt.nn)):
                self.change_FC_layer.add_module(
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
                self.change_FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))
            self.change_FC_layer.add_module("Class", Lin(in_feat, self.change_num_classes, bias=False))
            self.change_FC_layer.add_module("Softmax", nn.LogSoftmax(-1))

        #### INSTANCE DECODER ####
        self.mask_dim = 64
        self.num_queries = 160
        self.use_np_features = False
        hidden_dim = last_mlp_opt.nn[0] + self._num_categories
        self.mask_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.inst_feat_embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            FastBatchNorm1d(hidden_dim, momentum=last_mlp_opt.bn_momentum),
            nn.LeakyReLU(0.2),
        )
        if self.use_np_features:
            self.np_feature_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.class_embed_head = nn.Linear(hidden_dim, self.sem_num_classes)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.pos_enc = PositionEmbeddingCoordsSine(
            pos_type="fourier",
            d_pos=hidden_dim,
            gauss_scale=1.0,
            normalize=True,
        )
        self.mask_features_head = Lin(hidden_dim, hidden_dim, bias=False)
        self.change_activation = nn.Sigmoid() #nn.Sigmoid() #nn.Softmax()

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

        self.inst_concat_head = nn.ModuleList()
        self.tmp_change_conv = nn.ModuleList()
        PLANES = (128, 256, 512, 1024, 2048)
        self.tmp_change_conv.append(
            nn.Linear(PLANES[0] * 2, PLANES[0])
        )
        for i, hlevel in enumerate(self.hlevels):
            self.tmp_change_conv.append(
                nn.Linear(PLANES[hlevel] * 2, PLANES[hlevel])
            )
            self.inst_concat_head.append(
                nn.Linear(hidden_dim * (len(self.hlevels) - 1), PLANES[4 - hlevel])
            )
        self.inst_concat_head.append(
            nn.Linear(hidden_dim, PLANES[0])
        )

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

        ######### Inst Loss #########
        cost_class = 1.0
        cost_mask = 2.5
        cost_dice = 1.0

        weight_dict = {
            "loss_ce": cost_class,
            "loss_mask": cost_mask,
            "loss_dice": cost_dice,
        }
        aux_weight_dict = {}
        for i in range(len(self.hlevels) * self.num_decoders):
            aux_weight_dict.update(
                {k + f"_{i}": v for k, v in weight_dict.items()}
            )
        weight_dict.update(aux_weight_dict)
        eos_coef = 0.1
        losses = ["labels", "masks"]
        num_points = -1
        oversample_ratio = 3.0
        importance_sample_ratio = 0.75
        class_weights = -1

        self.matcher = HungarianMatcher(cost_class, cost_mask, cost_dice)
        self.set_criterion = SetCriterion(self.sem_num_classes, self.matcher, weight_dict, eos_coef, losses, num_points, oversample_ratio, importance_sample_ratio, class_weights)

        self.loss_names = ["loss_cd"]

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
        self.down_modules = nn.ModuleList()
        self.inner_modules = nn.ModuleList()
        self.up_sem_modules = nn.ModuleList()
        self.up_change_modules = nn.ModuleList()

        self.save_sampling_id_1 = opt.down_conv.get('save_sampling_id')

        # Factory for creating up and down modules
        factory_module_cls = self._get_factory(model_type, modules_lib)
        down_conv_cls_name = opt.down_conv.module_name
        up_conv_sem_name = opt.up_change_conv.module_name if opt.get('up_sem_conv') is not None else None
        up_conv_change_name = opt.up_sem_conv.module_name if opt.get('up_change_conv') is not None else None

        self._factory_module = factory_module_cls(
            down_conv_cls_name, up_conv_sem_name, up_conv_change_name, modules_lib
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
        for i in range(len(opt.down_conv.down_conv_nn)):
            args = self._fetch_arguments(opt.down_conv, i, "DOWN")
            conv_cls = self._get_from_kwargs(args, "conv_cls")
            down_module = conv_cls(**args)
            self._save_sampling_and_search(down_module)
            self.down_modules.append(down_module)

        # Up modules
        if up_conv_sem_name:
            for i in range(len(opt.up_sem_conv.up_conv_nn)):
                args = self._fetch_arguments(opt.up_sem_conv, i, "UP_SEM")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_sem_modules.append(up_module)
        if up_conv_change_name:
            for i in range(len(opt.up_change_conv.up_conv_nn)):
                args = self._fetch_arguments(opt.up_change_conv, i, "UP_CHANGE")
                conv_cls = self._get_from_kwargs(args, "conv_cls")
                up_module = conv_cls(**args)
                self._save_upsample(up_module)
                self.up_change_modules.append(up_module)

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
            self.target_y = data.target_y
            self.target_y_target = data.target_y_target
            self.change_y = data.change_y.to(device)
            self.change_y_target = data.change_y_target.to(device)
            pos_y = data.pos_b
            pos_y_target = data.pos_b_target
            full_target_y = data.full_target_y
            full_target_y_target = data.target_y_target
            full_pos = data.full_pos
            full_pos_target = data.full_pos_target

            full_rgb = data.full_rgb
            full_rgb_target = data.full_rgb_target

            full_change = data.full_change_y
            full_change_target = data.full_change_y_target
            full_inst_y = data.full_inst_y
            full_inst_y_target = data.full_inst_y_target
            self.labels = [self.change_y, self.change_y_target, self.target_y, self.target_y_target, full_target_y,
                           full_target_y_target, full_change, full_change_target, full_inst_y, full_inst_y_target,
                           full_pos, full_pos_target, full_rgb, full_rgb_target, pos_y, pos_y_target]
        else:
            self.input = data
            self.target_y = None
            self.target_y_target = None
            self.change_y = None
            self.change_y_target = None
            self.labels = None

    def _get_attn(self, init_mask, cluster_idx):
        output_mask = torch.zeros_like(init_mask, device=cluster_idx.device)
        uni_cluster_idxs = cluster_idx.unique()
        for uni_cluster_idx in uni_cluster_idxs:
            c_idx = torch.argwhere(uni_cluster_idx != cluster_idx).reshape(-1, )
            output_mask[c_idx, uni_cluster_idx] = 1
        return output_mask.bool()

    def get_pos_encs(self, b_coords):
        pos_encodings_pcd, b_l = [], []
        _batch = b_coords

        for i in range(len(b_coords)):
            _batch = b_coords[i].batch
            _coords = b_coords[i].pos
            _b = torch.unique(_batch)
            b_l.append([])
            pos_encodings_pcd.append([[]])
            for bid in _b:
                index = torch.argwhere(_batch == bid).squeeze(-1)
                b_l[-1].append(index)
                scene_min = _coords[index].min(dim=0)[0][None, ...]
                scene_max = _coords[index].max(dim=0)[0][None, ...]

                with autocast(enabled=False):
                    tmp = self.pos_enc(
                        _coords[index][None, ...].float(),
                        input_range=[scene_min, scene_max],
                    )
                pos_encodings_pcd[-1][0].append(tmp.squeeze(0).permute((1, 0)))
        return pos_encodings_pcd, b_l

    def decompose(self, f, batch_l):
        f_de = []
        for b in batch_l:
            f_b = f[b]
            f_de.append(f_b)
        return f_de

    def query_init(self, feats, coords, batch_l):
        de_coords = self.decompose(coords, batch_l)
        de_feats = self.decompose(feats, batch_l)

        fps_idx = [
            furthest_point_sample(
                de_coords[i][None, ...].float(),
                self.num_queries,
            )
                .squeeze(0)
                .long()
            for i in range(len(de_coords))
        ]

        sampled_coords = torch.stack(
            [
                de_coords[i][fps_idx[i].long(), :]
                for i in range(len(fps_idx))
            ]
        )

        mins = torch.stack(
            [
                de_coords[i].min(dim=0)[0]
                for i in range(len(de_coords))
            ]
        )
        maxs = torch.stack(
            [
                de_coords[i].max(dim=0)[0]
                for i in range(len(de_coords))
            ]
        )

        query_pos = self.pos_enc(
            sampled_coords.float(), input_range=[mins, maxs]
        )  # Batch, Dim, queries
        query_pos = self.query_projection(query_pos.permute((0, 2, 1))).permute((0, 2, 1))

        if not self.use_np_features:
            queries = torch.zeros_like(query_pos).permute((0, 2, 1))
        else:
            queries = torch.stack(
                [
                    de_feats[i][
                    fps_idx[i].long(), :
                    ]
                    for i in range(len(fps_idx))
                ]
            )
            queries = self.np_feature_projection(queries)
        query_pos = query_pos.permute((2, 0, 1))
        return queries, query_pos

    def _get_inst_feat(self, queries, mask_feats):
        inst_feats = []
        for query, mask_feat in zip(queries, mask_feats):
            inst_feat = mask_feat @ query
            inst_feats.append(inst_feat)
        inst_feats = torch.concat(inst_feats, dim=0)
        inst_feats = self.inst_feat_embed_head(inst_feats)
        return inst_feats

    def mask_module(self, query_feat, mask_features, num_pooling_steps, pcd_features, sampler_list, ret_attn_mask=True):
        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)

        output_masks = []

        for i in range(len(mask_features)):
            output_masks.append(
                mask_features[i] @ mask_embed[i].T
            )
        outputs_mask = torch.cat(output_masks)

        attn_mask = pcd_features.clone()
        attn_mask.x = outputs_mask
        for i in range(num_pooling_steps):
            attn_mask = sampler_list[i](attn_mask.clone())
        am_batch = attn_mask.batch
        de_b = []
        uni_b = torch.unique(am_batch)
        for b in uni_b:
            _bid = torch.argwhere(am_batch == b).reshape(-1, )
            de_b.append(_bid)
        inst_feat = self._get_inst_feat(query_feat, self.decompose(attn_mask.x, de_b))
        inst_feat = self.decompose(inst_feat, de_b)
        if ret_attn_mask:
            attn_mask = attn_mask.x
            cluster_idx = torch.argmax(attn_mask.detach(), dim=1)
            output_mask = self._get_attn(attn_mask.detach(), cluster_idx)
            attn_mask = output_mask

            # attn_mask = attn_mask.detach().sigmoid() < 0.5
            #### Decompose Batch ####
            attn_mask = self.decompose(attn_mask, de_b)
            #########################
            return (
                outputs_class,
                output_masks,
                attn_mask,
                inst_feat
            )
        return outputs_class, output_masks, inst_feat

    def feat_encoder(self):
        sem_stack_down_0, sem_stack_down_1, pos_stack_0, pos_stack_1 = [], [], [], []
        cd_stack_down_0, cd_stack_down_1 = [], []
        sampler_list = []

        data0 = self.input0
        data1 = self.input1

        #### Feature Encoder ####
        data0 = self.down_modules[0](data0, precomputed=self.pre_computed)
        data1 = self.down_modules[0](data1, precomputed=self.pre_computed_target)
        nn_list1 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
        nn_list0 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)

        sem_stack_down_0.append(data0.clone())
        sem_stack_down_1.append(data1.clone())
        pos_stack_0.append(data0.clone())
        pos_stack_1.append(data1.clone())

        diff0 = data0.clone()
        diff0.x = data0.x - data1.x[nn_list1[1, :], :]
        diff1 = data1.clone()
        diff1.x = data1.x - data0.x[nn_list0[1, :], :]

        cd_stack_down_0.append(diff0)
        cd_stack_down_1.append(diff1)

        data0.x = torch.cat((data0.x, diff0.x), axis=1)
        data1.x = torch.cat((data1.x, diff1.x), axis=1)

        for i in range(1, len(self.down_modules) - 1):
            data0 = self.down_modules[i](data0, precomputed=self.pre_computed)
            data1 = self.down_modules[i](data1, precomputed=self.pre_computed_target)
            nn_list1 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
            nn_list0 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)

            sem_stack_down_0.append(data0.clone())
            sem_stack_down_1.append(data1.clone())
            pos_stack_0.append(data0.clone())
            pos_stack_1.append(data1.clone())

            diff0 = data0.clone()
            diff0.x = data0.x - data1.x[nn_list1[1, :], :]
            diff1 = data1.clone()
            diff1.x = data1.x - data0.x[nn_list0[1, :], :]

            cd_stack_down_0.append(diff0)
            cd_stack_down_1.append(diff1)

            data0.x = torch.cat((data0.x, diff0.x), axis=1)
            data1.x = torch.cat((data1.x, diff1.x), axis=1)

            for sampler in self.down_modules[i].sampler:
                if sampler is not None:
                    sampler_list.append(sampler)

        data0 = self.down_modules[-1](data0, precomputed=self.pre_computed)
        data1 = self.down_modules[-1](data1, precomputed=self.pre_computed_target)
        nn_list1 = knn(data1.pos, data0.pos, 1, data1.batch, data0.batch)
        nn_list0 = knn(data0.pos, data1.pos, 1, data0.batch, data1.batch)

        diff0 = data0.clone()
        diff0.x = data0.x - data1.x[nn_list1[1, :], :]
        diff1 = data1.clone()
        diff1.x = data1.x - data0.x[nn_list0[1, :], :]

        cd_stack_down_0.append(diff0)
        cd_stack_down_1.append(diff1)

        for sampler in self.down_modules[-1].sampler:
            if sampler is not None:
                sampler_list.append(sampler)
        sem_stack_down_0.append(data0)
        sem_stack_down_1.append(data1)
        return cd_stack_down_0, cd_stack_down_1, sem_stack_down_0, sem_stack_down_1, pos_stack_0, pos_stack_1, sampler_list

    def sem_decoder(self, data0, data1, pos_stack_0, pos_stack_1, sem_stack_down_0, sem_stack_down_1):
        #### Sem Decoder ####
        aux0, aux1 = [], []
        sem_data_0 = data0.clone()
        sem_data_1 = data1.clone()
        pos_stack_0.append(data0.clone())
        pos_stack_1.append(data1.clone())
        aux0.append(data0.clone())
        aux1.append(data1.clone())
        innermost = False

        if not isinstance(self.inner_modules[0], Identity):
            sem_stack_down_0.append(data0)
            sem_stack_down_1.append(data1)
            sem_data_0 = self.inner_modules[0](sem_data_0)
            sem_data_1 = self.inner_modules[0](sem_data_1)
            innermost = True
        for i in range(len(self.up_sem_modules)):
            if i == 0 and innermost:
                sem_data_0 = self.up_sem_modules[i]((sem_data_0, sem_stack_down_0.pop()))
                sem_data_1 = self.up_sem_modules[i]((sem_data_1, sem_stack_down_1.pop()))
                aux0.append(sem_data_0.clone())
                aux1.append(sem_data_1.clone())
            else:
                sem_data_0 = self.up_sem_modules[i]((sem_data_0, sem_stack_down_0.pop()), precomputed=self.upsample_target)
                sem_data_1 = self.up_sem_modules[i]((sem_data_1, sem_stack_down_1.pop()), precomputed=self.upsample_target)
                aux0.append(sem_data_0.clone())
                aux1.append(sem_data_1.clone())

        last_feature_0 = sem_data_0.x
        last_feature_1 = sem_data_1.x
        return last_feature_0, last_feature_1, aux0, aux1

    def change_decoder(self, data, change_stack_down):
        innermost = False
        if not isinstance(self.inner_modules[0], Identity):
            change_stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True
        for i in range(len(self.up_change_modules)):
            if i == 0 and innermost:
                data = self.up_change_modules[i]((data, change_stack_down.pop()))
            else:
                data = self.up_change_modules[i]((data, change_stack_down.pop()), precomputed=self.upsample_target)
        last_feature = data.x
        if self._use_category:
            change_output = self.change_FC_layer(last_feature, self.category)
        else:
            change_output = self.change_FC_layer(last_feature)
        return change_output

    def concat_feat(self, inst_feat, indice):
        c_f = []
        for feat in inst_feat:
            c_f.append(torch.concat(feat[indice], dim=0))
        c_f = torch.concat(c_f, dim=-1)
        return c_f

    def change_branch(self, cd_stack_down_0, cd_stack_down_1, inst_feats0, inst_feats1):
        diff_list_0, diff_list_1 = [], []
        # Change Feature Extraction #
        inst_feat0 = self.inst_concat_head[0](self.concat_feat(inst_feats0, 0))
        inst_feat1 = self.inst_concat_head[0](self.concat_feat(inst_feats1, 0))
        diff_init_0 = self.change_extraction(cd_stack_down_0, cd_stack_down_1, inst_feat0, inst_feat1, -1)
        diff_init_1 = self.change_extraction(cd_stack_down_1, cd_stack_down_0, inst_feat1, inst_feat0, -1)
        # End #

        # 1、Change Extractor
        for i, hlevel in enumerate(self.hlevels):
            if i == len(self.hlevels) - 1:
                break
            inst_feat0 = self.inst_concat_head[hlevel + 1](self.concat_feat(inst_feats0, hlevel + 1))
            inst_feat1 = self.inst_concat_head[hlevel + 1](self.concat_feat(inst_feats1, hlevel + 1))
            diff_0 = self.change_extraction(cd_stack_down_0, cd_stack_down_1, inst_feat0, inst_feat1, -2 -hlevel)
            diff_1 = self.change_extraction(cd_stack_down_1, cd_stack_down_0, inst_feat1, inst_feat0, -2 -hlevel)
            diff_list_0.append(diff_0)
            diff_list_1.append(diff_1)

        # 2、Change Interpolation
        diff_list_0.reverse()
        diff_list_1.reverse()
        change_0 = self.change_decoder(diff_init_0, diff_list_0)
        change_1 = self.change_decoder(diff_init_1, diff_list_1)

        return change_0, change_1

    def inst_change_extraction(self, change_feat, inst_feat):
        diff_w_1 = self.change_activation(inst_feat)#.permute((1, 0)))
        diff_f_1 = diff_w_1 * change_feat
        return diff_f_1

    def inst_change_extraction_v2(self, data1, data0, inst_feat1, inst_feat0, conv_indice):
        nn_list = knn(data0[conv_indice].pos, data1[conv_indice].pos, 1, data0[conv_indice].batch, data1[conv_indice].batch)
        diff_feat = inst_feat1 - inst_feat0[nn_list[1, :], :]
        return diff_feat

    def change_extraction(self, data1, data0, inst_feat1, inst_feat0, conv_indice):
        diff = data1[conv_indice].clone()
        diff_feat = diff.x
        inst_change_1 = self.inst_change_extraction(diff_feat, inst_feat1)
        diff_init_1 = torch.concat([diff_feat, inst_change_1], dim=-1)
        diff.x = diff_init_1
        return diff

    def charge(self, list1, list2):
        for i, _ in enumerate(list1):
            if list1[i].shape[0] != list2[i].shape[0]:
                print("list1: {}".format(list1[i].shape[0]))
                print("list2: {}".format(list2[i].shape[0]))
                return True
        return False

    def get_batch_id(self, h_batch):
        hlevel_batch = []
        for h_b in h_batch:
            am_batch = h_b.batch
            de_b = []
            uni_b = torch.unique(am_batch)
            for b in uni_b:
                _bid = torch.argwhere(am_batch == b).reshape(-1, )
                de_b.append(_bid)
            hlevel_batch.append(de_b)
        return hlevel_batch

    def mask_branch(self, pcd_feat, aux, pos_encodings_pcd, batch_l, sampler_list, sampler_pt, is_eval=False):
        mask_features = self.mask_features_head(pcd_feat)
        mask_features = self.decompose(mask_features, batch_l[-1])
        queries, query_pos = self.query_init(aux[-1].x, aux[-1].pos, batch_l[-1])

        predictions_class = []
        predictions_mask = []
        inst_feats = []
        for decoder_counter in range(self.num_decoders):
            if self.shared_decoder:
                decoder_counter = 0
            inst_feat_per_decoder = []
            for i, hlevel in enumerate(self.hlevels):
                output_class, outputs_mask, attn_mask, inst_feat = self.mask_module(
                    queries,
                    mask_features,
                    len(aux) - hlevel - 1,
                    sampler_pt,
                    sampler_list,
                    ret_attn_mask=True,
                )

                decomposed_aux = self.decompose(aux[hlevel].x, batch_l[hlevel])
                decomposed_attn = attn_mask

                curr_sample_size = np.sort(
                    [pcd.shape[0] for pcd in decomposed_aux]
                )[int(len(decomposed_aux) / 2)]

                if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                    raise RuntimeError(
                        "only a single point gives nans in cross-attention"
                    )

                rand_idx = []
                mask_idx = []
                for k in range(len(decomposed_aux)):
                    pcd_size = min(decomposed_aux[k].shape[0], decomposed_attn[k].shape[0])
                    if pcd_size <= curr_sample_size:
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
                        idx = torch.randperm(
                            pcd_size, device=queries.device
                        )[:curr_sample_size]
                        midx = torch.zeros(
                            curr_sample_size,
                            dtype=torch.bool,
                            device=queries.device,
                        )  # attend to all

                    rand_idx.append(idx)
                    mask_idx.append(midx)

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
                        pos_encodings_pcd[hlevel][0][k][rand_idx[k], :]
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

                # cross-attention
                output = self.cross_attention[decoder_counter][i](
                    queries.permute((1, 0, 2)),
                    src_pcd,
                    memory_mask=batched_attn.repeat_interleave(
                        self.num_heads, dim=0
                    ).permute((0, 2, 1)),
                    memory_key_padding_mask=None,
                    pos=batched_pos_enc.permute((1, 0, 2)),
                    query_pos=query_pos,
                )

                # self-attention
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
                inst_feat_per_decoder.append(inst_feat)
            inst_feats.append(inst_feat_per_decoder)

        output_class, outputs_mask, inst_feat = self.mask_module(
            queries,
            mask_features,
            0,
            sampler_pt,
            sampler_list,
            ret_attn_mask=False,
        )

        predictions_class.append(output_class)
        predictions_mask.append(outputs_mask)

        return {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "aux_outputs": self._set_aux_loss(
                predictions_class, predictions_mask
            )
        }, inst_feats

    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    def forward(self, *args, **kwargs) -> Any:
        # 1、Encoder
        loss_weight = kwargs["loss_weight"]
        cd_stack_down_0, cd_stack_down_1, sem_stack_down_0, sem_stack_down_1, pos0, pos1, sampler_list = self.feat_encoder()
        data0, data1 = sem_stack_down_0.pop(), sem_stack_down_1.pop()
        pcd_feat0, pcd_feat1, aux0, aux1 = self.sem_decoder(data0, data1, pos0, pos1, sem_stack_down_0, sem_stack_down_1)

        # 2、Get Multilevel Encoded Pos
        pos0.reverse()
        pos1.reverse()
        enc_pos0, _ = self.get_pos_encs(pos0)
        enc_pos1, _ = self.get_pos_encs(pos1)

        # 3、Get batch id
        batch_l0 = self.get_batch_id(aux0)
        batch_l1 = self.get_batch_id(aux1)

        # 4、Mask Branch
        self.mask_output0, inst_feats0 = self.mask_branch(pcd_feat0, aux0, enc_pos0, batch_l0, sampler_list, self.input0)
        self.mask_output1, inst_feats1 = self.mask_branch(pcd_feat1, aux1, enc_pos1, batch_l1, sampler_list, self.input1)

        # 5、Change Branch
        change0, change1 = self.change_branch(cd_stack_down_0, cd_stack_down_1, inst_feats0, inst_feats1)
        self.change_output0 = change0
        self.change_output1 = change1

        # 5、Get input labels
        self.labels.append(self.decompose(self.input0.inst_y, batch_l0[-1]))
        self.labels.append(self.decompose(self.input1.inst_y, batch_l1[-1]))
        self.labels.append(self.decompose(self.input0.change_y, batch_l0[-1]))
        self.labels.append(self.decompose(self.input1.change_y, batch_l1[-1]))

        if self.labels is not None:
            self.compute_loss(loss_weight)

        self.data_visual0 = self.input0
        self.data_visual0.change_pred0 = torch.max(self.change_output0, -1)[1]
        self.data_visual1 = self.input1
        self.data_visual1.change_pred1 = torch.max(self.change_output1, -1)[1]
        self.output = [self.change_output0, self.change_output1, self.mask_output0, self.mask_output1,
                       self.decompose(self.change_output0, batch_l0[-1]), self.decompose(self.change_output1, batch_l1[-1])]

        return self.change_output0, self.change_output1, self.mask_output0, self.mask_output1

    def compute_loss(self, loss_weight):
        if self.change_weight_classes is not None:
            self.change_weight_classes = self.change_weight_classes.to(self.change_output0.device)
        if self.inst_weight_classes is not None:
            self.inst_weight_classes = self.inst_weight_classes.to(self.change_output0.device)
        self.loss = 0

        # Get regularization on weights
        if self.lambda_reg:
            self.loss_reg = self.get_regularization_loss(regularizer_type="l2", lambda_reg=self.lambda_reg)
            self.loss += self.loss_reg

        # Collect internal losses and set them with self and them to self for later tracking
        if self.lambda_internal_losses:
            print('lambda_internal_losses')
            self.loss += self.collect_internal_losses(lambda_weight=self.lambda_internal_losses)

        # 1、Change Loss
        if self._ignore_label is not None:
            self.loss_seg = F.nll_loss(self.change_output0, self.change_y, weight=self.change_weight_classes, ignore_index=self._ignore_label) + F.nll_loss(self.change_output1, self.change_y_target, weight=self.change_weight_classes, ignore_index=self._ignore_label)
        else:
            self.loss_seg = F.nll_loss(self.change_output0, self.change_y, weight=self.change_weight_classes) + F.nll_loss(self.change_output1, self.change_y_target, weight=self.change_weight_classes)

        # 2、Mask Loss
        mask_losses = {}
        if len(self.target_y) != 0:
            mask_losses0 = self.set_criterion(
                self.mask_output0, self.target_y, mask_type="masks"
            )
            new_loss0 = {}
            for k in list(mask_losses0.keys()):
                if k in self.set_criterion.weight_dict:
                    new_key = k + "_PC0"
                    mask_losses0[k] *= self.set_criterion.weight_dict[k]
                    new_loss0[new_key] = mask_losses0[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    print("PC0 Not weight {}".format(k))
                    mask_losses0.pop(k)
            mask_losses.update(new_loss0)
        if len(self.target_y_target) != 0:
            mask_losses1 = self.set_criterion(
                self.mask_output1, self.target_y_target, mask_type="masks"
            )
            new_loss1 = {}
            for k in list(mask_losses1.keys()):
                if k in self.set_criterion.weight_dict:
                    new_key = k + "_PC1"
                    mask_losses1[k] *= self.set_criterion.weight_dict[k]
                    new_loss1[new_key] = mask_losses1[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    print("PC1 Not weight {}".format(k))
                    mask_losses1.pop(k)
            mask_losses.update(new_loss1)
        mask_loss = sum(mask_losses.values())
        self.change_loss = loss_weight["change"] * self.loss_seg
        self.mask_loss = loss_weight["inst"] * mask_loss
        self.loss_all = self.change_loss  + self.mask_loss
        # 加权Loss
        lambda_c, lambda_i = 94.5, 1
        self.change_loss = lambda_c * self.loss_seg
        self.mask_loss = lambda_i * mask_loss
        self.loss_all = self.change_loss  + self.mask_loss

        self.loss += self.loss_all

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss.backward()  # calculate gradients of network G w.r.t. loss_G

#################### ATTENTION ####################
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