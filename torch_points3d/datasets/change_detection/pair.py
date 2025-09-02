import torch
from typing import List, Optional
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_points3d.datasets.multiscale_data import MultiScaleBatch, MultiScaleData
# from torch_points3d.core.data_transform.feature_augment_pair import ChromaticJitter, ChromaticTranslation, \
#     ChromaticAutoContrast
import re
import numpy as np
import torchvision.transforms as T
import random

KDTREE_KEY_PC0 = None
KDTREE_KEY_PC1 = None


class Pair(Data):

    def __init__(
            self,
            x=None,
            y=None,
            pos=None,
            x_target=None,
            pos_target=None,
            rgb=None,
            rgb_target=None,
            y1=None,
            y0=None,
            inst_y=None,
            inst_y_target=None,
            target_y=None,
            target_y_target=None,
            change_y_target=None,
            gt_change=None,
            pred1 = None,
            **kwargs,
    ):
        self.__data_class__ = Data
        self.KDTREE_KEY_PC0 = KDTREE_KEY_PC0
        self.KDTREE_KEY_PC1 = KDTREE_KEY_PC1
        super(Pair, self).__init__(x=x, pos=pos, rgb=rgb,
                                   x_target=x_target, pos_target=pos_target, rgb_target=rgb_target, inst_y=inst_y, inst_y_target=inst_y_target,
                                   target_y=target_y, target_y_target=target_y_target, change_y_target=change_y_target,
                                   y=y, y0=y0, y1=y1, gt_change= gt_change, pred1=pred1,**kwargs)

    @classmethod
    def make_pair(cls, data_source, data_target):
        """
        add in a Data object the source elem, the target elem.
        """
        # add concatenation of the point cloud
        batch = cls()
        for key in data_source.keys:
            batch[key] = data_source[key]
        for key_target in data_target.keys:
            batch[key_target + "_target"] = data_target[key_target]
        if (batch.x is None):
            batch["x_target"] = None
        return batch.contiguous()

    def to_data(self):
        data_source = self.__data_class__()
        data_target = self.__data_class__()
        for key in self.keys:
            match = re.search(r"(.+)_target$", key)
            if match is None:
                data_source[key] = self[key]
            else:
                new_key = match.groups()[0]
                data_target[new_key] = self[key]
        return data_source, data_target

    @property
    def num_nodes_target(self):
        for key, item in self('x_target', 'pos_target', 'norm_target', 'batch_target'):
            return item.size(self.__cat_dim__(key, item))
        return None

    def normalise(self, normValue=None):
        if normValue is None:
            min0 = torch.unsqueeze(self.pos.min(0)[0], 0)
            min1 = torch.unsqueeze(self.pos_target.min(0)[0], 0)
            minG = torch.cat((min0, min1), axis=0).min(0)[0]
        else:
            [minG, deltaG] = normValue
        # normalizing data with the same xmin ymin zmin
        # Keeping scale of data
        self.pos[:, 0] = (self.pos[:, 0] - minG[0])  # x
        self.pos[:, 1] = (self.pos[:, 1] - minG[1])  # y
        self.pos[:, 2] = (self.pos[:, 2] - minG[2])  # z

        self.pos_target[:, 0] = (self.pos_target[:, 0] - minG[0])  # x
        self.pos_target[:, 1] = (self.pos_target[:, 1] - minG[1])  # y
        self.pos_target[:, 2] = (self.pos_target[:, 2] - minG[2])  # z


    def data_augment(self, angle=2 * np.pi, paramGaussian=[0.01, 0.05], color_aug=False):
        """
        Random data augmentation
        """
        # random rotation around the Z axis
        angle = (np.random.random()-0.5) * angle
        M = torch.from_numpy(np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])).type(
            torch.float32)
        self.pos[:, :2] = torch.matmul(self.pos[:, :2], M)  # perform the rotation efficiently
        self.pos_target[:, :2] = torch.matmul(self.pos_target[:, :2], M)

        # random gaussian noise
        sigma, clip = paramGaussian
        # Hint: use torch.clip to clip and np.random.randn to generate gaussian noise
        self.pos = self.pos + torch.clip(torch.randn(self.pos.shape) * sigma, -clip, clip).type(torch.float32)
        self.pos_target = self.pos_target + torch.clip(torch.randn(self.pos_target.shape) * sigma, -clip, clip).type(
            torch.float32)
        # data color augmentation
        if color_aug:
            self._color_jitter()

    def _color_jitter(self):
        pass

        # if random.random() < 0.5:
        #     chromaticJitter = ChromaticJitter()
        #     self.rgb, self.rgb_target = chromaticJitter(self.rgb, self.rgb_target)
        #
        # if random.random() < 0.2:
        #     chromaticAutoContrast = ChromaticAutoContrast()
        #     self.rgb, self.rgb_target = chromaticAutoContrast(self.rgb, self.rgb_target)
        #
        # if random.random() < 0.5:
        #     chromaticTranslation = ChromaticTranslation()
        #     self.rgb, self.rgb_target = chromaticTranslation(self.rgb, self.rgb_target)


class MultiScalePair(Pair):
    def __init__(
            self,
            x=None,
            y=None,
            pos=None,
            multiscale: Optional[List[Data]] = None,
            upsample: Optional[List[Data]] = None,
            x_target=None,
            pos_target=None,
            multiscale_target: Optional[List[Data]] = None,
            upsample_target: Optional[List[Data]] = None,
            **kwargs,
    ):
        super(MultiScalePair, self).__init__(x=x, pos=pos,
                                             multiscale=multiscale,
                                             upsample=upsample,
                                             x_target=x_target, pos_target=pos_target,
                                             multiscale_target=multiscale_target,
                                             upsample_target=upsample_target,
                                             **kwargs)
        self.__data_class__ = MultiScaleData

    def apply(self, func, *keys):
        r"""Applies the function :obj:`func` to all tensor and Data attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            if torch.is_tensor(item):
                self[key] = func(item)
        for scale in range(self.num_scales):
            self.multiscale[scale] = self.multiscale[scale].apply(func)
            self.multiscale_target[scale] = self.multiscale_target[scale].apply(func)

        for up in range(self.num_upsample):
            self.upsample[up] = self.upsample[up].apply(func)
            self.upsample_target[up] = self.upsample_target[up].apply(func)
        return self

    @property
    def num_scales(self):
        """ Number of scales in the multiscale array
        """
        return len(self.multiscale) if self.multiscale else 0

    @property
    def num_upsample(self):
        """ Number of upsample operations
        """
        return len(self.upsample) if self.upsample else 0

    @classmethod
    def from_data(cls, data):
        ms_data = cls()
        for k, item in data:
            ms_data[k] = item
        return ms_data


class PairBatch(Pair):

    def __init__(self, batch=None, batch_target=None, **kwargs):
        r"""
        Pair batch for message passing
        """
        self.batch_target = batch_target
        self.batch = batch
        super(PairBatch, self).__init__(**kwargs)
        self.__data_class__ = Batch

    @staticmethod
    def get_instance_masks(data_list, label_offset=1):
        target = []

        for batch_id in range(len(data_list)):
            data = data_list[batch_id]
            label_ids = []
            masks = []
            segment_masks = []
            instance_ids = data.inst_y.unique()

            for instance_id in instance_ids:
                if instance_id == -1:
                    continue

                # TODO is it possible that a ignore class (255) is an instance???
                # instance == -1 ???

                tmp = data.sem_y[data.inst_y == instance_id]
                label_id = tmp[0]

                label_ids.append(label_id)
                masks.append(data.inst_y == instance_id)

            # if len(label_ids) == 0:
            #     return list()
            if len(label_ids) == 0:
                target.append({})
                continue

            label_ids = torch.stack(label_ids)
            masks = torch.stack(masks)

            l = torch.clamp(label_ids - label_offset, min=0)
            target.append({"labels": l, "masks": masks})

        return target
    @staticmethod
    def get_ori_instance_masks(data_list, label_offset=1):
        target = []

        for batch_id in range(len(data_list)):
            data = data_list[batch_id]
            label_ids = []
            masks = []
            segment_masks = []
            instance_ids = data.full_inst_y.unique()

            for instance_id in instance_ids:
                if instance_id == -1:
                    continue

                # TODO is it possible that a ignore class (255) is an instance???
                # instance == -1 ???

                tmp = data.full_sem_y[data.full_inst_y == instance_id]
                label_id = tmp[0]

                label_ids.append(label_id)
                masks.append(data.full_inst_y == instance_id)

            if len(label_ids) == 0:
                target.append({})
                continue

            label_ids = torch.stack(label_ids)
            masks = torch.stack(masks)

            l = torch.clamp(label_ids - label_offset, min=0)
            target.append({"labels": l, "masks": masks})

        return target
    @staticmethod
    def from_data_list(data_list):
        r"""
        from a list of torch_points3d.datasets.change_detection.pair.Pair objects, create
        a batch
        Warning : follow_batch is not here yet...
        """
        assert isinstance(data_list[0], Pair)
        data_list_s, data_list_t = list(map(list, zip(*[data.to_data() for data in data_list])))
        if hasattr(data_list_s[0], 'pair_ind'):
            pair_ind = concatenate_pair_ind(data_list_s, data_list_t)
        else:
            pair_ind = None
        target_s = PairBatch.get_instance_masks(data_list_s)
        target_t = PairBatch.get_instance_masks(data_list_t)
        full_target_s = PairBatch.get_ori_instance_masks(data_list_s)
        full_target_t = PairBatch.get_ori_instance_masks(data_list_t)
        batch_s = Batch.from_data_list(data_list_s)
        batch_t = Batch.from_data_list(data_list_t)
        pair = PairBatch.make_pair(batch_s, batch_t)
        pair.pair_ind = pair_ind

        pair.sem_y = None
        pair.sem_y_target = None
        pair.full_sem_y = None
        pair.full_sem_y_target = None

        # pair.inst_y = None
        # pair.inst_y_target = None

        pair.target_y = target_s
        pair.target_y_target = target_t
        pair.full_target_y = full_target_s
        pair.full_target_y_target = full_target_t
        pair.pos_b = [data_s.sample_ori_pos for data_s in data_list_s]
        pair.pos_b_target = [data_t.sample_ori_pos for data_t in data_list_t]
        pair.full_pos = [data_s.full_pos for data_s in data_list_s]
        pair.full_pos_target = [data_t.full_pos for data_t in data_list_t]

        pair.full_rgb = [data_s.full_rgb for data_s in data_list_s]
        pair.full_rgb_target = [data_t.full_rgb for data_t in data_list_t]

        pair.full_change_y = [data_s.full_change_y for data_s in data_list_s]
        pair.full_change_y_target = [data_t.full_change_y for data_t in data_list_t]
        pair.full_inst_y = [data_s.full_inst_y for data_s in data_list_s]
        pair.full_inst_y_target = [data_t.full_inst_y for data_t in data_list_t]
        return pair.contiguous()

    def to_data_list(self, target=1):
        """
        from a batch of element convert in to list of pair
        """
        batch_size = self.num_graphs
        data_l = []
        for p in range(batch_size):
            # for pair p in the batch
            idx_p = torch.where(self.batch == p)
            idx_target_p = torch.where(self.batch_target == p)
            pair = Pair()
            for key, item in self:
                # if torch.is_tensor(item) and key not in ['area', 'batch', 'batch_target', 'y', 'y0','y1','pred','pred1', "inst_y", "inst_y_target", "target_y", "target_y_target", "change_y_target"
                #                                          'ptr','ptr_target', 'gt_change']:
                if torch.is_tensor(item) and key not in ['area', 'batch', 'batch_target', 'y', 'y0','y1','pred','pred1', "inst_y", "inst_y_target", 'change_target_pred','change_pred',
                                                         'change_y', 'change_y_target', 'ptr','ptr_target', 'gt_change']:
                    if 'target' in key:
                        pair[key] = self[key][idx_target_p]
                    else:
                        pair[key] = self[key][idx_p]
            if hasattr(self, 'y') and self.y is not None:
                if target == 1:
                    pair.y = self.y[idx_target_p]
                if target == 0:
                    pair.y = self.y[idx_p]
            if hasattr(self, 'y0') and self.y0 is not None:
                pair.y0 = self.y0[idx_p]
            if hasattr(self, 'y1') and self.y1 is not None:
                pair.y1 = self.y1[idx_target_p]

            if hasattr(self, 'inst_y') and self.inst_y is not None:
                pair.inst_y = self.inst_y[idx_p]
            if hasattr(self, 'inst_y_target') and self.inst_y_target is not None:
                pair.inst_y_target = self.inst_y_target[idx_target_p]
            # if hasattr(self, 'target_y') and self.target_y is not None:
            #     pair.target_y = self.target_y[idx_p]
            # if hasattr(self, 'target_y_target') and self.target_y_target is not None:
            #     pair.target_y_target = self.target_y_target[idx_target_p]
            if hasattr(self, 'change_y_target') and self.change_y_target is not None:
                pair.change_y_target = self.change_y_target[idx_target_p]
            if hasattr(self, 'change_pred') and self.change_pred is not None:
                pair.change_pred = self.change_pred[idx_p]
            if hasattr(self, 'change_target_pred') and self.change_target_pred is not None:
                pair.change_target_pred = self.change_target_pred[idx_target_p]
            if hasattr(self, 'change_y') and self.change_y is not None:
                pair.change_y = self.change_y[idx_p]
            if hasattr(self, 'change_y_target') and self.change_y_target is not None:
                pair.change_y_target = self.change_y_target[idx_target_p]

            if hasattr(self, 'pred1') and self.pred1 is not None:
                pair.pred1 = self.pred1[idx_target_p]
            if hasattr(self, 'gt_change') and self.gt_change is not None:
                pair.gt_change = self.gt_change[idx_target_p]
            if hasattr(self, 'pred'):
                if target == 1:
                    pair.pred = self.pred[idx_target_p]
                if target == 0:
                    pair.pred = self.pred[idx_p]
            if hasattr(self, 'area'):
                pair.area = self.area[p]
            data_l.append(pair)
        return data_l

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class PairMultiScaleBatch(MultiScalePair):

    def __init__(self, batch=None, batch_target=None, **kwargs):
        self.batch = batch
        self.batch_target = batch_target
        super(PairMultiScaleBatch, self).__init__(**kwargs)
        self.__data_class__ = MultiScaleBatch

    @staticmethod
    def from_data_list(data_list):
        r"""
        from a list of torch_points3d.datasets.registation.pair.Pair objects, create
        a batch
        Warning : follow_batch is not here yet...
        """
        data_list_s, data_list_t = list(map(list, zip(*[data.to_data() for data in data_list])))
        if hasattr(data_list_s[0], 'pair_ind'):
            pair_ind = concatenate_pair_ind(data_list_s, data_list_t).to(torch.long)
        else:
            pair_ind = None
        batch_s = MultiScaleBatch.from_data_list(data_list_s)
        batch_t = MultiScaleBatch.from_data_list(data_list_t)
        pair = PairMultiScaleBatch.make_pair(batch_s, batch_t)
        pair.pair_ind = pair_ind
        return pair.contiguous()


class DensePairBatch(Pair):
    r""" A classic batch object wrapper with :class:`Pair`. Used for Dense Pair Batch (ie pointcloud with fixed size).
    """

    def __init__(self, batch=None, **kwargs):
        super(DensePairBatch, self).__init__(**kwargs)

        self.batch = batch
        self.__data_class__ = Data

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        """
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        # Check if all dimensions matches and we can concatenate data
        # if len(data_list) > 0:
        #    for data in data_list[1:]:
        #        for key in keys:
        #            assert data_list[0][key].shape == data[key].shape

        batch = DensePairBatch()
        batch.__data_class__ = data_list[0].__class__

        for key in keys:
            batch[key] = []

        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        for key in batch.keys:
            item = batch[key][0]
            if (
                    torch.is_tensor(item)
                    or isinstance(item, int)
                    or isinstance(item, float)
            ):
                if key != "pair_ind":
                    batch[key] = torch.stack(batch[key])
            else:
                raise ValueError("Unsupported attribute type")
        # add pair_ind for dense data too
        if hasattr(data_list[0], 'pair_ind'):
            pair_ind = concatenate_pair_ind(data_list, data_list).to(torch.long)
        else:
            pair_ind = None
        batch.pair_ind = pair_ind
        return batch.contiguous()
        # return [batch.x.transpose(1, 2).contiguous(), batch.pos, batch.y.view(-1)]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


def concatenate_pair_ind(list_data_source, list_data_target):
    """
    for a list of pair of indices batched, change the index it refers to wrt the batch index
    Parameters
    ----------
    list_data_source: list[Data]
    list_data_target: list[Data]
    Returns
    -------
    torch.Tensor
        indices of y corrected wrt batch indices


    """

    assert len(list_data_source) == len(list_data_target)
    assert hasattr(list_data_source[0], "pair_ind")
    list_pair_ind = []
    cum_size = torch.zeros(2)
    for i in range(len(list_data_source)):
        size = torch.tensor([len(list_data_source[i].pos),
                             len(list_data_target[i].pos)])
        list_pair_ind.append(list_data_source[i].pair_ind + cum_size)
        cum_size = cum_size + size
    return torch.cat(list_pair_ind, 0)
