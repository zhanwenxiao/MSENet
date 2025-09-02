import torchnet as tnt
from typing import NamedTuple, Dict, Any, List, Tuple
import torch, os
import numpy as np
from torch_scatter import scatter_add
from collections import OrderedDict, defaultdict
from torch_geometric.nn.unpool import knn_interpolate
from pathlib import Path
from torch_points3d.metrics.helper_ply import write_ply


from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.change_detection import IGNORE_LABEL
from torch_points3d.models import model_interface
from torch_geometric.nn import knn
from copy import deepcopy
from uuid import uuid4
import json

CLASS_LABELS = [
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]
VALID_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt["overlaps"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt["min_region_sizes"] = np.array([100])  # 100 for s3dis, scannet
# distance thresholds [m]
opt["distance_threshes"] = np.array([float("inf")])
# distance confidences
opt["distance_confs"] = np.array([-float("inf")])

def assign_instances_for_scan(pred: dict):
    pred_info = make_pred_info(pred)
    gt_ids = pred["gt_ids"]

    # get gt instances
    gt_instances = get_instances(
        gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL
    )
    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt["matched_pred"] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    for uuid in pred_info:
        label_id = int(pred_info[uuid]["label_id"])
        conf = pred_info[uuid]["conf"]
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info[uuid]["mask"]
        assert len(pred_mask) == len(gt_ids)
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt["min_region_sizes"][0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance["uuid"] = uuid
        pred_instance["pred_id"] = num_pred_instances
        pred_instance["label_id"] = label_id
        pred_instance["vert_count"] = num
        pred_instance["confidence"] = conf
        pred_instance["void_intersection"] = np.count_nonzero(
            np.logical_and(bool_void, pred_mask)
        )

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(
                np.logical_and(gt_ids == gt_inst["instance_id"], pred_mask)
            )
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)
        pred_instance["matched_gt"] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt

def evaluate_matches(matches):
    overlaps = opt["overlaps"]
    min_region_sizes = [opt["min_region_sizes"][0]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    # results: class x overlap
    ap = np.zeros(
        (len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float
    )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)
    ):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]["pred"]:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]["pred"][label_name]:
                            if "uuid" in p:
                                pred_visited[p["uuid"]] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]["pred"][label_name]
                    gt_instances = matches[m]["gt"][label_name]
                    # filter groups in ground truth
                    gt_instances = [
                        gt
                        for gt in gt_instances
                        if gt["instance_id"] >= 1000
                        and gt["vert_count"] >= min_region_size
                        and gt["med_dist"] <= distance_thresh
                        and gt["dist_conf"] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt["matched_pred"])
                        for pred in gt["matched_pred"]:
                            # greedy assignments
                            if pred_visited[pred["uuid"]]:
                                continue
                            overlap = float(pred["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - pred["intersection"]
                            )
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred["matched_gt"]:
                            overlap = float(gt["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - gt["intersection"]
                            )
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred["void_intersection"]
                            for gt in pred["matched_gt"]:
                                # group?
                                if gt["instance_id"] < 1000:
                                    num_ignore += gt["intersection"]
                                # small ground truth instances
                                if (
                                    gt["vert_count"] < min_region_size
                                    or gt["med_dist"] > distance_thresh
                                    or gt["dist_conf"] < distance_conf
                                ):
                                    num_ignore += gt["intersection"]
                            proportion_ignore = (
                                float(num_ignore) / pred["vert_count"]
                            )
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(
                        y_score_sorted, return_index=True
                    )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = (
                        y_true_sorted_cumsum[-1]
                        if len(y_true_sorted_cumsum) > 0
                        else 0
                    )
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(
                        recall_for_conv[0], recall_for_conv
                    )
                    recall_for_conv = np.append(recall_for_conv, 0.0)

                    stepWidths = np.convolve(
                        recall_for_conv, [-0.5, 0, 0.5], "valid"
                    )
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float("nan")
                ap[di, li, oi] = ap_current
    return ap

def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlaps"], 0.5))
    o25 = np.where(np.isclose(opt["overlaps"], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt["overlaps"], 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(
            aps[d_inf, li, oAllBut25]
        )
        avg_dict["classes"][label_name]["ap50%"] = np.average(
            aps[d_inf, li, o50]
        )
        avg_dict["classes"][label_name]["ap25%"] = np.average(
            aps[d_inf, li, o25]
        )
    return avg_dict

def make_pred_info(pred: dict):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    pred_info = {}
    assert (
        pred["pred_classes"].shape[0]
        == pred["pred_scores"].shape[0]
        == pred["pred_masks"].shape[1]
    )
    for i in range(len(pred["pred_classes"])):
        info = {}
        info["label_id"] = pred["pred_classes"][i]
        info["conf"] = pred["pred_scores"][i]
        info["mask"] = pred["pred_masks"][:, i]
        pred_info[uuid4()] = info  # we later need to identify these objects
    return pred_info

def get_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances

class iCDTracker(BaseTracker):
    def __init__(
        self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, ignore_label: int = IGNORE_LABEL
    ):
        """ This is a generic tracker for segmentation tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(iCDTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._change_classes = dataset.change_classes
        self._sem_classes = dataset.sem_classes
        self._ignore_label = ignore_label
        self.correct_class_num = False
        self._dataset = dataset
        self.reset(stage)
        self._metric_func = {
            "miou": max,
            "miou_ch":max, #miou over classes of change
            "macc": max,
            "acc": max,
            "loss": min,
            "map": max,
        }  # Those map subsentences to their optimization functions
        ##### Loss Weight #####
        self.loss_weight={"change": 1.0, "inst": 1.0}
        #######################

    def reset(self, stage="train"):
        super().reset(stage=stage)
        # self._num_classes=3
        self.change_confusion_matrix = ConfusionMatrix(self._change_classes + 1) #ConfusionMatrix(3)#

        self.apiou_confusion_matrix = ConfusionMatrix(self._change_classes + 1)
        self.ap50iou_confusion_matrix = ConfusionMatrix(self._change_classes + 1)
        self.ap25iou_confusion_matrix = ConfusionMatrix(self._change_classes + 1)

        self.change_acc = 0
        self.change_macc = 0
        self.change_miou = 0
        self.change_miou_per_class = {}
        self.change_miou_ch = 0
        self.change_loss = 0

        self._mAP = 0
        self._mAP25 = 0
        self._mAP50 = 0

        self._mAP_cmiou = 0
        self._mAP25_cmiou = 0
        self._mAP50_cmiou = 0

        # if self._stage == 'test':
        #     self._ds = self._dataset.test_data
        #     size = self._ds.grid_regular_centers.shape[0]
        # elif self._stage == 'val':
        #     self._ds = self._dataset.val_data
        #     size = self._ds._sample_per_epoch
        # else:
        #     self._ds = self._dataset.train_data
        #     size = self._ds._sample_per_epoch
        # self.inst_preds = [None] * size * 2
        #
        #
        # self.inst_idx = 0

        self.inst_preds = []
        # self.inst_preds = dict()
        # self.inst_idx = 0
        self.export = False #True
        self.area_num = 0

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: model_interface.TrackerInterface, data=None, conv_classes=None, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        if not self._dataset.has_inst_labels(self._stage):
            return

        super().track(model)
        outputs = model.get_output()
        targets = model.get_labels()

        try:
            self._loss = model.loss.item()
        except:
            self._loss = 0

        inst_gt_path = os.path.join(os.getcwd(), "output", "instance_segmentation", "groundtruth")
        inst_pred_path = os.path.join(os.getcwd(), "output", "instance_segmentation", "pred")
        ch_gt_path = os.path.join(os.getcwd(), "output", "change_detection", "groundtruth")
        ch_pred_path = os.path.join(os.getcwd(), "output", "change_detection", "pred")

        area = model.input0.file_names
        change_outputs0, change_outputs1, mask_outputs0, mask_outputs1, de_c_output0, de_c_output1 = outputs
        change_targets0, change_targets1, mask_targets0, mask_targets1, full_target_y, full_target_y_target, full_change, full_change_target, full_inst_y, full_inst_y_target, full_pos, full_pos_target, full_rgb, full_rgb_target, \
        pos_0, pos_1, inst_targets0, inst_targets1, de_c_target0, de_c_target1 = targets
        self._compute_change_metrics_list([change_outputs0, change_outputs1], [change_targets0, change_targets1], conv_classes)
        if len(mask_targets0) != 0 and self._stage != "train":
            inst_offset = self._compute_inst_metrics_list(de_c_output0, de_c_target0, mask_outputs0, mask_targets0, inst_targets0, full_target_y, full_change, full_inst_y, full_pos, pos_0, full_rgb, 0, area, inst_gt_path, inst_pred_path, ch_gt_path, ch_pred_path, 0)
        if len(mask_targets1) != 0 and self._stage != "train":
            _ = self._compute_inst_metrics_list(de_c_output1, de_c_target1, mask_outputs1, mask_targets1, inst_targets1, full_target_y_target, full_change_target, full_inst_y_target, full_pos_target, pos_1, full_rgb_target, 1, area, inst_gt_path, inst_pred_path, ch_gt_path, ch_pred_path, inst_offset)

        return outputs, targets

    def export_pred_inst(self, pred_masks, scores, pred_classes, file_names, base_path, pred_dir_path, dir_name):
        file_name = file_names
        pred_mask_path = f"{base_path}/pred_mask"
        Path(pred_mask_path).mkdir(parents=True, exist_ok=True)

        with open(f"{pred_dir_path}/{file_name}.txt", "w") as fout:
        # with open(output_file_name, "w") as fout:
            real_id = -1
            for instance_id in range(len(pred_classes)):
                real_id += 1
                pred_class = pred_classes[instance_id]
                score = scores[instance_id]
                mask = pred_masks[:, instance_id].astype("uint8")

                if score > 0.5: #self.config.general.export_threshold:
                    Path(os.path.join(pred_mask_path, dir_name)).mkdir(parents=True, exist_ok=True)

                    # reduce the export size a bit. I guess no performance difference
                    np.savetxt(
                        f"{pred_mask_path}/{dir_name}/{file_name}_{real_id}.txt",
                        mask,
                        fmt="%d",
                    )
                    fout.write(
                        f"pred_mask/{dir_name}/{file_name}_{real_id}.txt {pred_class} {score}\n"
                    )

    def export_pred_change(self, xyz, rgb, pred_change, gt_change, change_path, filename):
        xyz = xyz.detach().cpu().numpy()
        rgb = rgb.detach().cpu().numpy()

        output_path = os.path.join(change_path, filename + ".ply")
        write_ply(output_path, [xyz, rgb, pred_change, gt_change],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'scalar_label_ch', 'scalar_label_ch_gt'])

    def export_gt(self, xyz, rgb, inst, change, change_gt_path, filename):
        xyz = xyz.detach().cpu().numpy()
        rgb = rgb.detach().cpu().numpy()
        inst = inst.detach().cpu().numpy()
        sem = np.where(inst == -1, 0, 1)

        block = np.concatenate([xyz, rgb, np.expand_dims(sem, axis=-1), np.expand_dims(inst, axis=-1), np.expand_dims(change, axis=-1)], axis=-1)
        output_path = os.path.join(change_gt_path, filename + ".npy")
        np.save(output_path, block.astype(np.float32))

    def _compute_change_metrics_list(self, outputs, labels, conv_classes=None):
        output0, output1 = outputs
        labels0, labels1 = labels

        if self._ignore_label != None:
            # mask = labels != self._ignore_label
            # outputs = outputs[mask]
            # labels = labels[mask]

            mask0 = labels0 != self._ignore_label
            output0 = output0[mask0]
            labels0 = labels0[mask0]

            mask1 = labels1 != self._ignore_label
            output1 = output1[mask1]
            labels1 = labels1[mask1]

        output0, output1 = self._convert(output0), self._convert(output1)
        labels0, labels1 = self._convert(labels0), self._convert(labels1)
        pred0, pred1 = np.argmax(output0, 1), np.argmax(output1, 1)
        if len(labels) == 0:
            return
        if conv_classes is not None:
            pred0 = torch.from_numpy(conv_classes[pred0])
            pred1 = torch.from_numpy(conv_classes[pred1])
            # for key in conv_classes:
            #     pred[pred == key] = conv_classes[key]
        assert output0.shape[0] == len(labels0)
        assert output1.shape[0] == len(labels1)
        self.change_confusion_matrix.count_predicted_batch_list(labels0, pred0, labels1, pred1)

        self.change_acc = 100 * self.change_confusion_matrix.get_overall_accuracy()
        self.change_macc = 100 * self.change_confusion_matrix.get_mean_class_accuracy()
        self.change_miou = 100 * self.change_confusion_matrix.get_average_intersection_union()
        class_iou, present_class = self.change_confusion_matrix.get_intersection_union_per_class()
        self.change_miou_per_class = {
            i: "{:.2f}".format(100 * v)
            for i, v in enumerate(class_iou)
        }
        self.change_miou_ch = 100*np.mean(class_iou[1:])
        class_acc = self.change_confusion_matrix.confusion_matrix.diagonal() / self.change_confusion_matrix.confusion_matrix.sum(axis=1)
        self.change_acc_per_class = {
            k: "{:.2f}".format(100 * v)
            for k, v in enumerate(class_acc)
        }

        #### Inst Metrics ####

    def _compute_inst_change_metrics_list(self, outputs, labels, output, target_full_res, inst_targets, full_target_y,
                                       full_change, full_inst, full_pos, pos, full_rgb, time_series, area_names,
                                       inst_gt_path, inst_pred_path, ch_gt_path, ch_pred_path, inst_offset, conv_classes=None):

        if self.export:
            all_pred_changes = list()
            all_gt_changes = list()
            decoder_id = -1

            prediction = output["aux_outputs"]
            prediction.append(
                {
                    "pred_logits": output["pred_logits"],
                    "pred_masks": output["pred_masks"],
                }
            )
            decoder_id = -1
            prediction[decoder_id][
                "pred_logits"
            ] = torch.functional.F.softmax(
                prediction[decoder_id]["pred_logits"], dim=-1
            )[
                ..., :-1
            ]

            #### Inst Metrics ####
            for bid in range(len(prediction[decoder_id]["pred_masks"])):
                if self._stage == "test":
                    _pos = pos[bid]
                    _full_pos = full_pos[bid]
                    change = knn_interpolate(c_output[bid].detach().cpu(), pos[bid].detach().cpu(), full_pos[bid].detach().cpu(), k=1).numpy()
                    change = np.argmax(change, axis=-1).astype(np.int8)
                    change_target = full_change[bid].detach().cpu().numpy().astype(np.int8)
                else:
                    change = c_output[bid].detach().cpu().max(dim=-1)[1].numpy()  # change = torch.functional.F.softmax(change[bid], dim=-1).detach().cpu()
                    change_target = c_target[bid].detach().cpu().numpy()

                all_pred_changes.append(change)
                all_gt_changes.append(change_target)

            inst_num = len(prediction[decoder_id]["pred_masks"])
            self.area_num -= inst_offset
            for bid in range(len(prediction[decoder_id]["pred_masks"])):
                area_name = os.path.basename(os.path.dirname(area_names[bid]))

                ch_gt_area_path = os.path.join(ch_gt_path, area_name)
                ch_pred_area_path = os.path.join(ch_pred_path, area_name)
                if not os.path.exists(ch_gt_area_path):
                    os.makedirs(ch_gt_area_path)
                if not os.path.exists(ch_pred_area_path):
                    os.makedirs(ch_pred_area_path)

                file_names = "pointCloud" + str(time_series) + "_" + str(self.area_num + bid)
                self.export_pred_change(full_pos[bid], full_rgb[bid], all_pred_changes[bid], all_gt_changes[bid],
                                            ch_pred_area_path, file_names)
                self.export_gt(full_pos[bid], full_rgb[bid], inst_y[bid], all_gt_changes[bid], ch_gt_area_path,
                               file_names)
            self.area_num += inst_num

    def _compute_change_metrics(self, outputs, labels, conv_classes=None):
        if self._ignore_label != None:
            mask = labels != self._ignore_label
            outputs = outputs[mask]
            labels = labels[mask]

        outputs = self._convert(outputs)
        labels = self._convert(labels)
        pred = np.argmax(outputs, 1)
        if len(labels) == 0:
            return
        if conv_classes is not None:
            pred = torch.from_numpy(conv_classes[pred])
            # for key in conv_classes:
            #     pred[pred == key] = conv_classes[key]
        assert outputs.shape[0] == len(labels)
        self.change_confusion_matrix.count_predicted_batch(labels, pred)

        self.change_acc = 100 * self.change_confusion_matrix.get_overall_accuracy()
        self.change_macc = 100 * self.change_confusion_matrix.get_mean_class_accuracy()
        self.change_miou = 100 * self.change_confusion_matrix.get_average_intersection_union()
        class_iou, present_class = self.change_confusion_matrix.get_intersection_union_per_class()
        self.change_miou_per_class = {
            i: "{:.2f}".format(100 * v)
            for i, v in enumerate(class_iou)
        }
        self.change_miou_ch = 100*np.mean(class_iou[1:])
        class_acc = self.change_confusion_matrix.confusion_matrix.diagonal() / self.change_confusion_matrix.confusion_matrix.sum(axis=1)
        self.change_acc_per_class = {
            k: "{:.2f}".format(100 * v)
            for k, v in enumerate(class_acc)
        }

    def _compute_inst_metrics_list(self, c_output, c_target, output, target_full_res, inst_targets, full_target_y,
                                   full_change, full_inst, full_pos, pos, full_rgb, time_series, area_names, inst_gt_path, inst_pred_path, ch_gt_path, ch_pred_path, inst_offset):
        label_offset = 1
        prediction = output["aux_outputs"]
        prediction.append(
            {
                "pred_logits": output["pred_logits"],
                "pred_masks": output["pred_masks"],
            }
        )
        decoder_id = -1
        prediction[decoder_id][
            "pred_logits"
        ] = torch.functional.F.softmax(
            prediction[decoder_id]["pred_logits"], dim=-1
        )[
            ..., :-1
        ]

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_pred_changes = list()
        all_gt_changes = list()
        all_knn_idx = list()

        for bid in range(len(prediction[decoder_id]["pred_masks"])):
            masks = (
                prediction[decoder_id]["pred_masks"][bid]
                    .detach()
                    .cpu()
            )

            scores, masks, classes, heatmap = self.get_mask_and_scores(
                prediction[decoder_id]["pred_logits"][bid]
                    .detach()
                    .cpu(),
                masks,
                prediction[decoder_id]["pred_logits"][bid].shape[
                    0
                ],
                    self._sem_classes - 1,
                device="cpu",
            )

            if self._stage == "test":
                _pos = pos[bid]
                _full_pos = full_pos[bid]
                masks = knn_interpolate(masks, pos[bid].detach().cpu(), full_pos[bid].detach().cpu(), k=1).numpy().astype(np.int8)
                change = knn_interpolate(c_output[bid].detach().cpu(), pos[bid].detach().cpu(), full_pos[bid].detach().cpu(), k=1).numpy()
                change = np.argmax(change, axis=-1).astype(np.int8)
                change_target = full_change[bid].detach().cpu().numpy().astype(np.int8)
            else:
                masks = masks.numpy()
                change = c_output[bid].detach().cpu().max(dim=-1)[1].numpy()  # change = torch.functional.F.softmax(change[bid], dim=-1).detach().cpu()
                change_target = c_target[bid].detach().cpu().numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]
            sorted_masks = masks[:, sort_scores_index]

            all_pred_classes.append(sort_classes)
            all_pred_masks.append(sorted_masks)
            all_pred_scores.append(sort_scores_values)
            all_pred_changes.append(change)
            all_gt_changes.append(change_target)


        if self._stage == "test":
            label_y = full_target_y
            inst_y = full_inst
        else:
            label_y = target_full_res
            inst_y = inst_targets

        inst_num = len(prediction[decoder_id]["pred_masks"])
        self.area_num -= inst_offset
        for bid in range(len(prediction[decoder_id]["pred_masks"])):

            all_pred_classes[
                bid
            ] = self._remap_model_output(
                all_pred_classes[bid].cpu() + label_offset
            )

            if self.export:
                area_name = os.path.basename(os.path.dirname(area_names[bid]))
                inst_gt_area_path = os.path.join(inst_gt_path, area_name)
                inst_pred_area_path = os.path.join(inst_pred_path, area_name)
                if not os.path.exists(inst_gt_area_path):
                    os.makedirs(inst_gt_area_path)
                if not os.path.exists(inst_pred_area_path):
                    os.makedirs(inst_pred_area_path)

                ch_gt_area_path = os.path.join(ch_gt_path, area_name)
                ch_pred_area_path = os.path.join(ch_pred_path, area_name)
                if not os.path.exists(ch_gt_area_path):
                    os.makedirs(ch_gt_area_path)
                if not os.path.exists(ch_pred_area_path):
                    os.makedirs(ch_pred_area_path)

                file_names = "pointCloud" + str(time_series) + "_" + str(self.area_num + bid)
                if "labels" in label_y[bid]:
                    self.export_pred_inst(all_pred_masks[bid], all_pred_scores[bid], all_pred_classes[bid], file_names,
                                          inst_pred_path, inst_pred_area_path, area_name)
                self.export_pred_change(full_pos[bid], full_rgb[bid], all_pred_changes[bid], all_gt_changes[bid],
                                            ch_pred_area_path, file_names)
                self.export_gt(full_pos[bid], full_rgb[bid], inst_y[bid], all_gt_changes[bid], inst_gt_area_path,
                               file_names)
                self.export_gt(full_pos[bid], full_rgb[bid], inst_y[bid], all_gt_changes[bid], ch_gt_area_path,
                               file_names)

            if "labels" not in label_y[bid]:
                continue

            label_y[bid][
                "labels"
            ] = self._remap_model_output(
                label_y[bid]["labels"].cpu() + label_offset
            )
            self.inst_preds.append({
                "pred_masks": all_pred_masks[bid],
                "pred_scores": all_pred_scores[bid],
                "pred_classes": all_pred_classes[bid],
                "gt_target": inst_y[bid],
                "pred_changes": all_pred_changes[bid],
                "gt_changes": all_gt_changes[bid],
                "time": time_series,
            })

        self.area_num += inst_num
        return inst_num

    def _compute_inst_metrics(self, coord, out_masks, out_logits, target_masks, target_logits, ori_tgt, batch_l):
        ori_sem_label = ori_tgt[:, 0]
        ori_inst_label = ori_tgt[:, 1]

        _batch = batch_l.unique()

        all_pred_classes = list()
        all_pred_masks = list()
        all_pred_scores = list()
        all_heatmaps = list()
        all_query_pos = list()

        offset_coords_idx = 0

        for bid in _batch:
            b_index = torch.argwhere(batch_l == bid).squeeze(-1)
            _b_gt_masks = target_masks[bid]
            _b_gt_logits = target_logits[bid]
            if isinstance(_b_gt_masks, List):
                continue
            _b_masks = out_masks[b_index]
            _b_logits = out_logits[bid]
            _b_logits = torch.functional.F.softmax(_b_logits, dim=-1)[..., 1:]

            scores, masks, classes, heatmap = self.get_mask_and_scores(
                _b_logits.cpu(),
                _b_masks,
                _b_logits.shape[0],
                self._sem_classes - 1,
                device="cpu",
            )

            masks = masks.cpu().numpy()
            heatmap = heatmap.cpu().numpy()

            sort_scores = scores.sort(descending=True)
            sort_scores_index = sort_scores.indices.cpu().numpy()
            sort_scores_values = sort_scores.values.cpu().numpy()
            sort_classes = classes[sort_scores_index]

            sorted_masks = masks[:, sort_scores_index]
            sorted_heatmap = heatmap[:, sort_scores_index]

            all_pred_classes.append(sort_classes)
            all_pred_masks.append(sorted_masks)
            all_pred_scores.append(sort_scores_values)
            all_heatmaps.append(sorted_heatmap)

        flag = 0
        for bid in _batch:
            b_index = torch.argwhere(batch_l == bid).squeeze(-1)
            _b_gt_masks = target_masks[bid]
            _b_gt_logits = target_logits[bid]
            if isinstance(_b_gt_masks, List):
                flag += 1
                continue
            _b_ori_sem_label = ori_sem_label[b_index].cpu().numpy()
            _b_ori_inst_label = ori_inst_label[b_index].cpu().numpy()
            new_instance_ids = np.unique(
                _b_ori_inst_label, return_inverse=True
            )[1]
            assert (
                    new_instance_ids.max() < 1000
            ), "we cannot encode when there are more than 999 instances in a block"
            gt_data = (_b_ori_sem_label) * 1000 + new_instance_ids

            _b_coor = coord[b_index]
            _b_masks = out_masks[b_index]
            _b_logits = out_logits[bid]

            b_index = bid - flag
            bid = b_index

            self.inst_preds.append({
                "pred_masks": all_pred_masks[bid],
                "pred_scores": all_pred_scores[bid],
                "pred_classes": all_pred_classes[bid],
                "gt_ids": gt_data
            })

    def _remap_model_output(self, output):
        # output = np.array(output)
        # output_remapped = output.copy()
        # for i, k in enumerate(self.label_info.keys()):
        #     output_remapped[output == i] = k
        return np.array(output)

    def get_mask_and_scores(
        self, mask_cls, mask_pred, num_queries=100, num_classes=18, device=None, topk_per_image=100
    ):
        if device is None:
            device = self.device
        labels = (
            torch.arange(num_classes, device=device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        if topk_per_image != -1:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                topk_per_image, sorted=True
            )
        else:
            scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
                num_queries, sorted=True
            )

        labels_per_query = labels[topk_indices]
        topk_indices = topk_indices // num_classes
        mask_pred = mask_pred[:, topk_indices]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()

        mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6
        )
        score = scores_per_query.to(mask_scores_per_image.device) * mask_scores_per_image
        classes = labels_per_query

        return score, result_pred_mask, classes, heatmap

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)
        metrics["{}_change_loss".format(self._stage)] = self.change_loss
        metrics["{}_change_acc".format(self._stage)] = self.change_acc
        metrics["{}_change_macc".format(self._stage)] = self.change_macc
        metrics["{}_change_miou".format(self._stage)] = self.change_miou
        metrics["{}_change_miou_ch".format(self._stage)] = self.change_miou_ch
        # metrics["{}_iou_per_class".format(self._stage)] = self.change_miou_per_class

        metrics["{}_map".format(self._stage)] = self._mAP
        metrics["{}_map50".format(self._stage)] = self._mAP50
        metrics["{}_map25".format(self._stage)] = self._mAP25

        metrics["{}_map_ciou".format(self._stage)] = self._mAP_cmiou
        metrics["{}_map50_ciou".format(self._stage)] = self._mAP25_cmiou
        metrics["{}_map25_ciou".format(self._stage)] = self._mAP50_cmiou

        # if verbose:
        #     metrics["{}_iou_per_class".format(self._stage)] = self._miou_per_class
        #     metrics["{}_acc_per_class".format(self._stage)] = self._acc_per_class
        return metrics

    @property
    def metric_func(self):
        return self._metric_func

class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if instance_id == -1:
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(
            self.get_instance_verts(mesh_vert_instances, instance_id)
        )

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(
            self, default=lambda o: o.__dict__, sort_keys=True, indent=4
        )

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"] = self.label_id
        dict["vert_count"] = self.vert_count
        dict["med_dist"] = self.med_dist
        dict["dist_conf"] = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id = int(data["instance_id"])
        self.label_id = int(data["label_id"])
        self.vert_count = int(data["vert_count"])
        if "med_dist" in data:
            self.med_dist = float(data["med_dist"])
            self.dist_conf = float(data["dist_conf"])

    def __str__(self):
        return "(" + str(self.instance_id) + ")"

# import torchnet as tnt
# from typing import NamedTuple, Dict, Any, List, Tuple, Optional
# import torch
# import numpy as np
# from torch_scatter import scatter_add
# from collections import OrderedDict, defaultdict
#
# from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
# from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
# from torch_points3d.datasets.change_detection import IGNORE_LABEL
# from torch_points3d.models import model_interface
#
# from torch_points3d.models.model_interface import TrackerInterface
# from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
# from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
# from torch_points3d.models.panoptic.structures import PanopticResults, PanopticLabels
# from torch_points_kernels import instance_iou
# from .box_detection.ap import voc_ap
#
# if torch.cuda.is_available():
#     import torch_points_kernels.points_cuda as tpcuda
#
# class iCDTracker(BaseTracker):
#     def __init__(
#         self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, ignore_label: int = IGNORE_LABEL
#     ):
#         """ This is a generic tracker for segmentation tasks.
#         It uses a confusion matrix in the back-end to track results.
#         Use the tracker to track an epoch.
#         You can use the reset function before you start a new epoch
#
#         Arguments:
#             dataset  -- dataset to track (used for the number of classes)
#
#         Keyword Arguments:
#             stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
#             wandb_log {str} --  Log using weight and biases
#         """
#         super(iCDTracker, self).__init__(stage, wandb_log, use_tensorboard)
#         self._change_classes = dataset.change_classes
#         self._sem_classes = dataset.sem_classes
#         self._ignore_label = ignore_label
#         self.correct_class_num = False
#
#         self._dataset = dataset
#         self.reset(stage)
#         self._metric_func = {
#             "miou": max,
#             "miou_ch":max, #miou over classes of change
#             "macc": max,
#             "acc": max,
#             "loss": min,
#             "map": max,
#         }  # Those map subsentences to their optimization functions
#
#     def reset(self, stage="train"):
#         super().reset(stage=stage)
#         # self._num_classes=3
#         self.change_confusion_matrix = ConfusionMatrix(self._change_classes) #ConfusionMatrix(3)#
#         self.change_acc = 0
#         self.change_macc = 0
#         self.change_miou = 0
#         self.change_miou_per_class = {}
#         self.change_miou_ch = 0
#         self.change_loss = 0
#
#         self.inst_confusion_matrix = ConfusionMatrix(2)  # ConfusionMatrix(3)#
#         self.inst_pos = tnt.meter.AverageValueMeter()
#         self.inst_neg = tnt.meter.AverageValueMeter()
#         self.inst_acc_meter = tnt.meter.AverageValueMeter()
#         self.inst_ap_meter = InstanceAPMeter()
#         self._scan_id_offset = 0
#         self.inst_rec: Dict[str, float] = {}
#         self.inst_ap: Dict[str, float] = {}
#         self.inst_iou_per_class = {}
#
#     @staticmethod
#     def detach_tensor(tensor):
#         if torch.torch.is_tensor(tensor):
#             tensor = tensor.detach()
#         return tensor
#
#     @property
#     def confusion_matrix(self):
#         return self.change_confusion_matrix.confusion_matrix
#
#     def track(self, model: model_interface.TrackerInterface, data=None, conv_classes=None, **kwargs):
#         """ Add current model predictions (usually the result of a batch) to the tracking
#         """
#         if not self._dataset.has_labels(self._stage):
#             return
#
#         super().track(model)
#         batch, batch_target, ori_tgt0, ori_tgt1 = model.input0.batch, model.input1.batch, model.input0.y, model.input1.y
#         change_output, inst_output0, inst_output1, indice0, indice1 = model.get_output()
#         inst_output0, logits_output0, inst_output1, logits_output1 = inst_output0["pred_masks"], inst_output0["pred_logits"], inst_output1["pred_masks"], inst_output1["pred_logits"]
#         change_label, sem_label0, inst_label0, sem_label1, inst_label1 = model.get_labels()
#         try:
#             self._loss = model.loss.item()
#         except:
#             self._loss = 0
#
#         ########## CHANGE TRACK ##########
#         if type(change_output) == list:
#             self._compute_change_metrics_list(change_output, change_label, conv_classes)
#         else:
#             self._compute_change_metrics(change_output, change_label, conv_classes)
#         ########## CHANGE TRACK ##########
#
#         ########## INST TRACK ##########
#         self._compute_inst_metrics(inst_output0, logits_output0, inst_label0, sem_label0, ori_tgt0, indice0, batch)
#         self._compute_inst_metrics(inst_output1, logits_output1, inst_label1, sem_label1, ori_tgt1, indice1, batch_target)
#         ########## INST TRACK ##########
#
#         return change_output, change_label
#
#     def _compute_change_metrics_list(self, outputs, labels, conv_classes=None):
#         output0, output1 = outputs
#         labels0, labels1 = labels
#
#         if self._ignore_label != None:
#             # mask = labels != self._ignore_label
#             # outputs = outputs[mask]
#             # labels = labels[mask]
#
#             mask0 = labels0 != self._ignore_label
#             output0 = output0[mask0]
#             labels0 = labels0[mask0]
#
#             mask1 = labels1 != self._ignore_label
#             output1 = output1[mask1]
#             labels1 = labels1[mask1]
#
#         output0, output1 = self._convert(output0), self._convert(output1)
#         labels0, labels1 = self._convert(labels0), self._convert(labels1)
#         pred0, pred1 = np.argmax(output0, 1), np.argmax(output1, 1)
#         if len(labels) == 0:
#             return
#         if conv_classes is not None:
#             pred0 = torch.from_numpy(conv_classes[pred0])
#             pred1 = torch.from_numpy(conv_classes[pred1])
#             # for key in conv_classes:
#             #     pred[pred == key] = conv_classes[key]
#         assert output0.shape[0] == len(labels0)
#         assert output1.shape[0] == len(labels1)
#         self.change_confusion_matrix.count_predicted_batch_list(labels0, pred0, labels1, pred1)
#
#         self.change_acc = 100 * self.change_confusion_matrix.get_overall_accuracy()
#         self.change_macc = 100 * self.change_confusion_matrix.get_mean_class_accuracy()
#         self.change_miou = 100 * self.change_confusion_matrix.get_average_intersection_union()
#         class_iou, present_class = self.change_confusion_matrix.get_intersection_union_per_class()
#         self.change_miou_per_class = {
#             i: "{:.2f}".format(100 * v)
#             for i, v in enumerate(class_iou)
#         }
#         self.change_miou_ch = 100*np.mean(class_iou[1:])
#         class_acc = self.change_confusion_matrix.confusion_matrix.diagonal() / self.change_confusion_matrix.confusion_matrix.sum(axis=1)
#         self.change_acc_per_class = {
#             k: "{:.2f}".format(100 * v)
#             for k, v in enumerate(class_acc)
#         }
#
#         #### Inst Metrics ####
#
#     def _compute_change_metrics(self, outputs, labels, conv_classes=None):
#         if self._ignore_label != None:
#             mask = labels != self._ignore_label
#             outputs = outputs[mask]
#             labels = labels[mask]
#
#         outputs = self._convert(outputs)
#         labels = self._convert(labels)
#         pred = np.argmax(outputs, 1)
#         if len(labels) == 0:
#             return
#         if conv_classes is not None:
#             pred = torch.from_numpy(conv_classes[pred])
#             # for key in conv_classes:
#             #     pred[pred == key] = conv_classes[key]
#         assert outputs.shape[0] == len(labels)
#         self.change_confusion_matrix.count_predicted_batch(labels, pred)
#
#         self.change_acc = 100 * self.change_confusion_matrix.get_overall_accuracy()
#         self.change_macc = 100 * self.change_confusion_matrix.get_mean_class_accuracy()
#         self.change_miou = 100 * self.change_confusion_matrix.get_average_intersection_union()
#         class_iou, present_class = self.change_confusion_matrix.get_intersection_union_per_class()
#         self.change_miou_per_class = {
#             i: "{:.2f}".format(100 * v)
#             for i, v in enumerate(class_iou)
#         }
#         self.change_miou_ch = 100*np.mean(class_iou[1:])
#         class_acc = self.change_confusion_matrix.confusion_matrix.diagonal() / self.change_confusion_matrix.confusion_matrix.sum(axis=1)
#         self.change_acc_per_class = {
#             k: "{:.2f}".format(100 * v)
#             for k, v in enumerate(class_acc)
#         }
#
#     def _compute_inst_metrics(self, out_masks, out_logits, target_masks, target_logits, ori_tgt, indices, batch_l):
#         sem_tgt, inst_tgt = ori_tgt[:, 0], ori_tgt[:, 1]
#         iou_threshold = 0.25
#         # for batch_id, (map_id, target_id) in enumerate(indices):
#         #     b_index = torch.argwhere(batch_l == batch_id).squeeze(-1)
#         #     _b_gt_masks = target_masks[batch_id]
#         #     _b_gt_logits = target_logits[batch_id]
#         #     if isinstance(_b_gt_masks, List):
#         #         continue
#         #     _b_masks = out_masks[b_index]
#         #     _b_logits = out_logits[batch_id]
#         #
#         #     masks = _b_masks[:, map_id].T
#         #     masks = (masks > 0).long()
#         #     target_masks = _b_gt_masks[target_id].long()
#         #
#         #     sems = _b_logits[map_id, :]
#         #     sems = torch.functional.F.softmax(sems, dim=-1).max(1)[1]
#         #     target_sems = _b_gt_logits[target_id]
#         #
#         #     inst_mious = self._get_per_cluster_iou(target_masks, masks)
#         #
#         #     pass
#
#         pred_clusters, gt_clusters = [], []
#         for batch_id, (map_id, target_id) in enumerate(indices):
#             b_index = torch.argwhere(batch_l == batch_id).squeeze(-1)
#             _b_gt_masks = target_masks[batch_id]
#             _b_gt_logits = target_logits[batch_id]
#             if isinstance(_b_gt_masks, List):
#                 continue
#             _b_masks = out_masks[b_index]
#             _b_logits = out_logits[batch_id]
#
#             _b_inst_tgt = inst_tgt[b_index]
#             _b_sem_tgt = sem_tgt[b_index]
#
#             masks = _b_masks[:, map_id].T
#             masks = (masks > 0).float()
#             target_masks = _b_gt_masks[target_id]
#
#             sems = _b_logits[map_id, :]
#             sems = torch.functional.F.softmax(sems, dim=-1).max(1)[1]
#             target_sems = _b_gt_logits[target_id]
#
#             output_logits, target_logits, output_masks = self._get_data(masks, target_masks, sems, target_sems)
#
#             num_instances = target_sems.shape[0]
#             for i in range(num_instances):
#                 pred_clusters.append(
#                     _Instance(
#                         classname=target_sems[i],
#                         score=scores[i].item(),
#                         indices=indices,
#                         scan_id=self._scan_id_offset + batch_id,
#                     )
#                 )
#                 gt_clusters.append(
#                     _Instance(
#                         classname=target_sems[i],
#                         score=-1,
#                         indices=output_masks,
#                         scan_id=self._scan_id_offset + batch_id,
#                     )
#                 )
#
#             pass
#
#         self._scan_id_offset += batch_l[-1].item() + 1
#
#
#         # num_instances = 0
#         # for target_logit in target_logits:
#         #     if isinstance(target_logit, List):
#         #         continue
#         #     num_instances += target_logit.shape[0]
#         #
#         # iou_threshold = 0.25
#         # labels = [inst_tgt, target_logits, num_instances]
#         # tp, fp, acc = self._compute_acc(
#         #     out_masks, out_logits, labels, batch_l, num_instances, iou_threshold
#         # )
#
#     def _get_per_cluster_iou(self, masks, target_masks):
#         inst_mious = []
#         for mask, target_mask in zip(masks, target_masks):
#             inst_confusion_matrix = ConfusionMatrix(2)
#             mask = self._convert(mask)
#             target_mask = self._convert(target_mask)
#             inst_confusion_matrix.count_predicted_batch(mask, target_mask)
#             inst_miou = 100 * inst_confusion_matrix.get_average_intersection_union()
#             inst_mious.append(inst_miou)
#         return inst_mious
#
#
#
#
#     def _get_data(self, masks, target_masks, sems, target_sems):
#         clusters_num = target_sems.shape[0]
#         output_logits, target_logits, output_masks = [], [], []
#         for c_id in range(clusters_num):
#             output_sem = sems[c_id]
#             target_sem = target_sems[c_id]
#             output_mask = masks[c_id]
#             target_mask = target_masks[c_id]
#
#             output_logit = torch.where(output_mask == 1.0, output_sem, 0.0)
#             target_logit = torch.where(target_mask == True, target_sem, 0.0)
#
#             output_logits.append(output_logit)
#             target_logits.append(target_logit)
#             output_masks.append(output_mask)
#
#         output_logits = torch.stack(output_logits)
#         target_logits = torch.stack(target_logits)
#
#         return output_logits, target_logits, output_masks
#
#     def _compute_acc(self, clusters, predicted_labels, labels, batch, num_instances, iou_threshold):
#         """Computes the ratio of True positives, False positives and accuracy"""
#         iou_values, gt_ids = self.my_instance_iou(clusters, labels[0], batch).max(1)
#         gt_ids += 1
#         instance_offsets = torch.cat((torch.tensor([0]).to(num_instances.device), num_instances.cumsum(-1)))
#         tp = 0
#         fp = 0
#         for i, iou in enumerate(iou_values):
#             # Too low iou, no match in ground truth
#             if iou < iou_threshold:
#                 fp += 1
#                 continue
#
#             # Check that semantic is correct
#             sample_idx = batch[clusters[i][0]]
#             sample_mask = batch == sample_idx
#             instance_offset = instance_offsets[sample_idx]
#             gt_mask = labels.instance_labels[sample_mask] == (gt_ids[i] - instance_offset)
#             gt_classes = labels.y[sample_mask][torch.nonzero(gt_mask, as_tuple=False)]
#             gt_classes, counts = torch.unique(gt_classes, return_counts=True)
#             gt_class = gt_classes[counts.max(-1)[1]]
#             pred_class = predicted_labels[clusters[i][0]]
#             if gt_class == pred_class:
#                 tp += 1
#             else:
#                 fp += 1
#         acc = tp / len(clusters)
#         tp = tp / torch.sum(labels.num_instances).cpu().item()
#         fp = fp / torch.sum(labels.num_instances).cpu().item()
#         return tp, fp, acc
#
#     def my_instance_iou(
#         self,
#         instance_idx: List[torch.Tensor],
#         gt_instances: torch.Tensor,
#         batch: Optional[torch.Tensor] = None,
#     ):
#         if batch is None:
#             batch = torch.zeros_like(gt_instances)
#
#         # Gather number of gt instances per batch and size of those instances
#         gt_instance_sizes = []
#         num_gt_instances = []
#         batch_size = batch[-1] + 1
#         # for s in range(batch_size):
#         #     batch_mask = batch == s
#         #     sample_gt_instances = gt_instances[batch_mask]
#         #     sample_num_gt_instances = torch.max(sample_gt_instances).item()
#         #     num_gt_instances.append(sample_num_gt_instances)
#         #     for instance_id in range(1, sample_num_gt_instances + 1):
#         #         gt_instance_sizes.append(torch.sum(sample_gt_instances == instance_id))
#
#         sample_gt_instances = gt_instances
#         sample_num_gt_instances = torch.max(sample_gt_instances).item()
#         num_gt_instances.append(sample_num_gt_instances)
#         for instance_id in range(1, sample_num_gt_instances + 1):
#             gt_instance_sizes.append(torch.sum(sample_gt_instances == instance_id))
#
#
#         gt_instance_sizes = torch.stack(gt_instance_sizes)
#         num_gt_instances = torch.tensor(num_gt_instances)
#
#         # Instance offset when flatten
#         instance_offsets = [0]
#         cum_offset = 0
#         for instance in instance_idx:
#             cum_offset += instance.shape[0]
#             instance_offsets.append(cum_offset)
#
#         # Compute ious
#         instance_idx = torch.cat(instance_idx).long()
#         if gt_instances.is_cuda:
#             return tpcuda.instance_iou_cuda(
#                 instance_idx.cuda(),
#                 torch.tensor(instance_offsets).cuda(),
#                 gt_instances.cuda(),
#                 gt_instance_sizes.cuda(),
#                 num_gt_instances.cuda(),
#                 batch.cuda(),
#             )
#         else:
#             res = _instance_iou_cpu(
#                 instance_idx.numpy(),
#                 np.asarray(instance_offsets),
#                 gt_instances.numpy(),
#                 gt_instance_sizes.numpy(),
#                 num_gt_instances.numpy(),
#                 batch.numpy(),
#             )
#             return torch.tensor(res).float()
#
#
#     def get_metrics(self, verbose=False) -> Dict[str, Any]:
#         """ Returns a dictionnary of all metrics and losses being tracked
#         """
#         metrics = super().get_metrics(verbose)
#         metrics["{}_change_loss".format(self._stage)] = self.change_loss
#         metrics["{}_change_acc".format(self._stage)] = self.change_acc
#         metrics["{}_change_macc".format(self._stage)] = self.change_macc
#         metrics["{}_change_miou".format(self._stage)] = self.change_miou
#         metrics["{}_change_miou_ch".format(self._stage)] = self.change_miou_ch
#
#         metrics["{}_pos".format(self._stage)] = meter_value(self.inst_pos)
#         metrics["{}_neg".format(self._stage)] = meter_value(self.inst_neg)
#         metrics["{}_Iacc".format(self._stage)] = meter_value(self.inst_acc_meter)
#
#         if self._has_instance_data:
#             mAP = sum(self.inst_ap.values()) / len(self.inst_ap)
#             metrics["{}_map".format(self._stage)] = mAP
#
#         if verbose:
#             metrics["{}_iou_per_class".format(self._stage)] = self.inst_iou_per_class
#
#         if verbose and self._has_instance_data:
#             metrics["{}_class_rec".format(self._stage)] = self._dict_to_str(self.inst_rec)
#             metrics["{}_class_ap".format(self._stage)] = self._dict_to_str(self.inst_ap)
#
#         # if verbose:
#         #     metrics["{}_iou_per_class".format(self._stage)] = self._miou_per_class
#         #     metrics["{}_acc_per_class".format(self._stage)] = self._acc_per_class
#         return metrics
#
#
#
#     @property
#     def _has_instance_data(self):
#         return len(self.inst_rec)
#
#     @property
#     def metric_func(self):
#         return self._metric_func
#
# class _Instance(NamedTuple):
#     classname: str
#     score: float
#     indices: np.array  # type: ignore
#     scan_id: int
#
#     def iou(self, other: "_Instance") -> float:
#         assert self.scan_id == other.scan_id
#         intersection = float(len(np.intersect1d(other.indices, self.indices)))
#         return intersection / float(len(other.indices) + len(self.indices) - intersection)
#
#     def find_best_match(self, others: List["_Instance"]) -> Tuple[float, int]:
#         ioumax = -np.inf
#         best_match = -1
#         for i, other in enumerate(others):
#             iou = self.iou(other)
#             if iou > ioumax:
#                 ioumax = iou
#                 best_match = i
#         return ioumax, best_match
#
#
# class InstanceAPMeter:
#     def __init__(self):
#         self._pred_clusters = defaultdict(list)  # {classname: List[_Instance]}
#         self._gt_clusters = defaultdict(lambda: defaultdict(list))  # {classname:{scan_id: List[_Instance]}
#
#     def add(self, pred_clusters: List[_Instance], gt_clusters: List[_Instance]):
#         for instance in pred_clusters:
#             self._pred_clusters[instance.classname].append(instance)
#         for instance in gt_clusters:
#             self._gt_clusters[instance.classname][instance.scan_id].append(instance)
#
#     def _eval_cls(self, classname, iou_threshold):
#         preds = self._pred_clusters.get(classname, [])
#         allgts = self._gt_clusters.get(classname, {})
#         visited = {scan_id: len(gt) * [False] for scan_id, gt in allgts.items()}
#         ngt = 0
#         for gts in allgts.values():
#             ngt += len(gts)
#
#         # Start with most confident first
#         preds.sort(key=lambda x: x.score, reverse=True)
#         tp = np.zeros(len(preds))
#         fp = np.zeros(len(preds))
#         for p, pred in enumerate(preds):
#             scan_id = pred.scan_id
#             gts = allgts.get(scan_id, [])
#             if len(gts) == 0:
#                 fp[p] = 1
#                 continue
#
#             # Find best macth in ground truth
#             ioumax, best_match = pred.find_best_match(gts)
#
#             if ioumax < iou_threshold:
#                 fp[p] = 1
#                 continue
#
#             if visited[scan_id][best_match]:
#                 fp[p] = 1
#             else:
#                 visited[scan_id][best_match] = True
#                 tp[p] = 1
#
#         fp = np.cumsum(fp)
#         tp = np.cumsum(tp)
#         rec = tp / float(ngt)
#
#         # avoid divide by zero in case the first detection matches a difficult
#         # ground truth
#         prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
#         ap = voc_ap(rec, prec)
#         return rec, prec, ap
#
#     def eval(self, iou_threshold, processes=1):
#         rec = {}
#         prec = {}
#         ap = {}
#         for classname in self._gt_clusters.keys():
#             rec[classname], prec[classname], ap[classname] = self._eval_cls(classname, iou_threshold)
#
#         for i, classname in enumerate(self._gt_clusters.keys()):
#             if classname not in self._pred_clusters:
#                 rec[classname] = 0
#                 prec[classname] = 0
#                 ap[classname] = 0
#
#         return rec, prec, ap