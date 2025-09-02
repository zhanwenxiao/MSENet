from typing import Dict, Any
import logging
import torch
from torch_geometric.nn.unpool import knn_interpolate
import numpy as np
import os
import os.path as osp
import _pickle as pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.instance_change_detection_tracker import iCDTracker
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.datasets.change_detection import IGNORE_LABEL
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.models import model_interface
import json, math
from copy import deepcopy
from uuid import uuid4

log = logging.getLogger(__name__)

URB3DCD_NB_CLASS = 7

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

def softmax(weights):
    return torch.softmax(torch.stack([value for value in weights]), dim=-1)

def normalize(weights: dict):
    norm_const = 1 / sum(weights)
    return [value * norm_const for value in weights]

def assign_instances_for_scan(pred: dict):
    pred_info = make_pred_info(pred)
    gt_ids = pred["gt_target"].detach().cpu().numpy()
    non_building_idx = np.argwhere(gt_ids != -1).reshape(-1,)
    gt_ids[non_building_idx] += 1000
    gt_ids = np.where(gt_ids == -1, 0, gt_ids)

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
        if id == -1:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances

class Urb3DiCDTracker(iCDTracker):
    def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False,
                 ignore_label: int = IGNORE_LABEL, full_pc: bool = False, full_res: bool = False):
        super(Urb3DiCDTracker, self).__init__(dataset, stage, wandb_log, use_tensorboard, ignore_label)
        self.full_pc = full_pc
        self.full_res = full_res
        self.gt_tot = None
        self.pred_tot = None

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        if self._stage == 'test':
            self._ds = self._dataset.test_data
        elif self._stage == 'val':
            self._ds = self._dataset.val_data
        else:
            self._ds = self._dataset.train_data
        self._areas = [None] * self._ds.size()
        self._metric_per_areas = [None] * self._ds.size()
        self.gt_tot = None
        self.pred_tot = None

    def track(self, model: model_interface.TrackerInterface, data=None, full_pc=False, conv_classes=None,**kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        super().track(model, conv_classes=conv_classes)

        # Train mode or low res, nothing special to do
        if self._stage == "train" or not full_pc: #not full_pc:
            return
        inputs = data if data is not None else model.get_input()
        output = model.get_output()
        change_output0, change_output1, inst_output0, pred_logits0, inst_output1, pred_logits1 = output[0], output[1], output[2]["pred_masks"], output[2]["pred_logits"], output[3]["pred_masks"], output[3]["pred_logits"]
        inputs.change_pred = change_output0
        inputs.change_target_pred = change_output1
        data_l = inputs.to_data_list()
        num_change_pred = self._change_classes
        self.change_metrics(model, data_l, num_change_pred)

    def change_metrics(self, model, data_l, num_class_pred):
        for p in range(len(data_l)):
            area_sel = data_l[p].area
            # Test mode, compute votes in order to get full res predictions
            if self._areas[area_sel] is None:
                pair = self._ds._load_save(area_sel)
                self._areas[area_sel] = pair
                if self._areas[area_sel].change_y_target is None:
                    raise ValueError("It seems that the test area data does not have labels (attribute y).")
                self._areas[area_sel].prediction_count = torch.zeros(self._areas[area_sel].change_y.shape[0], dtype=torch.int)
                self._areas[area_sel].votes = torch.zeros((self._areas[area_sel].change_y.shape[0], num_class_pred), dtype=torch.float)
                self._areas[area_sel].prediction_count_target = torch.zeros(self._areas[area_sel].change_y_target.shape[0], dtype=torch.int)
                self._areas[area_sel].votes_target = torch.zeros((self._areas[area_sel].change_y_target.shape[0], num_class_pred), dtype=torch.float)

                self._areas[area_sel].to(model.device)

            # Gather origin ids and check that it fits with the test set
            if data_l[p].idx is None:
                raise ValueError("The inputs given to the model do not have a idx_target attribute.")
            if data_l[p].idx_target is None:
                raise ValueError("The inputs given to the model do not have a idx_target attribute.")

            originids = data_l[p].idx
            if originids.dim() == 2:
                originids = originids.flatten()
            if originids.max() >= self._areas[area_sel].pos.shape[0]:
                raise ValueError("Origin ids are larger than the number of points in the original point cloud.")

            originids_target = data_l[p].idx_target
            if originids_target.dim() == 2:
                originids_target = originids_target.flatten()
            if originids_target.max() >= self._areas[area_sel].pos_target.shape[0]:
                raise ValueError("Origin ids are larger than the number of points in the original point cloud.")

            # Set predictions
            self._areas[area_sel].votes[originids] += data_l[p].change_pred
            self._areas[area_sel].prediction_count[originids] += 1
            self._areas[area_sel].votes_target[originids_target] += data_l[p].change_target_pred
            self._areas[area_sel].prediction_count_target[originids_target] += 1

    def inst_metrics(self, dataset: str = "urb3d"):
        global CLASS_LABELS
        global VALID_CLASS_IDS
        global ID_TO_LABEL
        global LABEL_TO_ID
        global opt

        if dataset == "urb3d":
            # global CLASS_LABELS
            # global VALID_CLASS_IDS
            # global ID_TO_LABEL
            # global LABEL_TO_ID

            opt["min_region_sizes"] = np.array([10])

            CLASS_LABELS = [
                "Build",
            ]
            VALID_CLASS_IDS = np.array(
                [1]
            )

            ID_TO_LABEL = {}
            LABEL_TO_ID = {}
            for i in range(len(VALID_CLASS_IDS)):
                LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
                ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

        matches = {}
        for i, pred in enumerate(self.inst_preds):
            matches_key = i
            # assign gt to predictions
            gt2pred, pred2gt = assign_instances_for_scan(pred)
            matches[matches_key] = {}
            matches[matches_key]["gt"] = gt2pred
            matches[matches_key]["pred"] = pred2gt
        print("")
        ap_scores = evaluate_matches(matches)
        avgs = compute_averages(ap_scores)

        return avgs['all_ap'], avgs['all_ap_50%'], avgs['all_ap_25%']

    def get_kpi(self, metric, momentum):
        kpi = sum(metric) / len(metric) if len(metric) else None
        _kpi = kpi if math.isnan(kpi) else momentum * kpi + (1 - momentum) * kpi
        return _kpi

    def get_weight(self, kpis, potentials, focusing, normal_type='normalize'):
        weight_list = []
        for kpi, potential in zip(kpis, potentials):
            weight = (1 - kpi / potential) ** focusing
            weight_list.append(weight)
        _weight_norm = softmax if normal_type == 'softmax' else normalize
        normalized_weight = _weight_norm(weight_list)
        return normalized_weight

    def update_loss_weight(self, change_metrics, inst_metrics, loss_metric="amtl"):
        if loss_metric == "amtl":
            potential = [0.9259, 0.7425]
            focusing = 2
            margin = 0.05
            momentum = 0.1
            potential = [v * (1 + margin) for i, v in enumerate(potential)]
            change_kpi = self.get_kpi(change_metrics, momentum)
            inst_kpi = self.get_kpi(inst_metrics, momentum)
            weight = self.get_weight([change_kpi, inst_kpi], potential, focusing)

            self.loss_weight["change"] = weight[0]
            self.loss_weight["inst"] = weight[1]

    def finalise(self, save_pc=False, name_test="", saving_path=None, conv_classes=None, num_class_cm=None, **kwargs):
        # per_class_iou = self.change_confusion_matrix.get_intersection_union_per_class()[0]
        # self._iou_per_class = {self._dataset.INV_CHANGE_LABEL[k]: v for k, v in enumerate(per_class_iou)}

        # ap_scores, avgs = self.inst_metrics()
        # print("ap_scores {}".format(ap_scores))
        # print("avgs {}".format(avgs))

        if self._stage != "train":
            ap, ap50, ap25 = self.inst_metrics()
            self._mAP = ap
            self._mAP50 = ap50
            self._mAP25 = ap25

            change_miou = self.change_confusion_matrix.get_average_intersection_union()
            print("ap {} ap50 {} ap25 {} c_mIoU {}".format(ap, ap50, ap25, change_miou))

            # self.update_loss_weight([change_miou], [ap, ap50, ap25])
            # print("self.loss_weight: {}".format(self.loss_weight))

        if self.full_pc:
            if num_class_cm is None:
                if conv_classes is None:
                    num_class_cm = self._change_classes + 1
                else:
                    num_class_cm = np.max(conv_classes)

            gt_tot = []
            pred_tot = []
            gt_tot_target = []
            pred_tot_target = []
            for i, area in enumerate(self._areas):
                if area is not None:
                    # Complete for points that have a prediction
                    area = area.to("cpu")
                    has_prediction = area.prediction_count > 0
                    has_prediction_target = area.prediction_count_target > 0
                    pred = torch.argmax(area.votes[has_prediction], 1)
                    pred_target = torch.argmax(area.votes_target[has_prediction_target], 1)
                    if conv_classes is not None:
                        pred = torch.from_numpy(conv_classes[pred])
                        pred_target = torch.from_numpy(conv_classes[pred_target])

                    pos = area.pos[has_prediction]
                    pos_target = area.pos_target[has_prediction_target]
                    c = ConfusionMatrix(num_class_cm)
                    # If full res, knn interpolation
                    if self.full_res:
                        area_pos, area_orig_pos, _, _, _, _, gt0, gt1 = self._ds.clouds_loader(i, "vertex")
                        # _, area_orig_pos, gt = self._ds.clouds_loader(i)
                        area_pos = area_pos[:,:3]
                        area_orig_pos = area_orig_pos[:, :3]
                        # still on GPU no need for num_workers
                        pred = knn_interpolate(torch.unsqueeze(pred, 1), pos, area_pos, k=1).numpy()
                        pred = np.squeeze(pred)
                        pred = pred.astype(int)

                        pred_target = knn_interpolate(torch.unsqueeze(pred_target, 1), pos_target, area_orig_pos, k=1).numpy()
                        pred_target = np.squeeze(pred_target)
                        pred_target = pred_target.astype(int)

                        gt = gt0.numpy()
                        pos = area_pos

                        gt_target = gt1.numpy()
                        pos_target = area_orig_pos
                    else:
                        pred = pred.numpy()
                        gt = area.change_y[has_prediction].numpy()
                        pos = pos.cpu()

                        pred_target = pred_target.numpy()
                        gt_target = area.change_y_target[has_prediction_target].numpy()
                        pos_target = pos_target.cpu()

                    gt_tot.append(gt)
                    pred_tot.append(pred)

                    gt_tot_target.append(gt_target)
                    pred_tot_target.append(pred_target)
                    # Metric computation
                    c.count_predicted_batch_list(gt, pred, gt_target, pred_target) #c.count_predicted_batch(gt, pred)
                    acc = 100 * c.get_overall_accuracy()
                    macc = 100 * c.get_mean_class_accuracy()
                    miou = 100 * c.get_average_intersection_union()
                    recall = 100 * c.get_recall()
                    class_iou, present_class = c.get_intersection_union_per_class()
                    class_acc = c.confusion_matrix.diagonal()/c.confusion_matrix.sum(axis=1)
                    iou_per_class = {
                        k: "{:.2f}".format(100 * v)
                        for k, v in enumerate(class_iou)
                    }
                    acc_per_class = {
                        k: "{:.2f}".format(100 * v)
                        for k, v in enumerate(class_acc)
                    }
                    miou_ch = 100 * np.mean(class_iou[1:])
                    metrics = {}
                    metrics["{}_acc".format(self._stage)] = acc
                    metrics["{}_macc".format(self._stage)] = macc
                    metrics["{}_miou".format(self._stage)] = miou
                    metrics["{}_recall".format(self._stage)] = recall
                    metrics["{}_miou_ch".format(self._stage)] = miou_ch
                    metrics["{}_iou_per_class".format(self._stage)] = iou_per_class
                    metrics["{}_acc_per_class".format(self._stage)] = acc_per_class
                    self._metric_per_areas[i] = metrics

                    if (self._stage == 'test' or self._stage == 'val') and save_pc:
                        print('Saving PC %s' % (str(i)))
                        if saving_path is None:
                            saving_path = os.path.join(os.getcwd(), name_test)
                        if not os.path.exists(saving_path):
                            os.makedirs(saving_path)
                        self._dataset.to_ply(pos, pred,
                                             os.path.join(saving_path, "pointCloud" +
                                                          os.path.dirname(self._ds.filesPC0[i]).split('/')[
                                                              -1] + ".ply"),
                                             )
                        self._dataset.to_ply(pos_target, pred_target,
                                             os.path.join(saving_path, "pointCloud" +
                                                          os.path.dirname(self._ds.filesPC1[i]).split('/')[
                                                              -1] + ".ply"),
                                             )

            self.gt_tot = np.concatenate(gt_tot)
            self.pred_tot = np.concatenate(pred_tot)
            self.gt_tot_target = np.concatenate(gt_tot_target)
            self.pred_tot_target = np.concatenate(pred_tot_target)
            c = ConfusionMatrix(num_class_cm)
            c.count_predicted_batch_list(self.gt_tot, self.pred_tot, self.gt_tot_target, self.pred_tot_target) # c.count_predicted_batch(self.gt_tot, self.pred_tot)
            acc = 100 * c.get_overall_accuracy()
            macc = 100 * c.get_mean_class_accuracy()
            miou = 100 * c.get_average_intersection_union()
            recall = 100 * c.get_recall()
            class_iou, present_class = c.get_intersection_union_per_class()
            iou_per_class = {
                k: "{:.2f}".format(100 * v)
                for k, v in enumerate(class_iou)
            }
            class_acc = c.confusion_matrix.diagonal() / c.confusion_matrix.sum(axis=1)
            acc_per_class = {
                k: "{:.2f}".format(100 * v)
                for k, v in enumerate(class_acc)
            }
            miou_ch = 100 * np.mean(class_iou[1:])
            self.metric_full_cumul = {"acc": acc, "recall": recall, "macc": macc, "mIoU": miou, "miou_ch": miou_ch,
                                      "IoU per class": iou_per_class, "acc_per_class": acc_per_class, "all_ap": self._mAP,
                                      "all_ap_50": self._mAP50, "all_ap_25": self._mAP25}
            saving_path = os.path.join(os.getcwd(), self._stage, name_test)
            if not os.path.exists(saving_path):
                os.makedirs(saving_path)

            name_classes = [name for name, i in self._ds.change_labels.items()]
            self.save_metrics(name_test=name_test, saving_path=saving_path)
            try:
                self.plot_confusion_matrix(gt_tot, pred_tot, normalize=True, saving_path=saving_path + "cm.png",
                                           name_classes=name_classes)
            except:
                pass
            try:
                self.plot_confusion_matrix(gt_tot, pred_tot, normalize=True, saving_path=saving_path + "cm2.png")
            except:
                pass

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        if verbose:
            if self.full_pc:
                for i, area in enumerate(self._areas):
                    if area is not None:
                        metrics["%s_whole_pc_%s" % (self._stage, str(i) + "_" + osp.basename(self._ds.filesPC0[i]))] = \
                        self._metric_per_areas[i]
        return metrics

    def save_metrics(self, saving_path=None, name_test=""):
        metrics = self.get_metrics()
        if self.full_pc:
            for i, area in enumerate(self._areas):
                if area is not None:
                    metrics["%s_whole_pc_%s" % (self._stage, osp.basename(self._ds.filesPC0[i]))] = \
                    self._metric_per_areas[i]
        self._avg_metrics_full_pc = merge_avg_mappings(self._metric_per_areas)
        print("Average full pc res :\n")
        print(self._avg_metrics_full_pc)
        if saving_path is None:
            saving_path = os.path.join(os.getcwd(), self._stage, name_test)
        with open(osp.join(saving_path, "res.txt"), "w") as fi:
            for met, val in metrics.items():
                fi.write(met + " : " + str(val) + "\n")
            fi.write("\n")
            fi.write("Average full pc res \n")
            for met, val in self._avg_metrics_full_pc.items():
                fi.write(met + " : " + str(val) + "\n")
            fi.write("\n")
            fi.write("Cumulative full pc res \n")
            for met, val in self.metric_full_cumul.items():
                fi.write(met + " : " + str(val) + "\n")

    def plot_confusion_matrix(self, y_true, y_pred,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues,
                              saving_path="",
                              name_classes=None):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = unique_labels(y_true, y_pred)
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        if name_classes == None:
            name_classes = classes
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=name_classes, yticklabels=name_classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.xlim(-0.5, len(np.unique(y_pred)) - 0.5)
        plt.ylim(len(np.unique(y_pred)) - 0.5, -0.5)
        if saving_path != "":
            plt.savefig(saving_path)
        return ax


def merge_avg_mappings(dicts):
    """ Merges an arbitrary number of dictionaries based on the
    average value in a given mapping.

    Parameters
    ----------
    dicts : Dict[Any, Comparable]

    Returns
    -------
    Dict[Any, Comparable]
        The merged dictionary
    """
    merged = {}
    cpt = {}
    for d in dicts:  # `dicts` is a list storing the input dictionaries
        for key in d:
            if key not in merged:
                if type(d[key]) == dict:
                    merged[key] = {}
                    for key2 in d[key]:
                        merged[key][key2] = float(d[key][key2])
                else:
                    merged[key] = d[key]
                cpt[key] = 1
            else:
                if type(d[key]) == dict:
                    for key2 in d[key]:
                        merged[key][key2] += float(d[key][key2])
                else:
                    merged[key] += d[key]
                cpt[key] += 1
    for key in merged:
        if type(merged[key]) == dict:
            for key2 in merged[key]:
                merged[key][key2] /= cpt[key]
        else:
            merged[key] /= cpt[key]
    return merged

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
        # return 1
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
# from typing import NamedTuple, Dict, Any, List, Tuple
# import logging
# import torch
# from torch_geometric.nn.unpool import knn_interpolate
# import numpy as np
# import os
# import os.path as osp
# import _pickle as pickle
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# from sklearn.utils.multiclass import unique_labels
# from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
# from torch_points3d.metrics.instance_change_detection_tracker import iCDTracker
# from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
# from torch_points3d.datasets.change_detection import IGNORE_LABEL
# from torch_points3d.core.data_transform import SaveOriginalPosId
# from torch_points3d.models import model_interface
#
#
# log = logging.getLogger(__name__)
#
# URB3DCD_NB_CLASS = 7
#
# class Urb3DiCDTracker(iCDTracker):
#     def __init__(self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False,
#                  ignore_label: int = IGNORE_LABEL, full_pc: bool = False, full_res: bool = False):
#         super(Urb3DiCDTracker, self).__init__(dataset, stage, wandb_log, use_tensorboard, ignore_label)
#         self.full_pc = full_pc
#         self.full_res = full_res
#         self.gt_tot = None
#         self.pred_tot = None
#
#     def reset(self, *args, **kwargs):
#         super().reset(*args, **kwargs)
#         if self._stage == 'test':
#             self._ds = self._dataset.test_data
#         elif self._stage == 'val':
#             self._ds = self._dataset.val_data
#         else:
#             self._ds = self._dataset.train_data
#         self._areas = [None] * self._ds.size()
#         self._metric_per_areas = [None] * self._ds.size()
#         self.gt_tot = None
#         self.pred_tot = None
#
#     def track(self, model: model_interface.TrackerInterface, data=None, full_pc=False, conv_classes=None,**kwargs):
#         """ Add current model predictions (usually the result of a batch) to the tracking
#         """
#         super().track(model, conv_classes=conv_classes)
#
#         # Train mode or low res, nothing special to do
#         if self._stage == "train" or not full_pc: #not full_pc:
#             return
#         inputs = data if data is not None else model.get_input()
#         output = model.get_output()
#         change_output, inst_output0, pred_logits0, inst_output1, pred_logits1 = output[0], output[1]["pred_masks"], output[1]["pred_logits"], output[2]["pred_masks"], output[2]["pred_logits"]
#         inputs.pred = change_output
#         data_l = inputs.to_data_list()
#         num_change_pred = self._change_classes
#         self.change_metrics(model, data_l, num_change_pred)
#
#     def change_metrics(self, model, data_l, num_class_pred):
#         for p in range(len(data_l)):
#             area_sel = data_l[p].area
#             # Test mode, compute votes in order to get full res predictions
#             if self._areas[area_sel] is None:
#                 pair = self._ds._load_save(area_sel)
#                 self._areas[area_sel] = pair
#                 if self._areas[area_sel].y is None:
#                     raise ValueError("It seems that the test area data does not have labels (attribute y).")
#                 self._areas[area_sel].prediction_count = torch.zeros(self._areas[area_sel].y.shape[0], dtype=torch.int)
#                 self._areas[area_sel].votes = torch.zeros((self._areas[area_sel].y.shape[0], num_class_pred), dtype=torch.float)
#                 self._areas[area_sel].to(model.device)
#
#             # Gather origin ids and check that it fits with the test set
#             if data_l[p].idx_target is None:
#                 raise ValueError("The inputs given to the model do not have a idx_target attribute.")
#
#             originids = data_l[p].idx_target
#             if originids.dim() == 2:
#                 originids = originids.flatten()
#             if originids.max() >= self._areas[area_sel].pos_target.shape[0]:
#                 raise ValueError("Origin ids are larger than the number of points in the original point cloud.")
#             # Set predictions
#             self._areas[area_sel].votes[originids] += data_l[p].pred
#             self._areas[area_sel].prediction_count[originids] += 1
#
#     def inst_metrics(self, dataset: str = "urb3d"):
#         pass
#
#
#     def finalise(self, save_pc=False, name_test="", saving_path=None, conv_classes=None, num_class_cm=None, **kwargs):
#         per_class_iou = self.change_confusion_matrix.get_intersection_union_per_class()[0]
#         self._iou_per_class = {self._dataset.INV_OBJECT_LABEL[k]: v for k, v in enumerate(per_class_iou)}
#
#         ap_scores, avgs = self.inst_metrics()
#         print("ap_scores {}".format(ap_scores))
#         print("avgs {}".format(avgs))
#
#         if self.full_pc:
#             if num_class_cm is None:
#                 if conv_classes is None:
#                     num_class_cm = self._num_classes
#                 else:
#                     num_class_cm = np.max(conv_classes)
#
#             gt_tot = []
#             pred_tot = []
#             for i, area in enumerate(self._areas):
#                 if area is not None:
#                     # Complete for points that have a prediction
#                     area = area.to("cpu")
#                     has_prediction = area.prediction_count > 0
#                     pred = torch.argmax(area.votes[has_prediction], 1)
#                     if conv_classes is not None:
#                         pred = torch.from_numpy(conv_classes[pred])
#
#                     pos = area.pos_target[has_prediction]
#                     c = ConfusionMatrix(num_class_cm)
#                     # If full res, knn interpolation
#                     if self.full_res:
#                         _, area_orig_pos, gt = self._ds.clouds_loader(i)
#                         area_orig_pos = area_orig_pos[:,:3]
#                         # still on GPU no need for num_workers
#                         pred = knn_interpolate(torch.unsqueeze(pred, 1), pos,
#                                                area_orig_pos, k=1).numpy()
#                         pred = np.squeeze(pred)
#                         pred = pred.astype(int)
#                         gt = gt.numpy()
#                         pos = area_orig_pos
#                     else:
#                         pred = pred.numpy()
#                         gt = area.y[has_prediction].numpy()
#                         pos = pos.cpu()
#
#                     gt_tot.append(gt)
#                     pred_tot.append(pred)
#                     # Metric computation
#                     c.count_predicted_batch(gt, pred)
#                     acc = 100 * c.get_overall_accuracy()
#                     macc = 100 * c.get_mean_class_accuracy()
#                     miou = 100 * c.get_average_intersection_union()
#                     recall = 100 * c.get_recall()
#                     class_iou, present_class = c.get_intersection_union_per_class()
#                     class_acc = c.confusion_matrix.diagonal()/c.confusion_matrix.sum(axis=1)
#                     iou_per_class = {
#                         k: "{:.2f}".format(100 * v)
#                         for k, v in enumerate(class_iou)
#                     }
#                     acc_per_class = {
#                         k: "{:.2f}".format(100 * v)
#                         for k, v in enumerate(class_acc)
#                     }
#                     miou_ch = 100 * np.mean(class_iou[1:])
#                     metrics = {}
#                     metrics["{}_acc".format(self._stage)] = acc
#                     metrics["{}_macc".format(self._stage)] = macc
#                     metrics["{}_miou".format(self._stage)] = miou
#                     metrics["{}_recall".format(self._stage)] = recall
#                     metrics["{}_miou_ch".format(self._stage)] = miou_ch
#                     metrics["{}_iou_per_class".format(self._stage)] = iou_per_class
#                     metrics["{}_acc_per_class".format(self._stage)] = acc_per_class
#                     self._metric_per_areas[i] = metrics
#
#
#                     if (self._stage == 'test' or self._stage == 'val') and save_pc:
#                         print('Saving PC %s' % (str(i)))
#                         if saving_path is None:
#                             saving_path = os.path.join(os.getcwd(), name_test)
#                         if not os.path.exists(saving_path):
#                             os.makedirs(saving_path)
#                         self._dataset.to_ply(pos, pred,
#                                              os.path.join(saving_path, "pointCloud" +
#                                                           os.path.dirname(self._ds.filesPC0[i]).split('/')[
#                                                               -1] + ".ply"),
#                                              )
#             self.gt_tot = np.concatenate(gt_tot)
#             self.pred_tot = np.concatenate(pred_tot)
#             c = ConfusionMatrix(num_class_cm)
#             c.count_predicted_batch(self.gt_tot, self.pred_tot)
#             acc = 100 * c.get_overall_accuracy()
#             macc = 100 * c.get_mean_class_accuracy()
#             miou = 100 * c.get_average_intersection_union()
#             recall = 100 * c.get_recall()
#             class_iou, present_class = c.get_intersection_union_per_class()
#             iou_per_class = {
#                 k: "{:.2f}".format(100 * v)
#                 for k, v in enumerate(class_iou)
#             }
#             class_acc = c.confusion_matrix.diagonal() / c.confusion_matrix.sum(axis=1)
#             acc_per_class = {
#                 k: "{:.2f}".format(100 * v)
#                 for k, v in enumerate(class_acc)
#             }
#             miou_ch = 100 * np.mean(class_iou[1:])
#             self.metric_full_cumul = {"acc": acc, "recall": recall, "macc": macc, "mIoU": miou, "miou_ch": miou_ch,
#                                       "IoU per class": iou_per_class, "acc_per_class": acc_per_class}
#
#             saving_path = os.path.join(os.getcwd(), self._stage, name_test)
#             if not os.path.exists(saving_path):
#                 os.makedirs(saving_path)
#
#             name_classes = [name for name, i in self._ds.class_labels.items()]
#             self.save_metrics(name_test=name_test, saving_path=saving_path)
#             try:
#                 self.plot_confusion_matrix(gt_tot, pred_tot, normalize=True, saving_path=saving_path + "cm.png",
#                                            name_classes=name_classes)
#             except:
#                 pass
#             try:
#                 self.plot_confusion_matrix(gt_tot, pred_tot, normalize=True, saving_path=saving_path + "cm2.png")
#             except:
#                 pass
#
#     def get_metrics(self, verbose=False) -> Dict[str, Any]:
#         """ Returns a dictionnary of all metrics and losses being tracked
#         """
#         metrics = super().get_metrics(verbose)
#
#         if verbose:
#             if self.full_pc:
#                 for i, area in enumerate(self._areas):
#                     if area is not None:
#                         metrics["%s_whole_pc_%s" % (self._stage, str(i) + "_" + osp.basename(self._ds.filesPC0[i]))] = \
#                         self._metric_per_areas[i]
#         return metrics
#
#     def save_metrics(self, saving_path=None, name_test=""):
#         metrics = self.get_metrics()
#         if self.full_pc:
#             for i, area in enumerate(self._areas):
#                 if area is not None:
#                     metrics["%s_whole_pc_%s" % (self._stage, osp.basename(self._ds.filesPC0[i]))] = \
#                     self._metric_per_areas[i]
#         self._avg_metrics_full_pc = merge_avg_mappings(self._metric_per_areas)
#         print("Average full pc res :\n")
#         print(self._avg_metrics_full_pc)
#         if saving_path is None:
#             saving_path = os.path.join(os.getcwd(), self._stage, name_test)
#         with open(osp.join(saving_path, "res.txt"), "w") as fi:
#             for met, val in metrics.items():
#                 fi.write(met + " : " + str(val) + "\n")
#             fi.write("\n")
#             fi.write("Average full pc res \n")
#             for met, val in self._avg_metrics_full_pc.items():
#                 fi.write(met + " : " + str(val) + "\n")
#             fi.write("\n")
#             fi.write("Cumulative full pc res \n")
#             for met, val in self.metric_full_cumul.items():
#                 fi.write(met + " : " + str(val) + "\n")
#
#     def plot_confusion_matrix(self, y_true, y_pred,
#                               normalize=False,
#                               title=None,
#                               cmap=plt.cm.Blues,
#                               saving_path="",
#                               name_classes=None):
#         """
#         This function prints and plots the confusion matrix.
#         Normalization can be applied by setting `normalize=True`.
#         """
#         if not title:
#             if normalize:
#                 title = 'Normalized confusion matrix'
#             else:
#                 title = 'Confusion matrix, without normalization'
#
#         # Compute confusion matrix
#         cm = confusion_matrix(y_true, y_pred)
#         # Only use the labels that appear in the data
#         classes = unique_labels(y_true, y_pred)
#         # classes = classes[unique_labels(y_true, y_pred)]
#         if normalize:
#             cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#             print("Normalized confusion matrix")
#         else:
#             print('Confusion matrix, without normalization')
#
#         fig, ax = plt.subplots()
#         im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#         ax.figure.colorbar(im, ax=ax)
#         # We want to show all ticks...
#         if name_classes == None:
#             name_classes = classes
#         ax.set(xticks=np.arange(cm.shape[1]),
#                yticks=np.arange(cm.shape[0]),
#                # ... and label them with the respective list entries
#                xticklabels=name_classes, yticklabels=name_classes,
#                title=title,
#                ylabel='True label',
#                xlabel='Predicted label')
#
#         # Rotate the tick labels and set their alignment.
#         plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#                  rotation_mode="anchor")
#
#         # Loop over data dimensions and create text annotations.
#         fmt = '.2f' if normalize else 'd'
#         thresh = cm.max() / 2.
#         for i in range(cm.shape[0]):
#             for j in range(cm.shape[1]):
#                 ax.text(j, i, format(cm[i, j], fmt),
#                         ha="center", va="center",
#                         color="white" if cm[i, j] > thresh else "black")
#         fig.tight_layout()
#         plt.xlim(-0.5, len(np.unique(y_pred)) - 0.5)
#         plt.ylim(len(np.unique(y_pred)) - 0.5, -0.5)
#         if saving_path != "":
#             plt.savefig(saving_path)
#         return ax
#
#
# def merge_avg_mappings(dicts):
#     """ Merges an arbitrary number of dictionaries based on the
#     average value in a given mapping.
#
#     Parameters
#     ----------
#     dicts : Dict[Any, Comparable]
#
#     Returns
#     -------
#     Dict[Any, Comparable]
#         The merged dictionary
#     """
#     merged = {}
#     cpt = {}
#     for d in dicts:  # `dicts` is a list storing the input dictionaries
#         for key in d:
#             if key not in merged:
#                 if type(d[key]) == dict:
#                     merged[key] = {}
#                     for key2 in d[key]:
#                         merged[key][key2] = float(d[key][key2])
#                 else:
#                     merged[key] = d[key]
#                 cpt[key] = 1
#             else:
#                 if type(d[key]) == dict:
#                     for key2 in d[key]:
#                         merged[key][key2] += float(d[key][key2])
#                 else:
#                     merged[key] += d[key]
#                 cpt[key] += 1
#     for key in merged:
#         if type(merged[key]) == dict:
#             for key2 in merged[key]:
#                 merged[key][key2] /= cpt[key]
#         else:
#             merged[key] /= cpt[key]
#     return merged
#
# class Instance(object):
#     instance_id = 0
#     label_id = 0
#     vert_count = 0
#     med_dist = -1
#     dist_conf = 0.0
#
#     def __init__(self, mesh_vert_instances, instance_id):
#         if instance_id == -1:
#             return
#         self.instance_id = int(instance_id)
#         self.label_id = int(self.get_label_id(instance_id))
#         self.vert_count = int(
#             self.get_instance_verts(mesh_vert_instances, instance_id)
#         )
#
#     def get_label_id(self, instance_id):
#         return int(instance_id // 1000)
#
#     def get_instance_verts(self, mesh_vert_instances, instance_id):
#         return (mesh_vert_instances == instance_id).sum()
#
#     def to_json(self):
#         return json.dumps(
#             self, default=lambda o: o.__dict__, sort_keys=True, indent=4
#         )
#
#     def to_dict(self):
#         dict = {}
#         dict["instance_id"] = self.instance_id
#         dict["label_id"] = self.label_id
#         dict["vert_count"] = self.vert_count
#         dict["med_dist"] = self.med_dist
#         dict["dist_conf"] = self.dist_conf
#         return dict
#
#     def from_json(self, data):
#         self.instance_id = int(data["instance_id"])
#         self.label_id = int(data["label_id"])
#         self.vert_count = int(data["vert_count"])
#         if "med_dist" in data:
#             self.med_dist = float(data["med_dist"])
#             self.dist_conf = float(data["dist_conf"])
#
#     def __str__(self):
#         return "(" + str(self.instance_id) + ")"