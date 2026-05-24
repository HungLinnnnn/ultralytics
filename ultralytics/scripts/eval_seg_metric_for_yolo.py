import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import os
from ultralytics import YOLO

from skimage import io

import numpy as np
import torch

from torchmetrics.detection import PanopticQuality
import argparse
from tqdm.auto import tqdm

from torchmetrics.functional.segmentation import dice_score


import cv2
import numpy as np
import pandas as pd

def resize_and_crop_center(pred, gt_shape):
    """
    將正方形影像 pred 調整成與 gt 長邊相同，並中心裁切成 gt 的大小。
    
    Args:
        pred (np.ndarray): 正方形影像 (Hb, Wb[, C])
        gt_shape (tuple): 目標尺寸 (Ha, Wa)
    Returns:
        np.ndarray: 裁切後的影像，大小與 a 相同
    """
    H, W = gt_shape
    # 1. resize 成 (Wa, Wa)
    pred = cv2.resize(pred, (W, W), interpolation=cv2.INTER_NEAREST)

    if H == W:
        return pred

    # 2. 計算 crop 起點
    y_start = (W - H) // 2
    y_end = y_start + H

    # 3. 中心裁切
    pred = pred[y_start:y_end, :, ...]
    return pred

def visualized_diff(pred, gt, filename):
    gt = gt > 0
    pred = pred > 0
    diff_mask = np.zeros(pred.shape + (3, ), np.uint8)

    # gray: 正確
    diff_mask[np.logical_and(pred, gt)] = [200, 200, 200]

    # red: False Positive
    diff_mask[np.logical_and(pred, np.logical_not(gt))] = [255, 160, 80]

    # blue: Flalse Negative
    diff_mask[np.logical_and(np.logical_not(pred), gt)] = [80, 80, 255]

    cv2.imwrite(filename, diff_mask)


# --------------------------Optimised for Speed
def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.

    """
    # 空白情況
    if np.max(true) == 0 and np.max(pred) == 0:
        return 1.0  # 視為完美空白匹配
    elif np.max(true) == 0 or np.max(pred) == 0:
        return 0.0  # 一邊空白、一邊非空
    
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score


#####
def get_fast_aji_plus(true, pred):
    """AJI+, an AJI version with maximal unique pairing to obtain overall intersecion.
    Every prediction instance is paired with at most 1 GT instance (1 to 1) mapping, unlike AJI 
    where a prediction instance can be paired against many GT instances (1 to many).
    Remaining unpaired GT and Prediction instances will be added to the overall union.
    The 1 to 1 mapping prevents AJI's over-penalisation from happening.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.

    """
    # 空白情況
    if np.max(true) == 0 and np.max(pred) == 0:
        return 1.0  # 視為完美空白匹配
    elif np.max(true) == 0 or np.max(pred) == 0:
        return 0.0  # 一邊空白、一邊非空
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    #### Munkres pairing to find maximal unique pairing
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    ### extract the paired cost and remove invalid pair
    paired_iou = pairwise_iou[paired_true, paired_pred]
    # now select all those paired with iou != 0.0 i.e have intersection
    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]
    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score


#####
def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"

    if np.max(true) == 0 and np.max(pred) == 0:
        # 視為完美匹配
        return [1.0, 1.0, 1.0], [[], [], [], []]
    elif np.max(true) == 0 or np.max(pred) == 0:
        # 視為錯誤匹配
        return [0.0, 0.0, 0.0], [[], [], [], []]

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


#####
def get_fast_dice_2(true, pred):
    """Ensemble dice."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))

    overall_total = 0
    overall_inter = 0

    true_masks = [np.zeros(true.shape)]
    for t in true_id[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [np.zeros(true.shape)]
    for p in pred_id[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    for true_idx in range(1, len(true_id)):
        t_mask = true_masks[true_idx]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        try:  # blinly remove background
            pred_true_overlap_id.remove(0)
        except ValueError:
            pass  # just mean no background
        for pred_idx in pred_true_overlap_id:
            p_mask = pred_masks[pred_idx]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            overall_total += total
            overall_inter += inter

    return 2 * overall_inter / overall_total


#####--------------------------As pseudocode
def get_dice_1(true, pred):
    """Traditional dice."""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = np.sum(true * pred)
    denom = np.sum(true) + np.sum(pred)

    # inter = true * pred 
    # denom = true + pred
    

    if denom == 0:
        return 1.0  # both empty → perfect match
    return 2.0 * inter / denom
    # return 2.0 * np.sum(inter) / np.sum(denom)


####
def get_dice_2(true, pred):
    """Ensemble Dice as used in Computational Precision Medicine Challenge."""
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0
    true_id.remove(0)
    pred_id.remove(0)

    total_markup = 0
    total_intersect = 0
    for t in true_id:
        t_mask = np.array(true == t, np.uint8)
        for p in pred_id:
            p_mask = np.array(pred == p, np.uint8)
            intersect = p_mask * t_mask
            if intersect.sum() > 0:
                total_intersect += intersect.sum()
                total_markup += t_mask.sum() + p_mask.sum()
    return 2 * total_intersect / total_markup


#####
def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


#####
def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal 
    unique pairing (largest possible match) when pairing points in set B 
    against points in set A, using distance as cost function.

    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points 
        radius: valid area around a point in setA to consider 
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired

    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence 
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances 
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:,None], pairedB[:,None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB


def pq_stats_from_labels(true: np.ndarray, pred: np.ndarray, match_iou: float = 0.5):
    """
    極速版本：用 contingency table 取得交集，計算單張圖的 PQ 統計量。
    回傳: (tp, fp, fn, iou_sum)
    - true, pred: instance label map (0=背景, 1..N 實例)
    - 建議先呼叫 remap_label，使 id 連續，但就算不連續也可（這裡會內部壓縮）
    """
    assert match_iou >= 0.0

    # ---- 早退：全空、單邊空 ----
    if np.max(true) == 0 and np.max(pred) == 0:
        return 0, 0, 0, 0.0
    if np.max(true) == 0:
        P = len(np.unique(pred)) - 1
        return 0, P, 0, 0.0
    if np.max(pred) == 0:
        T = len(np.unique(true)) - 1
        return 0, 0, T, 0.0

    # ---- 壓成連續 id（不修改原資料） ----
    t_ids, t_inv = np.unique(true, return_inverse=True)     # 0..(T')
    p_ids, p_inv = np.unique(pred, return_inverse=True)     # 0..(P')
    Tn, Pn = len(t_ids), len(p_ids)

    # ---- 共現表（交集像素數）：O(HW) -> 一次 bincount 完成 ----
    # 把 (t,p) 對應到一維索引，再 bincount 計數後 reshape 回 (Tn, Pn)
    pair_1d = t_inv.astype(np.int64) * Pn + p_inv.astype(np.int64)
    inter_mat = np.bincount(pair_1d, minlength=Tn * Pn).reshape(Tn, Pn).astype(np.float64)

    # ---- 背景列/行（label==0）索引 ----
    t_bg = int(np.where(t_ids == 0)[0][0]) if (t_ids[0] == 0 or 0 in t_ids) else None
    p_bg = int(np.where(p_ids == 0)[0][0]) if (p_ids[0] == 0 or 0 in p_ids) else None

    # ---- 各實例面積（行/列和就是該實例像素數） ----
    t_area_all = inter_mat.sum(axis=1)  # shape (Tn,)
    p_area_all = inter_mat.sum(axis=0)  # shape (Pn,)

    # ---- 去掉背景的子矩陣 ----
    t_keep = np.arange(Tn) != t_bg if t_bg is not None else np.ones(Tn, dtype=bool)
    p_keep = np.arange(Pn) != p_bg if p_bg is not None else np.ones(Pn, dtype=bool)
    inter = inter_mat[np.ix_(t_keep, p_keep)]                # (T, P)
    if inter.size == 0:
        # 一邊沒有前景實例
        T = int(t_keep.sum()); P = int(p_keep.sum())
        return 0, P, T, 0.0

    t_area = t_area_all[t_keep][:, None]                     # (T, 1)
    p_area = p_area_all[p_keep][None, :]                     # (1, P)
    union = t_area + p_area - inter                          # (T, P)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0, inter / union, 0.0)

    T = iou.shape[0]
    P = iou.shape[1]

    if match_iou >= 0.5:
        # 唯一性天然保證，直接閾值
        mask = (iou > match_iou)
        # 罕見情況下同一行/列可能多個 True（數值邊界），做個最保守處理：
        #   取每行/每列的最大 IoU 位置的交集，避免重複配對
        if mask.any():
            row_best = (iou == iou.max(axis=1, keepdims=True)) & mask
            col_best = (iou == iou.max(axis=0, keepdims=True)) & mask
            pick = row_best & col_best
            tp = int(pick.sum())
            iou_sum = float((iou * pick).sum())
        else:
            tp = 0
            iou_sum = 0.0
        fp = P - tp
        fn = T - tp
        return tp, fp, fn, iou_sum
    else:
        # 需要最大唯一配對：在稀疏對上跑匈牙利
        # 為避免把 0-iou 對也拿去配，先把不達閾值的設為 -inf（或極小）
        score = iou.copy()
        score[score <= match_iou] = -1e9
        # 匈牙利要極小化成本，所以用負分數
        cost = -score
        row_ind, col_ind = linear_sum_assignment(cost)
        # 保留有效配對（>threshold）
        valid = iou[row_ind, col_ind] > match_iou
        tp = int(valid.sum())
        iou_sum = float(iou[row_ind[valid], col_ind[valid]].sum())
        fp = P - tp
        fn = T - tp
        return tp, fp, fn, iou_sum

# ---------------- class wrapper ----------------
class InstanceSegEvaluator:
    """
    pq_reduce: {'dataset', 'imagewise'}
      - 'dataset': 累積所有影像的 (TP, FP, FN, IoU_sum) 後一次計算（= TiSeg 的 mDQ/mSQ/mPQ）
      - 'imagewise': 每張圖各自算 DQ/SQ/PQ，再做影像平均（= TiSeg 的 imwDQ/imwSQ/imwPQ）
    """
    def __init__(self, only_dice=False, match_iou=0.5, pq_reduce='imagewise'):
        assert pq_reduce in {'dataset', 'imagewise'}
        self.only_dice = only_dice
        self.match_iou = match_iou
        self.pq_reduce = pq_reduce

        # 通用的影像級指標容器（Dice/AJI/AJI+ 一律是 per-image 再平均）
        self.dice_list = []
        self.aji_list = []
        self.aji_plus_list = []

        # ------- PQ 兩種彙整策略 -------
        if pq_reduce == 'dataset':
            # 累積整個 dataset 的 TP/FP/FN/IoU_sum
            self.tp_total = 0
            self.fp_total = 0
            self.fn_total = 0
            self.iou_total = 0.0
        else:
            # 每張圖各自的 DQ/SQ/PQ，最後再平均
            self.imw_dq = []
            self.imw_sq = []
            self.imw_pq = []

    def update_one(self, pred_np: np.ndarray, gt_np: np.ndarray):
        gt_np = remap_label(gt_np)
        pred_np = remap_label(pred_np)
        out = {
            "Dice": 0.0, "AJI": 0.0, "AJI+": 0.0,
            "DQ": 0.0, "SQ": 0.0, "PQ": 0.0,
        }

        # ---- per-image Dice/AJI/AJI+（遇到空圖做防呆）----

        dice = float(np.nan_to_num(get_dice_1(gt_np, pred_np), nan=1.0))
        self.dice_list.append(dice)
        out['Dice'] = dice
        if not self.only_dice:
            aji = float(np.nan_to_num(get_fast_aji(gt_np, pred_np), nan=1.0))
            aji_plus = float(np.nan_to_num(get_fast_aji_plus(gt_np, pred_np), nan=1.0))
            self.aji_list.append(aji)
            self.aji_plus_list.append(aji_plus)
            out["AJI"] = aji
            out["AJI+"] = aji_plus

        # ---- PQ 的統計 or 影像級 ----
        if self.only_dice:
            return

        tp, fp, fn, iou_sum = pq_stats_from_labels(gt_np, pred_np, self.match_iou)
        if self.pq_reduce == 'dataset':
            # 累積到 dataset 級
            self.tp_total += tp
            self.fp_total += fp
            self.fn_total += fn
            self.iou_total += iou_sum
        else:
            # 影像級：每張直接算一次 DQ/SQ/PQ
            if (tp + fp + fn) == 0:
                dq_i = sq_i = pq_i = 1.0
            else:
                dq_i = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
                sq_i = iou_sum / (tp + 1e-6)
                pq_i = dq_i * sq_i
            self.imw_dq.append(float(dq_i))
            self.imw_sq.append(float(sq_i))
            self.imw_pq.append(float(pq_i))
            out.update({
                "DQ": dq_i,
                "SQ": sq_i,
                "PQ": pq_i
            })

        return out

    def compute(self):
        out = {
            "Dice": float(np.mean(self.dice_list)) if len(self.dice_list) else 0.0,
        }
        # AJI/AJI+
        if not self.only_dice:
            out["AJI"] = float(np.mean(self.aji_list)) if len(self.aji_list) else 0.0
            out["AJI+"] = float(np.mean(self.aji_plus_list)) if len(self.aji_plus_list) else 0.0

        if self.only_dice:
            # 只要 Dice 的情境
            out.setdefault("DQ", 0.0); out.setdefault("SQ", 0.0); out.setdefault("PQ", 0.0)
            out.setdefault("AJI", 0.0); out.setdefault("AJI+", 0.0)
            return out

        if self.pq_reduce == 'dataset':
            # ---- Dataset-level（加總一次算）----
            tp, fp, fn, iou_sum = self.tp_total, self.fp_total, self.fn_total, self.iou_total
            if (tp + fp + fn) == 0:
                dq = sq = pq = 1.0
            else:
                dq = tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)
                sq = iou_sum / (tp + 1e-6)
                pq = dq * sq
            out.update({"DQ": float(dq), "SQ": float(sq), "PQ": float(pq)})
        else:
            # ---- Image-wise（每張算完再平均）----
            dq = float(np.mean(self.imw_dq)) if self.imw_dq else 0.0
            sq = float(np.mean(self.imw_sq)) if self.imw_sq else 0.0
            pq = float(np.mean(self.imw_pq)) if self.imw_pq else 0.0
            out.update({"DQ": dq, "SQ": sq, "PQ": pq})

        return out
 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir', type=str, default='/home/r12922169/ultralytics/cell_project/data/dsb2018_stardist/test/images'
    )
    parser.add_argument(
        '--gt_dir', type=str, default='/home/r12922169/ultralytics/cell_project/data/dsb2018_stardist/test/masks'
    )
    parser.add_argument(
        '--dst_dir', type=str, default=None
    )
    parser.add_argument(
        '--weight', type=str, required=True
    )
    parser.add_argument(
        '--imgsz', type=int, default=640
    )
    parser.add_argument(
        '--conf', type=float, default=0.35
    )
    parser.add_argument(
        '--diff', action='store_true'
    )
    parser.add_argument(
        '--global_coef', action='store_true'
    )
    parser.add_argument(
        '--save_global_coef', action='store_true'
    )
    parser.add_argument(
        '--auto', action='store_true'
    )
    parser.add_argument(
        '--binary', action='store_true'
    )
    parser.add_argument(
        '--save_csv', action='store_true'
    )
    parser.add_argument(
        "--csv_path", type=str, default="/home/r12922169/ultralytics/cell_project/experiment/new_dsb/info.csv"
    )
    parser.add_argument(
        "--device", type=str, default='3'
    )

    return parser.parse_args()


def eval(image_dir, gt_dir, dst_dir,  weight, imgsz, conf, diff, global_coef, save_global_coef, binary, save_csv, csv_path, device, auto=None):
    # auto是為了配合格式才傳進來的，用不到
    image_paths = sorted(os.listdir(image_dir))
    label_paths = sorted(os.listdir(gt_dir))

    assert len(image_paths) == len(label_paths)

    yolo = YOLO(weight)

    if dst_dir is not None:
        os.makedirs(dst_dir, exist_ok=True)

    only_dice = global_coef or binary
    evaluator = InstanceSegEvaluator(only_dice=only_dice)

    if save_csv:
        all_data = []
        column_name = ["Filename", "Dice", "AJI", "AJI+", "DQ", "SQ", "PQ"]

    for image_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
        result = yolo.predict(
            source=os.path.join(image_dir, image_path),
            imgsz=imgsz,
            show_boxes=False,
            rect=False,
            # save=True,
            show_labels=False,
            device=device,
            conf=conf,
            iou=0.45,
            verbose=False,
            half=False,
            retina_masks=True,
            # mask_ratio=2,
            # show_conf=True,
            # filename=image_path
        )

        if global_coef:
            # global coef模式底下只計算dice
            # 1) 讀取指定tmp.png
            pred_mask = cv2.imread('/home/r12922169/ultralytics/cell_project/experiment/global_coef.png', cv2.IMREAD_GRAYSCALE)
            
            # load gt
            gt = io.imread(os.path.join(gt_dir, label_path))
            # pad
            pred_mask = resize_and_crop_center(pred_mask, gt.shape)
            evaluator.update_one(pred_mask, gt)

            if save_global_coef:
                global_coef_dir = '/home/r12922169/ultralytics/cell_project/experiment/tmp_global_coef'
                basename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
                cv2.imwrite(os.path.join(global_coef_dir, basename), pred_mask)
           
        elif binary:
            # 1) 讀取指定binary.png
            pred_mask = cv2.imread('/home/r12922169/ultralytics/cell_project/experiment/binary.png', cv2.IMREAD_GRAYSCALE)
            # load gt
            gt = io.imread(os.path.join(gt_dir, label_path))
            pred_mask = resize_and_crop_center(pred_mask, gt.shape)
            evaluator.update_one(pred_mask, gt)

            # 一律save diff
            dst_dir = '/home/r12922169/ultralytics/cell_project/experiment/tmp_binary_diff'
            basename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
            visualized_diff(pred_mask, gt, os.path.join(dst_dir, basename))


        else :
            if dst_dir is not None:
                im = result[0].plot(color_mode='instance', boxes=False, labels=False, line_width=1)
                basename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
                cv2.imwrite(os.path.join(dst_dir, basename), im)
            

            # load gt
            gt = io.imread(os.path.join(gt_dir, label_path))
            #assert len(gt.shape) == 2, f"{gt.shape}, {image_path}"
            if len(gt.shape) == 3:
                gt = gt[0]
            # 產生mask
            if result[0].masks is not None:
                masks = result[0].masks.data
                instance_ids = torch.arange(1, masks.shape[0] + 1, device=masks.device).view(-1, 1, 1)
                # 每個 pixel 選擇最大 instance id (若重疊則取 id 較大的)
                merged = (masks * instance_ids).max(dim=0).values.cpu().numpy()  # shape [256, 256]
            else:
                # 沒有預設出任何mask
                merged = np.zeros_like(gt, dtype=np.int32)
            
            if diff:
                
                #dst_dir = dst_dir if dst_dir is not None else '/home/r12922169/ultralytics/cell_project/experiment/tmp_diff'
                diff_dir = '/home/r12922169/ultralytics/cell_project/experiment/tmp_result_diff'
                os.makedirs(diff_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(image_path))[0] + '.png'
                visualized_diff(merged, gt, os.path.join(diff_dir, basename))

                # im = result[0].plot(color_mode='instance', boxes=True, labels=False, line_width=1)
                # result_dir = '/home/r12922169/ultralytics/cell_project/experiment/tmp_result_diff'
                # cv2.imwrite(os.path.join(result_dir, basename), im)



            info = evaluator.update_one(merged, gt)
            if save_csv:
                basename = os.path.basename(image_path)
                basename_stem = os.path.splitext(basename)[0]
                info['Filename'] = basename_stem
                all_data.append(info)
    if save_csv:
        final_df = pd.DataFrame(all_data, columns=column_name)
        final_df.to_csv(csv_path, index=False, encoding='utf-8')
                

         

    metrics = evaluator.compute()
    print(metrics)
    print(f"{metrics['Dice']:.3f},{metrics['AJI']:.3f},{metrics['DQ']:.3f},{metrics['SQ']:.3f},{metrics['PQ']:.3f}")


def main():
    args = vars(get_args())

    if not args['auto']:
        eval(**args)
    else:
        # 自動inferecne模式，關掉所有繪製相關
        args['diff'] = False
        args['global_coef'] = False
        args['save_global_coef'] = False
        args['binary'] = False
        
        weight_dir = os.path.dirname(args['weight'])
        all_epoch_pt = [f for f in os.listdir(weight_dir) if f not in {'best.pt', 'last.pt'}]
        
        all_pt = sorted(all_epoch_pt, reverse=True, key=lambda x: int(x[5:-3]))
        used_pt = all_pt[:min(len(all_epoch_pt), 30)]  # 'best.pt', :min(len(all_epoch_pt), 30) ['last.pt'] + 
        for pt in used_pt:
            args['weight'] = os.path.join(weight_dir, pt)
            print(f"\n{pt}")
            eval(**args)



if  __name__ == '__main__':
    main()