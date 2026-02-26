# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Legacy instance segmentation metrics copied from eval_seg_metric_for_yolo.py.

This module intentionally keeps the legacy behavior and formula definitions so users can
switch between native and legacy metric backends during validation.
"""

from __future__ import annotations

import numpy as np

__all__ = (
    "stack_masks_to_label_map",
    "remap_label",
    "get_dice_1",
    "get_fast_aji",
    "get_fast_aji_plus",
    "pq_stats_from_labels",
    "compute_imagewise_pq",
)


def stack_masks_to_label_map(mask_stack: np.ndarray) -> np.ndarray:
    """Convert [N, H, W] instance mask stack to a single label map, matching legacy behavior.

    Overlapping pixels are assigned to the larger instance id (same as legacy script).
    """
    if mask_stack.ndim != 3:
        raise ValueError(f"mask_stack must be 3D [N,H,W], got shape={mask_stack.shape}")

    n, h, w = mask_stack.shape
    if n == 0:
        return np.zeros((h, w), dtype=np.int32)

    ids = np.arange(1, n + 1, dtype=np.int32).reshape(-1, 1, 1)
    return (mask_stack.astype(np.int32) * ids).max(axis=0).astype(np.int32)


def get_dice_1(true: np.ndarray, pred: np.ndarray) -> float:
    """Traditional binary Dice used by legacy script."""
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = np.sum(true * pred)
    denom = np.sum(true) + np.sum(pred)

    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def remap_label(pred: np.ndarray, by_size: bool = False) -> np.ndarray:
    """Rename all instance IDs so labels are contiguous [0, 1, 2, ...]."""
    pred_id = list(np.unique(pred))
    if 0 in pred_id:
        pred_id.remove(0)
    if len(pred_id) == 0:
        return pred

    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        pair_list = sorted(zip(pred_id, pred_size), key=lambda x: x[1], reverse=True)
        pred_id, _ = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_fast_aji(true: np.ndarray, pred: np.ndarray) -> float:
    """AJI implementation from legacy script (MoNuSeg style)."""
    if np.max(true) == 0 and np.max(pred) == 0:
        return 1.0
    if np.max(true) == 0 or np.max(pred) == 0:
        return 0.0

    # Keep legacy behavior while guarding against non-contiguous instance IDs.
    true = remap_label(np.copy(true))
    pred = remap_label(np.copy(pred))
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_ids = [int(t) for t in true_id_list[1:]]
    pred_ids = [int(p) for p in pred_id_list[1:]]
    true_masks = {t: np.array(true == t, np.uint8) for t in true_ids}
    pred_masks = {p: np.array(pred == p, np.uint8) for p in pred_ids}
    true_id_to_idx = {t: i for i, t in enumerate(true_ids)}
    pred_id_to_idx = {p: i for i, p in enumerate(pred_ids)}

    pairwise_inter = np.zeros([len(true_ids), len(pred_ids)], dtype=np.float64)
    pairwise_union = np.zeros([len(true_ids), len(pred_ids)], dtype=np.float64)

    for true_id in true_ids:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            pred_id = int(pred_id)
            if pred_id == 0 or pred_id not in pred_masks:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            ti = true_id_to_idx[true_id]
            pi = pred_id_to_idx[pred_id]
            pairwise_inter[ti, pi] = inter
            pairwise_union[ti, pi] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]

    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true_ids = {true_ids[i] for i in paired_true.tolist()}
    paired_pred_ids = {pred_ids[i] for i in paired_pred.tolist()}

    unpaired_true = [idx for idx in true_ids if idx not in paired_true_ids]
    unpaired_pred = [idx for idx in pred_ids if idx not in paired_pred_ids]
    for true_id in unpaired_true:
        overall_union += true_masks[int(true_id)].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[int(pred_id)].sum()

    return float(overall_inter / overall_union)


def get_fast_aji_plus(true: np.ndarray, pred: np.ndarray) -> float:
    """AJI+ from legacy script (kept for optional future reporting)."""
    from scipy.optimize import linear_sum_assignment

    if np.max(true) == 0 and np.max(pred) == 0:
        return 1.0
    if np.max(true) == 0 or np.max(pred) == 0:
        return 0.0

    # Keep legacy behavior while guarding against non-contiguous instance IDs.
    true = remap_label(np.copy(true))
    pred = remap_label(np.copy(pred))
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_ids = [int(t) for t in true_id_list[1:]]
    pred_ids = [int(p) for p in pred_id_list[1:]]
    true_masks = {t: np.array(true == t, np.uint8) for t in true_ids}
    pred_masks = {p: np.array(pred == p, np.uint8) for p in pred_ids}
    true_id_to_idx = {t: i for i, t in enumerate(true_ids)}
    pred_id_to_idx = {p: i for i, p in enumerate(pred_ids)}

    pairwise_inter = np.zeros([len(true_ids), len(pred_ids)], dtype=np.float64)
    pairwise_union = np.zeros([len(true_ids), len(pred_ids)], dtype=np.float64)

    for true_id in true_ids:
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(np.unique(pred_true_overlap))
        for pred_id in pred_true_overlap_id:
            pred_id = int(pred_id)
            if pred_id == 0 or pred_id not in pred_masks:
                continue
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            ti = true_id_to_idx[true_id]
            pi = pred_id_to_idx[pred_id]
            pairwise_inter[ti, pi] = inter
            pairwise_union[ti, pi] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    paired_iou = pairwise_iou[paired_true, paired_pred]

    paired_true = paired_true[paired_iou > 0.0]
    paired_pred = paired_pred[paired_iou > 0.0]
    paired_inter = pairwise_inter[paired_true, paired_pred]
    paired_union = pairwise_union[paired_true, paired_pred]

    paired_true_ids = {true_ids[i] for i in paired_true.tolist()}
    paired_pred_ids = {pred_ids[i] for i in paired_pred.tolist()}
    overall_inter = paired_inter.sum()
    overall_union = paired_union.sum()

    unpaired_true = [idx for idx in true_ids if idx not in paired_true_ids]
    unpaired_pred = [idx for idx in pred_ids if idx not in paired_pred_ids]
    for true_id in unpaired_true:
        overall_union += true_masks[int(true_id)].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[int(pred_id)].sum()

    return float(overall_inter / overall_union)


def pq_stats_from_labels(true: np.ndarray, pred: np.ndarray, match_iou: float = 0.5) -> tuple[int, int, int, float]:
    """Fast PQ stats from two label maps.

    Returns:
        (tp, fp, fn, iou_sum)
    """
    assert match_iou >= 0.0

    if np.max(true) == 0 and np.max(pred) == 0:
        return 0, 0, 0, 0.0
    if np.max(true) == 0:
        p = len(np.unique(pred)) - 1
        return 0, p, 0, 0.0
    if np.max(pred) == 0:
        t = len(np.unique(true)) - 1
        return 0, 0, t, 0.0

    t_ids, t_inv = np.unique(true, return_inverse=True)
    p_ids, p_inv = np.unique(pred, return_inverse=True)
    t_inv = t_inv.ravel()
    p_inv = p_inv.ravel()
    tn, pn = len(t_ids), len(p_ids)

    pair_1d = t_inv.astype(np.int64) * pn + p_inv.astype(np.int64)
    inter_mat = np.bincount(pair_1d, minlength=tn * pn).reshape(tn, pn).astype(np.float64)

    t_bg = int(np.where(t_ids == 0)[0][0]) if (t_ids[0] == 0 or 0 in t_ids) else None
    p_bg = int(np.where(p_ids == 0)[0][0]) if (p_ids[0] == 0 or 0 in p_ids) else None

    t_area_all = inter_mat.sum(axis=1)
    p_area_all = inter_mat.sum(axis=0)

    t_keep = np.arange(tn) != t_bg if t_bg is not None else np.ones(tn, dtype=bool)
    p_keep = np.arange(pn) != p_bg if p_bg is not None else np.ones(pn, dtype=bool)
    inter = inter_mat[np.ix_(t_keep, p_keep)]
    if inter.size == 0:
        t = int(t_keep.sum())
        p = int(p_keep.sum())
        return 0, p, t, 0.0

    t_area = t_area_all[t_keep][:, None]
    p_area = p_area_all[p_keep][None, :]
    union = t_area + p_area - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)

    t = iou.shape[0]
    p = iou.shape[1]

    if match_iou >= 0.5:
        mask = iou > match_iou
        if mask.any():
            row_best = (iou == iou.max(axis=1, keepdims=True)) & mask
            col_best = (iou == iou.max(axis=0, keepdims=True)) & mask
            pick = row_best & col_best
            tp = int(pick.sum())
            iou_sum = float((iou * pick).sum())
        else:
            tp = 0
            iou_sum = 0.0
        fp = p - tp
        fn = t - tp
        return tp, fp, fn, iou_sum

    from scipy.optimize import linear_sum_assignment

    score = iou.copy()
    score[score <= match_iou] = -1e9
    cost = -score
    row_ind, col_ind = linear_sum_assignment(cost)
    valid = iou[row_ind, col_ind] > match_iou
    tp = int(valid.sum())
    iou_sum = float(iou[row_ind[valid], col_ind[valid]].sum())
    fp = p - tp
    fn = t - tp
    return tp, fp, fn, iou_sum


def compute_imagewise_pq(tp: int, fp: int, fn: int, iou_sum: float, eps: float = 1e-6) -> float:
    """Legacy imagewise PQ reduction from (tp, fp, fn, iou_sum)."""
    if (tp + fp + fn) == 0:
        return 1.0
    dq = tp / (tp + 0.5 * fp + 0.5 * fn + eps)
    sq = iou_sum / (tp + eps)
    return float(dq * sq)
