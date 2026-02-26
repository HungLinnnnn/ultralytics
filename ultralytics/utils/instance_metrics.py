# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Instance-level segmentation metrics utilities for validation-time monitoring."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_flat_bool_masks(masks: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Convert masks to [N, HW] boolean arrays."""
    arr = np.asarray(masks)
    if arr.ndim == 1:
        if arr.size == 0:
            return np.zeros((0, 0), dtype=bool)
        raise ValueError(f"Expected mask array with ndim 2/3, got shape={arr.shape}")
    if arr.ndim == 2:
        return arr.astype(bool, copy=False)
    if arr.ndim == 3:
        if arr.shape[0] == 0:
            return np.zeros((0, int(arr.shape[1] * arr.shape[2])), dtype=bool)
        return arr.astype(bool, copy=False).reshape(arr.shape[0], -1)
    raise ValueError(f"Expected mask array with ndim 2/3, got shape={arr.shape}")


def compute_iou_matrix(gt_masks: np.ndarray | list[np.ndarray], pred_masks: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Compute IoU matrix between GT and prediction instance masks."""
    gt = _to_flat_bool_masks(gt_masks)
    pred = _to_flat_bool_masks(pred_masks)
    ng, npred = gt.shape[0], pred.shape[0]
    if ng == 0 or npred == 0:
        return np.zeros((ng, npred), dtype=np.float64)
    if gt.shape[1] != pred.shape[1]:
        raise ValueError(f"Mask size mismatch: gt HW={gt.shape[1]} vs pred HW={pred.shape[1]}")

    gt_i = gt.astype(np.int32, copy=False)
    pred_i = pred.astype(np.int32, copy=False)
    inter = gt_i @ pred_i.T
    gt_area = gt_i.sum(axis=1, dtype=np.int64)[:, None]
    pred_area = pred_i.sum(axis=1, dtype=np.int64)[None, :]
    union = gt_area + pred_area - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0).astype(np.float64)


def _match_hungarian(iou_mat: np.ndarray, thr: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    """One-to-one matching via Hungarian assignment."""
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
    if row_ind.size == 0:
        return [], np.zeros(0, dtype=np.float64)

    picked_ious = iou_mat[row_ind, col_ind]
    keep = picked_ious >= thr
    if not keep.any():
        return [], np.zeros(0, dtype=np.float64)
    matches = [(int(g), int(p)) for g, p, k in zip(row_ind, col_ind, keep) if k]
    return matches, picked_ious[keep].astype(np.float64)


def _match_greedy(iou_mat: np.ndarray, thr: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Greedy IoU-descending one-to-one matching."""
    ng, npred = iou_mat.shape
    if ng == 0 or npred == 0:
        return [], np.zeros(0, dtype=np.float64)

    flat = iou_mat.reshape(-1)
    order = np.argsort(flat)[::-1]
    used_g, used_p = set(), set()
    matches, matched_ious = [], []
    for idx in order:
        iou = float(flat[idx])
        if iou < thr:
            break
        g = int(idx // npred)
        p = int(idx % npred)
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        matches.append((g, p))
        matched_ious.append(iou)
    return matches, np.asarray(matched_ious, dtype=np.float64)


def match_instances(iou_mat: np.ndarray, thr: float = 0.5, method: str = "auto") -> dict[str, Any]:
    """Match instances from IoU matrix and return TP/FP/FN components."""
    if iou_mat.ndim != 2:
        raise ValueError(f"iou_mat must be 2D, got shape={iou_mat.shape}")
    if thr < 0.0:
        raise ValueError(f"thr must be >= 0, got {thr}")

    ng, npred = iou_mat.shape
    if ng == 0 or npred == 0:
        return {
            "matches": [],
            "matched_ious": np.zeros(0, dtype=np.float64),
            "tp": 0,
            "fp": int(npred),
            "fn": int(ng),
            "iou_sum": 0.0,
            "method_used": "none",
        }

    method = method.lower()
    if method not in {"auto", "hungarian", "greedy"}:
        raise ValueError(f"Unsupported matching method: {method}")

    matches: list[tuple[int, int]]
    matched_ious: np.ndarray
    method_used = method
    if method in {"auto", "hungarian"}:
        try:
            matches, matched_ious = _match_hungarian(iou_mat, thr)
            method_used = "hungarian"
        except Exception as e:
            if method == "hungarian":
                raise ImportError("Hungarian matching requires scipy.optimize.linear_sum_assignment") from e
            matches, matched_ious = _match_greedy(iou_mat, thr)
            method_used = "greedy"
    else:
        matches, matched_ious = _match_greedy(iou_mat, thr)
        method_used = "greedy"

    tp = int(len(matches))
    fp = int(npred - tp)
    fn = int(ng - tp)
    iou_sum = float(matched_ious.sum()) if matched_ious.size else 0.0
    return {
        "matches": matches,
        "matched_ious": matched_ious,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou_sum": iou_sum,
        "method_used": method_used,
    }


def compute_pq_components(
    gt_masks: np.ndarray | list[np.ndarray], pred_masks: np.ndarray | list[np.ndarray], thr: float = 0.5, method: str = "auto"
) -> dict[str, Any]:
    """Compute PQ decomposition (PQ/SQ/RQ) with TP/FP/FN details."""
    iou_mat = compute_iou_matrix(gt_masks, pred_masks)
    return compute_pq_components_from_iou(iou_mat, thr=thr, method=method)


def compute_pq_components_from_iou(iou_mat: np.ndarray, thr: float = 0.5, method: str = "auto") -> dict[str, Any]:
    """Compute PQ decomposition (PQ/SQ/RQ) directly from a precomputed IoU matrix."""
    if iou_mat.ndim != 2:
        raise ValueError(f"iou_mat must be 2D, got shape={iou_mat.shape}")
    m = match_instances(iou_mat, thr=thr, method=method)
    tp, fp, fn, iou_sum = m["tp"], m["fp"], m["fn"], m["iou_sum"]
    rq_denom = tp + 0.5 * fp + 0.5 * fn
    rq = float(tp / rq_denom) if rq_denom > 0 else 0.0
    sq = float(iou_sum / tp) if tp > 0 else 0.0
    pq = float(sq * rq)
    mean_iou_tp = sq

    return {
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "iou_sum": float(iou_sum),
        "mean_iou_tp": float(mean_iou_tp),
        "ng": int(iou_mat.shape[0]),
        "npred": int(iou_mat.shape[1]),
        "iou_mat": iou_mat,
        "matches": m["matches"],
        "matched_ious": m["matched_ious"],
        "method_used": m["method_used"],
    }


def compute_split_merge(
    gt_masks: np.ndarray | list[np.ndarray], pred_masks: np.ndarray | list[np.ndarray], iou_mat: np.ndarray, thr: float = 0.5
) -> dict[str, float]:
    """Compute split/merge rates from full IoU matrix."""
    ng = int(_to_flat_bool_masks(gt_masks).shape[0])
    npred = int(_to_flat_bool_masks(pred_masks).shape[0])
    return compute_split_merge_from_iou(iou_mat, ng=ng, npred=npred, thr=thr)


def compute_split_merge_from_iou(iou_mat: np.ndarray, ng: int, npred: int, thr: float = 0.5) -> dict[str, float]:
    """Compute split/merge rates directly from a precomputed IoU matrix."""
    if iou_mat.ndim != 2:
        raise ValueError(f"iou_mat must be 2D, got shape={iou_mat.shape}")
    if iou_mat.shape != (ng, npred):
        raise ValueError(f"iou_mat shape {iou_mat.shape} does not match ng/npred ({ng}, {npred})")

    if ng == 0:
        split_count = 0
    else:
        split_count = int(((iou_mat >= thr).sum(axis=1) >= 2).sum())
    if npred == 0:
        merge_count = 0
    else:
        merge_count = int(((iou_mat >= thr).sum(axis=0) >= 2).sum())

    split_rate = float(split_count / ng) if ng > 0 else 0.0
    merge_rate = float(merge_count / npred) if npred > 0 else 0.0
    return {
        "split_count": float(split_count),
        "merge_count": float(merge_count),
        "split_rate": split_rate,
        "merge_rate": merge_rate,
    }
