# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import builtins

import numpy as np
import pytest

from ultralytics.utils.instance_metrics import (
    compute_iou_matrix,
    compute_pq_components,
    compute_pq_components_from_iou,
    compute_split_merge,
    compute_split_merge_from_iou,
    match_instances,
)


def _mask(h=8, w=8):
    return np.zeros((h, w), dtype=np.uint8)


def test_perfect_match():
    gt = _mask()
    gt[1:5, 2:6] = 1
    pred = gt.copy()

    gt_masks = np.stack([gt], axis=0)
    pred_masks = np.stack([pred], axis=0)
    comp = compute_pq_components(gt_masks, pred_masks, thr=0.5, method="greedy")
    sm = compute_split_merge(gt_masks, pred_masks, comp["iou_mat"], thr=0.5)

    assert comp["tp"] == 1
    assert comp["fp"] == 0
    assert comp["fn"] == 0
    assert comp["pq"] == pytest.approx(1.0)
    assert comp["sq"] == pytest.approx(1.0)
    assert comp["rq"] == pytest.approx(1.0)
    assert sm["split_rate"] == pytest.approx(0.0)
    assert sm["merge_rate"] == pytest.approx(0.0)


def test_split_case():
    gt = _mask()
    gt[1:5, 2:6] = 1

    pred1 = _mask()
    pred1[1:5, 2:4] = 1
    pred2 = _mask()
    pred2[1:5, 4:6] = 1

    gt_masks = np.stack([gt], axis=0)
    pred_masks = np.stack([pred1, pred2], axis=0)
    comp = compute_pq_components(gt_masks, pred_masks, thr=0.5, method="greedy")
    sm = compute_split_merge(gt_masks, pred_masks, comp["iou_mat"], thr=0.5)

    assert sm["split_count"] >= 1
    assert sm["split_rate"] > 0.0


def test_merge_case():
    gt1 = _mask()
    gt1[1:3, 1:3] = 1
    gt2 = _mask()
    gt2[1:3, 4:6] = 1
    pred = _mask()
    pred[1:3, 1:3] = 1
    pred[1:3, 4:6] = 1

    gt_masks = np.stack([gt1, gt2], axis=0)
    pred_masks = np.stack([pred], axis=0)
    comp = compute_pq_components(gt_masks, pred_masks, thr=0.5, method="greedy")
    sm = compute_split_merge(gt_masks, pred_masks, comp["iou_mat"], thr=0.5)

    assert sm["merge_count"] >= 1
    assert sm["merge_rate"] > 0.0


def test_empty_pred():
    gt = _mask()
    gt[1:5, 2:6] = 1
    gt_masks = np.stack([gt], axis=0)
    pred_masks = np.zeros((0, *gt.shape), dtype=np.uint8)
    comp = compute_pq_components(gt_masks, pred_masks, thr=0.5, method="greedy")
    sm = compute_split_merge(gt_masks, pred_masks, comp["iou_mat"], thr=0.5)

    assert comp["tp"] == 0
    assert comp["fp"] == 0
    assert comp["fn"] == 1
    assert sm["merge_rate"] == pytest.approx(0.0)


def test_empty_gt():
    pred = _mask()
    pred[1:5, 2:6] = 1
    gt_masks = np.zeros((0, *pred.shape), dtype=np.uint8)
    pred_masks = np.stack([pred], axis=0)
    comp = compute_pq_components(gt_masks, pred_masks, thr=0.5, method="greedy")
    sm = compute_split_merge(gt_masks, pred_masks, comp["iou_mat"], thr=0.5)

    assert comp["tp"] == 0
    assert comp["fn"] == 0
    assert comp["fp"] == 1
    assert sm["split_rate"] == pytest.approx(0.0)


def test_matching_auto_fallback(monkeypatch):
    orig_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy.optimize":
            raise ImportError("mock scipy missing")
        return orig_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    iou_mat = np.array([[0.9, 0.2], [0.3, 0.8]], dtype=np.float64)
    out = match_instances(iou_mat, thr=0.5, method="auto")
    assert out["method_used"] == "greedy"
    assert out["tp"] == 2


def test_pq_from_iou_equivalence():
    gt1 = _mask()
    gt1[1:5, 1:5] = 1
    gt2 = _mask()
    gt2[4:7, 4:7] = 1
    pred1 = gt1.copy()
    pred2 = _mask()
    pred2[4:7, 4:7] = 1
    pred3 = _mask()
    pred3[0:2, 6:8] = 1

    gt_masks = np.stack([gt1, gt2], axis=0)
    pred_masks = np.stack([pred1, pred2, pred3], axis=0)
    iou_mat = compute_iou_matrix(gt_masks, pred_masks)

    direct = compute_pq_components(gt_masks, pred_masks, thr=0.5, method="greedy")
    from_iou = compute_pq_components_from_iou(iou_mat, thr=0.5, method="greedy")

    assert direct["tp"] == from_iou["tp"]
    assert direct["fp"] == from_iou["fp"]
    assert direct["fn"] == from_iou["fn"]
    assert direct["pq"] == pytest.approx(from_iou["pq"])
    assert direct["sq"] == pytest.approx(from_iou["sq"])
    assert direct["rq"] == pytest.approx(from_iou["rq"])
    assert direct["iou_sum"] == pytest.approx(from_iou["iou_sum"])


def test_split_merge_from_iou_equivalence():
    gt = _mask()
    gt[1:5, 2:6] = 1
    pred1 = _mask()
    pred1[1:5, 2:4] = 1
    pred2 = _mask()
    pred2[1:5, 4:6] = 1

    gt_masks = np.stack([gt], axis=0)
    pred_masks = np.stack([pred1, pred2], axis=0)
    iou_mat = compute_iou_matrix(gt_masks, pred_masks)

    direct = compute_split_merge(gt_masks, pred_masks, iou_mat=iou_mat, thr=0.5)
    from_iou = compute_split_merge_from_iou(iou_mat, ng=gt_masks.shape[0], npred=pred_masks.shape[0], thr=0.5)

    assert direct["split_count"] == pytest.approx(from_iou["split_count"])
    assert direct["merge_count"] == pytest.approx(from_iou["merge_count"])
    assert direct["split_rate"] == pytest.approx(from_iou["split_rate"])
    assert direct["merge_rate"] == pytest.approx(from_iou["merge_rate"])


def test_empty_iou_from_iou_path():
    iou_empty = np.zeros((0, 0), dtype=np.float64)
    comp_empty = compute_pq_components_from_iou(iou_empty, thr=0.5, method="greedy")
    sm_empty = compute_split_merge_from_iou(iou_empty, ng=0, npred=0, thr=0.5)
    assert comp_empty["tp"] == 0
    assert comp_empty["fp"] == 0
    assert comp_empty["fn"] == 0
    assert comp_empty["pq"] == pytest.approx(0.0)
    assert comp_empty["sq"] == pytest.approx(0.0)
    assert comp_empty["rq"] == pytest.approx(0.0)
    assert sm_empty["split_rate"] == pytest.approx(0.0)
    assert sm_empty["merge_rate"] == pytest.approx(0.0)

    iou_gt_only = np.zeros((2, 0), dtype=np.float64)
    comp_gt_only = compute_pq_components_from_iou(iou_gt_only, thr=0.5, method="greedy")
    sm_gt_only = compute_split_merge_from_iou(iou_gt_only, ng=2, npred=0, thr=0.5)
    assert comp_gt_only["tp"] == 0
    assert comp_gt_only["fp"] == 0
    assert comp_gt_only["fn"] == 2
    assert sm_gt_only["merge_rate"] == pytest.approx(0.0)

    iou_pred_only = np.zeros((0, 3), dtype=np.float64)
    comp_pred_only = compute_pq_components_from_iou(iou_pred_only, thr=0.5, method="greedy")
    sm_pred_only = compute_split_merge_from_iou(iou_pred_only, ng=0, npred=3, thr=0.5)
    assert comp_pred_only["tp"] == 0
    assert comp_pred_only["fp"] == 3
    assert comp_pred_only["fn"] == 0
    assert sm_pred_only["split_rate"] == pytest.approx(0.0)
