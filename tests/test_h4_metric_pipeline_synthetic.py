# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic H4 metric/data pipeline validation tests.

These tests use tiny in-memory DSB2018-style instance label maps only. They do
not run training, validation, prediction, benchmarking, or real-data diagnostics.
"""

import math

import numpy as np

from ultralytics.scripts import eval_seg_metric_for_yolo as metric


def _assert_close(actual, expected, *, tol=1e-6):
    assert math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol)


def _pq_value(stats):
    tp, fp, fn, iou_sum = stats
    if tp + fp + fn == 0:
        return 1.0
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = iou_sum / tp if tp else 0.0
    return dq * sq


def _intrusion_scores(pred_mask, neighbor_mask):
    intrusion = int(np.logical_and(pred_mask, neighbor_mask).sum())
    return intrusion, intrusion / int(pred_mask.sum()), intrusion / int(neighbor_mask.sum())


def test_empty_false_positive_and_false_negative_metric_contracts():
    empty = np.zeros((2, 2), dtype=np.int32)
    false_positive = np.array([[1, 0], [0, 0]], dtype=np.int32)
    false_negative_gt = np.array([[7, 0], [0, 0]], dtype=np.int32)

    _assert_close(metric.get_dice_1(empty, empty), 1.0)
    _assert_close(metric.get_fast_aji(empty, empty), 1.0)
    _assert_close(metric.get_fast_aji_plus(empty, empty), 1.0)
    assert metric.pq_stats_from_labels(empty, empty) == (0, 0, 0, 0.0)
    _assert_close(_pq_value(metric.pq_stats_from_labels(empty, empty)), 1.0)

    _assert_close(metric.get_dice_1(empty, false_positive), 0.0)
    _assert_close(metric.get_fast_aji(empty, false_positive), 0.0)
    _assert_close(metric.get_fast_aji_plus(empty, false_positive), 0.0)
    assert metric.pq_stats_from_labels(empty, false_positive) == (0, 1, 0, 0.0)

    _assert_close(metric.get_dice_1(false_negative_gt, empty), 0.0)
    _assert_close(metric.get_fast_aji(false_negative_gt, empty), 0.0)
    _assert_close(metric.get_fast_aji_plus(false_negative_gt, empty), 0.0)
    assert metric.pq_stats_from_labels(false_negative_gt, empty) == (0, 0, 1, 0.0)


def test_non_contiguous_instance_ids_remap_without_metric_change():
    gt = np.array([[0, 17], [0, 17]], dtype=np.int32)
    pred = np.array([[0, 5], [0, 5]], dtype=np.int32)

    gt_remapped = metric.remap_label(gt)
    pred_remapped = metric.remap_label(pred)

    assert gt_remapped.tolist() == [[0, 1], [0, 1]]
    assert pred_remapped.tolist() == [[0, 1], [0, 1]]
    _assert_close(metric.get_dice_1(gt, pred), 1.0)
    _assert_close(metric.get_fast_aji(gt_remapped, pred_remapped), 1.0)
    _assert_close(metric.get_fast_aji_plus(gt_remapped, pred_remapped), 1.0)
    assert metric.pq_stats_from_labels(gt, pred) == (1, 0, 0, 1.0)


def test_touching_instances_can_be_perfect_while_adjacency_is_positive():
    gt = np.array([[0, 0, 0, 0], [10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.int32)
    pred = np.array([[0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)

    _assert_close(metric.get_dice_1(gt, pred), 1.0)
    _assert_close(metric.get_fast_aji(metric.remap_label(gt), metric.remap_label(pred)), 1.0)
    _assert_close(metric.get_fast_aji_plus(metric.remap_label(gt), metric.remap_label(pred)), 1.0)
    assert metric.pq_stats_from_labels(gt, pred) == (2, 0, 0, 2.0)

    mask_10 = gt == 10
    mask_20 = gt == 20
    strict_horizontal_contact = np.logical_and(mask_10[:, :-1], mask_20[:, 1:]).any()
    assert strict_horizontal_contact


def test_merge_failure_is_exposed_by_aji_and_pq_but_hidden_by_foreground_dice():
    gt = np.array([[0, 0, 0, 0], [10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.int32)
    merged_pred = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.int32)

    gt_m = metric.remap_label(gt)
    pred_m = metric.remap_label(merged_pred)

    _assert_close(metric.get_dice_1(gt, merged_pred), 1.0)
    _assert_close(metric.get_fast_aji(gt_m, pred_m), 0.5)
    _assert_close(metric.get_fast_aji_plus(gt_m, pred_m), 1.0 / 3.0)
    assert metric.pq_stats_from_labels(gt, merged_pred) == (0, 1, 2, 0.0)
    _assert_close(_pq_value(metric.pq_stats_from_labels(gt, merged_pred)), 0.0)


def test_split_failure_is_exposed_by_aji_and_pq_but_hidden_by_foreground_dice():
    gt = np.array([[0, 0, 0, 0], [10, 10, 10, 10], [10, 10, 10, 10]], dtype=np.int32)
    split_pred = np.array([[0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)

    gt_m = metric.remap_label(gt)
    pred_m = metric.remap_label(split_pred)

    _assert_close(metric.get_dice_1(gt, split_pred), 1.0)
    _assert_close(metric.get_fast_aji(gt_m, pred_m), 1.0 / 3.0)
    _assert_close(metric.get_fast_aji_plus(gt_m, pred_m), 1.0 / 3.0)
    assert metric.pq_stats_from_labels(gt, split_pred) == (0, 2, 1, 0.0)
    _assert_close(_pq_value(metric.pq_stats_from_labels(gt, split_pred)), 0.0)


def test_duplicate_predictions_are_hidden_by_label_map_collapse_without_per_mask_metadata():
    gt = np.array([[0, 0], [1, 1], [1, 1]], dtype=np.int32)
    collapsed_label_map = np.array([[0, 0], [2, 2], [2, 2]], dtype=np.int32)

    _assert_close(metric.get_dice_1(gt, collapsed_label_map), 1.0)
    _assert_close(metric.get_fast_aji(metric.remap_label(gt), metric.remap_label(collapsed_label_map)), 1.0)
    _assert_close(metric.get_fast_aji_plus(metric.remap_label(gt), metric.remap_label(collapsed_label_map)), 1.0)
    _assert_close(_pq_value(metric.pq_stats_from_labels(gt, collapsed_label_map)), 1.0)

    # If two identical per-mask predictions survive NMS, instance-list accounting is TP=1, FP=1, FN=0.
    # The collapsed label map cannot represent that duplicate pressure by itself.
    _assert_close(_pq_value((1, 1, 0, 1.0)), 2.0 / 3.0)


def test_neighbor_intrusion_is_a_pairwise_error_not_only_an_image_metric():
    gt = np.array([[10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.int32)
    pred_for_10 = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int32) == 1
    neighbor_20 = gt == 20

    pixels, pred_norm, neighbor_norm = _intrusion_scores(pred_for_10, neighbor_20)

    assert pixels == 1
    _assert_close(pred_norm, 0.2)
    _assert_close(neighbor_norm, 0.25)


def test_imagewise_and_dataset_level_pq_can_diverge():
    empty = np.zeros((1, 1), dtype=np.int32)
    one_gt = np.ones((1, 1), dtype=np.int32)

    empty_pq = _pq_value(metric.pq_stats_from_labels(empty, empty))
    missed_pq = _pq_value(metric.pq_stats_from_labels(one_gt, empty))

    _assert_close((empty_pq + missed_pq) / 2.0, 0.5)
    _assert_close(_pq_value((0, 0, 1, 0.0)), 0.0)

    # Dense perfect image plus sparse failure image: image-wise PQ weights images equally,
    # while dataset-level PQ weights accumulated objects.
    _assert_close((1.0 + 0.0) / 2.0, 0.5)
    _assert_close(_pq_value((10, 0, 1, 10.0)), 10.0 / 10.5)

    # Sparse perfect image plus dense half-failure image shows the reverse divergence.
    _assert_close((1.0 + _pq_value((5, 0, 5, 5.0))) / 2.0, (1.0 + 2.0 / 3.0) / 2.0)
    _assert_close(_pq_value((6, 0, 5, 6.0)), 6.0 / 8.5)
