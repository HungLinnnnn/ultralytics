# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Synthetic tests for H4 internals export plumbing.

These tests do not run model inference, prediction, validation, benchmarking, or
real-data diagnostics.
"""

import json
from types import SimpleNamespace

import numpy as np
import torch

from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import ops
from ultralytics.utils.h4_export import (
    build_pre_nms_candidate_tables,
    compute_mask_logits,
    write_segmentation_internals,
)


def test_h4_export_flags_are_explicit_and_default_off():
    cfg = get_cfg()

    assert cfg.h4_export_internals is False
    assert cfg.h4_export_logits is False
    assert cfg.h4_export_max_pre_nms == 300

    enabled = get_cfg(
        overrides={
            "h4_export_internals": True,
            "h4_export_logits": True,
            "h4_export_max_pre_nms": 2,
        }
    )
    assert enabled.h4_export_internals is True
    assert enabled.h4_export_logits is True
    assert enabled.h4_export_max_pre_nms == 2


def test_pre_nms_candidate_table_is_bounded_and_keeps_raw_indices():
    prediction = torch.tensor(
        [
            [
                [5.0, 15.0, 25.0],  # x center
                [5.0, 15.0, 25.0],  # y center
                [4.0, 4.0, 4.0],  # width
                [4.0, 4.0, 4.0],  # height
                [0.60, 0.95, 0.80],  # class score
                [0.10, 0.20, 0.30],  # coeff 0
                [0.40, 0.50, 0.60],  # coeff 1
            ]
        ]
    )

    table = build_pre_nms_candidate_tables(prediction, 0.5, None, nc=1, max_candidates=2, max_nms=30000)[0]

    assert table["indices"].tolist() == [1, 2]
    assert table["rows"].shape == (2, 8)
    assert torch.allclose(table["rows"][:, 4], torch.tensor([0.95, 0.80]))


def test_mask_logits_match_standard_thresholded_mask_path():
    proto = torch.ones((1, 2, 2))
    coeff = torch.tensor([[1.0]])
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])

    logits = compute_mask_logits(proto, coeff, boxes, (2, 2), retina_masks=False)
    masks = ops.process_mask(proto, coeff, boxes, (2, 2), upsample=True)

    assert torch.equal(logits.gt(0.0).byte(), masks)


def test_h4_writer_creates_joinable_manifest_and_npz(tmp_path):
    row = write_segmentation_internals(
        save_dir=tmp_path,
        h4_export_dir=None,
        image_path="/synthetic/image_001.png",
        metadata={"dataset": "synthetic", "split": "unit", "nms_config": {"conf": 0.25}},
        arrays={
            "proto": torch.zeros((2, 4, 4)),
            "coeff_post_nms": torch.ones((1, 2)),
            "masks_post_nms": torch.ones((1, 4, 4), dtype=torch.uint8),
        },
    )

    manifest = tmp_path / "h4_internals" / "manifest.jsonl"
    assert manifest.exists()
    manifest_row = json.loads(manifest.read_text().splitlines()[0])
    assert manifest_row["schema_version"] == "h4_baseline_internals_v1"
    assert manifest_row["image_id"] == "image_001"
    assert manifest_row["arrays"]["proto"] == [2, 4, 4]
    assert row["array_path"] == manifest_row["array_path"]

    arrays = np.load(manifest_row["array_path"])
    assert arrays["coeff_post_nms"].shape == (1, 2)


def test_segmentation_construct_result_exports_internals_without_inference(tmp_path):
    predictor = SegmentationPredictor(
        overrides={
            "mode": "predict",
            "task": "segment",
            "h4_export_internals": True,
            "h4_export_dir": str(tmp_path),
            "h4_export_logits": True,
        }
    )
    predictor.model = SimpleNamespace(names={0: "cell"})

    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.90, 0.0, 1.0]])
    img = torch.zeros((1, 3, 2, 2))
    orig_img = np.zeros((2, 2, 3), dtype=np.uint8)
    proto = torch.ones((1, 2, 2))
    h4_export = {
        "pre_nms_candidates": {"indices": torch.tensor([7]), "rows": pred.clone()},
        "kept_pre_nms_indices": torch.tensor([7]),
        "nms_config": {"conf": 0.25, "iou": 0.7},
        "input_tensor_shape": [1, 3, 2, 2],
    }

    result = predictor.construct_result(pred, img, orig_img, "/synthetic/image_002.png", proto, h4_export)

    assert len(result.boxes) == 1
    manifest_row = json.loads((tmp_path / "manifest.jsonl").read_text().splitlines()[0])
    assert manifest_row["image_id"] == "image_002"
    assert manifest_row["mask_process"] == "process_mask"
    assert manifest_row["mask_logits_saved"] is True

    arrays = np.load(manifest_row["array_path"])
    assert arrays["proto"].shape == (1, 2, 2)
    assert arrays["coeff_post_nms"].shape == (1, 1)
    assert arrays["mask_logits_post_nms"].shape == (1, 2, 2)
    assert arrays["kept_pre_nms_indices"].tolist() == [7]
