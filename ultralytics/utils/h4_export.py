# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""H4 prototype ownership diagnostic export helpers.

The helpers in this module are inert unless an H4 export flag is enabled by a
caller. They collect tensors that standard YOLO segmentation ``Results`` do not
retain: prototypes, mask coefficients, pre/post-NMS identity, masks, optional
mask logits, and coordinate metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import ops

SCHEMA_VERSION = "h4_baseline_internals_v1"


def _jsonable(value: Any) -> Any:
    """Convert tensors, paths, and numpy scalars into JSON-serializable values."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensor-like values to CPU numpy arrays for npz storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _shape(value: Any) -> list[int] | None:
    """Return a JSON-friendly array shape."""
    if value is None:
        return None
    return list(value.shape)


def resolve_export_dir(save_dir: str | Path, h4_export_dir: str | Path | None = None) -> Path:
    """Resolve the H4 export root, using ``save_dir/h4_internals`` when unset."""
    if h4_export_dir:
        export_dir = Path(h4_export_dir)
        if not export_dir.is_absolute():
            export_dir = Path(save_dir) / export_dir
    else:
        export_dir = Path(save_dir) / "h4_internals"
    return export_dir.resolve()


def build_pre_nms_candidate_tables(
    prediction: torch.Tensor | list | tuple,
    conf_thres: float,
    classes: list[int] | torch.Tensor | None,
    nc: int,
    max_candidates: int,
    max_nms: int,
) -> list[dict[str, torch.Tensor]]:
    """Build bounded pre-NMS candidate tables without changing NMS behavior.

    The table follows the standard single-label YOLO NMS candidate filtering:
    confidence threshold, optional class filter, then descending confidence
    truncation. It intentionally does not decide suppressor relationships.
    """
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]

    if prediction.shape[-1] == 6:  # end-to-end style output, no mask coefficients
        return [
            {
                "indices": torch.arange(pred.shape[0], device=pred.device)[:max_candidates],
                "rows": pred[:max_candidates],
            }
            for pred in prediction
        ]

    bs = prediction.shape[0]
    nc = nc or (prediction.shape[1] - 4)
    extra = prediction.shape[1] - nc - 4
    mi = 4 + nc
    pred_bnc = prediction.transpose(-1, -2)
    boxes_xyxy = ops.xywh2xyxy(pred_bnc[..., :4])
    candidate_classes = pred_bnc[..., 4:mi]
    coeffs = pred_bnc[..., mi : mi + extra]
    class_filter = torch.as_tensor(classes, device=prediction.device) if classes is not None else None

    tables = []
    for bi in range(bs):
        conf, cls = candidate_classes[bi].max(1)
        keep = conf > conf_thres
        if class_filter is not None:
            keep &= (cls[:, None] == class_filter).any(1)

        indices = torch.nonzero(keep, as_tuple=False).view(-1)
        if indices.numel():
            order = conf[indices].argsort(descending=True)
            limit = min(max_candidates, max_nms, order.numel())
            indices = indices[order[:limit]]
            rows = torch.cat(
                (
                    boxes_xyxy[bi, indices],
                    conf[indices, None],
                    cls[indices, None].float(),
                    coeffs[bi, indices],
                ),
                1,
            )
        else:
            rows = torch.zeros((0, 6 + extra), device=prediction.device)

        tables.append({"indices": indices, "rows": rows})
    return tables


def compute_mask_logits(
    protos: torch.Tensor,
    masks_in: torch.Tensor,
    bboxes: torch.Tensor,
    shape: tuple[int, int],
    retina_masks: bool,
) -> torch.Tensor:
    """Compute signed mask logits with the same crop/scale order as standard mask assembly."""
    c, mh, mw = protos.shape
    logits = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)
    if retina_masks:
        logits = ops.scale_masks(logits[None], shape)[0]
        return ops.crop_mask(logits, bboxes)

    width_ratio = mw / shape[1]
    height_ratio = mh / shape[0]
    ratios = torch.tensor([[width_ratio, height_ratio, width_ratio, height_ratio]], device=bboxes.device)
    logits = ops.crop_mask(logits, boxes=bboxes * ratios)
    return F.interpolate(logits[None], shape, mode="bilinear")[0]


def write_segmentation_internals(
    save_dir: str | Path,
    h4_export_dir: str | Path | None,
    image_path: str | Path,
    metadata: dict[str, Any],
    arrays: dict[str, Any],
) -> dict[str, Any]:
    """Write one per-image H4 internals npz and append a JSONL manifest row."""
    export_dir = resolve_export_dir(save_dir, h4_export_dir)
    array_dir = export_dir / "arrays"
    array_dir.mkdir(parents=True, exist_ok=True)

    image_path = str(image_path)
    image_id = Path(image_path).stem
    path_hash = hashlib.sha1(image_path.encode()).hexdigest()[:10]
    npz_path = array_dir / f"{image_id}_{path_hash}.npz"

    np_arrays = {k: _to_numpy(v) for k, v in arrays.items() if v is not None}
    np.savez_compressed(npz_path, **np_arrays)

    manifest_row = {
        "schema_version": SCHEMA_VERSION,
        "image_id": image_id,
        "image_path": image_path,
        "array_path": str(npz_path),
        "arrays": {k: _shape(v) for k, v in arrays.items() if v is not None},
        **_jsonable(metadata),
    }
    manifest_path = export_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(manifest_row, sort_keys=True) + "\n")

    return manifest_row
