#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Preview the FA-OCD realization bridge using GT-derived local ambiguity targets."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.contact_targets import _morph_dilate
from ultralytics.utils.ocd_realization import decode_masks_with_ocd
from ultralytics.utils.ocd_targets import build_ocd_targets

IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
DEFAULT_OUTPUT_ROOT = Path("/home/r13922151/research_team/outputs/visualizations/ocd_realization_preview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview FA-OCD realization using YOLO-seg labels.")
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--proto-downsample", type=int, default=4)
    return parser.parse_args()


def resolve_images_dir(labels_dir: Path, images_dir: Path | None) -> Path:
    if images_dir is not None:
        return images_dir.resolve()
    candidate = labels_dir.resolve().parent / "images"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Could not infer images dir from {labels_dir}")
    return candidate


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / f"ocd_realize_{timestamp}").resolve()


def find_image_path(images_dir: Path, stem: str) -> Path:
    for suffix in IMG_SUFFIXES:
        path = images_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image found for stem={stem} under {images_dir}")


def read_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def load_instance_masks(label_path: Path, height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) < 7:
            continue
        coords = np.asarray(parts[1:], dtype=np.float32)
        if coords.size % 2:
            continue
        points = coords.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points = np.round(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        if points.shape[0] < 3:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        masks.append(mask)
    return masks


def panel(image: np.ndarray, title: str) -> np.ndarray:
    bar = np.full((32, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(bar, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return np.concatenate((bar, image), axis=0)


def colorize_stack(mask_stack: np.ndarray) -> np.ndarray:
    union = np.zeros((*mask_stack.shape[1:], 3), dtype=np.uint8)
    for idx, mask in enumerate(mask_stack, start=1):
        if not mask.any():
            continue
        hue = (idx * 0.61803398875) % 1.0
        hsv = np.array([[[int(hue * 179), 220, 255]]], dtype=np.uint8)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        union[mask.astype(bool)] = color
    return union


def boxes_from_masks(mask_stack: torch.Tensor) -> torch.Tensor:
    boxes = []
    for mask in mask_stack:
        ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
        if xs.numel() == 0:
            boxes.append(torch.tensor([0, 0, 1, 1], device=mask.device, dtype=torch.float32))
            continue
        boxes.append(torch.tensor([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], device=mask.device, dtype=torch.float32))
    return torch.stack(boxes, dim=0)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = resolve_images_dir(args.labels_dir, args.images_dir)
    label_paths = sorted(args.labels_dir.glob("*.txt"))[: args.limit]

    for label_path in label_paths:
        image_path = find_image_path(images_dir, label_path.stem)
        image_bgr = read_image_bgr(image_path)
        h, w = image_bgr.shape[:2]
        masks = load_instance_masks(label_path, h, w)
        if len(masks) < 2:
            continue

        index_mask = np.zeros((h, w), dtype=np.uint16)
        for idx, mask in enumerate(masks, start=1):
            index_mask[mask > 0] = idx
        proto_shape = (max(h // args.proto_downsample, 1), max(w // args.proto_downsample, 1))
        target_pack = build_ocd_targets(
            masks=torch.from_numpy(index_mask[None]).float(),
            proto_shape=proto_shape,
            overlap=True,
            pair_axis_enabled=True,
            return_debug=True,
        )

        instance_masks = []
        for instance_id in np.unique(index_mask):
            if instance_id <= 0:
                continue
            mask = torch.from_numpy((index_mask == instance_id).astype(np.float32))[None, None]
            mask = torch.nn.functional.interpolate(mask, size=proto_shape, mode="nearest")[0, 0]
            instance_masks.append(mask)
        proto = torch.stack([_morph_dilate(mask[None, None], 1)[0, 0] for mask in instance_masks], dim=0)
        mask_coeff = torch.eye(proto.shape[0], dtype=torch.float32)
        boxes = boxes_from_masks(torch.stack(instance_masks, dim=0))
        mult_logits = torch.nn.functional.one_hot(target_pack["mult_tgt"][0], num_classes=4).permute(2, 0, 1).float() * 6.0
        rho_logits = torch.log(target_pack["rho_tgt"][0].clamp_min(1e-4))
        xi_logits = torch.log(target_pack["xi_tgt"][0].clamp_min(1e-4))
        apd_code = target_pack["apd_tgt"][0]

        base_masks = proto.gt(0.0).cpu().numpy().astype(np.uint8)
        realized_masks, debug = decode_masks_with_ocd(
            proto=proto,
            mask_coeff=mask_coeff,
            boxes=boxes,
            shape=(h, w),
            mult_map=mult_logits,
            rho_map=rho_logits,
            xi_map=xi_logits,
            apd_code=apd_code,
            native=False,
        )
        realized_np = realized_masks.cpu().numpy().astype(np.uint8)
        amb = debug["amb_mask"].float().cpu().numpy()
        leakage = np.logical_and(realized_np.sum(axis=0) > 1, cv2.resize(amb, (w, h), interpolation=cv2.INTER_NEAREST) < 0.5)
        base_color = cv2.resize(colorize_stack(base_masks), (w, h), interpolation=cv2.INTER_NEAREST)
        realized_color = colorize_stack(realized_np)

        panels = [
            panel(image_bgr, "Original"),
            panel(base_color, "Base Dilated Candidates"),
            panel(realized_color, "Realized Masks"),
            panel(cv2.applyColorMap(np.clip(cv2.resize(amb, (w, h), interpolation=cv2.INTER_LINEAR) * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO), "Ambiguity"),
            panel((leakage[..., None] * np.array([0, 0, 255], dtype=np.uint8)).astype(np.uint8), "Leakage Outside Ambiguity"),
        ]
        sheet = np.concatenate(panels, axis=1)
        cv2.imwrite(str(output_dir / f"{label_path.stem}_realization.png"), sheet)

    print(f"Saved FA-OCD realization previews to {output_dir}")


if __name__ == "__main__":
    main()
