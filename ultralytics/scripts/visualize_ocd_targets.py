#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize FA-OCD supervision-rewritten targets from YOLO-seg label files."""

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

from ultralytics.utils.ocd_targets import build_ocd_targets

IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
DEFAULT_OUTPUT_ROOT = Path("/home/r13922151/research_team/outputs/visualizations/ocd_target_sanity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FA-OCD targets from a YOLO-seg labels directory.")
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--proto-downsample", type=int, default=4)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / f"ocd_target_{timestamp}").resolve()


def resolve_images_dir(labels_dir: Path, images_dir: Path | None) -> Path:
    if images_dir is not None:
        return images_dir.resolve()
    candidate = labels_dir.resolve().parent / "images"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Could not infer images dir from {labels_dir}")
    return candidate


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


def build_index_mask(instance_masks: list[np.ndarray]) -> np.ndarray:
    canvas = np.zeros_like(instance_masks[0], dtype=np.uint16)
    for idx, mask in enumerate(instance_masks, start=1):
        canvas[mask > 0] = idx
    return canvas


def colorize_index_mask(index_mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*index_mask.shape, 3), dtype=np.uint8)
    for idx in np.unique(index_mask):
        if idx <= 0:
            continue
        hue = (idx * 0.61803398875) % 1.0
        hsv = np.array([[[int(hue * 179), 220, 255]]], dtype=np.uint8)
        canvas[index_mask == idx] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return canvas


def panel(image: np.ndarray, title: str) -> np.ndarray:
    title_h = 32
    canvas = np.full((image.shape[0] + title_h, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[title_h:] = image
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def heatmap(mask: np.ndarray) -> np.ndarray:
    image = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_TURBO)


def draw_apd_overlay(base: np.ndarray, index_mask: np.ndarray, apd_codes: np.ndarray) -> np.ndarray:
    overlay = base.copy()
    ids = [x for x in np.unique(index_mask) if x > 0]
    for idx, instance_id in enumerate(ids):
        ys, xs = np.where(index_mask == instance_id)
        if xs.size == 0 or idx >= len(apd_codes):
            continue
        cx = int(np.round(xs.mean()))
        cy = int(np.round(ys.mean()))
        axis = apd_codes[idx, :2]
        length = max(12, int(20 * (0.5 + float(apd_codes[idx, 2]))))
        dx = int(np.round(axis[0] * length))
        dy = int(np.round(axis[1] * length))
        cv2.line(overlay, (cx - dx, cy - dy), (cx + dx, cy + dy), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), 2, (255, 255, 255), -1)
    return overlay


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = resolve_images_dir(args.labels_dir, args.images_dir)
    label_paths = sorted(args.labels_dir.glob("*.txt"))
    if args.limit is not None:
        label_paths = label_paths[: args.limit]

    for label_path in label_paths:
        image_path = find_image_path(images_dir, label_path.stem)
        image_bgr = read_image_bgr(image_path)
        h, w = image_bgr.shape[:2]
        masks = load_instance_masks(label_path, h, w)
        if not masks:
            continue

        index_mask = build_index_mask(masks)
        proto_shape = (max(h // args.proto_downsample, 1), max(w // args.proto_downsample, 1))
        index_tensor = torch.from_numpy(index_mask[None]).to(torch.float32)
        targets = build_ocd_targets(
            masks=index_tensor,
            proto_shape=proto_shape,
            overlap=True,
            pair_axis_enabled=True,
            return_debug=True,
        )
        mult = targets["mult_tgt"][0].cpu().numpy().astype(np.float32) / 3.0
        amb = targets["amb_mask"][0, 0].cpu().numpy().astype(np.float32)
        rho1 = targets["rho_tgt"][0, 0].cpu().numpy().astype(np.float32)
        rho2 = targets["rho_tgt"][0, 1].cpu().numpy().astype(np.float32)
        xi1 = targets["xi_tgt"][0, 0].cpu().numpy().astype(np.float32)
        apd = targets["apd_tgt"][0].cpu().numpy() if targets["apd_tgt"] else np.zeros((0, 4), dtype=np.float32)

        size = (w, h)
        panels = [
            panel(image_bgr, "Original"),
            panel(colorize_index_mask(index_mask), "Instance Index"),
            panel(cv2.resize(heatmap(mult), size), "Multiplicity"),
            panel(cv2.resize(heatmap(amb), size), "Ambiguity"),
            panel(cv2.resize(heatmap(rho1), size), "Slot-1 Ownership"),
            panel(cv2.resize(heatmap(rho2), size), "Slot-2 Ownership"),
            panel(cv2.resize(heatmap(xi1), size), "Pair-Axis Cue"),
            panel(draw_apd_overlay(image_bgr, index_mask, apd), "APD Overlay"),
        ]
        rows = [np.concatenate(panels[:4], axis=1), np.concatenate(panels[4:], axis=1)]
        sheet = np.concatenate(rows, axis=0)
        cv2.imwrite(str(output_dir / f"{label_path.stem}_sheet.png"), sheet)

    print(f"Saved FA-OCD target sanity sheets to {output_dir}")


if __name__ == "__main__":
    main()
