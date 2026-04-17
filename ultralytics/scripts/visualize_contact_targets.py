#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Visualize FA-CZS v1 contact targets from YOLO-seg label files."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.utils.contact_targets import build_contact_targets

IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")
DEFAULT_OUTPUT_ROOT = Path("/home/r13922151/research_team/outputs/visualizations/contact_target_sanity")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize FA-CZS v1 contact targets from a labels directory.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Path to YOLO-seg labels directory, e.g. /home/r13922151/cell_datasets/dataset/dsb2018/stardist_yolo/train/labels",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional images directory. Defaults to the sibling 'images' directory next to labels-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults under /home/r13922151/research_team/outputs/visualizations/contact_target_sanity/.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N labels for quick checks.",
    )
    parser.add_argument(
        "--proto-downsample",
        type=int,
        default=4,
        help="Proto stride relative to image space. Current FA-CZS v1 head defaults to image/4.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=3,
        help="Blur kernel forwarded to build_contact_targets().",
    )
    parser.add_argument(
        "--blur-passes",
        type=int,
        default=1,
        help="Blur passes forwarded to build_contact_targets().",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=4.0,
        help="Minimum proto-space instance area forwarded to build_contact_targets().",
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=1,
        help="Proto-space dilation radius forwarded to build_contact_targets().",
    )
    parser.add_argument(
        "--large-instance-thresh",
        type=float,
        default=256.0,
        help="Proto-space area threshold that switches to large-instance dilation.",
    )
    parser.add_argument(
        "--large-instance-radius",
        type=int,
        default=2,
        help="Proto-space dilation radius for larger instances.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Resolve the output directory, creating a timestamped default if omitted."""
    if args.output_dir is not None:
        return args.output_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels_slug = "__".join(args.labels_dir.resolve().parts[-3:])
    labels_slug = labels_slug.replace(":", "_")
    return (DEFAULT_OUTPUT_ROOT / f"{labels_slug}_{timestamp}").resolve()


def resolve_images_dir(labels_dir: Path, images_dir: Path | None) -> Path:
    """Resolve images directory from explicit arg or sibling layout."""
    if images_dir is not None:
        return images_dir.resolve()
    candidate = labels_dir.resolve().parent / "images"
    if not candidate.is_dir():
        raise FileNotFoundError(f"Could not infer images dir from labels dir: {candidate}")
    return candidate


def find_image_path(images_dir: Path, stem: str) -> Path:
    """Find the corresponding image file for a label stem."""
    for suffix in IMG_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for stem '{stem}' under {images_dir}")


def read_image_bgr(image_path: Path) -> np.ndarray:
    """Read an image as uint8 BGR for visualization."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
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
    """Parse YOLO-seg polygons into binary instance masks."""
    masks: list[np.ndarray] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        values = line.split()
        if len(values) < 7:
            continue
        coords = np.asarray(values[1:], dtype=np.float32)
        if coords.size % 2 != 0:
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
        masks.append(mask.astype(bool))
    return masks


def build_instance_index_mask(instance_masks: list[np.ndarray]) -> np.ndarray:
    """Convert per-instance boolean masks into an instance-index raster."""
    if not instance_masks:
        raise ValueError("No instance masks were found for this sample.")
    height, width = instance_masks[0].shape
    index_mask = np.zeros((height, width), dtype=np.uint16)
    for idx, mask in enumerate(instance_masks, start=1):
        index_mask[mask] = idx
    return index_mask


def colorize_instance_index(index_mask: np.ndarray) -> np.ndarray:
    """Render a colorized instance-index mask on a dark canvas."""
    canvas = np.zeros((*index_mask.shape, 3), dtype=np.uint8)
    ids = np.unique(index_mask)
    ids = ids[ids > 0]
    for instance_id in ids:
        color = color_for_instance(int(instance_id))
        canvas[index_mask == instance_id] = color
    return canvas


def color_for_instance(index: int) -> tuple[int, int, int]:
    """Return a deterministic BGR color for an instance id."""
    hue = (index * 0.61803398875) % 1.0
    hsv = np.array([[[int(hue * 179), 200, 255]]], dtype=np.uint8)
    return tuple(int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def resize_mask(mask: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    """Resize a single-channel mask to the target size."""
    width, height = size
    return cv2.resize(mask, (width, height), interpolation=interpolation)


def overlay_binary_core(image_bgr: np.ndarray, binary_core: np.ndarray) -> np.ndarray:
    """Overlay a binary contact core on top of the original image."""
    overlay = image_bgr.copy()
    red = np.zeros_like(image_bgr)
    red[..., 2] = 255
    mask = binary_core.astype(bool)
    overlay[mask] = cv2.addWeighted(image_bgr[mask], 0.3, red[mask], 0.7, 0.0)
    return overlay


def overlay_soft_target(image_bgr: np.ndarray, soft_target: np.ndarray) -> np.ndarray:
    """Overlay a soft contact target heatmap on top of the original image."""
    heat = np.clip(np.round(soft_target * 255.0), 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap, 0.4, 0.0)
    overlay[heat == 0] = image_bgr[heat == 0]
    return overlay


def make_panel(image_bgr: np.ndarray, title: str) -> np.ndarray:
    """Attach a title bar to a panel."""
    title_h = 34
    panel = np.full((image_bgr.shape[0] + title_h, image_bgr.shape[1], 3), 255, dtype=np.uint8)
    panel[title_h:] = image_bgr
    cv2.putText(panel, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return panel


def save_visualizations(
    image_path: Path,
    output_dir: Path,
    original_bgr: np.ndarray,
    instance_index_mask: np.ndarray,
    binary_core_small: np.ndarray,
    soft_target_small: np.ndarray,
) -> None:
    """Save the combined sanity sheet and component images."""
    height, width = original_bgr.shape[:2]
    instance_color = colorize_instance_index(instance_index_mask)
    binary_core = resize_mask(binary_core_small.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    soft_target = resize_mask(soft_target_small.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    core_overlay = overlay_binary_core(original_bgr, binary_core)
    soft_overlay = overlay_soft_target(original_bgr, soft_target)

    sheet = np.concatenate(
        [
            make_panel(original_bgr, "Original Image"),
            make_panel(instance_color, "Instance-Index Mask"),
            make_panel(core_overlay, "Binary Contact Core"),
            make_panel(soft_overlay, "Soft Contact Target Overlay"),
        ],
        axis=1,
    )

    stem = image_path.stem
    cv2.imwrite(str(output_dir / "sheets" / f"{stem}_contact_sheet.png"), sheet)
    cv2.imwrite(str(output_dir / "instance_index" / f"{stem}_instance_index.png"), instance_color)
    cv2.imwrite(str(output_dir / "binary_core" / f"{stem}_binary_core.png"), binary_core * 255)
    cv2.imwrite(
        str(output_dir / "soft_target" / f"{stem}_soft_target.png"),
        np.clip(np.round(soft_target * 255.0), 0, 255).astype(np.uint8),
    )


def ensure_output_dirs(output_dir: Path) -> None:
    """Create output directory structure."""
    for subdir in ("sheets", "instance_index", "binary_core", "soft_target"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Visualize contact targets for every label file in a directory."""
    args = parse_args()
    labels_dir = args.labels_dir.resolve()
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    images_dir = resolve_images_dir(labels_dir, args.images_dir)
    output_dir = resolve_output_dir(args)
    ensure_output_dirs(output_dir)

    label_paths = sorted(labels_dir.glob("*.txt"))
    if args.limit is not None:
        label_paths = label_paths[: args.limit]
    if not label_paths:
        raise FileNotFoundError(f"No .txt label files found in {labels_dir}")

    summary_lines = [
        f"labels_dir: {labels_dir}",
        f"images_dir: {images_dir}",
        f"output_dir: {output_dir}",
        f"proto_downsample: {args.proto_downsample}",
        f"count: {len(label_paths)}",
    ]

    for idx, label_path in enumerate(label_paths, start=1):
        image_path = find_image_path(images_dir, label_path.stem)
        image_bgr = read_image_bgr(image_path)
        height, width = image_bgr.shape[:2]

        instance_masks = load_instance_masks(label_path, height, width)
        if not instance_masks:
            continue

        instance_index_mask = build_instance_index_mask(instance_masks)
        proto_shape = (
            max(1, math.ceil(height / args.proto_downsample)),
            max(1, math.ceil(width / args.proto_downsample)),
        )
        target_tensor, debug = build_contact_targets(
            masks=torch.from_numpy(instance_index_mask[None]).float(),
            proto_shape=proto_shape,
            overlap=True,
            min_area=args.min_area,
            dilation_radius=args.dilation_radius,
            large_instance_thresh=args.large_instance_thresh,
            large_instance_radius=args.large_instance_radius,
            blur_kernel=args.blur_kernel,
            blur_passes=args.blur_passes,
            return_debug=True,
        )
        binary_core_small = debug["contact_core"][0, 0].cpu().numpy()
        soft_target_small = target_tensor[0, 0].cpu().numpy()

        save_visualizations(
            image_path=image_path,
            output_dir=output_dir,
            original_bgr=image_bgr,
            instance_index_mask=instance_index_mask,
            binary_core_small=binary_core_small,
            soft_target_small=soft_target_small,
        )

        summary_lines.append(
            f"{idx:04d} {label_path.name} instances={len(instance_masks)} core_sum={float(binary_core_small.sum()):.2f} target_sum={float(soft_target_small.sum()):.2f}"
        )

    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Saved contact target visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
