#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Generate triptych segmentation comparisons: GT | Baseline | SSM."""

from __future__ import annotations

import argparse
import colorsys
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils import LOGGER, TQDM, YAML

IMG_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command line args."""
    parser = argparse.ArgumentParser(description="Compare GT/Baseline/SSM segmentation in triptych panels.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("/home/r13922151/ultralytics/ultralytics/cfg/datasets/dsb2018.yaml"),
        help="Dataset YAML path.",
    )
    parser.add_argument("--split", type=str, default="val", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument(
        "--baseline-ckpt",
        type=Path,
        default=Path("/home/r13922151/ultralytics/runs/segment/DSB2018/yolov8-seg-baseline4/weights/best.pt"),
        help="Baseline checkpoint path.",
    )
    parser.add_argument(
        "--ssm-ckpt",
        type=Path,
        default=Path("/home/r13922151/ultralytics/runs/segment/DSB2018/yolov8-seg-ssm-pan9/weights/best.pt"),
        help="SSM checkpoint path.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/home/r13922151/ultralytics/runs/segment/compare_vis/baseline4_vs_ssm-pan9_val_conf025"),
        help="Output directory for triptych images.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Max detections per image.")
    parser.add_argument("--device", type=str, default="0", help="Inference device, e.g. 0 or cpu.")
    parser.add_argument(
        "--retina-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use retina masks if enabled.",
    )
    parser.add_argument("--alpha", type=float, default=0.35, help="Mask alpha blend ratio.")
    parser.add_argument("--line-width", type=int, default=1, help="Contour line width.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images for smoke tests.")
    parser.add_argument("--save-summary-csv", action="store_true", help="Save summary.csv (counts per image).")
    return parser.parse_args()


def _resolve_dataset_paths(data_yaml: Path, split: str) -> tuple[Path, Path]:
    """Resolve image and label directories from dataset YAML."""
    cfg = YAML.load(data_yaml)
    if split not in cfg:
        raise KeyError(f"Split '{split}' not found in dataset yaml: {data_yaml}")

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.resolve().parent / root).resolve()

    split_rel = cfg[split]
    if isinstance(split_rel, (list, tuple)):
        if not split_rel:
            raise ValueError(f"Dataset split '{split}' is empty in {data_yaml}")
        split_rel = split_rel[0]

    images_dir = Path(split_rel)
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    labels_dir = images_dir.parent / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")
    return images_dir, labels_dir


def _read_image_bgr(image_path: Path) -> np.ndarray:
    """Read image and normalize to uint8 BGR."""
    im = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.ndim == 3 and im.shape[2] == 1:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.ndim == 3 and im.shape[2] > 3:
        im = im[:, :, :3]

    if im.dtype != np.uint8:
        im = cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return im


def _gt_masks_from_yolo_seg(label_path: Path, height: int, width: int) -> list[np.ndarray]:
    """Load GT instance masks from YOLO-seg txt labels."""
    if not label_path.exists():
        return []

    masks: list[np.ndarray] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        vals = line.split()
        if len(vals) < 7:
            continue
        coords = np.asarray(vals[1:], dtype=np.float32)
        if coords.size % 2 != 0:
            continue

        pts = coords.reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        pts = np.round(pts).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        if pts.shape[0] < 3:
            continue

        m = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(m, [pts], color=1)
        masks.append(m.astype(bool))
    return masks


def _predict_instance_masks(model: YOLO, image_path: Path, hw: tuple[int, int], args: argparse.Namespace) -> list[np.ndarray]:
    """Run segmentation prediction and return boolean instance masks in original image shape."""
    h, w = hw
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        retina_masks=args.retina_masks,
        verbose=False,
    )
    res = results[0]
    if res.masks is None or res.masks.data is None or not len(res.masks.data):
        return []

    pred = res.masks.data.detach().float().cpu().numpy() > 0.5
    masks: list[np.ndarray] = []
    for m in pred:
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(m)
    return masks


def _instance_color(index: int) -> tuple[int, int, int]:
    """Get deterministic BGR color for instance id."""
    hue = ((index + 1) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _overlay_masks(
    image_bgr: np.ndarray,
    masks: list[np.ndarray],
    title: str,
    alpha: float,
    line_width: int,
    empty_text: str,
) -> np.ndarray:
    """Render mask overlay panel with title bar."""
    panel = image_bgr.copy()
    for idx, mask in enumerate(masks):
        m = mask.astype(bool)
        if not m.any():
            continue
        color = _instance_color(idx)
        layer = np.zeros_like(panel, dtype=np.uint8)
        layer[m] = color
        panel = cv2.addWeighted(panel, 1.0, layer, alpha, 0.0)

        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(panel, contours, -1, color, thickness=max(1, int(line_width)))

    if not masks:
        cv2.putText(panel, empty_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    bar_h = 36
    out = np.full((panel.shape[0] + bar_h, panel.shape[1], 3), 255, dtype=np.uint8)
    out[bar_h:, :] = panel
    cv2.putText(out, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def _validate_paths(args: argparse.Namespace) -> None:
    """Validate required file paths before execution."""
    missing = [p for p in (args.data, args.baseline_ckpt, args.ssm_ckpt) if not p.exists()]
    if missing:
        joined = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(f"Required paths do not exist:\n{joined}")


def main() -> None:
    """Script entrypoint."""
    args = parse_args()
    _validate_paths(args)
    args.outdir.mkdir(parents=True, exist_ok=True)

    images_dir, labels_dir = _resolve_dataset_paths(args.data, args.split)
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_SUFFIXES])
    if args.limit is not None and args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No images found in: {images_dir}")

    LOGGER.info("Loading models...")
    baseline_model = YOLO(str(args.baseline_ckpt))
    ssm_model = YOLO(str(args.ssm_ckpt))

    summary_rows: list[dict[str, int | str]] = []
    fail_count = 0
    for image_path in TQDM(image_paths, desc="Triptych"):
        try:
            image = _read_image_bgr(image_path)
            h, w = image.shape[:2]
            label_path = labels_dir / f"{image_path.stem}.txt"

            gt_masks = _gt_masks_from_yolo_seg(label_path, h, w)
            baseline_masks = _predict_instance_masks(baseline_model, image_path, (h, w), args)
            ssm_masks = _predict_instance_masks(ssm_model, image_path, (h, w), args)

            gt_panel = _overlay_masks(image, gt_masks, "GT", args.alpha, args.line_width, empty_text="No GT")
            baseline_panel = _overlay_masks(
                image, baseline_masks, "Baseline", args.alpha, args.line_width, empty_text="No prediction"
            )
            ssm_panel = _overlay_masks(image, ssm_masks, "SSM", args.alpha, args.line_width, empty_text="No prediction")

            triptych = cv2.hconcat([gt_panel, baseline_panel, ssm_panel])
            save_path = args.outdir / f"{image_path.stem}.png"
            cv2.imwrite(str(save_path), triptych)

            summary_rows.append(
                {
                    "image": image_path.name,
                    "gt_n": len(gt_masks),
                    "baseline_n": len(baseline_masks),
                    "ssm_n": len(ssm_masks),
                }
            )
        except Exception as e:
            fail_count += 1
            LOGGER.warning(f"Skipping {image_path.name}: {e}")

    if args.save_summary_csv:
        summary_csv = args.outdir / "summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "gt_n", "baseline_n", "ssm_n"])
            writer.writeheader()
            writer.writerows(summary_rows)
        LOGGER.info(f"Saved summary: {summary_csv}")

    LOGGER.info(f"Done. images={len(image_paths)}, success={len(summary_rows)}, failed={fail_count}, out={args.outdir}")


if __name__ == "__main__":
    main()
