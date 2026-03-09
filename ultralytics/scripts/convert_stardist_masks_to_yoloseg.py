#!/usr/bin/env python3
"""Convert StarDist instance masks (uint16 label maps) to YOLO-seg polygon labels."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert StarDist masks to YOLO-seg labels.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/home/r13922151/cell_datasets/dataset/dsb2018/stardist"),
        help="Source StarDist root containing split/{images,masks}.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/r13922151/cell_datasets/dataset/dsb2018/stardist_yolo"),
        help="Destination YOLO dataset root containing split/{images,labels}.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "test"),
        help="Splits to convert. Default: train test",
    )
    parser.add_argument(
        "--image-link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to place image files in dst split/images.",
    )
    parser.add_argument("--class-id", type=int, default=0, help="Class index for all nuclei.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination labels/images if already present.",
    )
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def mask_path_for_image(mask_dir: Path, stem: str) -> Path | None:
    tif = mask_dir / f"{stem}.tif"
    if tif.exists():
        return tif
    tiff = mask_dir / f"{stem}.tiff"
    if tiff.exists():
        return tiff
    matches = sorted(mask_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def ensure_image_target(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def polygon_line(contour: np.ndarray, width: int, height: int, class_id: int) -> str | None:
    pts = contour.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 3:
        return None
    pts[:, 0] = np.clip(pts[:, 0] / float(width), 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1] / float(height), 0.0, 1.0)
    flat = pts.reshape(-1)
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in flat.tolist())


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 1.0


def convert_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    image_link_mode: str,
    class_id: int,
    overwrite: bool,
) -> dict[str, float]:
    src_img = src_root / split / "images"
    src_mask = src_root / split / "masks"
    dst_img = dst_root / split / "images"
    dst_lb = dst_root / split / "labels"
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lb.mkdir(parents=True, exist_ok=True)

    images = list_images(src_img)
    n_images = len(images)
    n_instances = 0
    n_polygons = 0
    ious: list[float] = []
    n_missing_masks = 0
    n_empty_labels = 0

    for im_path in images:
        ensure_image_target(im_path.resolve(), dst_img / im_path.name, image_link_mode, overwrite)

        mask_path = mask_path_for_image(src_mask, im_path.stem)
        if mask_path is None:
            n_missing_masks += 1
            (dst_lb / f"{im_path.stem}.txt").write_text("", encoding="utf-8")
            n_empty_labels += 1
            continue

        label_map = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if label_map is None:
            raise RuntimeError(f"Failed to read mask file: {mask_path}")
        if label_map.ndim == 3:
            label_map = label_map[..., 0]

        h, w = label_map.shape[:2]
        lines: list[str] = []
        inst_ids = np.unique(label_map)
        inst_ids = inst_ids[inst_ids > 0]
        n_instances += int(inst_ids.size)

        for inst_id in inst_ids:
            inst_mask = (label_map == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            rec = np.zeros_like(inst_mask)
            cv2.fillPoly(rec, contours, 1)
            ious.append(iou(inst_mask > 0, rec > 0))

            for contour in contours:
                line = polygon_line(contour, w, h, class_id)
                if line is not None:
                    lines.append(line)

        n_polygons += len(lines)
        if not lines:
            n_empty_labels += 1

        text = ("\n".join(lines) + "\n") if lines else ""
        (dst_lb / f"{im_path.stem}.txt").write_text(text, encoding="utf-8")

    iou_arr = np.array(ious, dtype=np.float64) if ious else np.array([1.0], dtype=np.float64)
    return {
        "n_images": float(n_images),
        "n_instances": float(n_instances),
        "n_polygons": float(n_polygons),
        "n_missing_masks": float(n_missing_masks),
        "n_empty_labels": float(n_empty_labels),
        "iou_mean": float(iou_arr.mean()),
        "iou_p10": float(np.quantile(iou_arr, 0.1)),
        "iou_p50": float(np.quantile(iou_arr, 0.5)),
        "iou_p90": float(np.quantile(iou_arr, 0.9)),
        "iou_min": float(iou_arr.min()),
        "ratio_iou_lt_0_99": float((iou_arr < 0.99).mean()),
    }


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] src={src}")
    print(f"[INFO] dst={dst}")
    print(f"[INFO] splits={list(args.splits)} mode={args.image_link_mode} class_id={args.class_id}")

    for split in args.splits:
        if not (src / split / "images").exists():
            print(f"[WARN] skip split='{split}' because {src / split / 'images'} is missing")
            continue
        stats = convert_split(src, dst, split, args.image_link_mode, args.class_id, args.overwrite)
        print(f"\n[OK] split={split}")
        for k, v in stats.items():
            if k.startswith("n_"):
                print(f"  {k}: {int(v)}")
            else:
                print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()

