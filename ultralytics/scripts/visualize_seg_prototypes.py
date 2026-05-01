#!/usr/bin/env python3
"""Visualize YOLO segmentation prototype channels for single-image diagnosis.

This script is intentionally analysis-only: it loads a YOLOv8-seg style
checkpoint, runs one prediction, captures the segmentation prototype tensor,
and writes static images for prototype-channel inspection.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


FILE = Path(__file__).resolve()
PACKAGE_ROOT = FILE.parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if (REPO_ROOT / "ultralytics" / "__init__.py").exists():
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402


DEFAULT_OUT_DIR = PACKAGE_ROOT / "runs" / "prototype_vis"
PALETTE = (
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
    (23, 204, 146),
    (134, 219, 61),
    (52, 147, 26),
    (187, 212, 0),
    (168, 153, 44),
    (255, 194, 0),
    (147, 69, 52),
    (255, 115, 100),
    (236, 24, 0),
    (255, 56, 132),
    (133, 0, 82),
    (255, 56, 203),
    (200, 149, 255),
    (199, 55, 255),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one YOLOv8-seg inference and save segmentation prototype "
            "channel visualizations for prototype homogenization diagnosis."
        )
    )
    parser.add_argument("--weights", required=True, type=Path, help="Path to YOLOv8-seg weights, e.g. best.pt.")
    parser.add_argument("--source", required=True, type=Path, help="Path to one input image.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default="0", help="CUDA device id/string or 'cpu'. Default: 0.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask overlay alpha in [0, 1].")
    parser.add_argument("--max-channels", type=int, default=None, help="Maximum prototype channels to save.")
    parser.add_argument(
        "--channel-scale",
        type=int,
        default=6,
        help="Nearest-neighbor display scale for prototype_ch_XX.png and prototype_grid.png. Default: 6.",
    )
    parser.add_argument("--save-npy", action="store_true", help="Save raw prototype tensor as prototype_raw.npy.")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    device = str(device).strip()
    if device.lower() == "cpu":
        return "cpu"
    if not torch.cuda.is_available() and (device.isdigit() or device.startswith("cuda")):
        print(f"Warning: requested device '{device}' but CUDA is unavailable; falling back to CPU.")
        return "cpu"
    return device


def require_file(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
    return path


def normalize_to_uint8(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32, copy=False)
    finite = np.isfinite(channel)
    if not finite.all():
        channel = np.where(finite, channel, 0.0)
    min_val = float(channel.min())
    max_val = float(channel.max())
    denom = max_val - min_val
    if denom <= 1e-8:
        return np.zeros(channel.shape, dtype=np.uint8)
    return np.clip((channel - min_val) / denom * 255.0, 0, 255).astype(np.uint8)


def as_chw_proto(proto: torch.Tensor) -> torch.Tensor:
    """Normalize prototype tensor layout to [K, Hp, Wp] for one image."""
    if proto.ndim == 4:
        proto = proto[0]
    if proto.ndim != 3:
        raise ValueError(f"Expected prototype tensor with 3 or 4 dims, got shape {tuple(proto.shape)}.")
    return proto.detach().float().cpu()


def is_proto_tensor(value: Any, nm: int | None = None) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    if value.ndim == 4:
        return nm is None or value.shape[1] == nm
    if value.ndim == 3:
        return nm is None or value.shape[0] == nm
    return False


def find_proto_in_output(output: Any, nm: int | None = None) -> torch.Tensor | None:
    """Recursively find a proto-like tensor in a segmentation-head forward output."""
    if is_proto_tensor(output, nm):
        return output
    if isinstance(output, dict):
        for key in ("proto", "protos"):
            if key in output and is_proto_tensor(output[key], nm):
                return output[key]
        for key in ("one2many", "one2one"):
            found = find_proto_in_output(output.get(key), nm)
            if found is not None:
                return found
        for value in output.values():
            found = find_proto_in_output(value, nm)
            if found is not None:
                return found
    if isinstance(output, (tuple, list)):
        for value in output:
            found = find_proto_in_output(value, nm)
            if found is not None:
                return found
    return None


def resolve_core_model(yolo: YOLO) -> torch.nn.Module:
    core = getattr(yolo, "model", None)
    if core is None:
        raise RuntimeError("Could not resolve YOLO core model.")
    return core


def find_segment_head(core: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, module in core.named_modules():
        cls_name = module.__class__.__name__
        has_proto = hasattr(module, "proto")
        has_mask_count = hasattr(module, "nm")
        if has_proto and has_mask_count and ("Segment" in cls_name or hasattr(module, "cv4") or hasattr(module, "cv5")):
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError(
            "Could not find a segmentation head with a proto module. "
            "Debug by printing model.model.named_modules() and checking the final segmentation head."
        )
    return candidates[-1]


class PrototypeCapture:
    """Capture final segmentation proto from the segment head and base proto from head.proto.

    The head-level hook is the primary source because custom heads such as contact-aware
    variants may refine the base Proto output before returning final mask prototypes.
    The proto-module hook is retained as a debug fallback.
    """

    def __init__(self, yolo: YOLO):
        self.core = resolve_core_model(yolo)
        self.head_name, self.head = find_segment_head(self.core)
        self.nm = int(getattr(self.head, "nm", 0)) or None
        self.proto: torch.Tensor | None = None
        self.base_proto: torch.Tensor | None = None
        self.source = ""
        self.hooks: list[Any] = []

    def __enter__(self) -> "PrototypeCapture":
        self.hooks.append(self.head.register_forward_hook(self._capture_head_output))
        proto_module = getattr(self.head, "proto", None)
        if proto_module is not None:
            self.hooks.append(proto_module.register_forward_hook(self._capture_base_proto))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for hook in self.hooks:
            hook.remove()

    def _capture_head_output(self, module, inputs, output) -> None:  # noqa: ANN001
        proto = find_proto_in_output(output, self.nm)
        if proto is not None:
            self.proto = proto.detach()
            self.source = f"{self.head_name or '<root>'}.{module.__class__.__name__}.forward output"

    def _capture_base_proto(self, module, inputs, output) -> None:  # noqa: ANN001
        if is_proto_tensor(output, self.nm):
            self.base_proto = output.detach()
            if self.proto is None:
                self.source = f"{self.head_name or '<root>'}.proto ({module.__class__.__name__}) output"

    def get_proto(self) -> torch.Tensor:
        if self.proto is not None:
            return self.proto
        if self.base_proto is not None:
            print(
                "Warning: final segmentation-head proto was not captured; "
                "using the base proto module output instead."
            )
            return self.base_proto
        raise RuntimeError(
            "Prototype tensor was not captured. Debug by checking whether this checkpoint is a segmentation model "
            "and whether its head forward output contains a tensor shaped [B, nm, Hp, Wp]."
        )


def save_input_image(image_bgr: np.ndarray, out_dir: Path) -> Path:
    path = out_dir / "input.png"
    cv2.imwrite(str(path), image_bgr)
    return path


def draw_prediction_overlay(result: Any, image_bgr: np.ndarray, out_dir: Path, alpha: float) -> tuple[Path, int]:
    alpha = min(max(float(alpha), 0.0), 1.0)
    vis = image_bgr.copy()
    num_instances = 0

    if getattr(result, "masks", None) is not None and result.masks is not None:
        masks = result.masks.data.detach().cpu().numpy()
        num_instances = int(masks.shape[0])
        for i, mask in enumerate(masks):
            mask = mask.astype(np.uint8)
            if mask.shape[:2] != image_bgr.shape[:2]:
                mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            active = mask > 0
            if active.any():
                color = np.asarray(PALETTE[i % len(PALETTE)], dtype=np.float32)
                vis[active] = vis[active].astype(np.float32) * (1.0 - alpha) + color * alpha

    boxes = getattr(result, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        num_instances = max(num_instances, len(boxes))
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        cls = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
        names = getattr(result, "names", {}) or {}
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = np.round(box).astype(int).tolist()
            color = PALETTE[i % len(PALETTE)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label_name = names.get(int(cls[i]), str(int(cls[i]))) if isinstance(names, dict) else str(int(cls[i]))
            label = f"{label_name} {conf[i]:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    path = out_dir / "prediction_overlay.png"
    cv2.imwrite(str(path), vis)
    return path, num_instances


def save_channel_images(
    proto_chw: torch.Tensor,
    out_dir: Path,
    max_channels: int | None,
    channel_scale: int,
) -> tuple[list[Path], np.ndarray]:
    proto_np = proto_chw.numpy()
    total_channels, height, width = proto_np.shape
    channels_to_save = total_channels if max_channels is None else min(max(int(max_channels), 0), total_channels)
    scale = max(int(channel_scale), 1)
    channel_paths: list[Path] = []
    norm_channels: list[np.ndarray] = []

    for idx in range(channels_to_save):
        norm = normalize_to_uint8(proto_np[idx])
        norm_channels.append(norm)
        native_path = out_dir / f"prototype_native_ch_{idx:02d}.png"
        cv2.imwrite(str(native_path), norm)
        channel_paths.append(native_path)

        display = cv2.resize(norm, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
        path = out_dir / f"prototype_ch_{idx:02d}.png"
        cv2.imwrite(str(path), display)
        channel_paths.append(path)

    if not norm_channels:
        return channel_paths, np.zeros((0, height, width), dtype=np.uint8)
    return channel_paths, np.stack(norm_channels, axis=0)


def save_grid(norm_channels: np.ndarray, out_dir: Path, channel_scale: int) -> Path | None:
    if norm_channels.shape[0] == 0:
        return None
    count, height, width = norm_channels.shape
    scale = max(int(channel_scale), 1)
    tile_h, tile_w = height * scale, width * scale
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, channel in enumerate(norm_channels):
        row, col = divmod(idx, cols)
        display = cv2.resize(channel, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        text_scale = max(0.5, min(tile_h, tile_w) / 160.0)
        cv2.putText(
            tile,
            str(idx),
            (8, min(int(28 * text_scale), tile_h - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (0, 255, 255),
            max(1, int(round(text_scale * 2))),
            cv2.LINE_AA,
        )
        y1, x1 = row * tile_h, col * tile_w
        grid[y1 : y1 + tile_h, x1 : x1 + tile_w] = tile

    path = out_dir / "prototype_grid.png"
    cv2.imwrite(str(path), grid)
    return path


def main() -> None:
    args = parse_args()
    weights = require_file(args.weights, "weights")
    source = require_file(args.source, "source")
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read source image with OpenCV: {source}")

    device = resolve_device(args.device)
    print("Prototype visualization for YOLOv8-seg prototype homogenization/contact-region diagnosis.")
    print(f"Input image: {source}")
    print(f"Input image shape: {image_bgr.shape}")
    print(f"Weights: {weights}")
    print(f"Model input imgsz: {args.imgsz}")
    print(f"Device: {device}")
    print(f"Output directory: {out_dir}")

    yolo = YOLO(str(weights))
    core = resolve_core_model(yolo)
    was_training = bool(getattr(core, "training", False))
    core.eval()

    with PrototypeCapture(yolo) as capture, torch.no_grad():
        results = yolo.predict(
            # Pass the already-normalized BGR image instead of the path. Some grayscale
            # TIFFs are loaded by Ultralytics as 1-channel tensors, but YOLO seg
            # checkpoints expect 3-channel input.
            source=image_bgr,
            imgsz=args.imgsz,
            device=device,
            conf=args.conf,
            save=False,
            verbose=False,
            retina_masks=True,
        )
        proto = as_chw_proto(capture.get_proto())

    if was_training:
        core.train()

    if not results:
        raise RuntimeError("Prediction returned no Results objects.")
    result = results[0]

    saved_paths: list[Path] = []
    saved_paths.append(save_input_image(image_bgr, out_dir))
    overlay_path, num_instances = draw_prediction_overlay(result, image_bgr, out_dir, args.alpha)
    saved_paths.append(overlay_path)

    if args.save_npy:
        npy_path = out_dir / "prototype_raw.npy"
        np.save(npy_path, proto.numpy())
        saved_paths.append(npy_path)

    if args.max_channels is not None and args.max_channels < int(proto.shape[0]):
        print(
            f"Warning: --max-channels={args.max_channels} limits output; "
            f"omit it to save all {int(proto.shape[0])} prototype channels."
        )

    channel_paths, norm_channels = save_channel_images(proto, out_dir, args.max_channels, args.channel_scale)
    saved_paths.extend(channel_paths)
    grid_path = save_grid(norm_channels, out_dir, args.channel_scale)
    if grid_path is not None:
        saved_paths.append(grid_path)

    print(f"Hook target: {capture.head_name or '<root>'} ({capture.head.__class__.__name__})")
    print(f"Prototype source: {capture.source}")
    print(f"Prototype tensor shape: {list(proto.shape)}")
    print(f"Number of prototype channels: {int(proto.shape[0])}")
    print(f"Saved prototype channels: {int(norm_channels.shape[0])}")
    print(f"Saved prototype image files: {len(channel_paths)}")
    print(f"Number of predicted instances: {num_instances}")
    print("Purpose reminder: inspect whether channels specialize into foreground, boundary/contact, or background/noise.")
    print("Saved:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
