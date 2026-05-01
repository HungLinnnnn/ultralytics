import argparse
from pathlib import Path
import sys

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all .tif/.tiff files in a folder to .png."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder containing .tif/.tiff files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Folder to save converted .png files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subfolders for .tif/.tiff files.",
    )
    return parser.parse_args()


def list_tif_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
    return sorted(files)


def build_output_path(src: Path, input_dir: Path, output_dir: Path, recursive: bool) -> Path:
    if recursive:
        rel = src.relative_to(input_dir)
        return (output_dir / rel).with_suffix(".png")
    return output_dir / f"{src.stem}.png"


def convert_one(src: Path, dst: Path, overwrite: bool) -> str:
    if dst.exists() and not overwrite:
        return "skipped"

    dst.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("OpenCV failed to read this file.")

    ok = cv2.imwrite(str(dst), img)
    if not ok:
        raise RuntimeError("OpenCV failed to write PNG.")

    return "written"


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"[ERROR] Input folder does not exist: {input_dir}", file=sys.stderr)
        return 1

    tif_files = list_tif_files(input_dir, recursive=args.recursive)
    if not tif_files:
        print(f"[WARN] No .tif/.tiff files found in: {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    failed = 0

    for src in tif_files:
        dst = build_output_path(src, input_dir, output_dir, recursive=args.recursive)
        try:
            status = convert_one(src, dst, overwrite=args.overwrite)
            if status == "written":
                written = 1
            else:
                skipped = 1
        except Exception as exc:  # noqa: BLE001
            failed = 1
            print(f"[ERROR] {src} -> {dst}: {exc}", file=sys.stderr)

    print(f"Done. total={len(tif_files)}, written={written}, skipped={skipped}, failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    main()
