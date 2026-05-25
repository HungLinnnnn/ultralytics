#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash ultralytics/scripts/train_template.sh [KEY=value ...]

Configurable keys:
  GPU_ID       CUDA_VISIBLE_DEVICES value.
  MODEL        Model YAML or checkpoint path.
  DATA         Dataset YAML path.
  BATCH_SIZE   Training batch size.
  EPOCHS       Number of training epochs.
  DEVICE       Ultralytics device argument.
  PRETRAINED   Pretrained checkpoint or False.
  PROJECT      W&B project name/path.
  NAME         W&B run name.

Example:
  bash ultralytics/scripts/train_template.sh GPU_ID=0 DEVICE=0 BATCH_SIZE=32 EPOCHS=300 NAME=debug_run
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
        GPU_ID=*|MODEL=*|DATA=*|BATCH_SIZE=*|EPOCHS=*|DEVICE=*|PRETRAINED=*|PROJECT=*|NAME=*)
            export "$arg"
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

GPU_ID="${GPU_ID:-3}"
MODEL="${MODEL:-ultralytics/cfg/models/v8/yolov8-seg.yaml}"
DATA="${DATA:-ultralytics/cfg/datasets/dsb2018-stardist-yolo.yaml}"
BATCH_SIZE="${BATCH_SIZE:-100}"
EPOCHS="${EPOCHS:-600}"
DEVICE="${DEVICE:-$GPU_ID}"
PRETRAINED="${PRETRAINED:-yolov8l-seg.pt}"
PROJECT="${PROJECT:-DSB2018_stardist}"
NAME="${NAME:-yolov8-seg-baseline}"

yolo settings wandb=True

CUDA_VISIBLE_DEVICES="$GPU_ID" \
yolo segment train \
    model="$MODEL" \
    data="$DATA" \
    batch="$BATCH_SIZE" epochs="$EPOCHS" device="$DEVICE" \
    pretrained="$PRETRAINED" \
    project="$PROJECT" name="$NAME"
