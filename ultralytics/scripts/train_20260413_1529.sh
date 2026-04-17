#!/usr/bin/env bash
set -euo pipefail

# FA-CZS v1 training launcher for the fixed first-round matrix:
#   E1 = F-Base
#   E2 = F-Base + FA-CZS v1
#   E3 = F-Base + contact branch and supervision, modulation off
#
# Output locations:
# - Contact-target sanity sheets:
#   /home/r13922151/research_team/outputs/visualizations/contact_target_sanity/
# - Training runs, checkpoints, and Ultralytics logs:
#   /home/r13922151/ultralytics/runs/segment/DSB2018_stardist/<run_name>/
# - W&B logging follows the local yolo-ssm environment and `yolo settings wandb=True`.
#
# Recommended order:
#   1. ./train_20260413_1529.sh sanity
#   2. ./train_20260413_1529.sh smoke all
#   3. ALLOW_FULL=1 ./train_20260413_1529.sh full all

EXPECTED_BRANCH="feature/ssm-pannet-pgdsa-czs-v1"
REPO_ROOT="/home/r13922151/ultralytics"
RESEARCH_ROOT="/home/r13922151/research_team"
PYTHON_BIN="/home/r13922151/miniconda3/envs/yolo-ssm/bin/python"
YOLO_BIN="/home/r13922151/miniconda3/envs/yolo-ssm/bin/yolo"

DATA_YAML="/home/r13922151/ultralytics/ultralytics/cfg/datasets/dsb2018_stardist_yolo.yaml"
VIS_SCRIPT="/home/r13922151/ultralytics/ultralytics/scripts/visualize_contact_targets.py"
TRAIN_LABELS="/home/r13922151/cell_datasets/dataset/dsb2018/stardist_yolo/train/labels"
VAL_LABELS="/home/r13922151/cell_datasets/dataset/dsb2018/stardist_yolo/test/labels"
RUN_PROJECT_DIR="/home/r13922151/ultralytics/runs/segment/DSB2018_stardist"
SANITY_OUTPUT_ROOT="/home/r13922151/research_team/outputs/visualizations/contact_target_sanity"
PROJECT_NAME="${PROJECT_NAME:-DSB2018_stardist}"

# E1: retained F-Base anchor from the existing launcher.
E1_MODEL="/home/r13922151/ultralytics/ultralytics/cfg/models/v8/yolov8-seg-ssm-pan/alphainit0_2_noGabor.yaml"
# E2: F-Base + FA-CZS v1 contact-aware proto modulation.
E2_MODEL="/home/r13922151/ultralytics/ultralytics/cfg/models/v8/yolov8-seg-ssm-pan/fa_czs_v1.yaml"
# E3: F-Base + contact branch and supervision, modulation disabled.
E3_MODEL="/home/r13922151/ultralytics/ultralytics/cfg/models/v8/yolov8-seg-ssm-pan/fa_czs_v1_nomod.yaml"

PRETRAINED_WEIGHTS="yolov8n-seg.pt"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
MODE="${1:-help}"
TARGET="${2:-all}"

GPU_ID="${GPU_ID:-3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
DEVICE="${DEVICE:-$GPU_ID}"  # relative to CUDA_VISIBLE_DEVICES after masking
SEED="${SEED:-0}"

BATCH="${BATCH:-128}"
EPOCHS="${EPOCHS:-1500}"
PATIENCE="${PATIENCE:-300}"

# Smoke-test mode keeps the same data/config path but shortens the budget.
SMOKE_BATCH="${SMOKE_BATCH:-$BATCH}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-3}"
SMOKE_PATIENCE="${SMOKE_PATIENCE:-3}"

# Target sanity / pre-run check settings.
VIS_SPLIT="${VIS_SPLIT:-train}"
VIS_LABELS_DIR="${VIS_LABELS_DIR:-$TRAIN_LABELS}"
VIS_LIMIT="${VIS_LIMIT:-16}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-$SANITY_OUTPUT_ROOT/${RUN_TAG}_${VIS_SPLIT}}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    cat <<'EOF'
Usage:
  ./train_20260413_1529.sh sanity
  ./train_20260413_1529.sh smoke all
  ./train_20260413_1529.sh smoke E2
  ALLOW_FULL=1 ./train_20260413_1529.sh full all
  ALLOW_FULL=1 ./train_20260413_1529.sh full E3

Modes:
  sanity   Generate contact-target sanity sheets with visualize_contact_targets.py.
  smoke    Run a short dataset-backed smoke test for E1/E2/E3.
  full     Run the official long-training launcher for E1/E2/E3. Requires ALLOW_FULL=1.
  help     Print this message.

Targets:
  all | E1 | E2 | E3

Useful overrides:
  GPU_ID=3
  RUN_TAG=manual_tag
  VIS_LIMIT=24
  VIS_SPLIT=train
  VIS_LABELS_DIR=/abs/path/to/labels
  VIS_OUTPUT_DIR=/abs/path/to/output
  SMOKE_EPOCHS=5
  SMOKE_BATCH=128
  BATCH=128
  EPOCHS=1500
EOF
}

require_path() {
    local path="$1"
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 1
    fi
}

check_branch() {
    local branch
    branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
    if [[ "$branch" != "$EXPECTED_BRANCH" ]]; then
        echo "Expected branch '$EXPECTED_BRANCH' but found '$branch'." >&2
        exit 1
    fi
}

enable_wandb() {
    "$YOLO_BIN" settings wandb=True >/dev/null
}

resolve_case() {
    case "${1^^}" in
        E1) printf '%s|%s\n' "$E1_MODEL" "E1_fbase" ;;
        E2) printf '%s|%s\n' "$E2_MODEL" "E2_fa_czs_v1" ;;
        E3) printf '%s|%s\n' "$E3_MODEL" "E3_fa_czs_v1_nomod" ;;
        *)
            echo "Unknown target '$1'. Use all, E1, E2, or E3." >&2
            exit 1
            ;;
    esac
}

run_visual_sanity() {
    mkdir -p "$SANITY_OUTPUT_ROOT"
    echo "[sanity] labels_dir=$VIS_LABELS_DIR"
    echo "[sanity] output_dir=$VIS_OUTPUT_DIR"
    "$PYTHON_BIN" "$VIS_SCRIPT" \
        --labels-dir "$VIS_LABELS_DIR" \
        --output-dir "$VIS_OUTPUT_DIR" \
        --limit "$VIS_LIMIT"
    echo "[sanity] Inspect sheets under $VIS_OUTPUT_DIR/sheets before any long run."
}

run_train_case() {
    local exp="$1"
    local phase="$2"
    local model_path run_prefix epochs batch patience run_name
    IFS='|' read -r model_path run_prefix <<<"$(resolve_case "$exp")"

    if [[ "$phase" == "smoke" ]]; then
        epochs="$SMOKE_EPOCHS"
        batch="$SMOKE_BATCH"
        patience="$SMOKE_PATIENCE"
        run_name="${run_prefix}_smoke_${RUN_TAG}"
    else
        epochs="$EPOCHS"
        batch="$BATCH"
        patience="$PATIENCE"
        run_name="${run_prefix}_full_${RUN_TAG}"
    fi

    echo "[${phase}] ${exp} -> ${run_name}"
    "$YOLO_BIN" segment train \
        model="$model_path" \
        data="$DATA_YAML" \
        batch="$batch" \
        epochs="$epochs" \
        device="$DEVICE" \
        pretrained="$PRETRAINED_WEIGHTS" \
        project="$PROJECT_NAME" \
        name="$run_name" \
        seed="$SEED" \
        seg_metric_backend=native \
        seg_metric_legacy_pq_reduce=imagewise \
        patience="$patience"
}

run_target_group() {
    local phase="$1"
    local target_arg="${2^^}"
    if [[ "$target_arg" == "ALL" ]]; then
        run_train_case E1 "$phase"
        run_train_case E2 "$phase"
        run_train_case E3 "$phase"
    else
        run_train_case "$target_arg" "$phase"
    fi
}

require_full_ack() {
    if [[ "${ALLOW_FULL:-0}" != "1" ]]; then
        echo "Full training is still manually gated. Inspect the sanity sheets and smoke outputs first, then rerun with ALLOW_FULL=1." >&2
        exit 1
    fi
}

main() {
    require_path "$REPO_ROOT"
    require_path "$RESEARCH_ROOT"
    require_path "$PYTHON_BIN"
    require_path "$YOLO_BIN"
    require_path "$DATA_YAML"
    require_path "$VIS_SCRIPT"
    require_path "$TRAIN_LABELS"
    require_path "$VAL_LABELS"
    require_path "$E1_MODEL"
    require_path "$E2_MODEL"
    require_path "$E3_MODEL"

    check_branch
    enable_wandb
    mkdir -p "$RUN_PROJECT_DIR"
    cd "$REPO_ROOT"

    case "$MODE" in
        sanity)
            run_visual_sanity
            ;;
        smoke)
            run_target_group smoke "$TARGET"
            ;;
        full)
            require_full_ack
            run_target_group full "$TARGET"
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            echo "Unknown mode '$MODE'." >&2
            usage
            exit 1
            ;;
    esac
}

main "$@"
