#!/usr/bin/env bash
set -euo pipefail

# FA-OCD + APD-prior v1 (supervision-rewritten) launcher
# S0 = target / realization sanity tooling
# S1 = legacy pre-rewrite proxy control (FA-CZS no-mod)
# S2 = revised GT-slot ownership core
# S3 = pair-axis-only control
# S4 = revised law + pair-axis auxiliary + bounded APD prior

EXPECTED_BRANCH="feature/faocd-apd-suprewrite-v1"
REPO_ROOT="/home/r13922151/ultralytics"
RESEARCH_ROOT="/home/r13922151/research_team"
PYTHON_BIN="/home/r13922151/miniconda3/envs/yolo-ssm/bin/python"
YOLO_BIN="/home/r13922151/miniconda3/envs/yolo-ssm/bin/yolo"

DATA_YAML="/home/r13922151/ultralytics/ultralytics/cfg/datasets/dsb2018_stardist_yolo.yaml"
TARGET_VIS_SCRIPT="/home/r13922151/ultralytics/ultralytics/scripts/visualize_ocd_targets.py"
REALIZE_VIS_SCRIPT="/home/r13922151/ultralytics/ultralytics/scripts/preview_ocd_realization.py"
TRAIN_LABELS="/home/r13922151/cell_datasets/dataset/dsb2018/stardist_yolo/train/labels"
RUN_PROJECT_DIR="/home/r13922151/ultralytics/runs/segment/DSB2018_stardist"
SANITY_OUTPUT_ROOT="/home/r13922151/research_team/outputs/visualizations"
PROJECT_NAME="${PROJECT_NAME:-DSB2018_stardist}"

S1_MODEL="/home/r13922151/ultralytics/ultralytics/cfg/models/v8/yolov8-seg-ssm-pan/fa_czs_v1_nomod.yaml"
S2_MODEL="/home/r13922151/ultralytics/ultralytics/cfg/models/v8/yolov8-seg-ssm-pan/fa_ocd_suprewrite_s2_slotcore.yaml"
S3_MODEL="/home/r13922151/ultralytics/ultralytics/cfg/models/v8/yolov8-seg-ssm-pan/fa_ocd_suprewrite_s3_xionly.yaml"
S4_MODEL="/home/r13922151/ultralytics/ultralytics/cfg/models/v8/yolov8-seg-ssm-pan/fa_ocd_suprewrite_s4_full.yaml"

PRETRAINED_WEIGHTS="yolov8n-seg.pt"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
MODE="${1:-help}"
TARGET="${2:-all}"

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
DEVICE="${DEVICE:-$GPU_ID}"
SEED="${SEED:-0}"

BATCH="${BATCH:-64}"
EPOCHS="${EPOCHS:-1500}"
PATIENCE="${PATIENCE:-300}"
SMOKE_BATCH="${SMOKE_BATCH:-8}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_PATIENCE="${SMOKE_PATIENCE:-1}"
VIS_LIMIT="${VIS_LIMIT:-8}"
VIS_OUTPUT_DIR="${VIS_OUTPUT_DIR:-$SANITY_OUTPUT_ROOT/ocd_sanity_${RUN_TAG}}"
REALIZE_OUTPUT_DIR="${REALIZE_OUTPUT_DIR:-$SANITY_OUTPUT_ROOT/ocd_realization_${RUN_TAG}}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    cat <<'EOF'
Usage:
  ./train_20260417_1735.sh sanity-targets
  ./train_20260417_1735.sh sanity-realize
  ./train_20260417_1735.sh smoke S2
  ./train_20260417_1735.sh smoke all
  ALLOW_FULL=1 ./train_20260417_1735.sh full S4

Targets:
  all | S1 | S2 | S3 | S4
EOF
}

require_path() {
    [[ -e "$1" ]] || { echo "Missing required path: $1" >&2; exit 1; }
}

check_branch() {
    local branch
    branch="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
    [[ "$branch" == "$EXPECTED_BRANCH" ]] || { echo "Expected branch '$EXPECTED_BRANCH' but found '$branch'." >&2; exit 1; }
}

enable_wandb() {
    "$YOLO_BIN" settings wandb=True >/dev/null
}

resolve_case() {
    case "${1^^}" in
        S1) printf '%s|%s\n' "$S1_MODEL" "S1_legacy_proxy" ;;
        S2) printf '%s|%s\n' "$S2_MODEL" "S2_slotcore" ;;
        S3) printf '%s|%s\n' "$S3_MODEL" "S3_xionly" ;;
        S4) printf '%s|%s\n' "$S4_MODEL" "S4_full" ;;
        *) echo "Unknown target '$1'." >&2; exit 1 ;;
    esac
}

run_target_sanity() {
    "$PYTHON_BIN" "$TARGET_VIS_SCRIPT" \
        --labels-dir "$TRAIN_LABELS" \
        --output-dir "$VIS_OUTPUT_DIR" \
        --limit "$VIS_LIMIT"
}

run_realization_preview() {
    "$PYTHON_BIN" "$REALIZE_VIS_SCRIPT" \
        --labels-dir "$TRAIN_LABELS" \
        --output-dir "$REALIZE_OUTPUT_DIR" \
        --limit "$VIS_LIMIT"
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
        seg_metric_backend=legacy \
        seg_metric_legacy_pq_reduce=imagewise \
        patience="$patience"
}

run_target_group() {
    local phase="$1"
    local target_arg="${2^^}"
    if [[ "$target_arg" == "ALL" ]]; then
        run_train_case S1 "$phase"
        run_train_case S2 "$phase"
        run_train_case S3 "$phase"
        run_train_case S4 "$phase"
    else
        run_train_case "$target_arg" "$phase"
    fi
}

require_full_ack() {
    [[ "${ALLOW_FULL:-0}" == "1" ]] || { echo "Full training remains gated. Complete sanity + smoke first, then rerun with ALLOW_FULL=1." >&2; exit 1; }
}

main() {
    require_path "$REPO_ROOT"
    require_path "$RESEARCH_ROOT"
    require_path "$PYTHON_BIN"
    require_path "$YOLO_BIN"
    require_path "$DATA_YAML"
    require_path "$TARGET_VIS_SCRIPT"
    require_path "$REALIZE_VIS_SCRIPT"
    require_path "$TRAIN_LABELS"
    require_path "$S1_MODEL"
    require_path "$S2_MODEL"
    require_path "$S3_MODEL"
    require_path "$S4_MODEL"
    check_branch
    enable_wandb
    mkdir -p "$RUN_PROJECT_DIR"
    cd "$REPO_ROOT"

    case "$MODE" in
        sanity-targets) run_target_sanity ;;
        sanity-realize) run_realization_preview ;;
        smoke) run_target_group smoke "$TARGET" ;;
        full) require_full_ack; run_target_group full "$TARGET" ;;
        help|-h|--help) usage ;;
        *) echo "Unknown mode '$MODE'." >&2; usage; exit 1 ;;
    esac
}

main "$@"
