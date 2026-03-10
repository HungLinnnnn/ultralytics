yolo settings wandb=True

DATA=ultralytics/cfg/datasets/dsb2018_stardist_yolo.yaml
PROJECT=DSB2018_stardist
# ultralytics/cfg/datasets/dsb2018_from_ASFYOLO.yaml （asf-yolo用的）
# ultralytics/cfg/datasets/dsb2018.yaml （學長用的）
# /home/r13922151/miniconda3/envs/yolo-ssm/bin/python ultralytics/scripts/convert_stardist_masks_to_yoloseg.py \
#   --src /home/r13922151/cell_datasets/dataset/dsb2018/stardist \
#   --dst /home/r13922151/cell_datasets/dataset/dsb2018/stardist_yolo \
#   --splits train test --image-link-mode symlink --overwrite
#
# Optional: build selective-scan backend (for Mamba-YOLO SS2D selective path)
# cd /home/r13922151/Mamba-YOLO/selective_scan
# /home/r13922151/miniconda3/envs/yolo-ssm/bin/pip install -e .

# CUDA_VISIBLE_DEVICES=1 \
# yolo segment train \
#     model=ultralytics/cfg/models/v8/yolov8-seg.yaml \
#     data=$DATA \
#     batch=128 epochs=1000 device=1 \
#     pretrained=yolov8n-seg.pt \
#     project=$PROJECT name=baseline \
#     seg_metric_backend=native seg_metric_legacy_pq_reduce=imagewise \
#     patience=300


# SSM-PAN variant example (keeps baseline command above unchanged):
# CUDA_VISIBLE_DEVICES=3 \
# yolo segment train \
#     model=ultralytics/cfg/models/v8/yolov8-seg-ssm-pan.yaml \
#     data=ultralytics/cfg/datasets/dsb2018.yaml \
#     batch=128 epochs=1000 device=3 \
#     pretrained=yolov8n-seg.pt \
#     project=DSB2018 name=yolov8-seg-ssm-pan-LLDLowFixed-LLDHighGaborFixed-alpheInit0.2-native \
#     seg_metric_backend=native seg_metric_legacy_pq_reduce=imagewise \
#     patience=250

CUDA_VISIBLE_DEVICES=1 \
yolo segment train \
    model=ultralytics/cfg/models/v8/yolov8-seg-ssm-pan/alphainit0_2_noGabor.yaml \
    data=$DATA \
    batch=128 epochs=1000 device=1 \
    pretrained=yolov8n-seg.pt \
    project=$PROJECT name=alpheInit0.2_noGabor_fallback \
    seg_metric_backend=native seg_metric_legacy_pq_reduce=imagewise \
    patience=300


# Optional segmentation metric backend switch examples:
# yolo segment val model=... data=... seg_metric_backend=native (default, faster but less accurate than legacy)
# yolo segment val model=... data=... seg_metric_backend=legacy seg_metric_legacy_pq_reduce=imagewise (slower but more accurate, especially for small objects)
