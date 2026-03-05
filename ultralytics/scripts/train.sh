yolo settings wandb=True
# CUDA_VISIBLE_DEVICES=1 \
# yolo segment train \
#     model=ultralytics/cfg/models/v8/yolov8-seg.yaml \
#     data=ultralytics/cfg/datasets/dsb2018.yaml \
#     batch=128 epochs=1000 device=1 \
#     pretrained=yolov8n-seg.pt \
#     project=DSB2018 name=yolov8-seg-baseline-epoch1000-native \
#     seg_metric_backend=native seg_metric_legacy_pq_reduce=imagewise \
#     patience=250

# SSM-PAN variant example (keeps baseline command above unchanged):
CUDA_VISIBLE_DEVICES=3 \
yolo segment train \
    model=ultralytics/cfg/models/v8/yolov8-seg-ssm-pan.yaml \
    data=ultralytics/cfg/datasets/dsb2018.yaml \
    batch=128 epochs=1000 device=3 \
    pretrained=yolov8n-seg.pt \
    project=DSB2018 name=yolov8-seg-ssm-pan-LLDLowFixed-LLDHighGaborFixed-alpheInit0.2-native \
    seg_metric_backend=native seg_metric_legacy_pq_reduce=imagewise \
    patience=250

# Optional segmentation metric backend switch examples:
# yolo segment val model=... data=... seg_metric_backend=native (default, faster but less accurate than legacy)
# yolo segment val model=... data=... seg_metric_backend=legacy seg_metric_legacy_pq_reduce=imagewise (slower but more accurate, especially for small objects)
