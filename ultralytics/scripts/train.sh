CUDA_VISIBLE_DEVICES=3 \
yolo segment train \
    model=ultralytics/cfg/models/v8/yolov8-seg.yaml \
    data=ultralytics/cfg/datasets/dsb2018.yaml \
    batch=64 epochs=600 device=3 \
    pretrained=yolov8n-seg.pt