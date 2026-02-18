from ultralytics import YOLO
import torch
m = YOLO("ultralytics/cfg/models/v8/yolov8-seg-ssm.yaml")
x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    y = m.model(x)
print([t.shape if hasattr(t, "shape") else type(t) for t in y])
print(m.model)