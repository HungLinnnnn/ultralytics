import torch
from ultralytics import YOLO

# 你的模型 YAML 或 pt
model = YOLO("ultralytics/cfg/models/v8/yolov8-seg-ssm-pan/alphainit0.yaml", task="segment")
model.model.to("cuda:0").eval()

# 跑一次 GPU forward
x = torch.randn(1, 3, 640, 640, device="cuda:0")
with torch.no_grad():
    _ = model.model(x)

# 你指定的 layer 16
m16 = model.model.model[16]
print("mode =", m16.ss2d.mode)
print("has_backend =", m16.ss2d.has_backend)
print("last_route =", m16.ss2d.last_route)