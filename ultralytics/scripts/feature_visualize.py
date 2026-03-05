import os
import cv2
import torch
import numpy as np

# 【關鍵】：設定 matplotlib 在背景執行，避免遠端伺服器報錯
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from ultralytics import YOLO
from pathlib import Path

# --- 定義要攔截熱力圖的目標模組 ---
TARGET_MODULES = ['LLDLow', 'LLDHigh', 'MambaCM']

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.model.named_modules():
            class_name = module.__class__.__name__
            if class_name in TARGET_MODULES:
                hook = module.register_forward_hook(self._hook_fn(class_name, name))
                self.hooks.append(hook)

    def _hook_fn(self, class_name, layer_name):
        def hook(module, input, output):
            # 處理可能回傳 tuple 的情況
            feat = output[0].detach() if isinstance(output, tuple) else output.detach()
            self.features[f"{class_name}_{layer_name}"] = feat
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def process_heatmap(tensor: torch.Tensor, strategy: str = 'max') -> np.ndarray:
    """將高維特徵張量降維並歸一化"""
    feat = tensor.squeeze(0) # [C, H, W]
    
    if strategy == 'max':
        feat = torch.max(feat, dim=0)[0]
    else:
        feat = torch.mean(feat, dim=0)

    feat = feat.cpu().numpy()
    feat = np.maximum(feat, 0) # ReLU
    feat_max, feat_min = np.max(feat), np.min(feat)
    
    if feat_max - feat_min > 1e-6:
        feat = (feat - feat_min) / (feat_max - feat_min)
    else:
        feat = np.zeros_like(feat)
    return feat

def analyze_and_visualize(model_path: str, image_path: str, output_dir: str, step: int = 32):
    """
    執行推理並將熱力圖與 Offset 向量圖存入指定資料夾。
    """
    # 1. 建立輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] 輸出路徑已設定為: {output_dir}")

    # 2. 載入模型與圖片
    print("[*] 正在載入模型與圖片...")
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    # 3. 註冊 Hooks 以提取熱力圖
    extractor = FeatureExtractor(model)

    # 4. 執行推論 (這會觸發 hooks 並填充 BDFWarpUp 的 last_offset)
    print("[*] 執行網路前向傳播...")
    _ = model.predict(image_path, imgsz=640, conf=0.25, verbose=False)

    # ==========================================
    # 任務 A：繪製並儲存特徵熱力圖 (Heatmaps)
    # ==========================================
    print("[*] 正在產生特徵熱力圖...")
    for name, tensor in extractor.features.items():
        # 高頻與邊界特徵用 max，低頻與語意特徵用 mean
        strategy = 'max' if 'High' in name else 'mean'
        feat_map = process_heatmap(tensor, strategy=strategy)
        
        # 上色與疊加
        feat_map_8u = np.uint8(255 * feat_map)
        feat_map_resized = cv2.resize(feat_map_8u, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.applyColorMap(feat_map_resized, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.6 + img * 0.4
        
        save_path = os.path.join(output_dir, f"heatmap_{name}.jpg")
        cv2.imwrite(save_path, superimposed_img)
        print(f"    -> 儲存: {save_path}")

    extractor.remove_hooks()

    # ==========================================
    # 任務 B：繪製並儲存 Offset 流場向量圖 (Quiver)
    # ==========================================
    print("[*] 正在產生 Offset 流場向量圖...")
    found_offset = False
    for name, module in model.model.named_modules():
        if module.__class__.__name__ == 'BDFWarpUp':
            if hasattr(module, 'last_offset') and module.last_offset is not None:
                found_offset = True
                
                offset = module.last_offset.detach().cpu()
                offset_xy = offset[0] 
                h_feat, w_feat = offset_xy.shape[1:]

                upsampled_offset_xy = torch.nn.functional.interpolate(
                    offset_xy.unsqueeze(0), 
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0) 

                scale_h, scale_w = orig_h / h_feat, orig_w / w_feat
                upsampled_offset_xy[0] *= scale_w 
                upsampled_offset_xy[1] *= scale_h 

                dx = upsampled_offset_xy[0].numpy()
                dy = upsampled_offset_xy[1].numpy()

                y_grid, x_grid = np.mgrid[step//2:orig_h:step, step//2:orig_w:step]
                u = dx[y_grid, x_grid]  
                v = dy[y_grid, x_grid]  

                fig, ax = plt.subplots(figsize=(12, 12 * orig_h / orig_w))
                ax.imshow(img_rgb)
                
                # 繪製向量
                ax.quiver(x_grid, y_grid, u, v, color='lime', angles='xy', scale_units='xy', scale=1, headwidth=4, width=0.002)

                ax.set_title(f"Offset Deformation Field - {name}")
                ax.set_aspect('equal')
                plt.axis('off')
                
                save_path = os.path.join(output_dir, f"offset_quiver_{name.replace('.', '_')}.png")
                fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=200)
                plt.close(fig) # 釋放記憶體
                print(f"    -> 儲存: {save_path}")

    if not found_offset:
        print("[!] 警告：未找到 BDFWarpUp 層，或其 last_offset 為空。請確認推論時模型處於正確狀態。")

    print("[*] 視覺化完成！")

if __name__ == '__main__':
    # ---------------------------------------------------------
    # 使用者配置區 (請依據你的遠端伺服器路徑進行修改)
    # ---------------------------------------------------------
    MODEL_FILE = '/home/r13922151/ultralytics/runs/segment/DSB2018/yolov8-seg-ssm-pan-LLDLowFixed-LLDHighGaborFixed-alpheInit0.2-native/weights/best.pt'   # 你的權重檔
    IMAGE_FILE = '/data_28T/r12_jason/R12922169_Master_Thesis_Project/dataset/dsb2018/standard/test/images/0bda515e370294ed94efd36bd53782288acacb040c171df2ed97fd691fc9d8fe.tif'           # 測試圖片
    OUTPUT_DIR = '/home/r13922151/ultralytics/runs/segment/DSB2018/yolov8-seg-ssm-pan-LLDLowFixed-LLDHighGaborFixed-alpheInit0.2-native/visualizations'                     # 輸出的資料夾名稱
    VECTOR_STEP = 16                                    # 向量圖的稀疏度 (越小箭頭越密)
    
    analyze_and_visualize(
        model_path=MODEL_FILE, 
        image_path=IMAGE_FILE, 
        output_dir=OUTPUT_DIR,
        step=VECTOR_STEP
    )