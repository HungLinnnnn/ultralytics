# GEMINI.md — SSM-Net 實作與修正計畫 (v2.0)

**目標**：修復 SSM-PAN 架構中的邏輯漏洞，引入嚴格的物理約束（頻率、幾何），並針對細胞分割任務（密集、邊界模糊）進行特化，最終超越 ASF-YOLO 與 YOLOv8-seg。

---

## Phase 1: 數學嚴謹性注入 (Mathematical Rigor)

**核心邏輯**：不能讓網絡「假裝」在做頻率分解，必須強制它做。

### 1.1 重構 `LLDLow`: 強制低頻 (Hard-Coded Low Pass)

單純的 Depthwise Conv 會退化成普通特徵提取器。我們需要真正的「模糊」來提取語意。

* **Action**: 修改 `ssmnet_pan.py` 中的 `LLDLow`。
* **Design Choice**: 使用 `Gaussian Low-Pass Filter` 作為物理過濾器。

### 1.2 — 可學習高斯濾波器實作規範

**目標**：將原有的 `LLDLow` (普通卷積) 替換為 **「物理約束的可學習高斯濾波器 (Learnable Gaussian Filter)」**，以確保該模組嚴格執行低通濾波，並能自適應學習不同特徵層級所需的模糊程度 ($\sigma$)。

### 1.3 核心動機 (Motivation)

在 SSM-Net 中，`LLD` 的任務是將特徵嚴格分離為「低頻語意」與「高頻細節」。

* **舊版問題**：標準 `Conv2d` 沒有低通約束，訓練後可能退化為全通或高通濾波器，導致頻率分解失效。
* **新版優勢**：透過參數化高斯函數 $G(\sigma)$，我們限制卷積核必須呈鐘形分佈。網絡只能改變 $\sigma$ (模糊半徑)，無法改變其物理特性。這確保了 `High = Input - Low` 得到的必然是邊界與紋理。

### 1.4 數學定義 (Mathematical Formulation)

對於每個輸入通道 $c$，我們定義一個獨立的標準差 $\sigma_c$。
二維高斯核 $K_c(x, y)$ 定義如下：

$$K_c(x, y) = \frac{1}{Z} \exp\left(-\frac{x^2 + y^2}{2\sigma_c^2}\right)$$

其中：

* $(x, y)$ 為卷積核內的相對坐標。
* $\sigma_c$ 為可學習參數 (必須 $>0$)。
* $Z$ 為歸一化常數，確保 $\sum K_c(x, y) = 1$ (能量守恆)。

### 1.5 程式碼實作 (Code Specification)

請將以下程式碼完全替換 `ultralytics/nn/modules/ssmnet_pan.py` 中的 `LLDLow` 類別。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LLDLow(nn.Module):
    """
    Learnable Gaussian Low-Pass Filter (LLDLow).
    
    Mechanism:
        Instead of learning weights directly, this module learns the 'sigma' (standard deviation)
        of a Gaussian kernel per channel. This enforces a strict physical low-pass constraint
        while allowing the network to adapt the blur radius for different features.
        
    Args:
        c1 (int): Input channels.
        k (int): Kernel size (recommend odd numbers, e.g., 5 or 7).
    """
    def __init__(self, c1: int, k: int = 5):
        super().__init__()
        self.k = k
        self.c1 = c1
        self.padding = k // 2
        self.groups = c1
        
        # 1. Learnable Parameter: Sigma
        # Initialize sigma=1.0. A value of 1.0 is a balanced blur.
        # Shape: [c1, 1, 1, 1] to broadcast over spatial dimensions later.
        self.sigma = nn.Parameter(torch.ones(c1, 1, 1, 1))
        
        # 2. Fixed Coordinate Grid (Buffer)
        # We pre-calculate (x^2 + y^2) for the kernel grid.
        # This does not change during training, so we register it as a buffer.
        self.register_buffer('dist_sq', self._build_dist_sq(k))

    @staticmethod
    def _build_dist_sq(k: int) -> torch.Tensor:
        """Generates the squared distance grid (x^2 + y^2) for a kxk kernel."""
        # Range: [-(k-1)/2, ..., (k-1)/2]
        ax = torch.arange(k, dtype=torch.float32) - (k - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        # Shape: [k, k]
        return xx.pow(2) + yy.pow(2)

    def get_kernel(self) -> torch.Tensor:
        """
        Dynamically generates the Gaussian kernel based on current sigma.
        Returns:
            kernel (Tensor): Shape [c1, 1, k, k]
        """
        # Constraint: Sigma must be positive. 
        # Softplus ensures positivity, +0.1 prevents division by zero and extreme sharpness.
        sigma = F.softplus(self.sigma) + 0.1
        
        # Calculate Gaussian: exp(-dist^2 / (2*sigma^2))
        # self.dist_sq shape: [k, k] -> broadcast to [c1, 1, k, k]
        # sigma shape: [c1, 1, 1, 1]
        
        # We move dist_sq to the same device as sigma
        dist = self.dist_sq.to(sigma.device)
        
        gamma = 1.0 / (2 * sigma.pow(2))
        kernel = torch.exp(-dist * gamma)
        
        # Normalize: Sum of kernel elements must be 1 to preserve brightness/energy.
        # Sum over spatial dimensions (-2, -1)
        kernel_sum = kernel.sum(dim=(-2, -1), keepdim=True)
        return kernel / kernel_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input feature map [B, C, H, W]
        Returns:
            out (Tensor): Low-frequency feature map [B, C, H, W]
        """
        # 1. Generate the kernel on-the-fly
        kernel = self.get_kernel()
        
        # 2. Apply Depthwise Convolution
        # groups=c1 ensures each channel is convolved with its own sigma-based kernel
        return F.conv2d(x, kernel, stride=1, padding=self.padding, groups=self.groups)

```
### 1.6 驗證計畫 (Verification)

在訓練開始前，建議編寫一個簡單的腳本驗證梯度是否能正確流向 $\sigma$。

```python
# test_lld.py
import torch
from ultralytics.nn.modules.ssmnet_pan import LLDLow

def test_sigma_grad():
    # Setup
    c1 = 4
    model = LLDLow(c1, k=5)
    x = torch.randn(1, c1, 32, 32, requires_grad=True)
    
    # Forward
    out = model(x)
    loss = out.mean() # Dummy loss
    
    # Backward
    loss.backward()
    
    print(f"Sigma values: {model.sigma.flatten().tolist()}")
    print(f"Sigma grads: {model.sigma.grad.flatten().tolist()}")
    
    assert model.sigma.grad is not None, "Gradient did not flow to sigma!"
    print("Pass: LLDLow is learnable.")

if __name__ == "__main__":
    test_sigma_grad()

```

### 1.7 設計參數建議 (Configuration)

在 `yolov8-seg-ssm-pan.yaml` 中使用此模組時：

* **`k` (Kernel Size)**:
* 對於細胞分割 (小目標)，建議使用 **`k=5`**。這提供了足夠的空間讓 $\sigma \approx 1.0 \sim 1.5$ 的高斯分佈展開。
* 如果使用 `k=3`，高斯會被截斷成類似 `AvgPool` 的形狀，失去可學習的精細度。


* **初始化**:
* 目前代碼將 `sigma` 初始化為 1.0。這是一個穩健的起點（輕微模糊）。模型會根據 Loss 自動調整：如果需要更多上下文，$\sigma$ 會變大；如果需要保留更多細節，$\sigma$ 會變小（接近 0.1）。



### 1.8 重構 `LLDHigh`: 混合頻率與邊緣 (Hybrid High Frequency)

我們需要結合「拉普拉斯殘差」（全向細節）與「Gabor 紋理」（方向性邊界）。

* **Action**: 修改 `ssmnet_pan.py` 中的 `LLDHigh`。
* **Design Choice**: `Residual + Gabor -> Conv`。
* **Code Spec**:
```python
class LLDHigh(nn.Module):
    """Decomposes high frequency using Laplacian Residual + Gabor priors."""
    def __init__(self, c1: int):
        super().__init__()
        # 這裡實作一個簡化的 Gabor-like 卷積，或直接用可學習卷積提取邊緣
        # 為了效率，我們先用一個專門初始化的 Conv 來模擬
        self.edge_conv = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.fusion = nn.Conv2d(c1 * 2, c1, 1, 1, 0)

        # 初始化 edge_conv 為 Sobel 或隨機高頻
        # (進階實作：可在此處插入真正的 Gabor Bank)

    def forward(self, xs):
        # xs = [Raw_Feature, Low_Feature]
        f_raw, f_low = xs

        # 1. 拉普拉斯殘差 (各向同性高頻)
        if f_low.shape[-2:] != f_raw.shape[-2:]:
            f_low = F.interpolate(f_low, size=f_raw.shape[-2:], mode='bilinear')
        high_res = f_raw - f_low 

        # 2. 邊緣卷積 (方向性高頻)
        high_edge = self.edge_conv(f_raw)

        # 3. 融合
        return self.fusion(torch.cat([high_res, high_edge], dim=1))

```



---

## Phase 2: 流形對齊升級 (Manifold Alignment)

**核心邏輯**：給予模型更大的變形自由度，但同時施加物理約束。

### 2.1 升級 `BDFWarpUp`: 擴大感受野與輸出接口

* **Action**:
1. 將 `max_offset` 預設值改為 **2.0** 或 **3.0**。
2. **關鍵修改**：`forward` 必須返回 `(aligned_feature, offset_field)`。我們需要拿到 `offset_field` 來計算 TV Loss。


* **Code Spec**:
```python
class BDFWarpUp(nn.Module):
    # ... __init__ 保持不變，將 max_offset default 改為 2.0 ...

    def forward(self, xs):
        # ... 前面計算 offset 邏輯不變 ...
        offset = self.max_offset * torch.tanh(offset_raw)

        # ... grid_sample 邏輯 ...
        aligned = F.grid_sample(...)

        # 修改返回值：若在訓練模式，返回 tuple 以供 Loss 計算
        if self.training:
            return aligned, offset
        return aligned

```


*(注意：這會影響下游 `SGF` 的輸入，下游模組需要適配接受 Tuple 或只取第一個元素)*

---

## Phase 3: 損失函數工程 (Loss Engineering)

**核心邏輯**：沒有 Loss 的約束，特徵分離只是空談。

### 3.1 新增 `SsmLoss` 類別

不要直接修改原本的 `v8SegmentLoss`，而是繼承並擴展它。

* **Location**: `ultralytics/utils/loss.py`
* **Specs**:
1. **Offset Smoothness Loss (TV Loss)**:

$$L_{tv} = \sum |offset_{x+1} - offset_x| + |offset_{y+1} - offset_y|$$



防止偏移場出現劇烈的雜訊。
2. **Boundary Aux Loss**:
取 `LLDHigh` 的輸出，計算 `BCE(Sigmoid(H), Canny(GT))`。
這強迫 `LLDHigh` 真的去學細胞邊界，而不是學背景紋理。


* **Implementation Strategy**:
你需要修改 `ultralytics/engine/trainer.py` 或 `tasks.py` 中的 `init_criterion`，讓它使用你自定義的 `SsmLoss`。

---

## Phase 4: 參數調優與實驗 (Configuration)

### 4.1 YAML 最終確認 (`yolov8-seg-ssm-pan.yaml`)

* **Backbone**: 保持原樣 (P3-P5)。
* **Head**:
* LLD: 使用新版 `LLDLow` (AvgPool) 和 `LLDHigh`。
* Mamba: 設定 `d_state=32` (增加記憶容量)。
* BDF: 設定 `max_offset=2.0`。



### 4.2 訓練超參數建議

* **Epochs**: 200+ (Mamba 收斂較慢)。
* **Warmup**: 增加 Warmup epochs (例如 5 -> 10)，讓 BDF 的 offset 有時間穩定初始化（從 0 開始）。
* **Optimizer**: `AdamW` (對 Transformer/Mamba 結構更友善)。

---

## 執行清單 Checklist

* [ ] **[P1]** 將 `LLDLow` 改寫為基於 `Gaussian Low-Pass Filter` 的卷積。
* [ ] **[P1]** 實作 `LLDHigh` 的殘差+卷積融合邏輯。
* [ ] **[P2]** 修改 `BDFWarpUp`，使其在訓練時返回 `offset`。
* [ ] **[P2]** 修改 `SGF` 與 `GateConcat` 以適配 BDF 可能返回的 tuple 輸入（簡單解法：檢查輸入如果是 tuple，取 `[0]`）。
* [ ] **[P3]** (選做) 在 `loss.py` 實作 TV Loss 並掛載到訓練流程（若太複雜可先跳過，但強烈建議加上）。
* [ ] **[P4]** 啟動訓練，監控 `train/box_loss`, `train/seg_loss` 以及自定義的 `offset_norm` (如果有的話)。

這份文件現在是你的**作戰守則**。請從 P1 開始執行。