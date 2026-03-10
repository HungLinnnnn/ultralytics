可以。下面我直接把你的原始 Codex Plan 改造成一份可真正落地的 Phase-Guided DSA 版本完整方案。我會以你目前的 YOLOv8-seg + SSM-Net(PAN-like) 架構為基底，保留你原本的 PAN 拓樸與 LowAggP5 + MambaCM + top-down + bottom-up 主幹，只把最關鍵的 BDFWarpUp / SGF / GateConcat 系統化重構成「高低頻解耦 + 相位引導對齊 + 高頻殘差修復」版本。

我會分成這幾部分：

1. 設計目標與核心假設
2. 整體架構流程
3. 數學公式設計
4. 每個模組的輸入/輸出/參數/插入點
5. 具體參數建議
6. 訓練策略與 loss
7. 消融與除錯路線
8. 對你原始 YAML / module 的修改建議
9. 最後給你一版「建議實作順序」

---

## 1. 設計目標與核心假設

你這個模組不是單純要做一個「更複雜的 upsample」，而是要處理兩種本質不同的問題：

**問題 A：跨尺度融合時的幾何錯位**

當 P5 上採樣到 P4、P4 上採樣到 P3 時，source feature 雖然語義較強，但其局部結構位置可能和 target scale 的高頻邊界不完全對齊。

**問題 B：上採樣造成高頻細節流失**

即使 source feature 對齊了，它的高頻邊界也可能已被插值模糊化，無法單靠 warp 恢復。

所以核心原則是：

* 低頻負責語義與拓撲穩定
* 高頻負責局部邊界與幾何對齊
* 相位差只用來提供局部幾何先驗，不直接宣稱精確解析求得真實 flow
* 對齊與修復必須解耦：Align first, then Restore

---

## 2. 整體架構流程

你目前原始 PAN-like 架構可以保留，改動如下：

**Backbone 保持不變**

輸出：

* P3
* P4
* P5

**Neck 中加入高低頻分解**

對 P3/P4/P5 做：

$$L_i = \mathrm{LowPass}(P_i), \qquad H_i = P_i - L_i, \quad i\in\{3,4,5\}$$

得到：

* L3, H3
* L4, H4
* L5, H5

---

**P5 建立 global anchor**

沿用你的設計：

$$Z_5 = \mathrm{LowAggP5}(L_3, L_4, L_5)$$

$$G_5 = \mathrm{MambaCM}(Z_5)$$

再與 L5 合成 top-down source：

$$S_5 = \mathrm{Proj}([G_5, L_5])$$

---

**Top-down 兩段改成 PG-BDFWarpUp + SGF-R**

**Stage 1: P5 → P4**

輸入：

* source: S5
* target low/high: L4, H4

執行：

1. `src_up = upsample(S5)`
2. 對 `src_up` 做低高頻分解，取 `src_H4`
3. 用 H4 與 `src_H4` 做局部複數濾波，取得 phase-guided cues
4. 預測 offset `δp4`
5. warp `src_up` 或只 warp 其高頻分量
6. 做高頻殘差修復
7. 再由 SGF-R 融合 low/high/global

得到：

* F4_in
* 經 C2f 得 F4

**Stage 2: P4 → P3**

完全對應：

* source = F4
* target = L3, H3

得到：

* F3_in
* 經 C2f 得 F3

---

**Bottom-up 先保留你的 GateConcat**

你原本 bottom-up：

* P3→P4 guide = F4
* P4→P5 guide = G5

這部分先不用強行相位化，因為 bottom-up 本質是 routing / feedback selection，不是跨尺度高頻修復主戰場。
因此建議：

* Top-down 用 Phase-Guided DSA
* Bottom-up 保留 GateConcat 作 selective routing

這是最穩的第一版。

---

## 3. 數學公式設計

---

### 3.1 高低頻分解

對輸入特徵 $F \in \mathbb{R}^{B\times C\times H\times W}$：

$$L = \mathrm{DWConv}_{lp}(F)$$

$$H = F - L$$

其中：

* `DWConv_lp` 為 depthwise low-pass operator
* 建議 kernel size = 3 或 5
* 初版可直接用 depthwise average-like conv 或可學 depthwise smoothing conv

---

### 3.2 局部複數濾波器群

對 target high 與 source-up high：

$$H_t \in \mathbb{R}^{B\times C_h\times H\times W}, \qquad H_s \in \mathbb{R}^{B\times C_h\times H\times W}$$

先做 channel projection：

$$\tilde{H}_t = \mathrm{Proj}_h(H_t), \qquad \tilde{H}_s = \mathrm{Proj}_h(H_s)$$

其中：

* $\tilde{H}_t, \tilde{H}_s \in \mathbb{R}^{B\times C_p\times H\times W}$
* $C_p$ 建議遠小於原始通道數，例如 16 / 32 / 64

這一步很重要，因為不是每個原始 feature channel 都適合直接做 phase analysis。

---

**固定方向 complex filter bank**

對每個方向 $m=1,\dots,M$，使用一組固定複數濾波器：

$$R_{t,m} = \tilde{H}_t * G^{(m)}_{\text{real}}, \qquad I_{t,m} = \tilde{H}_t * G^{(m)}_{\text{imag}}$$

$$R_{s,m} = \tilde{H}_s * G^{(m)}_{\text{real}}, \qquad I_{s,m} = \tilde{H}_s * G^{(m)}_{\text{imag}}$$

振幅與相位：

$$A_{t,m} = \sqrt{R_{t,m}^2 + I_{t,m}^2 + \epsilon}$$

$$A_{s,m} = \sqrt{R_{s,m}^2 + I_{s,m}^2 + \epsilon}$$

$$\phi_{t,m} = \operatorname{atan2}(I_{t,m}, R_{t,m})$$

$$\phi_{s,m} = \operatorname{atan2}(I_{s,m}, R_{s,m})$$

---

### 3.3 Wrapped phase difference

注意這裡不是 unwrapping，而是 principal wrapped phase difference：

$$\Delta \phi_m = \operatorname{wrap}(\phi_{t,m} - \phi_{s,m})$$

其中：

$$\operatorname{wrap}(x)= ((x+\pi)\bmod 2\pi)-\pi$$

---

### 3.4 相位置信度權重

建議用以下之一：

$$W_m = \min(A_{t,m}, A_{s,m})$$

或較平滑版本：

$$W_m = \sqrt{A_{t,m}A_{s,m}}$$

再正規化：

$$\hat{W}_m = \frac{W_m}{\operatorname{mean}(W_m)+\epsilon}$$

相位有效特徵：

$$P_m = \Delta \phi_m \odot \hat{W}_m$$

---

### 3.5 Phase-guided offset predictor

將下列特徵聚合：

$$F_{\text{guide}} = \operatorname{Concat}\Big[\tilde{H}_t,\ \tilde{H}_s,\ P_1,\dots,P_M,\ A_{t,1},\dots,A_{t,M},\ A_{s,1},\dots,A_{s,M},\ L_t,\ src_{up}\Big]$$

其中 $L_t$ 與 $src_{up}$ 是否都納入，可依版本選擇：

* 第一版建議保留 $L_t$，因為低頻有助穩定 offset 預測
* $src_{up}$ 可先做 1x1 projection 再 concat，避免通道過重

offset 預測：

$$O_{\text{raw}} = \mathrm{Conv}_{3\times3}\Big(\sigma(\mathrm{Conv}_{1\times1}(F_{\text{guide}}))\Big)$$

$$\delta p = \Delta p_{\max}\cdot\tanh(O_{\text{raw}})$$

其中：

$$\delta p \in \mathbb{R}^{B\times 2\times H\times W}$$

表示每個位置的 $(\delta x,\delta y)$。

---

### 3.6 幾何對齊

**方案 A：warp 整個 upsampled source**

最穩、最簡單：

$$\hat{src} = \operatorname{grid\_sample}(src_{up}, \mathcal{G} + \delta p)$$

其中 $\mathcal{G}$ 是 base sampling grid。

**方案 B：只 warp 高頻，再與低頻重組**

更符合你的頻率解耦理念：

先對 `src_up` 分解：

$$src_{up} = src_L + src_H$$

只對高頻 warp：

$$\hat{src}_H = \operatorname{grid\_sample}(src_H, \mathcal{G} + \delta p)$$

低頻保持剛性：

$$\hat{src} = src_L + \hat{src}_H$$

建議你採用方案 B。
因為這和你原本「低頻不應被隨意變形」的想法一致，也更符合細胞分割中 topology stability 的需求。

---

### 3.7 高頻殘差修復

warp 完只處理對齊，還要補細節。
定義：

$$H_{\text{align}} = \hat{src}_H$$

用一個輕量修復模組 $R$：

$$H_{\text{res}} = R\big(\operatorname{Concat}[H_{\text{align}}, H_t]\big)$$

$$H_{\text{refine}} = H_{\text{align}} + \beta \cdot H_{\text{res}}$$

其中 $\beta$ 可設：

* 固定小值，例如 0.5
* 或 learnable scalar，初值 0

---

### 3.8 最終 top-down 融合

source 低頻、target 低頻、修復後高頻三者融合：

$$F_{\text{td}} = \operatorname{Fuse}\big([src_L,\ L_t,\ H_{\text{refine}}]\big)$$

初版建議用：

$$F_{\text{td}} = \mathrm{Conv}_{1\times1}(\operatorname{Concat}[src_L,\ L_t,\ H_{\text{refine}}])$$

再加 residual safety：

$$F_{\text{out}} = src_{up} + \alpha \cdot F_{\text{td}}$$

其中 $\alpha$ 初值建議 0 或 0.1。

這就是你原本 SGF 的相位版重構。

---

## 4. 模組設計總表

---

### 4.1 LLDLow

**功能**

低頻提取

**輸入**

* F: (B,C,H,W)

**輸出**

* L: (B,C,H,W)

**建議實作**

* depthwise conv, kernel=3 or 5, padding=same
* 初版可固定近似平滑核
* 後續可改 learnable

**參數**

* k=3
* mode="dwconv"
* norm="gn" optional

---

### 4.2 LLDHigh

**功能**

高頻殘差

**輸入**

* F
* L

**輸出**

* H = F - L

---

### 4.3 LowAggP5

沿用你原設計即可。

**輸入**

* L3, L4, L5

**輸出**

* Z5

**建議**

* d=128 for n/s
* d=192 or 256 for m/l

---

### 4.4 MambaCM

沿用。

**輸入**

* Z5

**輸出**

* G5

---

### 4.5 PhaseFilterBank

這是新模組。

**功能**

對 projected high-frequency feature 施加固定複數方向濾波器

**輸入**

* Ht_proj: (B,Cp,H,W)
* Hs_proj: (B,Cp,H,W)

**輸出**

* phase_feats
* amp_t_feats
* amp_s_feats

**參數**

* num_orientations M = 4 起步
* kernel_size = 5
* sigma = 1.5
* lambda = 3.0
* gamma = 0.5
* requires_grad=False

**備註**

用 depthwise groups=Cp。

---

### 4.6 PhaseGuidedOffsetHead

**功能**

從 phase/amplitude/high/low cues 預測 offset

**輸入**

建議是：

* Ht_proj
* Hs_proj
* phase_feats
* amp_t_feats
* amp_s_feats
* Lt_proj
* src_up_proj

**輸出**

* delta_p: (B,2,H,W)

**建議結構**

Conv1x1 -> SiLU
Conv3x3 -> SiLU
Conv3x3 -> 2ch
tanh * max_offset

**參數**

* hidden_ch = 64 for n/s
* hidden_ch = 96 or 128 for m/l
* max_offset = 0.5 ~ 1.0 pixel at target grid

**初始化**

最後一層 conv：

* weight = 0
* bias = 0

這樣一開始就是 identity warp。

---

### 4.7 PG-BDFWarpUp

這是原本 BDFWarpUp 的相位版。

**功能**

phase-guided deformable upsample

**輸入**

* src: 低解析 source feature
* high_t: target high-frequency
* low_t: target low-frequency

**輸出**

* aligned_src: target scale 對齊後 source
* optional: offset, srcL, srcH_align, H_refine

**內部流程**

1. `src_up = interpolate(src, scale_factor=2)`
2. `src_L = LLDLow(src_up)`
3. `src_H = src_up - src_L`
4. `Ht_proj = proj_h(high_t)`
5. `Hs_proj = proj_h(src_H)`
6. `Lt_proj = proj_l(low_t)`
7. `Src_proj = proj_s(src_up)`
8. phase filter bank
9. offset predictor
10. warp `src_H`
11. `H_refine = restore([warp(src_H), high_t])`
12. `aligned_src = src_L + H_refine`

**插入點**

替換你原本：

* P5→P4 的 BDFWarpUp
* P4→P3 的 BDFWarpUp

---

### 4.8 SGF-R

這是原本 SGF 的修正版。

**功能**

將 aligned_src, low_t, high_t 做 residual-safe fusion

**輸入**

* M = aligned_src
* B = low_t
* E = high_t

**輸出**

* F_td_in

**建議公式**

$$w_b = \sigma(\mathrm{Conv}(E))$$

$$B' = B \odot w_b$$

$$w_m = 1 + \tanh(\mathrm{Conv}(E))$$

$$M' = M \odot w_m$$

$$F = \mathrm{Conv}_{1\times1}([M', B'])$$

$$out = M + \alpha F$$

**參數**

* alpha_init = 0.0
* residual=True

---

### 4.9 GateConcat

bottom-up 保留即可。

**輸入**

* x
* y
* guide

**輸出**

* concat([x_gated,y])

**建議不改核心邏輯**

因為第一版先集中火力把 top-down 做穩。

---

## 5. 模組插入點

依你的原始 neck：

---

### 5.1 LLD 插入在 backbone 輸出後

對 layer 4, 6, 9：

* P3 -> L3/H3
* P4 -> L4/H4
* P5 -> L5/H5

---

### 5.2 Phase-Guided BDFWarpUp 插入 top-down upsample 位置

原本

* S5 -> BDFWarpUp -> A4
* F4 -> BDFWarpUp -> A3

改成

* S5 + H4 + L4 -> PG-BDFWarpUp -> A4
* F4 + H3 + L3 -> PG-BDFWarpUp -> A3

---

### 5.3 SGF 換成 SGF-R

原本

* [A4, L4, H4] -> SGF -> F4_in
* [A3, L3, H3] -> SGF -> F3_in

保留位置，只改內部邏輯

* [A4, L4, H4] -> SGF-R -> F4_in
* [A3, L3, H3] -> SGF-R -> F3_in

---

### 5.4 Bottom-up 不動

先保持：

* GateConcat(x=D4, y=F4, guide=F4)
* GateConcat(x=D5, y=S5, guide=G5)

---

## 6. 建議參數選擇

這部分我直接給你第一版可用設定。

---

### 6.1 頻率分解

**LLDLow**

* kernel size = 3 起步
* 若邊界仍太粗，可試 5
* groups=C
* 初版權重可設為近似均值核，後續再放開學習

**LLDHigh**

* H = F - L

---

### 6.2 phase filter bank

初版建議

* M = 4 directions
* 0°
* 45°
* 90°
* 135°
* kernel_size = 5
* sigma = 1.2 ~ 1.8
* lambda = 3.0
* gamma = 0.5

若太重可先：

* M = 2 directions 做快速驗證

---

### 6.3 projection channels

不要對全通道直接做相位分析，太重也太 noisy。

建議

* YOLOv8n/s:
* Cp = 16 or 32


* YOLOv8m/l:
* Cp = 32 or 64



也就是：

* proj_h: C -> Cp
* proj_l: C -> 16 or 32
* proj_s: C -> 16 or 32

---

### 6.4 offset range

初版

* max_offset = 0.5

因為相鄰金字塔層經上採樣後，多數局部修正應是 sub-pixel 到 1 pixel 左右。
若你直接設 2.0，訓練早期容易亂飄。

後續

* 若對齊不夠再試 1.0

---

### 6.5 restore block

初版簡單版

Concat([H_align, H_t]) -> Conv3x3 -> SiLU -> Conv3x3

輸出

* C channels
* 殘差式：

$$H_{\text{refine}} = H_{\text{align}} + \beta H_{\text{res}}$$



beta_init

* 0 或 0.1

---

### 6.6 SGF-R

初版

* alpha_init = 0.0
* residual on

---

### 6.7 GateConcat

保持你原本設定：

* guide conv bias = +2.0

---

## 7. Loss 設計

你若只靠 segmentation loss，模組未必穩。建議加幾個弱正則。

---

### 7.1 主 loss

沿用 YOLOv8-seg 原本：

* box / cls / dfl / mask loss

若你只做細胞分割，可依任務調整，但主訓練機制不動。

---

### 7.2 offset smoothness loss

對兩個 phase warp 的 offset：

$$\mathcal{L}_{tv} = \sum_{i\in\{3,4\}} \left( \|\nabla_x \delta p_i\|_1 + \|\nabla_y \delta p_i\|_1 \right)$$

建議權重：

* λ_tv = 0.01 起步
* 若 offset 很亂，可升到 0.05

---

### 7.3 offset magnitude regularization

避免早期 offset 過大：

$$\mathcal{L}_{mag} = \sum_i \|\delta p_i\|_1$$

建議很小權重：

* λ_mag = 1e-4 ~ 5e-4

---

### 7.4 可選：high-frequency consistency loss

若你願意額外做一點 supervision，可鼓勵對齊後高頻靠近 target：

$$\mathcal{L}_{hf} = \| \hat{src}_H - H_t \|_1$$

但這個 loss 要小心，因為 source/target 不一定應完全一樣。
建議：

* 初版先不加
* 若後續發現 warp 學不到，再小權重加入 0.01

---

## 8. 你原始 YAML / module 的具體修改方向

下面我不重寫整份 YAML，而是告訴你哪些地方該換。

---

### 8.1 BDFWarpUp 改成 PGBDFWarpUp

你原本：

```yaml
- [[19, 13, 12], 1, BDFWarpUp, [1.0]]

```

改為概念上：

```yaml
- [[19, 13, 12], 1, PGBDFWarpUp, [cp, max_offset, ...]]

```

同理：

```yaml
- [[22, 11, 10], 1, PGBDFWarpUp, [cp, max_offset, ...]]

```

新 args 建議

例如：

* cp=32
* num_orient=4
* k_gabor=5
* max_offset=0.5
* warp_high_only=True

---

### 8.2 SGF 改成 SGFR

```yaml
- [[20, 12, 13], 1, SGFR, [512, True, 0.0]]
- [[23, 10, 11], 1, SGFR, [256, True, 0.0]]

```

---

### 8.3 LowAggP5 / MambaCM 可保留

這部分不用動。

---

### 8.4 GateConcat 保留

不建議第一版同時改太多。

---

## 9. parse_model 需要新增的 channel 規則

這很重要，因為 Ultralytics 會在 parse_model 推斷 channel。

你新增這些 module 時要明確定義：

---

**PGBDFWarpUp**

輸入

* src
* high_t
* low_t

輸出

* 通常輸出 channel = c_src

若你內部有 projection 再還原，最後輸出仍建議回到 source channel 數，這樣最穩。

所以：

* c2 = c_src

---

**SGFR**

輸入

* M
* B
* E

輸出

* c_out 由 YAML args 指定

---

**PhaseFilterBank**

若單獨實作成中間工具模組，未必直接出現在 YAML。
我建議它內嵌在 PGBDFWarpUp 裡，避免 parse_model 複雜化。

---

## 10. 建議的 PGBDFWarpUp 內部結構

我直接給你一個最推薦的內部流程。

---

**輸入**

* src: (B,Cs,Hs,Ws)
* high_t: (B,Ct,H,W)
* low_t: (B,Ct,H,W)

其中 H = 2Hs, W = 2Ws

---

**流程**

Step 1. 上採樣

$$src_{up} = \mathrm{Interp}(src)$$

Step 2. source 分解

$$src_L = \mathrm{LowPass}(src_{up}), \qquad src_H = src_{up} - src_L$$

Step 3. high projection

$$\tilde{H}_t = \mathrm{Proj}_h(high_t)$$

$$\tilde{H}_s = \mathrm{Proj}_h(src_H)$$

Step 4. 低頻/語義投影

$$\tilde{L}_t = \mathrm{Proj}_l(low_t)$$

$$\tilde{S} = \mathrm{Proj}_s(src_{up})$$

Step 5. local complex filtering

得到：

* phase_feats
* amp_t_feats
* amp_s_feats

Step 6. offset regression

$$\delta p = \Delta p_{\max} \tanh(\mathrm{OffsetHead}(\cdot))$$

Step 7. warp high only

$$\hat{src}_H = \mathrm{Warp}(src_H,\delta p)$$

Step 8. restore

$$H_{res} = R([\hat{src}_H, high_t])$$

$$H_{refine} = \hat{src}_H + \beta H_{res}$$

Step 9. recomposition

$$A = src_L + H_{refine}$$

輸出：

* A 作為 aligned feature 給 SGFR
* optional 回傳 delta_p

---

## 11. 建議的實驗順序

這很關鍵，不然一次改太多你不知道哪裡有效。

---

**Stage A：最小可跑版**

先做：

* LLDLow/High
* PGBDFWarpUp 但先關掉 phase，只用：
* [high_t, low_t, src_up] -> offset_head
* warp_high_only=True


* restore block
* SGFR

這一版先驗證：

* 單靠 align+restore 是否比原本 blind upsample 好

---

**Stage B：加入 phase cues**

把以下加入 offset head：

* phase_feats
* amp_feats

比較：

* 無 phase
* phase only
* phase + amplitude confidence

---

**Stage C：phase filter bank 消融**

比較：

* 2 directions
* 4 directions
* 8 directions

通常 4 就夠了，8 可能過重。

---

**Stage D：fusion 消融**

比較：

* direct sum
* SGFR
* SGFR + restore

---

## 12. 實作風險與提醒

---

### 12.1 最大風險：相位特徵太 noisy

所以一定要做：

* channel projection
* amplitude weighting
* small max_offset
* zero-init offset head last layer

---

### 12.2 第二個風險：模組太重

你在 YOLOv8 neck 上不能一下塞太大。
所以建議：

* phase analysis 只在 Cp=16/32 的子空間做
* filter bank 固定，不學習
* restore block 輕量化

---

### 12.3 第三個風險：bottom-up 也同時大改

先不要。
把 top-down phase-guided 做穩再說。

---

## 13. 最終建議版架構摘要

你這個版本可以命名成：

Phase-Guided Decoupled Spectral Alignment Neck

簡稱：

* PG-DSA Neck
* 或 Phase-Guided DSA

其核心流程是：

1. backbone 的 P3/P4/P5 做高低頻分解
2. low-frequency 聚合到 P5，經 Mamba 得全域 anchor G5
3. top-down 時，不再用黑箱 upsample，而改為：
* source upsample
* source high / target high 做局部複數濾波
* 相位差與振幅置信度引導 offset 預測
* 只 warp source high
* 再做 high-frequency residual restoration
* 與低頻做 residual-safe fusion


4. bottom-up 保留 gate routing
5. head 接 F3, F4_out, F5_out

---

## 14. 一段可直接放進你的設計文件的正式描述

你可以這樣寫：

In the proposed neck, cross-scale top-down fusion is reformulated as a decoupled alignment-restoration process. For each top-down stage, the upsampled source feature is first decomposed into low- and high-frequency components. The target high-frequency feature and the source high-frequency feature are then projected into a compact latent space and processed by a fixed local complex filter bank to extract wrapped phase-difference cues and amplitude-aware confidence maps. These phase-guided cues are not used as a closed-form solver for displacement, but as geometrically meaningful priors for a lightweight offset predictor. The predicted offset field is applied only to the source high-frequency branch, while the low-frequency branch remains rigid to preserve global topology stability. After geometric alignment, a residual restoration block further compensates for high-frequency details lost during interpolation. Finally, the restored high-frequency branch is fused with the rigid low-frequency branch through a residual-safe fusion block. This design explicitly separates spatial misalignment from spectral degradation, which is especially suitable for boundary-sensitive cell segmentation.

---

是否需要執行最後一節的提議，將上述架構進階轉換為 PyTorch module 骨架程式碼，或是生成對應的新版 YAML 配置檔？