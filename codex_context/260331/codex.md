下面是一份可直接交給 Codex 的 codex_plan.md 內容。
我把它設計成：
- 維持你目前最穩的主線：LLDLow = low pass、LLDHigh = F-L、fallback
- 吸收 FreqFusion 的 repair + offset + high-frequency enhancement 思想
- 吸收 SFFNet 的顯式低/高頻角色分工
- 先不用 phase / wavelet
- 先把 top-down 做穩，再談更前瞻的 offset operator

這份計劃的理論依據是：
- FreqFusion 的三個核心元件是 ALPF、offset generator、AHPF，其目的分別是減少 intra-category inconsistency、依 local similarity 做 resampling、補回 downsampling 遺失的高頻細節。 ￼
- SFFNet 的核心是顯式引入 low-frequency 與 high-frequency features，再讓 global branch 更搭 low-frequency、local branch 更搭 high-frequency，最後靠 MDAF 做 representation bridging，而不是直接相加。 ￼  ￼
- 你目前穩定可用的 YOLOv8n 主線，就是 LLDLow + LLDHigh_noGabor + MambaCM(fallback) + PBDFWarpUp/SGFR 這類保守版拓樸。 ￼
- 你原本的 codex 規劃，也把 Stage A 定義成 先做 decoupled align+restore，再往更強的 offset cue 走。 ￼

---

codex_plan.md

# Codex Plan — FHF-Neck v1
## Frequency-Hierarchical Fusion Neck for YOLOv8n-Seg
### Preserve `Low pass + (F - L)`, add rigid-low repair + similarity-guided high warp + high restore

---

## 0) Goal

We want a stable next-step neck for cell instance segmentation on top of the current best `scale=n` line:

- `LLDLow = low pass`
- `LLDHigh = F - L`
- `noGabor`
- `MambaCM = fallback`

The immediate objective is **not** to jump to wavelet or phase again.

The objective is:

1. preserve the current stable explicit low/high decomposition
2. upgrade top-down fusion into:
   - **Rigid low-frequency repair path**
   - **Elastic high-frequency alignment path**
   - **High-frequency residual restoration**
3. keep bottom-up unchanged for now
4. use a **FreqFusion-style similarity-guided offset head** first
5. only after it is stable, test more advanced offset operators

This design is motivated by:
- FreqFusion: ALPF + offset generator + AHPF for inconsistency reduction, resampling, and high-frequency recovery
- SFFNet: explicit low/high-frequency role assignment and semantic bridging instead of direct add/concat
- our current experiments: noGabor + fallback + conservative top-down is the most stable direction

---

## 1) Design principles

### 1.1 Keep decomposition simple and stable
We keep:
\[
L = \mathrm{LowPass}(F), \qquad H = F - L
\]

Reason:
- already stable in our current runs
- same resolution / same channel count
- easy to integrate into PAN-like neck
- allows clear validation of the main hypothesis:
  - low-frequency should be repaired but not warped
  - high-frequency should be aligned and restored

### 1.2 Separate “repair” from “alignment”
Low-frequency path:
- semantic / topology support
- no geometric warp
- only repair / consistency enhancement

High-frequency path:
- edge / boundary / local detail
- offset-guided alignment
- then residual restoration

### 1.3 Start with similarity-guided offset
Do not use phase for now.
Do not use wavelet for now.
Do not use SS2D for now.

Use:
- local similarity map
- small conv head
- small offset range
- warp high only

### 1.4 Conservative settings first
Default initial settings:
- `max_offset = 0.5`
- `alpha_init = 0.0`
- `beta_init = 0.0`
- `fallback` for MambaCM
- no bottom-up redesign

---

## 2) Target architecture summary

Backbone unchanged:
- YOLOv8n-seg backbone
- P3, P4, P5

Head / neck overall flow:
1. decompose P3/P4/P5 into low/high
2. aggregate low-frequency at P5
3. produce global anchor G5 using fallback MambaCM
4. top-down:
   - low-frequency repair
   - high-frequency similarity-guided warp
   - high-frequency restore
   - conservative fusion
5. bottom-up unchanged with GateConcat
6. segment head unchanged

---

## 3) Module plan

### 3.1 Keep existing modules

Keep as-is initially:
- `LLDLow`
- `LLDHigh_noGabor`
- `LowAggP5`
- `MambaCM`
- `GateConcat`

Do not redesign these in v1 unless necessary.

---

### 3.2 New module: `LowRepairUp`
Purpose:
- repair low-frequency inconsistency after upsampling
- keep low path rigid
- avoid geometric tearing

#### Inputs
- `src_low`: low-frequency from upsampled source
- `low_t`: target low-frequency
- optional projected context from source / global anchor

#### Output
- `low_refine`

#### First implementation
Simple residual repair:
\[
R_L = \mathrm{Conv}_{3\times3}([\mathrm{Proj}(src\_low), \mathrm{Proj}(low_t)])
\]
\[
low\_refine = src\_low + \alpha_L \cdot R_L
\]

Recommended:
- `alpha_L` as learnable scalar
- initialize `alpha_L = 0.0`

#### Notes
- no grid_sample
- no offset
- no deformable op
- low path must remain rigid

---

### 3.3 New module: `LocalSimilarityMap`
Purpose:
- compute local similarity between target high-frequency and source high-frequency
- serve as geometry cue for offset prediction

#### Inputs
- `high_t`
- `src_high`

#### Output
- local similarity tensor `S`

#### First implementation
Use local patch correlation:
1. project high features to small channel dimension
2. unfold local neighborhood from `src_high`
3. compute dot-product or cosine similarity with `high_t`

Recommended local window:
- `k = 5` first
- can try `k = 7` later

Recommended projected channels:
- `Cp = 16` or `32`

#### Output shape
One of:
- `(B, k*k, H, W)`  preferred
- or grouped similarity channels

---

### 3.4 New module: `SimilarityOffsetHead`
Purpose:
- predict small 2D offset from local similarity + high/low context

#### Inputs
- projected `high_t`
- projected `src_high`
- similarity map `S`
- projected `low_t` (optional but recommended)

#### Output
- `delta_p` with shape `(B, 2, H, W)`

#### Formula
\[
O_{\text{raw}} = \mathrm{Head}([H_t^{proj}, H_s^{proj}, S, L_t^{proj}])
\]
\[
\delta p = \Delta p_{\max} \cdot \tanh(O_{\text{raw}})
\]

#### Recommended head
- Conv1x1 -> SiLU
- Conv3x3 -> SiLU
- Conv3x3 -> 2 channels

#### Initialization
Last conv:
- weight = 0
- bias = 0

This ensures identity initialization.

#### Recommended default
- `max_offset = 0.5`

Do not use `2.0` in v1.

---

### 3.5 New module: `HighWarpRestore`
Purpose:
- warp only the high-frequency source
- restore sharpened high-frequency details using target high feature

#### Inputs
- `src_high`
- `high_t`
- `delta_p`

#### Output
- `high_refine`

#### Step A: warp
\[
\hat{H}_s = \mathrm{grid\_sample}(src\_high, \mathcal{G} + \delta p)
\]

#### Step B: restore
\[
R_H = \mathrm{Restore}([\hat{H}_s, high_t])
\]
\[
high\_refine = \hat{H}_s + \beta \cdot R_H
\]

#### Recommended restore block
- Conv3x3 -> SiLU -> Conv3x3
- output same channels as `src_high`

#### Initialization
- `beta` learnable scalar
- initialize `beta = 0.0`

---

### 3.6 New module: `FreqSplitWarpUp`
This is the top-down replacement block.

Purpose:
- replace `PBDFWarpUp`
- explicitly split low/high
- repair low rigidly
- align high elastically
- recombine

#### Inputs
- `src`
- `high_t`
- `low_t`

#### Outputs
- aligned source at target resolution
- optionally `delta_p`

#### Internal pipeline
1. `src_up = interpolate(src, scale_factor=2 or target_size)`
2. decompose:
   \[
   src_L = LowPass(src_up), \qquad src_H = src_up - src_L
   \]
3. low repair:
   \[
   low_{ref} = LowRepairUp(src_L, low_t)
   \]
4. local similarity:
   \[
   S = LocalSimilarityMap(high_t, src_H)
   \]
5. offset:
   \[
   \delta p = SimilarityOffsetHead(high_t, src_H, S, low_t)
   \]
6. warp high:
   \[
   high_{align} = grid\_sample(src_H, grid + \delta p)
   \]
7. restore high:
   \[
   high_{ref} = HighWarpRestore(high_{align}, high_t)
   \]
8. recomposition:
   \[
   A = low_{ref} + high_{ref}
   \]

#### Important
- only high-frequency is warped
- low-frequency is never warped

---

### 3.7 New module: `SGFRv2`
Purpose:
- conservative fusion after low/high recomposition
- replace the more aggressive current SGFR use

#### Inputs
- aligned source `A`
- target low-frequency `low_t`
- optional target high-frequency `high_t`

#### Output
- top-down fused feature

#### First implementation
\[
F = \mathrm{Conv}_{1\times1}([A, low_t])
\]
\[
out = A + \alpha \cdot F
\]

#### Recommended
- learnable scalar `alpha`
- initialize `alpha = 0.0`

#### Notes
Do not overcomplicate SGFRv2 in v1.
Keep it identity-biased.

---

## 4) Neck flow (step-by-step)

Assume:
- P3 = layer 4
- P4 = layer 6
- P5 = layer 9

### 4.1 Frequency decomposition
- `L3 = LLDLow(P3)`
- `H3 = LLDHigh_noGabor(P3, L3)`
- `L4 = LLDLow(P4)`
- `H4 = LLDHigh_noGabor(P4, L4)`
- `L5 = LLDLow(P5)`

### 4.2 Global anchor at P5
- `Z5 = LowAggP5(L3, L4, L5)`
- `G5 = MambaCM(Z5, mode="fallback")`
- `S5 = Conv1x1([G5, L5])`

### 4.3 Top-down P5 -> P4
- `A4 = FreqSplitWarpUp(src=S5, high_t=H4, low_t=L4)`
- `F4_in = SGFRv2(A4, L4, H4)`
- `F4 = C2f(F4_in)`

### 4.4 Top-down P4 -> P3
- `A3 = FreqSplitWarpUp(src=F4, high_t=H3, low_t=L3)`
- `F3_in = SGFRv2(A3, L3, H3)`
- `F3 = C2f(F3_in)`

### 4.5 Bottom-up unchanged
- `D4 = Down(F3)`
- `BU4 = GateConcat(D4, F4, guide=F4)`
- `F4_out = C2f(BU4)`

- `D5 = Down(F4_out)`
- `BU5 = GateConcat(D5, S5, guide=G5)`
- `F5_out = C2f(BU5)`

### 4.6 Head
- `Segment(F3, F4_out, F5_out)`

---

## 5) YAML changes

Start from current stable Stage A style YAML.

### Replace top-down blocks only

#### Before
- `PBDFWarpUp`
- `SGFR`

#### After
- `FreqSplitWarpUp`
- `SGFRv2`

### Conceptual YAML skeleton
```yaml
# top-down P5 -> P4
- [[18, 13, 12], 1, FreqSplitWarpUp, [32, 5, 0.5, 0.0, "corr"]]   # A4
- [[19, 12, 13], 1, SGFRv2, [512, True, 0.0]]                     # F4_in
- [-1, 3, C2f, [512]]                                             # F4

# top-down P4 -> P3
- [[21, 11, 10], 1, FreqSplitWarpUp, [32, 5, 0.5, 0.0, "corr"]]   # A3
- [[22, 10, 11], 1, SGFRv2, [256, True, 0.0]]                     # F3_in
- [-1, 3, C2f, [256]]                                             # F3
```

Suggested argument meaning:
- mid_ch = 32
- sim_kernel = 5
- max_offset = 0.5
- beta_init = 0.0
- offset_mode = "corr"

Keep unchanged
- decomposition layers
- LowAggP5
- MambaCM(fallback)
- GateConcat
- bottom-up
- Segment head

---

## 6) parse_model / channel inference

Add parse support for:
- FreqSplitWarpUp
- SGFRv2

Rules

FreqSplitWarpUp
Inputs:
- src
- high_t
- low_t

Output:
- same channel count as src after upsample path
So:
- c2 = c_src

SGFRv2
Output:
- c_out from args

Avoid lazy modules.

---

## 7) Loss and regularization

### 7.1 Main loss

Keep YOLOv8-seg default:
- box
- cls
- dfl
- mask

Do not redesign task loss in v1.

### 7.2 Offset smoothness regularization

For each top-down offset field:
[
L_{tv} = |\nabla_x \delta p|_1 + |\nabla_y \delta p|_1
]

Recommended:
- lambda_tv = 0.01

### 7.3 Offset magnitude regularization

[
L_{mag} = |\delta p|_1
]

Recommended:
- lambda_mag = 1e-4

### 7.4 Optional high-frequency consistency loss

Only if needed later:
[
L_{hf} = | high_{align} - high_t |_1
]

Do not enable in the first version.

---

## 8) Debug hooks

Add optional debug flag in FreqSplitWarpUp.

When enabled, log occasionally:
- src_L.mean/std
- src_H.mean/std
- similarity.mean/std/max
- delta_p.abs().mean()
- delta_p.abs().max()
- beta
- alpha_L

Also check:
- no NaN / Inf in similarity
- no NaN / Inf in offsets
- offset grads are finite

---

## 9) Implementation order

### Step 1

Implement FreqSplitWarpUp without offset:
- low repair
- high identity
- restore
- recomposition

Purpose:
- verify new low/high recomposition path is stable

### Step 2

Enable local similarity map
- compute S
- log shapes/stats
- still keep delta_p = 0

Purpose:
- verify similarity branch is numerically stable

### Step 3

Enable offset head
- small max_offset = 0.5
- zero-init final layer

Purpose:
- verify offsets become nonzero during training

### Step 4

Enable full top-down replacement
- both P5->P4 and P4->P3

### Step 5

Run full training

---

## 10) Experiment plan

### Exp-1: recomposition-only sanity

FreqSplitWarpUp with no offset (high identity)
Compare against current best baseline.

Goal:
- ensure no collapse
- verify low repair / high restore path is stable

### Exp-2: similarity-guided offset

Enable offset with:
- max_offset=0.5
- alpha=0.0
- beta=0.0

Goal:
- check whether offset brings gains without exploding split/FP

### Exp-3: low-repair ablation

Turn off LowRepairUp
Keep high align + high restore.

Goal:
- measure contribution of rigid low-frequency repair

### Exp-4: high-restore ablation

Turn off restore
Keep low repair + high warp.

Goal:
- measure whether alignment and restoration must be decoupled

### Exp-5: offset range ablation

Try:
- 0.5
- 1.0

Do not try 2.0 in the first round.

---

## 11) Success criteria

A version is considered promising if compared to current orange branch it gives:
	1.	equal or higher mAP50(M) / mAP50-95(M)
	2.	no obvious increase in split_rate
	3.	FP does not blow up
	4.	PQ / AJI / RQ stay stable or improve
	5.	offsets are nonzero and train normally

A version is not considered successful if:
- mAP rises but split / FP / PQ get clearly worse
- or offsets stay near zero throughout training
- or training becomes unstable

---

## 12) Things NOT to do in v1
- do not reintroduce Gabor
- do not switch to phase
- do not switch to wavelet yet
- do not redesign bottom-up
- do not use SS2D
- do not use large offset ranges
- do not change backbone scale during this phase

---

## 13) Naming suggestion

Main experiment name:
- fhf_v1_corr_noGabor_fallback_scaleN

Ablations:
- fhf_v1_noOffset_noGabor_fallback_scaleN
- fhf_v1_noLowRepair_noGabor_fallback_scaleN
- fhf_v1_noHighRestore_noGabor_fallback_scaleN
- fhf_v1_corr_offset1.0_noGabor_fallback_scaleN

---

## 14) Deliverables expected from Codex
1.	new modules in ssmnet_pan.py
    - LowRepairUp
    - LocalSimilarityMap
    - SimilarityOffsetHead
    - HighWarpRestore
    - FreqSplitWarpUp
    - SGFRv2
2.	parse_model support in tasks.py
3.	new YAML based on current stage-A stable config
4.	optional debug logging hooks
5.	no changes to data pipeline or trainer unless strictly necessary

---

## 15) Final instruction to Codex

Implement the minimum viable stable version first.

Priority order:
1.	stability
2.	correct low/high role separation
3.	conservative offset learning
4.	clean ablation

Do not over-engineer the first version.
Keep interfaces simple.
Preserve current working parts.