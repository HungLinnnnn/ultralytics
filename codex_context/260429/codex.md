請你在我的研究專案中執行一個「只讀分析 + 新增可視化腳本」任務。請務必遵守以下要求：
## 任務目標
我目前在研究 YOLOv8-seg 細胞實例分割，想觀察 segmentation prototype channels 到底學到了什麼。因此請你研究目前 `/home/r13922151/ultralytics` 專案結構，然後在 `scripts/` 目錄下新增一支小腳本，可以：
1. 讀取一張輸入圖片；
2. 載入指定 YOLOv8-seg 權重；
3. 執行單張圖 inference；
4. 抽出 segmentation prototype tensor；
5. 將每一個 prototype channel 視覺化成圖片；
6. 另外輸出一張 prototype grid overview；
7. 儘量也輸出原圖、predicted masks/boxes overlay，方便對照。
請注意：這次不是訓練、不是改模型、不是跑實驗，只是寫一支分析工具。
\---
## 必須先閱讀/理解的檔案與資料夾
請先閱讀以下內容，掌握目前研究規範與專案結構：
1. `/home/r13922151/research_team/AGENTS.md`
2. `/home/r13922151/research_team/team_skill_map.yaml`
3. `/home/r13922151/research_team/context/project_brief.md`
4. `/home/r13922151/research_team/context/current_plan_context.md`
5. `/home/r13922151/research_team/context/repo_map.md`
6. `/home/r13922151/research_team/research_log.md`
7. `/home/r13922151/ultralytics/ultralytics/`
8. `/home/r13922151/ultralytics/scripts/`，若不存在請建立
請根據 `team_skill_map.yaml` 自動路由：
- `pi_committee`：確認任務邊界與風險；
- `experiment_team`：確認腳本不影響訓練流程、可重現；
- `method_team`：確認 prototype visualization 對目前 prototype decoupling / contact ownership 研究是否有用；
- `evaluation_team`：提出輸出圖應該如何命名與保存，方便後續比較。
這次不需要產出大型報告，但請在最後簡短回報你新增了什麼檔案、怎麼執行、輸出會長什麼樣。
\---
## 嚴格限制
1. 不要修改任何 training / validation / model architecture 的核心程式。
2. 不要改 YAML model config。
3. 不要啟動 train / val / predict 大量流程。
4. 不要覆蓋既有檔案。
5. 只能新增一支或少數幾支輔助腳本到 `scripts/`。
6. 如果需要新增 import 或 helper function，優先寫在同一支 script 裡，避免污染主程式。
7. 所有輸出請寫到使用者指定的 output directory，預設可用：
   `/home/r13922151/ultralytics/ultralytics/runs/prototype_vis/`
8. 腳本必須能用 CLI 執行。
\---
## 期望新增腳本
請新增：
`/home/r13922151/ultralytics/ultralytics/scripts/visualize_seg_prototypes.py`
此腳本需要支援以下參數：
```bash
python scripts/visualize_seg_prototypes.py \
  --weights /path/to/best.pt \
  --source /path/to/image.png \
  --out-dir runs/prototype_vis/example \
  --imgsz 640 \
  --device 0 \
  --conf 0.25 \
  --save-npy
```

必要 CLI 參數：

* --weights：YOLOv8-seg 權重路徑，例如 runs/segment/train/weights/best.pt
* --source：單張影像路徑
* --out-dir：輸出資料夾
* --imgsz：inference size，預設 640
* --device：cuda device 或 cpu，預設 0
* --conf：confidence threshold，預設 0.25
* --save-npy：可選，若指定則保存 raw prototype tensor 為 .npy
* --alpha：overlay alpha，預設 0.45
* --max-channels：最多輸出幾個 prototype channel，預設全部

---

核心技術要求

YOLOv8-seg 的 segmentation prototype 通常來自 segment head / proto module。請你研究目前 ultralytics 版本中 segmentation inference 的輸出格式，找到能取得 prototype tensor 的方式。

請優先嘗試以下方向：

1. 使用 Ultralytics model forward / predict 流程，觀察 segmentation model raw outputs；
2. 若 standard model.predict() 不直接暴露 prototype，請使用 forward hook 掛在 segmentation proto module 上；
3. 請不要硬編死 layer index，除非你已確認目前 model 結構且在註解中說明原因；
4. 優先透過 module name / module type 辨識 proto module；
5. 如果必須 fallback 到 layer index，請把 fallback 寫清楚，並在執行時 print warning。

請在腳本中做到：

* 載入模型；
* 對單張 image inference；
* 取得 prototype tensor，常見 shape 可能是：
    * [K, Hp, Wp]
    * [1, K, Hp, Wp]
    * 或其他 batch-first 格式
* 將 tensor normalize 後每個 channel 輸出成灰階或 heatmap 圖；
* 保存：
    * input.png
    * prediction_overlay.png
    * prototype_grid.png
    * prototype_ch_00.png
    * prototype_ch_01.png
    * …
    * 若 --save-npy，保存 prototype_raw.npy

---

視覺化細節

每個 prototype channel 請輸出至少兩種資訊：

1. normalize 後的 channel 圖：
    * min-max normalize 到 0–255；
    * 若 max-min 太小，避免除零；
    * 檔名：prototype_ch_XX.png
2. grid overview：
    * 將所有 channel 排成 grid；
    * 每格標上 channel index；
    * 檔名：prototype_grid.png

prediction overlay：

* 若模型有預測出 masks，請把 masks 疊在原圖上；
* 若有 boxes，請畫 boxes；
* 若沒有偵測結果，也要正常輸出原圖與 prototypes；
* 不要因為沒有 detection 就中斷，因為 prototype 仍然有分析價值。

---

重要：prototype resolution 與原圖對齊

prototype tensor 的空間大小可能不是原圖大小，例如 (H_p \times W_p) 小於輸入影像。請：

1. 每個 prototype channel 單獨保存原始 prototype resolution 的圖；
2. 另外可選擇 upsample 到 input image size 後保存 overlay 版本；
3. 在 console print：
    * input image shape；
    * model input imgsz；
    * prototype tensor shape；
    * number of prototype channels；
    * number of predicted instances。

---

程式品質要求

請讓腳本具備基本健壯性：

* 檢查 weights 是否存在；
* 檢查 source 是否存在；
* 自動建立 output directory；
* 清楚處理 CPU/GPU；
* inference 使用 torch.no_grad()；
* 避免影響 model training state；
* 輸出完成後 print 所有主要輸出檔案路徑。

請加入簡短註解說明：

* prototype tensor 是從哪裡取得；
* hook 掛在哪個 module；
* 若找不到 prototype，應該如何 debug。

---

預期回報格式

完成後請回報：

1. 新增檔案路徑；
2. 腳本功能摘要；
3. 執行範例；
4. 實際 tested command；
5. 輸出檔案列表；
6. prototype tensor shape；
7. 是否使用 hook；hook 掛在哪個 module；
8. 有無修改核心 ultralytics 程式；
9. 下一步建議：如何用這個工具觀察 shared/contact prototype channel 是否分工。

---

驗收標準

請至少用一張圖片做 smoke test。若沒有可用資料圖，請不要亂找外部資料；可以只保證腳本 syntax 正確並說明待我提供 image/weights 後可執行。

理想情況下，執行後應該能看到：

```text
Input image: ...
Prototype shape: [K, Hp, Wp]
Saved:
- input.png
- prediction_overlay.png
- prototype_grid.png
- prototype_ch_00.png
...
```

---

研究脈絡提醒

我目前關心的是 YOLOv8-seg 的 prototype homogenization 問題，也就是相鄰細胞 A/B 的 prototype pixel feature (p_A, p_B) 是否太像，導致：

\[
c_A^\top p_A \approx c_A^\top p_B
\]

進而造成 mask spillover。

所以這支腳本未來要幫助我觀察：

1. 哪些 prototype channel 像 cell foreground；
2. 哪些像 boundary/contact region；
3. 哪些可能只是 background/noise；
4. 相鄰細胞接觸區是否有獨立 channel 反應；
5. 若之後加入 shared/contact prototype split，能否視覺化比較 (P^{sh}) 和 (P^{ct})。

因此請在腳本註解或 README-style console output 中提醒這個用途。

我建議你先讓 Codex 只做這支 visualization script。等能穩定抽出 prototype 之後，下一步再下 prompt 讓 research_team 寫 diagnostic：計算相鄰 GT pair 的 prototype cosine similarity、contact band activation、以及每個 prototype channel 對 mask logit 的貢獻 $c_{i,k}P_k(x)$。
