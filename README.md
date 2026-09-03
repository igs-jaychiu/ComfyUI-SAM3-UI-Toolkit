# ComfyUI-SAM3-UI-Toolkit

提供 ComfyUI 原生 SAM3 工作流程使用的 UI 資產檢查、裁切、人工核准與
確定性補洞節點。

這套節點特別針對遊戲 UI 分層：先用 SAM3 找出文字、Icon、按鈕，再由使用者
檢查編號遮罩。補洞使用 Telea、Navier-Stokes 或局部漸層，不使用擴散模型，
因此不會在空白位置生成假文字或新 UI 元素。

## 安裝

> 若只用 Git URL 安裝，不必先上架 Comfy Registry。

### ComfyUI Manager

在 Manager 選擇 **Install via Git URL**，貼上發布後的 GitHub Repository URL，
安裝完成後重新啟動 ComfyUI。

### Git Clone

```powershell
cd D:\ComfyUI\ComfyUI\custom_nodes
git clone https://github.com/igs-jaychiu/ComfyUI-SAM3-UI-Toolkit.git
D:\ComfyUI\python_embeded\python.exe -m pip install -r D:\ComfyUI\ComfyUI\custom_nodes\ComfyUI-SAM3-UI-Toolkit\requirements.txt
```

重新啟動 ComfyUI。若你的 portable Python 不在
`D:\ComfyUI\python_embeded\python.exe`，請換成實際路徑。

## 節點

### Review and Filter SAM3 Masks

- 為每個 SAM3 mask 顯示固定的 `#N`。
- `exclude_indices` 支援 `2,5,8-10` 格式。
- 綠框為保留，紅框 `XN` 為排除。
- 黃色區域是 grow 後真正會被修改的範圍。

### Require SAM3 Preview Approval

- `approved=false` 時，在 Preview 儲存後停止後續裁切與補圖。
- 確認遮罩並設為 `true` 後，才允許該階段繼續。

### Deterministic UI Inpaint

- `telea`：適合小文字與小型 Icon。
- `navier_stokes`：可與 Telea 比較邊緣延伸效果。
- `gradient`：適合平滑面板或按鈕移除後的背景。
- 遮罩以外的 tensor 像素會原值保留。

### 全自動節點（V4）

- **Auto Filter SAM3 Masks**：把 SAM3 individual masks 自動整理成「每個 UI 元件一張遮罩」。
  去重（IoU）、丟掉其實是容器的遮罩（`exclude_masks_*` 接入按鈕／緞帶遮罩）、丟掉被整行包含的單字碎片、
  `row_merge` 把同一行文字碎片接回、`close_holes` 補遮罩內洞。不需要人工填 `exclude_indices`。
- **Concat SAM3 Mask Batches**：把多個提示詞的遮罩批次接成一批，再交給 Auto Filter。
- **Crop SAM3 Masks To RGBA Sprites**：每張遮罩切成透明 PNG，並把座標寫到 `output/<prefix>_coords.json`。
- **Deterministic UI Inpaint** 新增 `interp` 方法（預設）：邊緣感知線性插值＋內部平滑，
  `grow` / `shadow_reach` 讓反鋸齒邊與軟陰影一起被移除；`bg_std_max` / `max_expand` 防止吃到框線或按鈕光澤。

### 通用自動分層節點（V5）

- **Auto Layer SAM3 Masks (z-order)**：把多個提示詞的遮罩倒進同一個池子，依**包含關係**自動排出
  z-order 層級。葉節點（文字／圖示／道具）是 LAYER_1，承載它們的按鈕與緞帶是 LAYER_2，
  再往上是卡片、面板、外框。分層依據是幾何包含，不是哪個提示詞找到的，所以換圖不用改參數。
  `min_votes` 是共識門檻：同一個元件要有幾個提示詞同時找到才算數，預設 2 可濾掉單一提示詞的幻覺；
  調成 3 會更乾淨但漏抓變多，調成 1 最完整但雜訊最多。
- **Concat SAM3 Mask Batches** 擴充到 8 個輸入，並輸出 `LABELS_JSON`，讓每張遮罩帶著來源提示詞名稱
  一路傳到資產命名。

### 其他節點

- Crop SAM3 Batch To Objects
- Merge SAM3 Mask Batch
- Overlay SAM3 Selection
- MAT Inpaint SAM3 Objects Sequentially（舊版相容）

最後一個 MAT 節點是選用功能，需要另外安裝
[`Acly/comfyui-inpaint-nodes`](https://github.com/Acly/comfyui-inpaint-nodes)
並提供 `INPAINT_MODEL`。V3 確定性 workflow 不需要 MAT。

## 範例 Workflow

### V5 通用自動分層（建議）

`example_workflows/SAM3_3_Generic_Auto_Layer_V5.json`

同一套流程適用任何 UI 或場景圖，不需要針對圖片改提示詞。28 個通用提示詞全部在原圖上跑，
遮罩池交給 Auto Layer 自動分層，每層依序切出透明資產再確定性補洞。
在 5 張風格完全不同的測試圖（設定視窗、商城、低對比紅絲絨結算板、遊戲主選單、中秋場景）上，
對 118 個人工標註元件的抓取率是 98.3%，單張最低 95%。

### V4 四階段分層（舊版）

`example_workflows/SAM3_2_Auto_Layered_UI_Extraction_V4.json`

無人工核准。所有 SAM3 偵測在原圖上平行執行，再依層次順序處理：
文字 → 物件 icon → 緞帶／按鈕 → 欄位卡片 → 主面板，每層先切出透明資產，再用確定性補洞把該層從畫布移除。
輸出在 `output/sam3_auto_v4/<stage>/`：`preview`（編號檢查圖）、`asset`（透明 PNG）、`asset_coords.json`（座標）、
`filled`（該層移除後的畫布），最後 `06_background/final` 是乾淨背景。換圖時只需改各階段提示詞與數量。

### V3 人工核准（舊版）

`example_workflows/SAM3_1_Four_Stage_UI_Extraction_V3_Deterministic.json`

此 workflow 還需要 ComfyUI 原生 SAM3 節點與 SAM3 checkpoint。所有 approval
Gate 預設關閉；請依照文字、Icon、按鈕順序逐階段檢查。

## 相容性

- Python 3.10+
- ComfyUI
- PyTorch（由 ComfyUI 提供）
- Pillow（由 ComfyUI 提供）
- OpenCV 4.8+

Windows portable、ComfyUI Desktop 與 NVIDIA CUDA 環境皆可使用。節點本身
不綁定特定 GPU；SAM3 模型的硬體需求由 ComfyUI 決定。

## 發布到 GitHub / Comfy Registry

1. 建立一個名為 `ComfyUI-SAM3-UI-Toolkit` 的 GitHub repository。
2. 將整個專案提交並推送到 GitHub；之後即可透過 Manager 的 Git URL 安裝。
3. 若要讓節點出現在 Registry 搜尋結果，再建立 Comfy Registry Publisher，並把
   `replace-with-your-comfy-publisher-id` 換成實際 Publisher ID 後發布。

Git URL 安裝不需要第 3 步。
