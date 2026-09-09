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

### V7 還原度：把切出來的素材貼回去,量它還原多少

以前只能看預覽圖判斷切得好不好。V7 把每張 sprite 依座標貼回背景上重畫整個畫面,
再跟原始截圖逐像素比較,得到一個數字。

- **Score Asset Reconstruction**（`SAM3ReconstructScore`）：吃原圖、背景與八層 sprite＋座標,
  輸出重建圖、誤差熱圖與 JSON 報告。`score` 是誤差在 `tolerance`（預設 10/255）以內的像素比例;
  報告同時給 `score_tight`（2/255）、`mae255`、`psnr`,以及 `opaque_of_alpha`、`alpha_px_per_screen`
  這兩個防呆值 —— 只要 sprite 退化成不透明方塊,分數會漂亮但這兩個值會跟著跑掉。
- **Crop 節點新增 `under` 輸入**：接該層 Deterministic UI Inpaint 的輸出。
  「元件底下是什麼顏色」以前只能估,而且估出來的還要跟後面補洞節點另一套參數估的結果吻合,
  不吻合邊緣就錯。補洞結果不是估的 —— 重建時 sprite 底下真的就是它。
  有了 `under`,Crop 直接量補洞改動了哪些像素、用 `C = a*F + (1-a)*U` 解出顏色,
  再把 alpha 取成線段 `U → F` 上離原像素最近的點。
- **陰影跟著元件走**：補洞會把元件連同陰影一起擦掉,但 sprite 以前只切到元件本體,
  陰影因此兩邊都不見（素材沒有、背景被咬掉一圈）。現在補洞改動的每個像素都指派給
  **最近**的元件（不是每個附近的元件,否則相鄰按鈕會互相蓋掉共用的間隙)。
- **解不出來的像素照原樣保留**：這種畫風每個元件都有近黑描邊,而描邊不是任何顏色的混合,
  線段解不到,以前描邊會被洗淡。現在殘差超過 `under_exact` 就保留原像素並標為不透明 —— 墨線本來就是不透明的。
- **`feather` 預設改 0**：模糊過的 alpha 不再代表該像素真正的覆蓋率,貼回去就對不上。

實測（三張圖,`tolerance=10`）:

| 圖 | 尺寸 | score | score_tight (2/255) | MAE | PSNR |
| --- | --- | --- | --- | --- | --- |
| Cocos 遊戲截圖 | 750x1334 | 99.66% | 97.29% | 0.21 | 39.5 dB |
| Cocos 遊戲截圖 | 1500x2668 | 99.87% | 98.11% | 0.13 | 42.6 dB |
| 商城面板 | 752x1344 | 99.60% | 96.17% | 0.29 | 37.4 dB |

同一批素材仍然是真正的挖空圖:101 張 sprite 的 alpha 平均覆蓋率 0.71,
只有 4 張超過 95%（其中一張是整螢幕背景）。

#### `score` 高不代表每張素材單獨可用

這點要講清楚,不然數字會被誤讀。`score` 只回答「**疊回去看不出差別**」。
容器素材本來就該把子元件挖掉,而挖掉的那塊補成什麼樣子, 疊回去時剛好被子元件蓋住 ——
所以補得再爛 `score` 也不會掉。這是三個不同的問題,報告分別給三個數字:

| 問題 | 指標 | 實測 |
| --- | --- | --- |
| 疊回去像不像原畫面 | `score` / `score_tight` | 99.66% / 97.3% |
| 每張素材自己像不像原圖 | `asset_real`（分層給） | layer1 **1.000**、layer2 0.75、layer4 0.09 |
| 補洞有沒有溢出到子元件以外 | `asset_unexplained` | 0.0016（最差單張 0.147） |

`asset_real` 在 layer 1 必須是 1.0（葉節點沒有東西被剝離,所以它就該逐像素等於原圖）,
容器低於 1 是正常的；真正該看的是 `asset_unexplained` —— 編出來的面積扣掉子元件蓋住的面積。

#### 補洞改用重複紋理複製

`interp` 線性插值對平面板子是對的,對有花紋的東西是錯的:棋盤木板會補成一團漸層,
所以容器素材看起來跟原畫面完全不像。同一塊紋理幾乎都在一個週期外的位置,
所以現在先用自相關找出週期再**複製**（不是猜),整條流程仍然沒有生成模型。

量法是在畫面上「沒有任何細元件」的背景挖出元件形狀的洞,補完跟真實像素比:

| 洞尺寸 | MAE | 誤差 ≤24/255 | 誤差 >64/255 | 90 百分位 |
| --- | --- | --- | --- | --- |
| 30-130 px　interp | 5.26 | 0.948 | 2.3% | 11 |
| 30-130 px　週期複製 | **2.99** | **0.987** | **0.7%** | **6** |
| 120-260 px　interp | 9.11 | 0.893 | 3.9% | 30 |
| 120-260 px　週期複製 | **8.78** | **0.930** | 4.9% | **17** |

大洞會切成 `periodic_block` 大小的塊各自找 offset —— 一個 offset 要對整個大洞負責時,
它可以通過邊界檢查而中間卻落在隔壁的 sprite 上（棋盤中間會多一隻貓）。
超過 `periodic_max_span` 就交回 `interp`:那邊實測贏不了,因為內部的塊已經沒有可信邊界可以驗。
另外來源區的局部變異不能明顯高於洞周圍的環帶,否則同一塊板子上的另一行文字會被抄進來。

#### Pack 兩種切法都裝, 並標註每張該用哪一份

依包含關係分層, 對「按鈕包著標籤」是對的, 對「一行文字包著自己的字」是錯的 ——
把後者剝掉只剩一塊沒有字的底板, 那不是任何人要的素材。所以 zip 裡兩份都有:

- `all/layerN/` — **asset**（下層已剝離）
- `flat/layerN/` — **flat**（直接從原圖切）
- `curated/` — 每個元件挑「可用的那一份」, manifest 的 `cut` 欄位記錄挑了哪種

manifest 每個元件帶三個判斷依據 (`use` 是結論):

| 欄位 | 意思 | 觸發 flat 的條件 |
| --- | --- | --- |
| `covered_by_children` | 更細的層蓋掉它多少 | > 0.5 → 剝完幾乎都是補出來的 |
| `peeled_matches_screen` | 剝離版還在的部分跟原圖吻合多少 | < 0.6 → 剩下的東西已經不對 |
| `peeled_lost_area` | 少掉的面積扣掉子元件該蓋的 | > 0.25 → 元件被吃掉一塊 |

第三個是必要的:一行文字的最後一個字被別層的偵測吃掉時, 前兩個指標都很漂亮
（covered 0.06、matches 0.94）但素材就是缺字。分母必須是元件自己的 mask 面積,
不能拿 flat 的面積比 —— flat 的 matte 帶軟陰影裙邊, 會讓每個切得很緊的葉節點看起來像少了一半。

實測 101 個元件:葉節點全部 `lost = 0`、`matches = 1.0`、判 asset;
只有 1 個元件 `lost > 0.05`（就是上面那行文字, 0.42）;22 個判 flat。

### V6 新增 / 強化

- **SAM3 Prompt Bank**：一個節點跑完整份提示詞清單,格式 `名稱 | 提示詞 | 門檻`,一行一個。
  改提示詞只要編輯文字框,不用重拉線。工作流節點數從 114 降到 62。
- **差異遮罩 alpha**：SAM3 給文字的是填滿的方塊,切出來會帶著底板。Crop 節點改用
  「估出底下的底色再依色差算 alpha」,把字形摳出來並保留抗鋸齒。文字不透明佔比 0.70 → 0.40。
  三道護欄避免誤傷實心物件:遮罩邊界已貼合真實邊緣就跳過、挖掉的區域若被自己包住就跳過、
  保留比例過低就退回原遮罩。
- **同款元件對齊**:重複版面的元件輸出成同尺寸,跨層歸組。實測商城圖卡片寬度差 8.1% → 0.9%,
  按鈕 1.7%/14.7% → 0%/0%,緞帶 1.0%/2.4% → 0%/0%。
- **補洞自適應**:`auto_scale` 依該層元件實際大小推算外擴與陰影範圍。場景圖背景殘差 0.094 → 0.040。
- **分層上限 8 層**,並在座標 JSON 帶上 `uid`、`layer`、`label`、`votes`、`parent`、`area`,
  可以直接重建 UI 樹,也能事後依票數過濾而不必重跑偵測。

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

### V6 提示詞庫 + 自動分層（建議）

`example_workflows/SAM3_4_Prompt_Bank_Auto_Layer_V6.json`

74 個節點。提示詞集中在一個文字框,8 層 z-order,每層輸出 `asset`（乾淨容器）與
`flat`（原圖外觀）兩份透明 PNG 加座標 JSON。5 張測試圖抓取率 98.3%,單張最低 95%。
每層的 `asset` 已接上該層補洞結果作為 `under`,最後由 Score Asset Reconstruction
把素材貼回去重畫並輸出還原度（實測 99.6%,見上表）。

### V5 通用自動分層（舊版）

`example_workflows/SAM3_3_Generic_Auto_Layer_V5.json`

同一套流程適用任何 UI 或場景圖，不需要針對圖片改提示詞。28 個通用提示詞全部在原圖上跑，
遮罩池交給 Auto Layer 自動分層，每層依序切出透明資產再確定性補洞。
在 5 張風格完全不同的測試圖（設定視窗、商城、低對比紅絲絨結算板、遊戲主選單、中秋場景）上，
對 118 個人工標註元件的抓取率是 98.3%，單張最低 95%。

每層會輸出兩份透明 PNG：

- `asset`：把下層元件移除後才切下的**乾淨容器**。按鈕、卡片、面板要用這份，裡面不會殘留文字。
- `flat`：直接從原圖切下的**原樣**。整段文字、logo、角色要用這份，不會被下層的移除弄糊。

座標寫在同層的 `asset_coords.json` / `flat_coords.json`。

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
