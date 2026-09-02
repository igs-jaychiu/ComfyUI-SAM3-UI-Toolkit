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

### 其他節點

- Crop SAM3 Batch To Objects
- Merge SAM3 Mask Batch
- Overlay SAM3 Selection
- MAT Inpaint SAM3 Objects Sequentially（舊版相容）

最後一個 MAT 節點是選用功能，需要另外安裝
[`Acly/comfyui-inpaint-nodes`](https://github.com/Acly/comfyui-inpaint-nodes)
並提供 `INPAINT_MODEL`。V3 確定性 workflow 不需要 MAT。

## 範例 Workflow

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
