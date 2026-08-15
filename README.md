# 植物病蟲害智慧辨識系統 v1.1

基於 **DINOv3 (ViT Large)** 的深度學習影像分類模型，專為植物病害與蟲害精準辨識而設計。本系統整合了現代化的參數管理與資料平衡技術，提供高效且易於擴充的解決方案。

## 支援的病蟲害類別

本系統目前支援以下 8 種類別（與新資料集 `disease` 同步）：

| 類別 | 中文名稱 | 類型 |
|------|---------|------|
| `healthy` | 健康 | - |
| `algal_leaf_spot` | 藻斑病 | 病害 |
| `anthracnose` | 炭疽病 | 病害 |
| `pest_kanzawa_spider_mite` | 神澤氏葉蟎 | 蟲害 |
| `pest_mealybug` | 粉介殼蟲 | 蟲害 |
| `pest_tea_mite` | 茶葉蟎 | 蟲害 |
| `pest_whitefly` | 番荔枝粉蝨 | 蟲害 |
| `rust` | 銹病 | 病害 |

## 核心特色

- **先進模型**: 採用 Meta 最新推出的 **DINOv3 (ViT Large)** 架構，具備卓越的特徵提取能力。
- **參數設定化**: 透過 `config.yaml` 統一管理模型、訓練、資料分割與重採樣設定，無需更動程式碼。
- **資料平衡 (Resample)**: 內建資料集重採樣功能，支援固定數量或比例平衡各類別樣本，解決資料不均問題。
- **現代化管理**: 使用 `uv` 進行極速依賴管理與環境隔離。

## FastAPI 與 Streamlit 串接

本分支新增非同步植物病蟲害辨識 API，並將結果串接至 `app_api.py`：

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/v1/health` | 健康檢查 |
| `POST` | `/v1/tasks` | 建立辨識任務 |
| `GET` | `/v1/tasks/{task_id}` | 輪詢任務狀態與結果 |

API 會依序回傳 `PENDING`、`PROCESSING`、`SUCCEEDED` 或 `FAILED`。執行前請確認 `output/best_model.pth` 與 `output/classes.json` 已準備完成。

## 安裝與快速開始

### 環境需求
- Python 3.10+
- NVIDIA GPU (建議) 或 CPU
- [uv](https://github.com/astral-sh/uv) 套件管理器

### 安裝步驟

```bash
# 複製專案
git clone <your-repo-url>
cd fruit-DL

# 初始化與同步環境
uv sync
```

## 資料集準備

### 1. 原始資料結構
將照片按類別放入 `disease/` 目錄：
```
disease/
  ├── healthy/
  ├── rust/
  └── ...
```

### 2. 分割資料集
系統會根據 `config.yaml` 中的比例（預設 70/15/15）將資料切分為 Train/Val/Test：
```bash
# 執行分割
python split_dataset.py --copy
```

## 參數設定 (`config.yaml`)

所有關鍵參數皆可在 `config.yaml` 中調整：
```yaml
model:
  name: "vit_large_patch16_dinov3.lvd1689m"
dataset:
  split: { val_ratio: 0.15, test_ratio: 0.15 }
  resample: { enabled: true, target: 200 }
train:
  batch_size: 8
  epochs: 30
  lr: 0.0001
```

## 使用方式

### 1. 訓練模型
```bash
# 開始訓練 (自動讀取 config.yaml)
python train.py

# 或使用 uv 指定參數覆蓋
uv run python train.py --batch-size 4 --epochs 50
```

### 2. 模型評估 (測試集)
```bash
# 使用訓練後最佳模型評估獨立測試集
python evaluate.py --model-path output/best_model.pth --data-dir dataset/test
```

### 3. Web 介面（FastAPI + Streamlit）

本機測試需在專案根目錄開啟三個終端機。

終端機 1：啟動 FastAPI（模型只載入一次，因此使用一個 worker）

```powershell
$env:ACADEMY_API_TOKEN = "local-development-token"
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Swagger 文件：http://127.0.0.1:8000/docs

終端機 2：啟動圖片伺服器，模擬正式環境的 Signed URL

```powershell
New-Item -ItemType Directory -Force ".\runtime\streamlit_uploads" | Out-Null
uv run python -m http.server 9000 --bind 127.0.0.1 --directory ".\runtime\streamlit_uploads"
```

終端機 3：啟動已串接 API 的 Streamlit

```powershell
$env:ACADEMY_API_TOKEN = "local-development-token"
uv run streamlit run app_api.py
```

開啟 http://localhost:8501 後，即可上傳同一植株的 1～5 張 JPEG 或 PNG 圖片進行測試。

若要執行自動化測試：

```bash
uv run pytest -q
```

## 技術規格

- **模型架構**: DINOv3 ViT Large (`vit_large_patch16_dinov3.lvd1689m`)
- **參數規模**: ~1.2GB 權重檔案
- **優化技術**: AdamW Optimizer, AMP (混合精度訓練)
- **硬體需求**: 
  - 訓練 VRAM: >= 8GB (建議)
  - 推論 VRAM: ~2GB

### **NPUST DN-LAB © 2026**
