# 植物病蟲害智慧辨識系統

本專案以 DINOv3 ViT Large 影像分類模型，提供釋迦、茶葉、豆科及設施小葉菜的病蟲害辨識。FastAPI 負責非同步辨識任務；Streamlit 提供選擇作物、上傳圖片與查看結果的介面。

原始碼套件版本為 `1.1.0`。Docker image 使用 `rong0130/fruit-dl-api:1.0.1` 標籤；兩者是不同的版本標示。更新模型或程式後須重新建置 image；標籤相同不代表 Docker Hub 上的內容已同步更新。

## 支援作物與類別

四個部署模型均不含 `healthy` 類別。模型分類結果不能取代現場調查或病原鑑定。

| 作物 | API 代碼 | 模型標籤 | 中文名稱 |
| --- | --- | --- | --- |
| 釋迦 | `annona` | `Algal_leaf_spot` | 藻斑病 |
| 釋迦 | `annona` | `Anthracnose` | 炭疽病 |
| 釋迦 | `annona` | `Kanzawa_spider_mite` | 神澤氏葉蟎 |
| 釋迦 | `annona` | `Mealybug` | 粉介殼蟲 |
| 釋迦 | `annona` | `Rust` | 銹病 |
| 釋迦 | `annona` | `Sugar_apple_whitefly` | 番荔枝粉蝨 |
| 釋迦 | `annona` | `Tea_red_spider_mite` | 茶葉蟎 |
| 茶葉 | `tea` | `Agromyzidae` | 潛葉蠅 |
| 茶葉 | `tea` | `Caloptilia_theivora` | 茶細蛾 |
| 茶葉 | `tea` | `Oligonychus_coffeae` | 葉螨 |
| 茶葉 | `tea` | `Tortrix` | 捲葉蛾 |
| 豆科 | `legume` | `Bemisia_tabaci` | 銀葉粉蝨 |
| 豆科 | `legume` | `Liriomyza_spp` | 斑潛蠅 |
| 豆科 | `legume` | `Rust` | 銹病 |
| 豆科 | `legume` | `Scale_insects` | 介殼蟲 |
| 豆科 | `legume` | `Small_green_leafhopper` | 小綠葉蟬 |
| 豆科 | `legume` | `bean_aphid` | 蚜蟲 |
| 設施小葉菜 | `facility_leafy_greens` | `Aphids` | 蚜蟲 |
| 設施小葉菜 | `facility_leafy_greens` | `Phaedon_brassicae_Baly` | 十字花科小猿葉蟲 |
| 設施小葉菜 | `facility_leafy_greens` | `Pieris_rapae` | 紋白蝶 |
| 設施小葉菜 | `facility_leafy_greens` | `Striped_flea_beetle` | 黃條葉蚤 |
| 設施小葉菜 | `facility_leafy_greens` | `White_rust` | 白銹病 |

API 以「作物＋模型標籤」查中文名稱、問題類型與學名，避免不同作物的同名標籤互相混用。部分標籤只能支持屬、科或目層級的學名；對照依據見 `docs/scientific_name_sources.md`。

## 系統組成

- **FastAPI**：Bearer Token 驗證、非同步任務、SQLite 任務紀錄及標準化 `result_blocks`。
- **模型管理器**：依作物切換模型，同一時間只持有一個 predictor，以配合 8 GB GPU。
- **Streamlit**：選擇作物、上傳 1～5 張 JPEG／PNG 圖片、輪詢任務及顯示結果。
- **Docker image**：只封裝 FastAPI 與四組推論模型；Streamlit 在 image 外執行。

## 模型檔案

從原始碼啟動 API 或建置 Docker image 前，以下每個目錄都必須有 `best_model.pth` 與 `classes.json`：

```text
runs/
├── annona/20261026-173435_batch8-lr2e-5/
├── tea/20261026-191542_batch8-lr2e-5/
├── legume/20261026-203804_batch8-lr2e-5/
└── facility_leafy_greens/20261027-170444_batch8-lr2e-5/
```

GitHub 不包含模型權重、資料集或執行中的 SQLite 資料庫。依本專案 Dockerfile 建置的 image 會包含上述四組推論模型。

## 開發環境

- Python 3.10 以上；本機 `.python-version` 指定 3.12。
- `uv` 套件管理器。
- NVIDIA GPU 建議用於模型推論及訓練；目前 Docker 設定使用 NVIDIA CUDA 與 `--gpus all`。
- 本機 API、圖片伺服器及 Streamlit 分別使用 8000、9000、8501 port。

在 VS Code PowerShell 終端機進入專案後：

```powershell
# 安裝鎖定的專案依賴
uv sync

# 從可提交的範例建立本機訓練設定
Copy-Item .\config.yaml.example .\config.yaml
```

## 資料集準備與分割

沿用原專案「先依類別整理照片，再分割為訓練、驗證、測試集」的流程。四種作物須各自準備資料夾，**類別資料夾名稱必須使用上表的模型標籤**，不要以中文資料夾名稱直接訓練。例如茶葉資料：

```text
disease/
└── tea/
    ├── Agromyzidae/
    ├── Caloptilia_theivora/
    ├── Oligonychus_coffeae/
    └── Tortrix/
```

釋迦、豆科及設施小葉菜分別放在 `disease/annona/`、`disease/legume/`、`disease/facility_leafy_greens/`，並依上表建立各自的類別資料夾。目前部署的四個模型均不含 `healthy`，因此重現這四組模型時不要將 `healthy` 放進來源資料。

`split_dataset.py` 會讀取 `config.yaml` 的分割比例；目前範例設定為訓練 70%、驗證 15%、測試 15%。以下使用 `--copy` 保留原始照片。請在專案根目錄執行，每次使用新的目標目錄，避免混入舊資料：

```powershell
# 分割釋迦資料
uv run python .\split_dataset.py --source-dir .\disease\annona --target-dir .\datasets\annona\v1 --copy

# 分割茶葉資料
uv run python .\split_dataset.py --source-dir .\disease\tea --target-dir .\datasets\tea\v1 --copy

# 分割豆科資料
uv run python .\split_dataset.py --source-dir .\disease\legume --target-dir .\datasets\legume\v1 --copy

# 分割設施小葉菜資料
uv run python .\split_dataset.py --source-dir .\disease\facility_leafy_greens --target-dir .\datasets\facility_leafy_greens\v1 --copy
```

每個分割後的資料集應有相同的類別目錄：

```text
datasets/tea/v1/
├── train/
│   ├── Agromyzidae/
│   ├── Caloptilia_theivora/
│   ├── Oligonychus_coffeae/
│   └── Tortrix/
├── val/
│   └── 與 train 相同的類別目錄
└── test/
    └── 與 train 相同的類別目錄
```

分割後可先檢查各資料集。每個類別在 `train`、`val`、`test` 都應有足夠圖片；照片太少時，分割或後續訓練可能失敗。

```powershell
# 檢查釋迦資料集
uv run python .\check_data.py --data-dir .\datasets\annona\v1

# 檢查茶葉資料集
uv run python .\check_data.py --data-dir .\datasets\tea\v1

# 檢查豆科資料集
uv run python .\check_data.py --data-dir .\datasets\legume\v1

# 檢查設施小葉菜資料集
uv run python .\check_data.py --data-dir .\datasets\facility_leafy_greens\v1
```

## 版本化訓練

訓練前確認專案根目錄已有由 `config.yaml.example` 複製而來的 `config.yaml`。範例設定的批次大小為 8、學習率為 `0.00002`。`train_versioned.py` 預設使用 CUDA；若明確要使用 CPU，才加入 `--device cpu`。

四種作物分開執行訓練：

```powershell
# 訓練釋迦模型
uv run python .\train_versioned.py --plant annona --data-dir .\datasets\annona\v1 --run-name batch8-lr2e-5 --config .\config.yaml

# 訓練茶葉模型
uv run python .\train_versioned.py --plant tea --data-dir .\datasets\tea\v1 --run-name batch8-lr2e-5 --config .\config.yaml

# 訓練豆科模型
uv run python .\train_versioned.py --plant legume --data-dir .\datasets\legume\v1 --run-name batch8-lr2e-5 --config .\config.yaml

# 訓練設施小葉菜模型
uv run python .\train_versioned.py --plant facility_leafy_greens --data-dir .\datasets\facility_leafy_greens\v1 --run-name batch8-lr2e-5 --config .\config.yaml
```

每次訓練會建立新的 `runs/<作物>/<時間>_<run-name>/`，不覆蓋舊 run。輸出包含 `best_model.pth`、`classes.json`、逐 epoch 的 `metrics.csv`，以及 `evaluation/` 中的訓練／驗證 loss 曲線、驗證集和獨立測試集報告。訓練期間只使用驗證集選出最佳模型；訓練結束後才用同一份 `best_model.pth` 各評估一次 val 和 test。test 成績不參與選模。

`config.yaml` 的 `dataset.resample` 預設啟用每類固定 **300 次／epoch** 的取樣。原圖多於 300 張時，每個 epoch 無放回抽取 300 張；不足時，每張原圖先取一次，再有放回抽樣補足。抽樣索引每個 epoch 重新產生，重複取到的原圖會各自套用隨機資料增強；不會複製圖片檔案。只有 `train/` 使用此設定，`val/` 和 `test/` 維持原有圖片。訓練取樣次數可能高於原圖張數，因此訓練時間也可能增加。

若單次訓練要改用其他張數，在主要訓練入口加入 `--samples-per-class N`，例如 `--samples-per-class 150`。這個參數會覆蓋設定檔的重採樣方式與目標，啟用每類固定張數；省略時使用 `config.yaml`。實際設定寫入 run 的 `effective_config.yaml`，每個 epoch 的取樣總數記錄在 `summary.json`。

訓練期間以驗證集 **Macro Avg Recall**（各類別 recall 的簡單平均）選出 `best_model.pth`。報表同時保留整體 Accuracy、各類 precision／recall／F1／support、Weighted Avg、Loss 與混淆矩陣；Weighted Avg 僅供參考。

新訓練的 run 會自動產生 `evaluation/validation_report.txt`、`confusion_matrix.png`、`val_summary.json`、`test_report.txt`、`test_summary.json`、`test_confusion_matrix.png` 與 `loss_curve.png`。曲線讀取 `metrics.csv`，標示最佳 checkpoint 的 epoch，呈現完整訓練過程。報告包含主要 Macro Avg Recall、checkpoint／epoch、Loss、Accuracy，以及完整 classification report（含 Accuracy、Macro Avg、Weighted Avg）。缺少某類 test 圖片時，主要 Macro Avg Recall 標為 N/A；單類 support 少於 20 張時會附低樣本數提示。每個作物仍使用各自的 train／val／test 目錄與模型，不合併圖片。

新訓練的 run **不會自動成為 API 使用的模型**。若要更新部署模型，須先確認 `classes.json` 的類別與順序，並更新 `api/model_registry.py` 中對應的 run 路徑，再重新建置 Docker image。

## Streamlit 相似參考照片（FAISS demo）

此功能只在 `app_api.py` 展示，不改 API 回應，也不加入 Docker image。先為四個作物各建立索引；索引只收錄各自 `train/` 的原始圖片，每張一次，不收錄 `val/`、`test/` 或訓練重採樣的次數。索引與圖片路徑清單儲存在已忽略的 `runtime/faiss/`，本機須保留來源圖片。

```powershell
# 安裝 demo 專用依賴；Docker 不安裝這組選用依賴
uv sync --extra demo

# API 與 Streamlit 暫停時建立四作物索引，避免同時占用 GPU 或讀取更新中的索引
uv run --extra demo python .\build_faiss_demo.py --plant all

# 若本機沒有可用的 CUDA，可加入 --device cpu（建立時間會較久）
```

如只需重建一個作物，可將 `all` 改成 `annona`、`tea`、`legume` 或 `facility_leafy_greens`。每次更新 `api/model_registry.py` 登錄的模型，須以同一模型重建該作物索引；若本機索引模型版本與目前登錄模型不同，demo 會提示重建。**API 實際執行的模型也須與本機登錄表一致**；目前 API 回應不含可供自動核對的權重版本資訊。

若建立索引時使用 `--output-root` 指定其他目錄，啟動 Streamlit 前也須將 `STREAMLIT_FAISS_INDEX_ROOT` 設成該目錄。

收到 API 合併結果後，Streamlit 會在本機 CPU 逐張計算該病徵分數，選最高分的一張作為檢索圖，僅從相同作物、相同病徵的 `train/` 參考照片顯示最多三張。這會增加 demo 等待時間，但不會重新訓練模型。低信心而要求補拍時不展示參考照片。相似照片是視覺比對資料，不是診斷正確性的證明。

## 本機啟動 FastAPI 與 Streamlit

以下三個程序請分別放在三個 PowerShell 終端機，並保持終端機開啟。執行前請確認四組模型檔案已放在上述 `runs/` 路徑。

**終端機 1：FastAPI**

```powershell
# 設定本機測試用 Token；其他終端機須使用相同值
$env:ACADEMY_API_TOKEN = "<自行設定的測試 Token>"

# 只啟動一個 worker，避免重複載入大型模型
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

**終端機 2：圖片伺服器**

```powershell
# 建立圖片暫存目錄
New-Item -ItemType Directory -Force ".\runtime\streamlit_uploads" | Out-Null

# 提供 Streamlit 上傳圖片，供本機 FastAPI 下載
uv run python -m http.server 9000 --bind 127.0.0.1 --directory ".\runtime\streamlit_uploads"
```

**終端機 3：Streamlit**

```powershell
# 與 FastAPI 使用相同的 Token
$env:ACADEMY_API_TOKEN = "<與終端機 1 相同的測試 Token>"

# 指定本機 API 與圖片伺服器
$env:FRUIT_API_BASE_URL = "http://127.0.0.1:8000"
$env:STREAMLIT_IMAGE_BASE_URL = "http://127.0.0.1:9000"

# 啟動前端
uv run --extra demo streamlit run .\app_api.py
```

開啟 `http://127.0.0.1:8501`，選擇作物並上傳相同作物的 1～5 張圖片。

## 使用 Docker image 與本機 Streamlit

Docker image 只提供 API。使用這份原始碼與上述四組模型檔案在專案根目錄建置 image：

```powershell
docker build -t rong0130/fruit-dl-api:1.0.1 .
```

如果要使用 Docker Hub 的 image，可以改用 `docker pull rong0130/fruit-dl-api:1.0.1`；請先確認該標籤已推送本次建置的內容。推送指令為 `docker push rong0130/fruit-dl-api:1.0.1`，推送前須登入有權限的 Docker Hub 帳號。

在執行 `docker run` 的目錄建立 `.env`，至少包含一行：

```text
ACADEMY_API_TOKEN=<自行設定的 Token>
```

請勿提交或公開 `.env`。在 PowerShell 執行：

```powershell
# 啟動 API；SQLite 資料保存於 Docker volume
docker run -d --name fruit-dl-api --restart unless-stopped --gpus all --env-file ".\.env" -p 127.0.0.1:8000:8000 -v fruit-dl-runtime:/app/runtime rong0130/fruit-dl-api:1.0.1

# 查看容器狀態
docker ps --filter "name=fruit-dl-api"
```

此範例假設 8000 port 尚未被其他 API 佔用，且沒有同名 container。若已啟動舊版 `fruit-dl-api` 容器，重新建置或拉取 image 後須停止並移除舊容器，再執行上述 `docker run`；`fruit-dl-runtime` volume 仍保存 SQLite 資料。

若要在 Windows 主機的 Streamlit 顯示 FAISS 參考照片，主機仍須備有 `api/model_registry.py` 登錄的四組本機權重、`datasets/<作物>/v1/train/` 原圖及已建立的 `runtime/faiss/` 索引；Docker image 中的模型不會供 Streamlit 直接讀取。缺少這些檔案時，原有 API 診斷仍可使用，但參考照片無法顯示。

若 Streamlit 和圖片伺服器在 Windows 主機上執行，圖片伺服器改用以下指令，讓 Docker 容器能下載圖片：

```powershell
# 在專案目錄建立上傳圖片目錄
New-Item -ItemType Directory -Force ".\runtime\streamlit_uploads" | Out-Null

# 讓容器可經由主機網址連接 9000 port
uv run python -m http.server 9000 --bind 0.0.0.0 --directory ".\runtime\streamlit_uploads"
```

啟動 Streamlit 前，設定：

```powershell
# Token 必須與 Docker 容器 .env 中的 ACADEMY_API_TOKEN 相同
$env:ACADEMY_API_TOKEN = "<與 Docker 容器相同的 Token>"

# Streamlit 從 Windows 主機連到 API
$env:FRUIT_API_BASE_URL = "http://127.0.0.1:8000"

# FastAPI 從 Docker 容器連回主機上的圖片伺服器
$env:STREAMLIT_IMAGE_BASE_URL = "http://host.docker.internal:9000"

# 啟動 Streamlit
uv run --extra demo streamlit run .\app_api.py
```

## API 使用方式

三支端點都需要 Bearer Token：

| Method | Path | 用途 |
| --- | --- | --- |
| `GET` | `/v1/health` | 檢查模型及背景 Worker 是否可服務 |
| `POST` | `/v1/tasks` | 建立非同步辨識任務 |
| `GET` | `/v1/tasks/{task_id}` | 查詢狀態、進度或結果 |

`POST /v1/tasks` 的 `task.plant` 可填 `annona`、`tea`、`legume`、`facility_leafy_greens`；省略時預設為 `annona`。請求可包含 1～5 張 JPEG 或 PNG 圖片的可下載 URL。`expires_at` 必須包含時區。

請求結構範例：

```json
{
  "platform_task_id": "example-tea-001",
  "task": {
    "academy_type": "plant_diagnosis",
    "plant": "tea"
  },
  "images": [
    {
      "image_url": "https://example.org/example.jpg",
      "expires_at": "2026-12-31T23:59:59+08:00",
      "mime_type": "image/jpeg"
    }
  ]
}
```

POST 回傳 `task_id` 和 `PENDING`。接著以 GET 查詢，直到狀態為 `SUCCEEDED` 或 `FAILED`。`PROCESSING` 時可能附帶 `progress`；成功時有 `result`，失敗時有 `error`。成功結果包含標準化的 `result_blocks`。

HTTP 錯誤使用 `{"code": "...", "message": "..."}` 格式，例如認證失敗的 `401 UNAUTHORIZED`、請求驗證失敗的 `400 VALIDATION_ERROR`、模型不可用的 `503 MODEL_UNAVAILABLE`，以及未預期錯誤的 `500 INTERNAL_ERROR`。


## 測試

```powershell
# 執行專案測試
uv run pytest -q
```
