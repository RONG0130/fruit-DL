# AGENTS.md

本檔案提供協助維護此專案的 AI 助理使用。請以繁體中文溝通，修改前先確認實際程式與 Git 狀態，保留工作區既有變更。

## 專案概述

本專案使用 DINOv3 ViT Large 辨識四種作物的病蟲害：

| 作物代碼 | 作物 | 模型類別數 |
| --- | --- | ---: |
| `annona` | 釋迦 | 7 |
| `tea` | 茶葉 | 4 |
| `legume` | 豆科 | 3 |
| `facility_leafy_greens` | 設施小葉菜 | 4 |

目前部署的四個模型均不含 `healthy` 類別。不要將舊版單一釋迦模型的類別名稱或 `output/` 路徑套用到四作物 API。

## 主要程式

- `api/main.py`：FastAPI 端點、Bearer Token 驗證、統一錯誤格式及服務生命週期。
- `api/schemas.py`：請求與回應的資料驗證；`task.plant` 未提供時預設為 `annona`。
- `api/model_registry.py`：四作物模型路徑、類別順序檢查及單一 GPU 模型管理器。
- `api/worker.py`：背景任務、圖片下載、模型推論與結果保存。
- `api/store.py`：SQLite 任務紀錄及未完成任務恢復。
- `api/result_mapper.py`：以「作物代碼＋模型標籤」查詢中文名稱、問題類型及學名。
- `docs/scientific_name_sources.md`：學名對照的來源與分類精度說明。
- `app_api.py`、`streamlit_api_client.py`、`streamlit_plant.py`：Streamlit 介面與 API 串接。
- `train_versioned.py`：版本化訓練入口；每次訓練寫入獨立的 `runs/<plant>/<run>/` 目錄。
- `Dockerfile`：只封裝 FastAPI 推論服務與四組模型；Streamlit 在 Docker image 外執行。

## API 契約

三支端點都需要 `Authorization: Bearer <ACADEMY_API_TOKEN>`：

- `GET /v1/health`
- `POST /v1/tasks`
- `GET /v1/tasks/{task_id}`

建立任務時，`task.academy_type` 必須為 `plant_diagnosis`；`task.plant` 可為 `annona`、`tea`、`legume` 或 `facility_leafy_greens`。舊請求省略 `plant` 時仍使用釋迦模型。

POST 回傳 `task_id` 與 `PENDING`。GET 任務狀態可能為 `PENDING`、`PROCESSING`、`SUCCEEDED` 或 `FAILED`；`progress` 僅在處理中回傳。成功結果使用既有 `result_blocks` 契約。維護時請保留 Bearer Token 驗證、任務狀態、錯誤格式及舊請求相容性。

HTTP 錯誤回應使用 `{"code": "...", "message": "..."}`。重要代碼包括 `UNAUTHORIZED`、`VALIDATION_ERROR`、`MODEL_UNAVAILABLE`、`TASK_NOT_FOUND`、`TASK_PLANT_CONFLICT` 與 `INTERNAL_ERROR`。背景圖片下載或推論失敗則由 GET 任務結果回傳 `FAILED` 與 `error`。

## 模型與 GPU

四組模型路徑固定於 `api/model_registry.py`：

- `runs/annona/20261006-215715_batch8-lr2e-5/`
- `runs/tea/20261006-223812_batch8-lr2e-5/`
- `runs/legume/20261006-232420_batch8-lr2e-5/`
- `runs/facility_leafy_greens/20261007-000803_batch8-lr2e-5/`

每組需要 `best_model.pth` 與 `classes.json`。模型權重與資料集不進 GitHub；從原始碼執行 API 或建置 Docker image 前，須另行準備這些檔案。Docker Hub 已建置的 image 則包含推論所需模型。

開發機 GPU 僅有 8 GB VRAM。`ModelManager` 一次只保留一個作物的 predictor，並以鎖保護模型切換及推論。不要改成四個 DINOv3 Large 模型同時常駐 GPU。部署 FastAPI 時使用一個 Uvicorn worker，避免每個程序重複載入模型。

同一個模型標籤可能出現在不同作物，例如 `Rust`。查中文名稱、問題類型和學名時，必須同時使用作物代碼與原始模型標籤。不要根據照片或通稱自行推定沒有來源支持的物種學名。

## 本機開發

專案使用 `uv` 管理依賴。`pyproject.toml` 支援 Python 3.10 以上；本機 `.python-version` 為 3.12，Docker 使用 Python 3.10。

```powershell
# 安裝依賴
uv sync

# 複製訓練設定範例；config.yaml 為本機檔案，不提交 Git
Copy-Item .\config.yaml.example .\config.yaml

# 執行測試
uv run pytest -q
```

本機啟動 FastAPI 時，須設定 `ACADEMY_API_TOKEN`。Streamlit 使用相同 Token，並透過 `FRUIT_API_BASE_URL` 指定 API。圖片由本機 HTTP 伺服器提供，網址透過 `STREAMLIT_IMAGE_BASE_URL` 設定。

若 API 在 Docker、圖片伺服器在 Windows 主機，圖片網址應使用 `http://host.docker.internal:9000`，而圖片伺服器須監聽 `0.0.0.0`；容器中的 `127.0.0.1` 指向容器自身。

## 訓練與輸出

`train_versioned.py` 以 `--plant`、`--data-dir`、`--run-name` 和 `--config` 指定訓練。資料目錄須有類別一致的 `train/`、`val/`、`test/`。預設裝置為 CUDA；CUDA 不可用時不會自動切換至 CPU，明確指定 `--device cpu` 才會使用 CPU。

每次訓練建立獨立的 `runs/<plant>/<timestamp>_<run-name>/`，包含模型權重、`classes.json`、有效設定、指標、驗證報告與混淆矩陣。部署時使用的 run 仍須與 `api/model_registry.py` 登錄路徑及類別順序一致。

## 修改與驗證原則

1. 修改前先執行 `git status --short`，辨識並保留使用者已有的變更。
2. 修改 API 契約時，檢查 `api/schemas.py`、Streamlit client、背景任務與相關測試是否一致。
3. 修改模型標籤或中文對照時，同步核對 `classes.json`、`api/result_mapper.py`、`streamlit_plant.py` 與學名來源文件。
4. 修改 Docker 封裝時，確認四組模型路徑、`.dockerignore` 和單 worker 設定。
5. 執行與修改相關的測試，並如實說明尚未驗證的部分。
6. 不將 `.env`、Token、資料集、模型權重、SQLite 資料庫或 `runtime/` 內容提交到 Git。