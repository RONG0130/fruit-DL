# AGENTS.md

本檔案提供協助維護此專案的 AI 助理使用。請以繁體中文溝通，修改前先確認實際程式與 Git 狀態，保留工作區既有變更。

## 專案概述

本專案使用 DINOv3 ViT Large 辨識四種作物的病蟲害：

| 作物代碼 | 作物 | 模型類別數 |
| --- | --- | ---: |
| `annona` | 釋迦 | 6 |
| `tea` | 茶葉 | 5 |
| `legume` | 豆科 | 7 |
| `facility_leafy_greens` | 設施小葉菜 | 6 |

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
- `evaluate.py`：單作物 val／test 評估、Macro Avg Recall、報告與 loss 曲線。
- `train.py`：訓練資料增強與每類固定次數的重採樣。
- `build_faiss_demo.py`、`similarity_demo.py`：本機 Streamlit Demo 的圖片特徵與相似照片索引。
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

- `runs/annona/20261102-175010_batch8-lr2e-5/`
- `runs/tea/20261102-205448_batch8-lr2e-5/`
- `runs/legume/20261102-231927_batch8-lr2e-5/`
- `runs/facility_leafy_greens/20261103-202056_batch8-lr2e-5/`

每組需要 `best_model.pth` 與 `classes.json`。模型權重與資料集不進 GitHub；從原始碼執行 API 或建置 Docker image 前，須另行準備這些檔案。Docker Hub 已建置的 image 則包含推論所需模型。

開發機 GPU 僅有 8 GB VRAM。`ModelManager` 一次只保留一個作物的 predictor，並以鎖保護模型切換及推論。不要改成四個 DINOv3 Large 模型同時常駐 GPU。部署 FastAPI 時使用一個 Uvicorn worker，避免每個程序重複載入模型。

同一個模型標籤可能出現在不同作物，例如 `Rust`。查中文名稱、問題類型和學名時，必須同時使用作物代碼與原始模型標籤。不要根據照片或通稱自行推定沒有來源支持的物種學名。

釋迦 `Oligonychus_coffeae` 是合併神澤氏葉螨與茶葉螨資料後的模型標籤；對外只稱「葉螨」，學名用共同的葉蟎科 `Tetranychidae`，不得把整類說成單一 *Oligonychus coffeae*。豆科新增 `Agromyzidae`，設施小葉菜新增 `Liriomyza_spp`；四個作物的最新 `classes.json`、API 對照、Streamlit 側欄及 Docker 複製路徑須保持一致。

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

每次訓練建立獨立的 `runs/<plant>/<timestamp>_<run-name>/`，包含模型權重、`classes.json`、有效設定、指標、驗證與獨立測試報告、混淆矩陣，以及 train／val loss 曲線。部署時使用的 run 仍須與 `api/model_registry.py` 登錄路徑及類別順序一致。

### 本次評估與重採樣決策

- 四個作物各自訓練與評估，不合併圖片。主要指標是各類別 recall 的簡單平均，即 Macro Avg Recall；Accuracy、Weighted Avg、各類 precision／recall／F1／support 與混淆矩陣仍保留供參考。
- 訓練期間以 val Macro Avg Recall 選擇 `best_model.pth`；訓練結束後自動以最佳模型各產生一次 val 與獨立 test 報告。不要把 val 成績稱為 test 成績。若某類別沒有測試圖片，主要指標為 N/A；support 少於 20 張時提示低樣本數。
- `config.yaml.example` 預設啟用每類每 epoch 300 次的 `fixed_count` 重採樣。原圖多於 300 張時無放回抽 300 張；不足時保留所有原圖各一次，再有放回抽樣補足。每個 epoch 重新抽樣；重複抽到的圖片重新套用訓練隨機增強，不產生新檔。僅作用於 train；可用 `train_versioned.py --samples-per-class N` 覆蓋。

### FAISS Streamlit Demo

- 使用現有各作物 DINOv3 模型的分類前特徵向量，不使用 CLIP。執行 `uv run --extra demo python .\build_faiss_demo.py --plant all`，將四作物各病徵 `train/` 原圖建立成 `runtime/faiss/` 索引；不收錄 val／test 或重採樣次數。更換模型權重後須重建對應索引。
- API 逐張分類後，對各類別分數取平均並回傳前三名；它不回傳每張圖片的分數或特徵向量。Streamlit Demo 因此以本機同版模型再推論上傳圖片，選出對 API 第一名病徵分數最高的一張，僅在同作物、同病徵索引中顯示最相似的三張照片。FAISS 只在 Streamlit Demo，不加入 API 與 Docker image。
- Docker image 仍使用 `rong0130/fruit-dl-api:1.0.1` 標籤；標籤相同不表示 Docker Hub 內容已更新。Streamlit 顯示 FAISS 參考照片時，主機需要本機模型權重、train 原圖與索引。

## 修改與驗證原則

1. 修改前先執行 `git status --short`，辨識並保留使用者已有的變更。
2. 修改 API 契約時，檢查 `api/schemas.py`、Streamlit client、背景任務與相關測試是否一致。
3. 修改模型標籤或中文對照時，同步核對 `classes.json`、`api/result_mapper.py`、`streamlit_plant.py` 與學名來源文件。
4. 修改 Docker 封裝時，確認四組模型路徑、`.dockerignore` 和單 worker 設定。
5. 執行與修改相關的測試，並如實說明尚未驗證的部分。
6. 不將 `.env`、Token、資料集、模型權重、SQLite 資料庫或 `runtime/` 內容提交到 Git。
