# Streamlit 植物選擇實作規劃

> 本文件供審閱；尚未實作。實作時逐項驗證，並保留工作區既有變更。

**目標：** 讓使用者在 Streamlit 選擇釋迦、茶葉、豆科或設施小葉菜，送出的 `POST /v1/tasks` 包含正確 `task.plant`，且畫面不顯示其他植物的舊結果或舊類別清單。

**架構：** 植物代碼是前端與 API 的共同識別值；前端只維護中文選項與類別顯示文字，不載入模型。`PlantDiagnosisAPIClient` 負責組成請求，`app_api.py` 負責選擇、狀態重設及呈現。API 回傳的 `result_blocks` 仍是診斷名稱、問題類型與學名的權威來源。

**技術：** Python、Streamlit、httpx、pytest；不修改 FastAPI、模型、Docker 或訓練程式。

**依據：** 本專案現有 `app_api.py`、`streamlit_api_client.py`，以及已實作的 FastAPI `task.plant` 契約。

## 現況與檔案責任

- `app_api.py:1190` 上傳圖片；`:1306` 案件表單；`:1337` 呼叫 `run_diagnosis`；`:529` 建立並輪詢任務；`:1110` 側欄仍列釋迦舊類別。
- `streamlit_api_client.py:92` 的 `create_task()` 目前未接受 `plant`，請求中的 `task` 只有 `academy_type`。
- `tests/test_streamlit_api_client_auth.py` 只檢查 Token；`tests/test_api_progress.py` 以 AST 確認表單、輪詢及完整 GET 回應的既有行為。
- 新增前端測試檔 `tests/test_streamlit_plant_selection.py`，以輕量檢查驗證選項、切換後的狀態重設和畫面資料來源；既有測試保留。

## 固定行為與資料契約

- 選項依序為「釋迦 `annona`」、「茶葉 `tea`」、「豆科 `legume`」、「設施小葉菜 `facility_leafy_greens`」；預設釋迦。
- 選擇元件放在圖片上傳區之前、案件表單之外，使切換植物時立即更新提示與側欄，而填寫案件表單的文字仍保持原有一次送出行為。
- 使用者送件時將**當次選定的代碼**傳給 `run_diagnosis(..., plant: str)`，再傳給 `create_task(..., plant: str = "annona")`；HTTP body 只在 `task` 新增 `plant`，其餘欄位與 Bearer Token 不變。
- 切換植物時清除 `latest_task_id`、`latest_result`、`latest_task_response`、`latest_error`；圖片保留供使用者自行確認是否屬於所選植物。圖片指紋與植物代碼共同決定目前結果是否仍有效，避免同一張圖片換植物後顯示前一植物的結果。
- 側欄「支援辨識類別」改成所選植物的實際模型類別；結果表格、長條圖及學名仍讀 API 的 `result_blocks`，不以側欄清單覆寫 API 結果。未確認學名遵循 API 的選填欄位，不補猜。
- 舊程式呼叫 `create_task()` 未傳 `plant` 時仍送 `annona`；維持原有圖片限制、進度、錯誤顯示、完整 API 回應和輪詢時間。

## 實作順序

### 1. API Client 傳送植物代碼

**修改：** `streamlit_api_client.py:92`；擴充 `tests/test_streamlit_api_client_auth.py`。

- [ ] 先新增測試：呼叫 `create_task(..., plant="tea")` 時，攔截的 POST JSON 含 `task.plant == "tea"`，且 Bearer Token、`academy_type`、images、context 維持原值；省略 plant 時為 `annona`。
- [ ] 執行該測試，確認因現有函式不接受 `plant` 而失敗。
- [ ] 為 `create_task()` 新增 `plant: str = "annona"` 參數，僅將代碼加進 `task` 物件。
- [ ] 重跑該測試與既有 API Client 認證測試，確認通過。

### 2. Streamlit 選擇、送件與結果狀態

**修改：** `app_api.py:225-265`、`:529-580`、`:1110-1145`、`:1184-1235`、`:1306-1345`；新增 `tests/test_streamlit_plant_selection.py`。

- [ ] 先新增測試：四個中文選項精確對應四個 API 代碼，預設釋迦；切換代碼與原圖指紋相同時，仍會清除上次任務與結果；`run_diagnosis` 將所選代碼送入 Client。
- [ ] 執行新測試，確認因缺少植物選項與參數而失敗。
- [ ] 加入集中定義的植物選項與每種植物的類別顯示清單；選擇元件放在上傳區之前，以 `st.session_state` 記錄選擇。
- [ ] 將植物代碼納入現有圖片變更判定，切換時清除舊任務／結果；在上傳區提示圖片須屬於所選植物。
- [ ] 擴充 `run_diagnosis()` 參數並轉送給 Client；側欄清單依所選植物切換。結果名稱仍優先使用 API item 的 `label`。
- [ ] 重跑新測試、`tests/test_api_progress.py` 與全套 `pytest -q`，確認既有表單、進度和結果契約未回歸。

### 3. 手動驗收

- [ ] 啟動本機 FastAPI、圖片伺服器與 Streamlit；預設釋迦可送件，茶葉選項送件時在「完整 API 回應」看到茶葉結果。
- [ ] 用同一組已上傳圖片切換植物，確認舊任務編號、診斷、錯誤提示不再顯示；側欄類別即時改變。
- [ ] 依序檢查四個選項都能送出，並在開發測試中核對 POST 的 `task.plant`。這項檢查不宣稱四個模型的診斷準確度已驗證；各植物圖片端到端測試留到下一階段。
- [ ] 核對無 Token、API 不可用、任務 `FAILED` 時的既有提示仍可理解。

## 審閱重點與待確認

1. 植物選單預設釋迦是否符合目前使用者習慣。
2. 切換植物時保留已上傳圖片、但清除診斷結果的行為是否合適。
3. 側欄是否顯示該植物的所有模型類別，或僅顯示中文名稱。
4. 本階段維持 Streamlit 一次只送一種植物、同一植株的 1～5 張圖片。
