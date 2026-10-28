# 使用符合規範的 NVIDIA CUDA 12.8 Runtime
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Python 與 uv 執行環境設定
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON=python3 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

# 安裝 Ubuntu 22.04 內建的 Python 3.10
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 從官方 Image 複製固定版本的 uv
COPY --from=ghcr.io/astral-sh/uv:0.12.3 /uv /uvx /bin/

WORKDIR /app

# 先複製依賴設定，讓 Docker 可以使用快取
COPY pyproject.toml uv.lock README.md ./

# 完全按照 uv.lock 安裝正式環境依賴
# --no-dev：不安裝 pytest 等開發套件
# --no-install-project：此專案不是 Python package，不需要安裝專案本身
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# 只複製 API 推論真正需要的程式
COPY api/ ./api/
COPY predict.py ./

# 四組模型須保留 api/model_registry.py 登錄的相對路徑。
# 只複製推論需要的權重及類別檔，不將驗證報告、圖表放入映像。
COPY runs/annona/20261026-173435_batch8-lr2e-5/best_model.pth runs/annona/20261026-173435_batch8-lr2e-5/classes.json ./runs/annona/20261026-173435_batch8-lr2e-5/
COPY runs/tea/20261026-191542_batch8-lr2e-5/best_model.pth runs/tea/20261026-191542_batch8-lr2e-5/classes.json ./runs/tea/20261026-191542_batch8-lr2e-5/
COPY runs/legume/20261026-203804_batch8-lr2e-5/best_model.pth runs/legume/20261026-203804_batch8-lr2e-5/classes.json ./runs/legume/20261026-203804_batch8-lr2e-5/
COPY runs/facility_leafy_greens/20261027-170444_batch8-lr2e-5/best_model.pth runs/facility_leafy_greens/20261027-170444_batch8-lr2e-5/classes.json ./runs/facility_leafy_greens/20261027-170444_batch8-lr2e-5/

# 建立 SQLite 任務資料庫目錄，並在建置時核對四組檔案及類別順序。
RUN mkdir -p /app/runtime \
    && /app/.venv/bin/python -c "from api.model_registry import MODEL_SPECS, check_artifacts; [check_artifacts(plant) for plant in MODEL_SPECS]"

# 規格要求容器內固定監聽 8000
EXPOSE 8000

# 因目前 /v1/health 也需要 Bearer Token，
# 健康檢查會自動讀取執行容器時傳入的 ACADEMY_API_TOKEN
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import os, urllib.request; request = urllib.request.Request('http://127.0.0.1:8000/v1/health', headers={'Authorization': 'Bearer ' + os.environ['ACADEMY_API_TOKEN']}); urllib.request.urlopen(request, timeout=5).read()"]

# 啟動 FastAPI；只有一個 Worker，由模型管理器依植物切換 GPU 模型。
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
