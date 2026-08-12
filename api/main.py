"""
植物病蟲害辨識 FastAPI 入口。
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from predict import PlantDiseasePredictor

from api import store
from api.schemas import (
    CreateTaskRequest,
    CreateTaskResponse,
    HealthResponse,
)
from api.worker import cleanup_loop, worker_loop


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


class APIError(Exception):
    """專案 API 使用的自訂錯誤。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message


def load_predictor() -> PlantDiseasePredictor:
    """從專案 output 資料夾載入模型。"""

    return PlantDiseasePredictor(
        model_path=str(
            PROJECT_ROOT
            / "output"
            / "best_model.pth"
        ),
        classes_path=str(
            PROJECT_ROOT
            / "output"
            / "classes.json"
        ),
        verbose=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """初始化資料庫、模型、佇列及 Worker。"""

    await store.init_db()

    LOGGER.info("正在載入 DINOv3 模型")

    # 模型約 1.2 GB，載入時可能需要一段時間。
    app.state.predictor = await asyncio.to_thread(
        load_predictor
    )

    app.state.task_queue = asyncio.Queue()

    unfinished_tasks = (
        await store.list_unfinished_tasks()
    )

    for task in unfinished_tasks:
        app.state.task_queue.put_nowait(
            task["task_id"]
        )

    app.state.worker_task = asyncio.create_task(
        worker_loop(
            app.state.task_queue,
            app.state.predictor,
        )
    )

    app.state.cleanup_task = asyncio.create_task(
        cleanup_loop()
    )

    LOGGER.info(
        "API 啟動完成，恢復未完成任務數量：%s",
        len(unfinished_tasks),
    )

    try:
        yield

    finally:
        app.state.worker_task.cancel()
        app.state.cleanup_task.cancel()

        with suppress(asyncio.CancelledError):
            await app.state.worker_task

        with suppress(asyncio.CancelledError):
            await app.state.cleanup_task


app = FastAPI(
    title="植物病蟲害辨識 API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(APIError)
async def api_error_handler(
    request: Request,
    error: APIError,
):
    """回傳統一的業務錯誤格式。"""

    return JSONResponse(
        status_code=error.status_code,
        content={
            "code": error.code,
            "message": error.message,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    error: RequestValidationError,
):
    """將 Pydantic 驗證錯誤轉成 HTTP 400。"""

    LOGGER.warning(
        "API 請求格式驗證失敗：%s",
        error.errors(),
    )

    return JSONResponse(
        status_code=400,
        content={
            "code": "VALIDATION_ERROR",
            "message": "請求格式錯誤或必填欄位缺漏",
        },
    )


@app.get(
    "/v1/health",
    response_model=HealthResponse,
)
async def health():
    """確認 FastAPI 是否成功啟動"""

    return {"status": "ok"}

# /v1/tasks: 建立辨識任務。
@app.post(
    "/v1/tasks",
    response_model=CreateTaskResponse,
    status_code=200,
)
async def create_task(
    body: CreateTaskRequest,
):
    """建立非同步辨識任務。"""

    if body.task.academy_type != "plant_diagnosis":
        raise APIError(
            status_code=400,
            code="UNSUPPORTED_ACADEMY_TYPE",
            message="本服務僅支援 plant_diagnosis",
        )

    task, is_new = await store.create_or_get_task(
        body
    )

    if is_new:
        app.state.task_queue.put_nowait(
            task["task_id"]
        )

    return {
        "task_id": task["task_id"],
        # POST 的回應契約固定為 PENDING。
        # 重複請求即使對應任務已完成，也只回傳原 task_id；
        # 平台再透過 GET 取得任務的實際狀態與結果。
        "status": "PENDING",
    }

# /v1/tasks/{task_id}: 查詢任務
@app.get("/v1/tasks/{task_id}")
async def get_task(task_id: str):
    """查詢任務狀態及結果。"""

    task = await store.get_task(task_id)

    if task is None:
        raise APIError(
            status_code=404,
            code="TASK_NOT_FOUND",
            message="找不到任務，或結果已超過保存期限",
        )

    status = task["status"]

    response = {
        "task_id": task["task_id"],
        "status": status,
    }

    # 正式規格只在推論處理期間附帶 progress。
    # PENDING、SUCCEEDED 與 FAILED 都不回傳此欄位。
    if (
        status == "PROCESSING"
        and task["progress"] is not None
    ):
        response["progress"] = task["progress"]

    if status in {"PENDING", "PROCESSING"}:
        return response

    if status == "SUCCEEDED":
        response["result"] = task["result"]
        return response

    if status == "FAILED":
        response["error"] = task["error"]
        return response

    raise APIError(
        status_code=500,
        code="INTERNAL_ERROR",
        message="任務狀態異常",
    )
