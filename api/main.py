"""
植物病蟲害辨識 FastAPI 入口。
"""
import asyncio
import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

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
    """統一 API 對外使用的業務錯誤。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers

        super().__init__(message)


# auto_error=False 很重要。
# 如果使用預設值，缺少 Token 時 FastAPI 可能自行回傳其他格式，
# 無法符合 UNAUTHORIZED 的統一錯誤格式。
bearer_scheme = HTTPBearer(
    auto_error=False,
)


def verify_api_token(
    credentials: (
        HTTPAuthorizationCredentials | None
    ) = Depends(bearer_scheme),
) -> None:
    """
    驗證一站式平台送來的 Bearer Token。

    正確格式：
    Authorization: Bearer <token>
    """

    expected_token = os.getenv(
        "ACADEMY_API_TOKEN",
        "",
    ).strip()

    has_valid_scheme = (
        credentials is not None
        and credentials.scheme.lower() == "bearer"
    )

    received_token = (
        credentials.credentials
        if credentials is not None
        else ""
    )

    is_authenticated = (
        bool(expected_token)
        and has_valid_scheme
        and secrets.compare_digest(
            received_token,
            expected_token,
        )
    )

    if not is_authenticated:
        raise APIError(
            status_code=401,
            code="UNAUTHORIZED",
            message="認證失敗，請提供有效的 Bearer Token",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )


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
    """初始化資料庫、模型、佇列及背景 Worker。"""

    await store.init_db()

    # 先建立所有狀態欄位。
    # 即使模型載入失敗，端點也能安全檢查這些欄位。
    app.state.predictor = None
    app.state.task_queue = asyncio.Queue()
    app.state.worker_task = None

    # 資料清理工作與模型無關，因此仍然可以啟動。
    app.state.cleanup_task = asyncio.create_task(
        cleanup_loop()
    )

    LOGGER.info("正在載入 DINOv3 模型")

    try:
        # 模型約 1.2 GB，放到其他執行緒載入，
        # 避免阻塞 FastAPI 的事件迴圈。
        app.state.predictor = await asyncio.to_thread(
            load_predictor
        )

    except Exception:
        # 這裡不能再次拋出例外。
        # 必須讓 FastAPI 繼續啟動，端點才能回傳 503。
        LOGGER.exception(
            "模型載入失敗，API 將以不可用狀態啟動"
        )

    else:
        # 只有模型成功載入後，才恢復未完成任務並啟動 Worker。
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

        LOGGER.info(
            "API 啟動完成，恢復未完成任務數量：%s",
            len(unfinished_tasks),
        )

    try:
        yield

    finally:
        background_tasks = [
            (
                "worker_task",
                app.state.worker_task,
            ),
            (
                "cleanup_task",
                app.state.cleanup_task,
            ),
        ]

        # 先通知仍在運作的背景工作停止。
        for _, background_task in background_tasks:
            if (
                background_task is not None
                and not background_task.done()
            ):
                background_task.cancel()

        # 等待背景工作結束，避免關閉 API 時留下工作。
        for task_name, background_task in background_tasks:
            if background_task is None:
                continue

            try:
                await background_task
            except asyncio.CancelledError:
                pass
            except Exception:
                LOGGER.exception(
                    "背景工作 %s 異常結束",
                    task_name,
                )


app = FastAPI(
    title="植物病蟲害辨識 API",
    version="1.0.0",
    lifespan=lifespan,

    # 套用至三支 API，避免遺漏其中一支。
    dependencies=[
        Depends(verify_api_token),
    ],
)


def ensure_model_service_ready() -> None:
    """
    確認模型與背景 Worker 都可以接受新任務。

    模型沒有載入，或 Worker 已經停止時，
    必須同步回傳 HTTP 503。
    """

    predictor = getattr(
        app.state,
        "predictor",
        None,
    )
    worker_task = getattr(
        app.state,
        "worker_task",
        None,
    )

    worker_is_running = (
        worker_task is not None
        and not worker_task.done()
    )

    if predictor is None or not worker_is_running:
        raise APIError(
            status_code=503,
            code="MODEL_UNAVAILABLE",
            message="模型服務目前無法使用，請稍後再試",
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
        headers=error.headers,
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

@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    error: Exception,
):
    """
    攔截請求處理期間未被捕捉的內部錯誤。

    對外只回傳固定錯誤格式，
    完整例外僅記錄於伺服器 Log。
    """

    LOGGER.error(
        "API 發生未預期錯誤：%s %s",
        request.method,
        request.url.path,

        # 保留完整 traceback 於伺服器 Log，
        # 但不可放進 Response 回傳給呼叫端。
        exc_info=(
            type(error),
            error,
            error.__traceback__,
        ),
    )

    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "伺服器內部錯誤",
        },
    )


@app.get(
    "/v1/health",
    response_model=HealthResponse,
)
async def health():
    """確認模型與背景 Worker 是否可以正常服務。"""

    ensure_model_service_ready()

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

    # 必須在寫入資料庫前檢查。
    # 模型不可用時不能建立一個永遠無法處理的 PENDING 任務。
    ensure_model_service_ready()

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
