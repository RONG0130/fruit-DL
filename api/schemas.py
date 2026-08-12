"""
API Request 與 Response 的資料格式。

本模組只負責：
1. 驗證平台送入的 JSON
2. 忽略未定義的額外欄位
3. 限制圖片數量及格式
4. 定義建立任務的回應格式
"""

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
)


class APIModel(BaseModel):
    """
    所有 API Schema 的共同基底。

    extra="ignore"：
    平台可能傳入本服務不認識的額外欄位，
    依會議規格必須忽略，而不能因此拒絕請求。
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )


class ImageInput(APIModel):
    """單張輸入圖片的資訊。"""

    image_url: HttpUrl
    expires_at: datetime
    mime_type: str

    @field_validator("expires_at")
    @classmethod
    def check_expires_at(cls, value: datetime) -> datetime:
        """確認 Signed URL 到期時間包含可比較的時區。"""

        if value.tzinfo is None:
            raise ValueError("expires_at 必須包含時區")

        # 是否過期屬於背景圖片下載流程。
        # POST 仍須先建立任務，之後由 GET 回傳
        # FAILED + IMAGE_DOWNLOAD_FAILED。
        return value

    @field_validator("mime_type")
    @classmethod
    def check_mime_type(cls, value: str) -> str:
        """目前僅接受網頁與模型支援的 JPEG、PNG。"""

        normalized = value.lower().strip()

        allowed_types = {
            "image/jpeg",
            "image/png",
        }

        if normalized not in allowed_types:
            raise ValueError("mime_type 僅支援 image/jpeg 或 image/png")

        return normalized


class TaskInfo(APIModel):
    """平台指定的任務種類。"""

    # 不使用 Literal，因為錯誤 academy_type
    # 必須回 UNSUPPORTED_ACADEMY_TYPE，而不是一般驗證錯誤。
    academy_type: str


class TaskContext(APIModel):
    """平台提供的選填案件資訊。"""

    user_description: str | None = None
    location: str | None = None
    environment: str | None = None


class CreateTaskRequest(APIModel):
    """POST /v1/tasks 的 Request Body。"""

    platform_task_id: str = Field(
        min_length=1,
        max_length=128,
    )

    images: list[ImageInput] = Field(
        min_length=1,
        max_length=5,
    )

    task: TaskInfo
    context: TaskContext | None = None


class CreateTaskResponse(APIModel):
    """POST /v1/tasks 的 Response。"""

    task_id: str
    # 建立任務端點只負責接受任務並回傳識別碼，
    # 實際狀態必須由 GET /v1/tasks/{task_id} 查詢。
    status: Literal["PENDING"]


class HealthResponse(APIModel):
    """GET /v1/health 的 Response。"""

    status: Literal["ok"]
