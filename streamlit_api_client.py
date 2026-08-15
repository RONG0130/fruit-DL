"""
Streamlit 呼叫植物病蟲害 FastAPI 的 Client。

此檔案只負責 HTTP 通訊，不處理 Streamlit 畫面。
"""

from typing import Any

import httpx


class APIClientError(Exception):
    """呼叫 FastAPI 失敗時使用的錯誤。"""

    def __init__(
        self,
        code: str,
        message: str,
    ):
        self.code = code
        self.message = message
        super().__init__(message)


class PlantDiagnosisAPIClient:
    """封裝三支植物病蟲害 API。"""

    def __init__(
        self,
        base_url: str,
        api_token: str,
        timeout_seconds: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(
            timeout_seconds
        )
        self.headers = (
            {
                "Authorization": (
                    f"Bearer {api_token}"
                )
            }
            if api_token
            else {}
        )

    def _parse_response(
        self,
        response: httpx.Response,
    ) -> dict[str, Any]:
        """解析成功回應，或轉換 API 錯誤。"""

        if response.is_success:
            return response.json()

        try:
            error_data = response.json()
        except ValueError:
            error_data = {}

        raise APIClientError(
            code=error_data.get(
                "code",
                f"HTTP_{response.status_code}",
            ),
            message=error_data.get(
                "message",
                "FastAPI 回傳無法解析的錯誤",
            ),
        )

    def health_check(self) -> bool:
        """呼叫 GET /v1/health。"""

        try:
            response = httpx.get(
                f"{self.base_url}/v1/health",
                headers=self.headers,
                timeout=self.timeout,
            )

            data = self._parse_response(response)
            return data.get("status") == "ok"

        except (
            httpx.RequestError,
            APIClientError,
        ):
            return False

    def create_task(
        self,
        platform_task_id: str,
        images: list[dict],
        context: dict | None = None,
    ) -> dict:
        """呼叫 POST /v1/tasks。"""

        payload = {
            "platform_task_id": platform_task_id,
            "images": images,
            "task": {
                "academy_type": "plant_diagnosis",
            },
            "context": context,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/v1/tasks",
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )

        except httpx.RequestError as error:
            raise APIClientError(
                code="API_CONNECTION_FAILED",
                message="無法連線到 FastAPI",
            ) from error

        return self._parse_response(response)

    def get_task(self, task_id: str) -> dict:
        """呼叫 GET /v1/tasks/{task_id}。"""

        try:
            response = httpx.get(
                f"{self.base_url}/v1/tasks/{task_id}",
                headers=self.headers,
                timeout=self.timeout,
            )

        except httpx.RequestError as error:
            raise APIClientError(
                code="API_CONNECTION_FAILED",
                message="查詢任務時無法連線到 FastAPI",
            ) from error

        return self._parse_response(response)
