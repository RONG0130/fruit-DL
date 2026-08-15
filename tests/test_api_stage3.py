"""第三階段 API 基礎功能測試。"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api import image_downloader, store
from api import main
from api.main import app
from api.schemas import CreateTaskRequest


TEST_API_TOKEN = "test-only-api-token"

AUTH_HEADERS = {
    "Authorization": (
        f"Bearer {TEST_API_TOKEN}"
    ),
}


def valid_request_payload() -> dict:
    """建立尚未過期且包含額外欄位的合法測試資料。"""

    return {
        "platform_task_id": "platform_test_001",
        "images": [
            {
                "image_url": "https://example.com/leaf.jpg",
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(minutes=30)
                ).isoformat(),
                "mime_type": "image/jpeg",
                "unknown_image_field": "會被忽略",
            }
        ],
        "task": {
            "academy_type": "plant_diagnosis",
            "unknown_task_field": "會被忽略",
        },
        "context": {
            "user_description": "葉片出現斑點",
            "location": "屏東縣",
            "environment": "露天",
        },
        "unknown_root_field": "會被忽略",
    }


def test_health_and_database_initialization(tmp_path, monkeypatch):
    """啟動 FastAPI 時應建立資料庫，健康檢查應回傳 ok。"""

    test_db_path = tmp_path / "tasks.db"
    monkeypatch.setattr(store, "DB_PATH", test_db_path)
    monkeypatch.setenv(
        "ACADEMY_API_TOKEN",
        TEST_API_TOKEN,
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/health",
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert test_db_path.exists()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/health"),
        ("POST", "/v1/tasks"),
        ("GET", "/v1/tasks/test-task-id"),
    ],
)
def test_protected_endpoints_return_401(
    method: str,
    path: str,
    monkeypatch,
):
    """三支 API 缺少或使用錯誤 Token 時都必須回傳 401。"""

    monkeypatch.setenv(
        "ACADEMY_API_TOKEN",
        TEST_API_TOKEN,
    )

    client = TestClient(app)
    request_arguments = {}

    if method == "POST":
        request_arguments["json"] = (
            valid_request_payload()
        )

    missing_response = client.request(
        method,
        path,
        **request_arguments,
    )
    invalid_response = client.request(
        method,
        path,
        headers={
            "Authorization": "Bearer wrong-token",
        },
        **request_arguments,
    )

    expected_error = {
        "code": "UNAUTHORIZED",
        "message": (
            "認證失敗，請提供有效的 Bearer Token"
        ),
    }

    for response in [
        missing_response,
        invalid_response,
    ]:
        assert response.status_code == 401
        assert response.json() == expected_error
        assert response.headers[
            "www-authenticate"
        ] == "Bearer"


def test_model_load_failure_returns_503(
    tmp_path,
    monkeypatch,
):
    """模型載入失敗時，健康檢查及建立任務都必須回傳 503。"""

    test_db_path = tmp_path / "tasks.db"
    monkeypatch.setenv(
        "ACADEMY_API_TOKEN",
        TEST_API_TOKEN,
    )

    def broken_load_predictor():
        """模擬模型檔損壞或無法載入。"""

        raise RuntimeError("測試用模型載入失敗")

    async def must_not_create_task(body):
        """模型不可用時不得建立資料庫任務。"""

        raise AssertionError(
            "模型不可用時不應建立任務"
        )

    monkeypatch.setattr(
        store,
        "DB_PATH",
        test_db_path,
    )
    monkeypatch.setattr(
        main,
        "load_predictor",
        broken_load_predictor,
    )
    monkeypatch.setattr(
        main.store,
        "create_or_get_task",
        must_not_create_task,
    )

    # lifespan 應捕捉模型載入錯誤，
    # 因此 TestClient 仍然可以正常啟動。
    with TestClient(app) as client:
        health_response = client.get(
            "/v1/health",
            headers=AUTH_HEADERS,
        )
        create_response = client.post(
            "/v1/tasks",
            json=valid_request_payload(),
            headers=AUTH_HEADERS,
        )

    expected_error = {
        "code": "MODEL_UNAVAILABLE",
        "message": "模型服務目前無法使用，請稍後再試",
    }

    assert health_response.status_code == 503
    assert health_response.json() == expected_error
    assert create_response.status_code == 503
    assert create_response.json() == expected_error


def test_finished_worker_is_model_unavailable(
    monkeypatch,
):
    """模型雖已載入，但 Worker 結束時仍視為不可用。"""

    class FinishedWorkerTask:
        """模擬已經異常結束的背景 Worker。"""

        def done(self) -> bool:
            return True

    monkeypatch.setattr(
        main.app.state,
        "predictor",
        object(),
        raising=False,
    )
    monkeypatch.setattr(
        main.app.state,
        "worker_task",
        FinishedWorkerTask(),
        raising=False,
    )

    with pytest.raises(main.APIError) as captured:
        main.ensure_model_service_ready()

    assert captured.value.status_code == 503
    assert captured.value.code == (
        "MODEL_UNAVAILABLE"
    )
    assert captured.value.message == (
        "模型服務目前無法使用，請稍後再試"
    )


def test_schema_accepts_valid_request_and_ignores_unknown_fields():
    """合法資料應通過，未知欄位不應造成錯誤。"""

    request = CreateTaskRequest.model_validate(valid_request_payload())
    serialized = request.model_dump(mode="json")

    assert request.platform_task_id == "platform_test_001"
    assert "unknown_root_field" not in serialized
    assert "unknown_image_field" not in serialized["images"][0]
    assert "unknown_task_field" not in serialized["task"]


def test_schema_accepts_expired_url_for_async_failure():
    """URL 過期屬背景失敗，POST Request 本身仍可建立任務。"""

    payload = valid_request_payload()
    payload["images"][0]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()

    request = CreateTaskRequest.model_validate(
        payload
    )

    assert request.images[0].expires_at < (
        datetime.now(timezone.utc)
    )


def test_expired_url_becomes_image_download_failed(
    tmp_path,
):
    """背景下載發現 URL 過期時應回正式非同步錯誤碼。"""

    async def scenario():
        with pytest.raises(
            image_downloader.ImageDownloadError
        ) as captured:
            await image_downloader.download_images(
                images=[
                    {
                        "image_url": (
                            "https://example.com/expired.jpg"
                        ),
                        "expires_at": (
                            datetime.now(timezone.utc)
                            - timedelta(minutes=1)
                        ).isoformat(),
                        "mime_type": "image/jpeg",
                    }
                ],
                target_directory=tmp_path,
            )

        assert captured.value.code == (
            "IMAGE_DOWNLOAD_FAILED"
        )

    asyncio.run(scenario())


def test_store_idempotency(tmp_path, monkeypatch):
    """相同 platform_task_id 必須取得同一個 task_id。"""

    test_db_path = tmp_path / "tasks.db"
    monkeypatch.setattr(store, "DB_PATH", test_db_path)
    request = CreateTaskRequest.model_validate(valid_request_payload())

    async def scenario():
        await store.init_db()

        first_task, first_is_new = await store.create_or_get_task(request)
        second_task, second_is_new = await store.create_or_get_task(request)

        assert first_is_new is True
        assert second_is_new is False
        assert first_task["task_id"] == second_task["task_id"]
        assert first_task["status"] == "PENDING"

    asyncio.run(scenario())


def test_duplicate_post_always_returns_pending(monkeypatch):
    """重複 POST 應回原 task_id，但回應狀態仍固定為 PENDING。"""

    request = CreateTaskRequest.model_validate(
        valid_request_payload()
    )

    async def fake_create_or_get_task(body):
        return (
            {
                "task_id": "acad_existing_task",
                "status": "SUCCEEDED",
            },
            False,
        )

    # 此測試只驗證重複 POST 的冪等行為，
    # 不啟動真實 lifespan，因此略過模型就緒檢查。
    monkeypatch.setattr(
        main,
        "ensure_model_service_ready",
        lambda: None,
    )

    monkeypatch.setattr(
        main.store,
        "create_or_get_task",
        fake_create_or_get_task,
    )

    response = asyncio.run(
        main.create_task(request)
    )

    assert response == {
        "task_id": "acad_existing_task",
        "status": "PENDING",
    }

def test_unhandled_exception_returns_standard_500(
    monkeypatch,
):
    """未捕捉的內部例外必須轉成標準 HTTP 500 格式。"""

    async def broken_create_or_get_task(body):
        """模擬建立任務時發生未預期錯誤。"""

        raise RuntimeError(
            "測試用內部敏感錯誤"
        )

    monkeypatch.setenv(
        "ACADEMY_API_TOKEN",
        TEST_API_TOKEN,
    )

    # 這個測試不啟動 lifespan，
    # 因此略過模型就緒檢查。
    monkeypatch.setattr(
        main,
        "ensure_model_service_ready",
        lambda: None,
    )
    monkeypatch.setattr(
        main.store,
        "create_or_get_task",
        broken_create_or_get_task,
    )

    # 預設 TestClient 會把伺服器例外重新拋給 pytest。
    # 設為 False 後，才能檢查實際 HTTP 500 Response。
    client = TestClient(
        app,
        raise_server_exceptions=False,
    )

    response = client.post(
        "/v1/tasks",
        json=valid_request_payload(),
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 500
    assert response.json() == {
        "code": "INTERNAL_ERROR",
        "message": "伺服器內部錯誤",
    }

    # 對外不得使用 FastAPI 預設 detail 格式。
    assert "detail" not in response.json()

    # 不得把實際例外內容或 stack trace 洩漏出去。
    assert "測試用內部敏感錯誤" not in (
        response.text
    )
    assert "Traceback" not in response.text