"""第三階段 API 基礎功能測試。"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from api import image_downloader, store
from api import main
from api.main import app
from api.schemas import CreateTaskRequest


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

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert test_db_path.exists()


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
