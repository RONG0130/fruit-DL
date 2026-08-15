"""Streamlit API Client 的 Bearer Token 測試。"""

import httpx

from streamlit_api_client import (
    PlantDiagnosisAPIClient,
)


def test_client_sends_bearer_token_to_all_endpoints(
    monkeypatch,
):
    """Streamlit 呼叫三支 API 時都必須送出相同的 Token。"""

    captured_headers: list[dict] = []

    def fake_get(url: str, **kwargs):
        captured_headers.append(
            kwargs.get("headers", {})
        )

        if url.endswith("/v1/health"):
            return httpx.Response(
                200,
                json={"status": "ok"},
            )

        return httpx.Response(
            200,
            json={
                "task_id": "academy-test-task",
                "status": "PENDING",
            },
        )

    def fake_post(url: str, **kwargs):
        captured_headers.append(
            kwargs.get("headers", {})
        )

        return httpx.Response(
            200,
            json={
                "task_id": "academy-test-task",
                "status": "PENDING",
            },
        )

    monkeypatch.setattr(
        httpx,
        "get",
        fake_get,
    )
    monkeypatch.setattr(
        httpx,
        "post",
        fake_post,
    )

    client = PlantDiagnosisAPIClient(
        base_url="http://127.0.0.1:8000",
        api_token="streamlit-test-token",
    )

    assert client.health_check() is True

    client.create_task(
        platform_task_id="platform-test-task",
        images=[],
    )
    client.get_task("academy-test-task")

    assert captured_headers == [
        {
            "Authorization": (
                "Bearer streamlit-test-token"
            )
        },
        {
            "Authorization": (
                "Bearer streamlit-test-token"
            )
        },
        {
            "Authorization": (
                "Bearer streamlit-test-token"
            )
        },
    ]
