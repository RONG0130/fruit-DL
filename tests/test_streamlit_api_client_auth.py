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


def test_create_task_sends_selected_plant_and_defaults_to_annona(monkeypatch):
    """選定作物必須進入 POST task；舊呼叫仍使用釋迦。"""
    payloads = []

    def fake_post(url: str, **kwargs):
        payloads.append(kwargs["json"])
        return httpx.Response(200, json={"task_id": "one", "status": "PENDING"})

    monkeypatch.setattr(httpx, "post", fake_post)
    client = PlantDiagnosisAPIClient("http://127.0.0.1:8000", "token")
    client.create_task("first", [{"image_url": "https://example.com/one.jpg"}], plant="tea")
    client.create_task("second", [], context={"location": "屏東"})

    assert payloads[0]["task"] == {"academy_type": "plant_diagnosis", "plant": "tea"}
    assert payloads[0]["images"] == [{"image_url": "https://example.com/one.jpg"}]
    assert payloads[1]["task"] == {"academy_type": "plant_diagnosis", "plant": "annona"}
    assert payloads[1]["context"] == {"location": "屏東"}
