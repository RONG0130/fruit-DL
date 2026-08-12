"""會議規格中的輪詢結果與真實工作進度測試。"""

import asyncio
import ast
from pathlib import Path

from api import main, worker


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_polling_interval_matches_meeting_spec():
    """第一次輪詢至少等 1 秒，後續間隔必須介於 1.5～3 秒。"""

    source = (
        PROJECT_ROOT / "app_api.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    constants = {}
    sleep_argument_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
            ):
                constants[node.targets[0].id] = node.value.value

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "time"
            and node.func.attr == "sleep"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            sleep_argument_names.append(node.args[0].id)

    assert constants["INITIAL_POLL_DELAY_SECONDS"] >= 1.0
    assert 1.5 <= constants["POLL_INTERVAL_SECONDS"] <= 3.0
    assert constants["TASK_TIMEOUT_SECONDS"] == 30 * 60
    assert constants["SIGNED_URL_VALIDITY_MINUTES"] >= 30
    assert "INITIAL_POLL_DELAY_SECONDS" in sleep_argument_names
    assert "POLL_INTERVAL_SECONDS" in sleep_argument_names


def test_streamlit_defines_need_more_info_before_display():
    """補拍提示必須在結果顯示函式內先取得，避免 NameError。"""

    source = (
        PROJECT_ROOT / "app_api.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    display_function = next(
        node
        for node in tree.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name == "display_api_result"
        )
    )

    assigned_lines = [
        node.lineno
        for node in ast.walk(display_function)
        if (
            isinstance(node, ast.Name)
            and node.id == "need_more_info"
            and isinstance(node.ctx, ast.Store)
        )
    ]
    loaded_lines = [
        node.lineno
        for node in ast.walk(display_function)
        if (
            isinstance(node, ast.Name)
            and node.id == "need_more_info"
            and isinstance(node.ctx, ast.Load)
        )
    ]

    assert assigned_lines
    assert loaded_lines
    assert min(assigned_lines) < min(loaded_lines)


def test_case_inputs_submit_through_one_streamlit_form():
    """案件欄位應使用表單，避免每次輸入都重新執行頁面。"""

    source = (
        PROJECT_ROOT / "app_api.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    diagnosis_form = next(
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.With)
            and isinstance(
                node.items[0].context_expr,
                ast.Call,
            )
            and isinstance(
                node.items[0].context_expr.func,
                ast.Attribute,
            )
            and node.items[0].context_expr.func.attr
            == "form"
        )
    )

    form_calls = {
        node.func.attr
        for node in ast.walk(diagnosis_form)
        if (
            isinstance(node, ast.Call)
            and isinstance(
                node.func,
                ast.Attribute,
            )
        )
    }

    assert {
        "text_area",
        "text_input",
        "form_submit_button",
    } <= form_calls
    assert "button" not in form_calls


def test_streamlit_displays_complete_get_response():
    """開發測試區應顯示完整 GET 信封，而非只有 result 內層。"""

    source = (
        PROJECT_ROOT / "app_api.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    display_function = next(
        node
        for node in tree.body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "display_api_result"
        )
    )

    json_calls = [
        node
        for node in ast.walk(display_function)
        if (
            isinstance(node, ast.Call)
            and isinstance(
                node.func,
                ast.Attribute,
            )
            and node.func.attr == "json"
        )
    ]

    assert len(json_calls) == 1
    assert isinstance(
        json_calls[0].args[0],
        ast.Name,
    )
    assert json_calls[0].args[0].id == (
        "task_response"
    )


def test_succeeded_response_omits_progress(monkeypatch):
    """成功任務應回傳完整結果，但不附帶 PROCESSING 專用進度。"""

    async def fake_get_task(task_id: str):
        return {
            "task_id": task_id,
            "status": "SUCCEEDED",
            "progress": 100,
            "result": {
                "schema_version": "academy_model_result.v1",
            },
            "error": None,
        }

    monkeypatch.setattr(
        main.store,
        "get_task",
        fake_get_task,
    )

    response = asyncio.run(
        main.get_task("acad_progress_test")
    )

    assert response["status"] == "SUCCEEDED"
    assert "progress" not in response
    assert response["result"]["schema_version"] == (
        "academy_model_result.v1"
    )


def test_pending_omits_progress(monkeypatch):
    """PENDING 即使資料庫保存 0，也不應回傳 progress。"""

    async def fake_get_task(task_id: str):
        return {
            "task_id": task_id,
            "status": "PENDING",
            "progress": 0,
            "result": None,
            "error": None,
        }

    monkeypatch.setattr(
        main.store,
        "get_task",
        fake_get_task,
    )

    response = asyncio.run(
        main.get_task("acad_pending_test")
    )

    assert response == {
        "task_id": "acad_pending_test",
        "status": "PENDING",
    }


def test_processing_contains_progress(monkeypatch):
    """只有 PROCESSING 應附帶 0～100 的實際進度。"""

    async def fake_get_task(task_id: str):
        return {
            "task_id": task_id,
            "status": "PROCESSING",
            "progress": 60,
            "result": None,
            "error": None,
        }

    monkeypatch.setattr(
        main.store,
        "get_task",
        fake_get_task,
    )

    response = asyncio.run(
        main.get_task("acad_processing_test")
    )

    assert response == {
        "task_id": "acad_processing_test",
        "status": "PROCESSING",
        "progress": 60,
    }


def test_worker_updates_progress_by_real_work_stage(monkeypatch):
    """兩張圖片應依下載、逐張推論、合併及保存階段更新進度。"""

    recorded_progress: list[int] = []
    saved_result: dict = {}

    class FakePredictor:
        """不載入真實模型的輕量測試預測器。"""

        num_classes = 3

        def predict(self, image: Path, top_k: int):
            return [
                ("healthy", 70.0),
                ("anthracnose", 20.0),
                ("rust", 10.0),
            ][:top_k]

        def get_model_info(self):
            return {"accuracy": 94.61}

    async def fake_get_task(task_id: str):
        return {
            "task_id": task_id,
            "status": "PENDING",
            "request": {
                "images": [
                    {"image_url": "https://example.com/1.jpg"},
                    {"image_url": "https://example.com/2.jpg"},
                ]
            },
        }

    async def fake_download_images(
        images: list[dict],
        target_directory: Path,
        progress_callback=None,
    ):
        paths = []

        for index, _ in enumerate(images, start=1):
            paths.append(
                target_directory / f"image_{index}.jpg"
            )

            if progress_callback is not None:
                await progress_callback(index, len(images))

        return paths

    async def fake_mark_processing(
        task_id: str,
        progress: int,
    ):
        recorded_progress.append(progress)

    async def fake_mark_succeeded(
        task_id: str,
        result: dict,
    ):
        recorded_progress.append(100)
        saved_result.update(result)

    async def fake_mark_failed(task_id: str, error: dict):
        raise AssertionError(f"任務不應失敗：{error}")

    monkeypatch.setattr(
        worker.store,
        "get_task",
        fake_get_task,
    )
    monkeypatch.setattr(
        worker,
        "download_images",
        fake_download_images,
    )
    monkeypatch.setattr(
        worker.store,
        "mark_processing",
        fake_mark_processing,
    )
    monkeypatch.setattr(
        worker.store,
        "mark_succeeded",
        fake_mark_succeeded,
    )
    monkeypatch.setattr(
        worker.store,
        "mark_failed",
        fake_mark_failed,
    )

    asyncio.run(
        worker.process_task(
            "acad_progress_test",
            FakePredictor(),
        )
    )

    assert recorded_progress == [
        5,
        20,
        35,
        40,
        62,
        85,
        90,
        95,
        100,
    ]
    assert saved_result["raw_output"] == {
        "model_label": "healthy",
        "model_score": 0.7,
    }


def test_worker_maps_image_error_to_contract_code(monkeypatch):
    """圖片內部錯誤對外必須統一為 IMAGE_DOWNLOAD_FAILED。"""

    saved_error: dict = {}

    async def fake_get_task(task_id: str):
        return {
            "task_id": task_id,
            "status": "PENDING",
            "request": {
                "images": [
                    {
                        "image_url": (
                            "https://example.com/broken.jpg"
                        )
                    }
                ]
            },
        }

    async def fake_download_images(*args, **kwargs):
        raise worker.ImageDownloadError(
            code="INVALID_IMAGE",
            message="圖片無法解析",
        )

    async def fake_mark_processing(
        task_id: str,
        progress: int,
    ):
        return None

    async def fake_mark_failed(
        task_id: str,
        error: dict,
    ):
        saved_error.update(error)

    monkeypatch.setattr(
        worker.store,
        "get_task",
        fake_get_task,
    )
    monkeypatch.setattr(
        worker,
        "download_images",
        fake_download_images,
    )
    monkeypatch.setattr(
        worker.store,
        "mark_processing",
        fake_mark_processing,
    )
    monkeypatch.setattr(
        worker.store,
        "mark_failed",
        fake_mark_failed,
    )

    asyncio.run(
        worker.process_task(
            "acad_image_error",
            object(),
        )
    )

    assert saved_error == {
        "code": "IMAGE_DOWNLOAD_FAILED",
        "message": "圖片無法解析",
    }


def test_worker_maps_model_error_to_internal_error(monkeypatch):
    """模型推論失敗對外必須使用正式的 INTERNAL_ERROR。"""

    saved_error: dict = {}

    class BrokenPredictor:
        num_classes = 1

        def predict(self, image: Path, top_k: int):
            raise RuntimeError("測試用模型錯誤")

    async def fake_get_task(task_id: str):
        return {
            "task_id": task_id,
            "status": "PENDING",
            "request": {
                "images": [
                    {
                        "image_url": (
                            "https://example.com/leaf.jpg"
                        )
                    }
                ]
            },
        }

    async def fake_download_images(*args, **kwargs):
        return [Path("leaf.jpg")]

    async def fake_mark_processing(
        task_id: str,
        progress: int,
    ):
        return None

    async def fake_mark_failed(
        task_id: str,
        error: dict,
    ):
        saved_error.update(error)

    monkeypatch.setattr(
        worker.store,
        "get_task",
        fake_get_task,
    )
    monkeypatch.setattr(
        worker,
        "download_images",
        fake_download_images,
    )
    monkeypatch.setattr(
        worker.store,
        "mark_processing",
        fake_mark_processing,
    )
    monkeypatch.setattr(
        worker.store,
        "mark_failed",
        fake_mark_failed,
    )

    asyncio.run(
        worker.process_task(
            "acad_model_error",
            BrokenPredictor(),
        )
    )

    assert saved_error == {
        "code": "INTERNAL_ERROR",
        "message": "模型服務異常",
    }
