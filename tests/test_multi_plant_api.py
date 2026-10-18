"""多植物 API 契約與模型選擇。"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from api.schemas import CreateTaskRequest
from api.result_mapper import build_api_result, get_metadata
from api.model_registry import MODEL_SPECS, ModelManager
from api import store
from api import main
from api.main import app
from streamlit_plant import PLANT_CLASSES


def payload(plant=None):
    task = {"academy_type": "plant_diagnosis"}
    if plant is not None:
        task["plant"] = plant
    return {
        "platform_task_id": "one",
        "images": [{
            "image_url": "https://example.com/leaf.jpg",
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "mime_type": "image/jpeg",
        }],
        "task": task,
    }


def test_legacy_request_defaults_to_annona():
    assert CreateTaskRequest.model_validate(payload()).task.plant == "annona"


def test_invalid_plant_is_rejected():
    with pytest.raises(ValidationError):
        CreateTaskRequest.model_validate(payload("banana"))


def test_invalid_plant_keeps_standard_http_error(monkeypatch):
    monkeypatch.setenv("ACADEMY_API_TOKEN", "test-token")
    response = TestClient(app).post(
        "/v1/tasks", json=payload("banana"),
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 400
    assert response.json() == {
        "code": "VALIDATION_ERROR",
        "message": "請求格式錯誤或必填欄位缺漏",
    }


def test_all_registered_classes_have_metadata():
    for plant, spec in MODEL_SPECS.items():
        for label in spec.classes:
            item = build_api_result([(label, 90)], 1, plant=plant)["result_blocks"][0]["items"][0]
            assert item["label"] != label
            assert any(a["key"] == "issue_type" for a in item["attributes"])


def test_diagnosis_labels_match_plant_translation_tables():
    """API 結果應與各植物的中英對照表名稱一致。"""
    for plant, classes in PLANT_CLASSES.items():
        for model_label, chinese_name in classes:
            result = build_api_result([(model_label, 90)], 1, plant=plant)
            item = result["result_blocks"][0]["items"][0]
            assert item["label"] == chinese_name, (plant, model_label)
            assert chinese_name in item["description"]
            assert chinese_name in result["summary"]["description"]


@pytest.mark.parametrize(
    ("plant", "model_label", "scientific_name"),
    [
        ("annona", "Algal_leaf_spot", "Cephaleuros spp."),
        ("annona", "Anthracnose", "Colletotrichum spp."),
        ("annona", "Kanzawa_spider_mite", "Tetranychus kanzawai"),
        ("annona", "Mealybug", "Pseudococcidae"),
        ("annona", "Rust", "Pucciniales"),
        ("annona", "Sugar_apple_whitefly", "Aleyrodidae"),
        ("annona", "Tea_red_spider_mite", "Oligonychus coffeae"),
        ("tea", "Agromyzidae", "Agromyzidae"),
        ("tea", "Caloptilia_theivora", "Caloptilia theivora"),
        ("tea", "Oligonychus_coffeae", "Oligonychus coffeae"),
        ("tea", "Tortrix", "Tortricidae"),
        ("legume", "Liriomyza_spp", "Liriomyza spp."),
        ("legume", "Rust", "Pucciniales"),
        ("legume", "bean_aphid", "Aphididae"),
        ("facility_leafy_greens", "Aphids", "Aphididae"),
        ("facility_leafy_greens", "Phaedon_brassicae_Baly", "Phaedon brassicae"),
        ("facility_leafy_greens", "Pieris_rapae", "Pieris rapae"),
        ("facility_leafy_greens", "Striped_flea_beetle", "Phyllotreta striolata"),
    ],
)
def test_each_model_class_has_supported_scientific_taxon(
    plant, model_label, scientific_name
):
    """每個模型標籤提供有來源支持的最低共同分類層級。"""
    result = build_api_result([(model_label, 90)], 1, plant=plant)
    item = result["result_blocks"][0]["items"][0]
    assert {
        "key": "scientific_name",
        "label": "學名",
        "value": scientific_name,
    } in item["attributes"]


def test_retired_model_label_is_not_accepted():
    with pytest.raises(ValueError):
        get_metadata("anthracnose", "annona")


def test_rust_uses_plant_specific_metadata():
    annona = build_api_result([("Rust", 90)], 1, plant="annona")
    legume = build_api_result([("Rust", 90)], 1, plant="legume")
    annona_item = annona["result_blocks"][0]["items"][0]
    legume_item = legume["result_blocks"][0]["items"][0]
    assert annona_item["label"] == "銹病"
    assert legume_item["label"] == "銹病"
    assert next(a["value"] for a in annona_item["attributes"] if a["key"] == "scientific_name") == "Pucciniales"
    assert next(a["value"] for a in legume_item["attributes"] if a["key"] == "scientific_name") == "Pucciniales"


def test_manager_replaces_model_before_loading_next_one(monkeypatch):
    loaded = []
    class FakePredictor:
        def __init__(self, plant):
            self.plant = plant
    def fake_load(plant):
        loaded.append(plant)
        return FakePredictor(plant)
    manager = ModelManager(fake_load)
    async def scenario():
        async with manager.use("annona") as first:
            assert first.plant == "annona"
        async with manager.use("tea") as second:
            assert second.plant == "tea"
        assert manager.loaded_plant == "tea"
    asyncio.run(scenario())
    assert loaded == ["annona", "tea"]


def test_same_platform_id_cannot_switch_plant(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "tasks.db")
    async def scenario():
        await store.init_db()
        await store.create_or_get_task(CreateTaskRequest.model_validate(payload("annona")))
        with pytest.raises(store.TaskPlantConflict):
            await store.create_or_get_task(CreateTaskRequest.model_validate(payload("tea")))
    asyncio.run(scenario())


def test_create_task_does_not_wait_for_model_switch(monkeypatch):
    """POST 只需檢查模型檔並排隊；耗時載入交給背景 Worker。"""
    request = CreateTaskRequest.model_validate(payload("tea"))

    class SlowManager:
        def use(self, plant):
            raise AssertionError("POST 不應等待模型切換")

    async def fake_store(body):
        return {"task_id": "already-created"}, False

    monkeypatch.setattr(main, "ensure_model_service_ready", lambda: None)
    monkeypatch.setattr(main, "check_artifacts", lambda plant: None)
    monkeypatch.setattr(main.app.state, "model_manager", SlowManager(), raising=False)
    monkeypatch.setattr(main.store, "create_or_get_task", fake_store)

    assert asyncio.run(main.create_task(request)) == {
        "task_id": "already-created", "status": "PENDING"
    }
