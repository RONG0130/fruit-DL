"""植物選項與前端任務狀態測試。"""

from pathlib import Path

from streamlit_plant import PLANT_OPTIONS, PLANT_CLASSES, selection_fingerprint, clear_task_state


def test_plant_options_match_api_codes_and_class_lists():
    assert list(PLANT_OPTIONS.items()) == [
        ("annona", "釋迦"),
        ("tea", "茶葉"),
        ("legume", "豆科"),
        ("facility_leafy_greens", "設施小葉菜"),
    ]
    assert set(PLANT_CLASSES) == set(PLANT_OPTIONS)
    assert ("Caloptilia_theivora", "茶細蛾") in PLANT_CLASSES["tea"]
    assert ("Rust", "銹病") in PLANT_CLASSES["legume"]


def test_sidebar_names_match_plant_translation_tables():
    """對照四份中英對照表，避免側欄自行改寫原始中文。"""
    assert PLANT_CLASSES == {
        "annona": (
            ("Algal_leaf_spot", "藻斑病"),
            ("Anthracnose", "炭疽病"),
            ("Kanzawa_spider_mite", "神澤氏葉蟎"),
            ("Mealybug", "粉介殼蟲"),
            ("Rust", "銹病"),
            ("Sugar_apple_whitefly", "番荔枝粉蝨"),
            ("Tea_red_spider_mite", "茶葉蟎"),
        ),
        "tea": (
            ("Agromyzidae", "潛葉蠅"),
            ("Caloptilia_theivora", "茶細蛾"),
            ("Oligonychus_coffeae", "葉螨"),
            ("Tortrix", "捲葉蛾"),
        ),
        "legume": (
            ("Bemisia_tabaci", "銀葉粉蝨"),
            ("Liriomyza_spp", "斑潛蠅"),
            ("Rust", "銹病"),
            ("Scale_insects", "介殼蟲"),
            ("Small_green_leafhopper", "小綠葉蟬"),
            ("bean_aphid", "蚜蟲"),
        ),
        "facility_leafy_greens": (
            ("Aphids", "蚜蟲"),
            ("Phaedon_brassicae_Baly", "十字花科小猿葉蟲"),
            ("Pieris_rapae", "紋白蝶"),
            ("Striped_flea_beetle", "黃條葉蚤"),
            ("White_rust", "白銹病"),
        ),
    }


def test_changing_plant_invalidates_existing_image_result():
    assert selection_fingerprint("same-image", "annona") != selection_fingerprint(
        "same-image", "tea"
    )
    state = {
        "latest_task_id": "old-task",
        "latest_result": {"summary": "old"},
        "latest_task_response": {"status": "SUCCEEDED"},
        "latest_error": "old-error",
        "latest_similarity": {"selected_index": 0},
        "selected_plant": "tea",
    }
    clear_task_state(state)
    assert state == {
        "latest_task_id": None,
        "latest_result": None,
        "latest_task_response": None,
        "latest_error": None,
        "latest_similarity": None,
        "selected_plant": "tea",
    }


def test_ui_passes_selected_plant_to_diagnosis_and_client():
    """此頁面在匯入時即啟動 Streamlit；核對 UI 呼叫邊界。"""
    source = (Path(__file__).resolve().parent.parent / "app_api.py").read_text(encoding="utf-8")
    assert "plant=selected_plant" in source
    assert "plant=plant" in source
