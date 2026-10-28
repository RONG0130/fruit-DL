"""Streamlit 使用的植物選項與任務畫面狀態。"""


# 第一項是舊版預設作物；代碼須與 FastAPI 的 task.plant 一致。
PLANT_OPTIONS = {
    "annona": "釋迦",
    "tea": "茶葉",
    "legume": "豆科",
    "facility_leafy_greens": "設施小葉菜",
}

# 僅供側欄說明模型支援範圍；正式診斷文字一律使用 API result_blocks。
PLANT_CLASSES = {
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


def selection_fingerprint(image_fingerprint: str, plant: str) -> str:
    """同一圖片改選植物時，視為不同診斷案件。"""
    return f"{plant}:{image_fingerprint}"


def clear_task_state(state) -> None:
    """清除前一次診斷資料，保留植物選擇與已上傳圖片。"""
    for key in (
        "latest_task_id", "latest_result", "latest_task_response", "latest_error",
        "latest_similarity",
    ):
        state[key] = None
