"""
將模型預測結果轉換成 API 對外回傳格式。
"""

import os


def load_min_diagnosis_confidence() -> float:
    """讀取低信心判定門檻，設定錯誤時安全回到 50%。"""

    configured_value = os.getenv(
        "MIN_DIAGNOSIS_CONFIDENCE",
        "50",
    )

    try:
        threshold = float(configured_value)
    except ValueError:
        threshold = 50.0

    return max(
        0.0,
        min(threshold, 100.0),
    )


# 低於此信心度時，不直接宣稱單一病因。
# 可在正式部署時透過環境變數調整，但數值固定限制於 0～100。
MIN_DIAGNOSIS_CONFIDENCE = (
    load_min_diagnosis_confidence()
)


# 以「植物＋原始標籤」查表；同名 Rust 不得跨作物套用物種名。
# 廣義標籤只填有來源支持的屬、科或目；各名稱與層級見 docs/scientific_name_sources.md。
PLANT_METADATA = {
    "annona": {
        "Algal_leaf_spot": {"zh_name": "藻斑病", "issue_type": "病害", "scientific_name": "Cephaleuros spp."},
        "Anthracnose": {"zh_name": "炭疽病", "issue_type": "病害", "scientific_name": "Colletotrichum spp."},
        "Kanzawa_spider_mite": {"zh_name": "神澤氏葉蟎", "issue_type": "蟲害", "scientific_name": "Tetranychus kanzawai"},
        "Mealybug": {"zh_name": "粉介殼蟲", "issue_type": "蟲害", "scientific_name": "Pseudococcidae"},
        "Rust": {"zh_name": "銹病", "issue_type": "病害", "scientific_name": "Pucciniales"},
        "Sugar_apple_whitefly": {"zh_name": "番荔枝粉蝨", "issue_type": "蟲害", "scientific_name": "Aleyrodidae"},
        "Tea_red_spider_mite": {"zh_name": "茶葉蟎", "issue_type": "蟲害", "scientific_name": "Oligonychus coffeae"},
    },
    "tea": {
        "Agromyzidae": {"zh_name": "潛葉蠅", "issue_type": "蟲害", "scientific_name": "Agromyzidae"},
        "Caloptilia_theivora": {"zh_name": "茶細蛾", "issue_type": "蟲害", "scientific_name": "Caloptilia theivora"},
        "Oligonychus_coffeae": {"zh_name": "葉螨", "issue_type": "蟲害", "scientific_name": "Oligonychus coffeae"},
        "Tortrix": {"zh_name": "捲葉蛾", "issue_type": "蟲害", "scientific_name": "Tortricidae"},
    },
    "legume": {
        "Liriomyza_spp": {"zh_name": "斑潛蠅", "issue_type": "蟲害", "scientific_name": "Liriomyza spp."},
        "Rust": {"zh_name": "銹病", "issue_type": "病害", "scientific_name": "Pucciniales"},
        "bean_aphid": {"zh_name": "蚜蟲", "issue_type": "蟲害", "scientific_name": "Aphididae"},
    },
    "facility_leafy_greens": {
        "Aphids": {"zh_name": "蚜蟲", "issue_type": "蟲害", "scientific_name": "Aphididae"},
        "Phaedon_brassicae_Baly": {"zh_name": "十字花科小猿葉蟲", "issue_type": "蟲害", "scientific_name": "Phaedon brassicae"},
        "Pieris_rapae": {"zh_name": "紋白蝶", "issue_type": "蟲害", "scientific_name": "Pieris rapae"},
        "Striped_flea_beetle": {"zh_name": "黃條葉蚤", "issue_type": "蟲害", "scientific_name": "Phyllotreta striolata"},
    },
}


def get_metadata(class_name: str, plant: str = "annona") -> dict:
    """取得類別的繁體中文名稱與問題類型。"""

    if class_name in PLANT_METADATA.get(plant, {}):
        return PLANT_METADATA[plant][class_name]
    raise ValueError(f"找不到 {plant}/{class_name} 的結果對照")

def normalize_prediction_percentages(
    predictions: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """
    將整批模型信心度統一轉換成 0～100 的百分比。

    模型可能回傳兩種格式：
    1. 機率格式：0.0～1.0，例如 0.85
    2. 百分比格式：0.0～100.0，例如 85.0

    使用整批資料的最高信心度判斷維度，避免將原本的
    0.19% 誤判成 19%。
    """

    numeric_predictions = [
        (
            class_name,
            float(confidence),
        )
        for class_name, confidence in predictions
    ]

    highest_confidence = max(
        confidence
        for _, confidence in numeric_predictions
    )

    # 如果整批資料的最高值都沒有超過 1，
    # 代表這批資料使用的是 0～1 機率格式。
    multiplier = (
        100.0
        if highest_confidence <= 1.0
        else 1.0
    )

    return [
        (
            class_name,

            # 將結果限制在 0～100，避免異常數值
            # 造成對外 API 的 confidence 超出規格。
            max(
                0.0,
                min(
                    confidence * multiplier,
                    100.0,
                ),
            ),
        )
        for class_name, confidence
        in numeric_predictions
    ]

def normalize_confidence(value: float) -> int:
    """依會議規格將信心度轉成 0～100 整數。"""

    return max(
        0,
        min(int(round(value)), 100),
    )


def build_api_result(
    merged_predictions: list[tuple[str, float]],
    image_count: int,
    plant: str = "annona",
) -> dict:
    """將多張圖片合併預測轉成 API 結果。"""

    if not merged_predictions:
        raise ValueError("模型沒有回傳任何預測結果")

    # 在門檻判斷及 API 格式轉換前，
    # 先將整批信心度統一成 0～100。
    normalized_predictions = (
        normalize_prediction_percentages(
            merged_predictions
        )
    )

    best_class, best_confidence = (
        normalized_predictions[0]
    )
    best_metadata = get_metadata(best_class, plant)
    normalized_best_confidence = normalize_confidence(
        best_confidence
    )
    is_uncertain = (
        best_confidence
        < MIN_DIAGNOSIS_CONFIDENCE
    )

    result_items = []

    if is_uncertain:
        # 推論流程已正常完成，所以仍回傳 SUCCEEDED。
        # 但信心度不足時不得把最高候選直接當成明確病因。
        result_items.append(
            {
                "label": "無法明確判定",
                "confidence": (
                    normalized_best_confidence
                ),
                "description": (
                    "模型已完成影像分類，但最高候選"
                    f"「{best_metadata['zh_name']}」的信心度為 "
                    f"{normalized_best_confidence}%，低於 "
                    f"{MIN_DIAGNOSIS_CONFIDENCE:g}% 的判定門檻；"
                    "目前影像資訊不足以提供明確病因。"
                ),
            }
        )

    else:
        for class_name, confidence in normalized_predictions:
            metadata = get_metadata(class_name, plant)
            normalized_confidence = normalize_confidence(
                confidence
            )

            attributes = []

            # 會議允許的 issue_type value 為病害、蟲害、
            # 生理障礙或環境因素。健康與未知類別不附此屬性。
            issue_type = metadata.get("issue_type")

            if issue_type:
                attributes.append(
                    {
                        "key": "issue_type",
                        "label": "問題類型",
                        "value": issue_type,
                    }
                )

            scientific_name = metadata.get("scientific_name")

            if scientific_name:
                attributes.append(
                    {
                        "key": "scientific_name",
                        "label": "學名",
                        "value": scientific_name,
                    }
                )

            item = {
                "label": metadata["zh_name"],
                "confidence": normalized_confidence,
                "description": (
                    "影像分類模型將此候選判定為"
                    f"「{metadata['zh_name']}」，信心度為 "
                    f"{normalized_confidence}%。"
                    "目前模型未輸出病徵區域定位，"
                    "仍需配合現場症狀與專業判斷確認。"
                ),
            }

            # attributes 是選填欄位；只有確實有資料時才輸出。
            if attributes:
                item["attributes"] = attributes

            result_items.append(item)

    if is_uncertain:
        summary = {
            "title": "目前無法明確判斷",
            "description": (
                f"系統已合併分析 {image_count} 張同一植株圖片，"
                f"但最高候選信心度只有 {normalized_best_confidence}%，"
                "建議補充更清晰且包含不同部位的照片。"
            ),
            "confidence": normalized_best_confidence,
        }
    else:
        summary = {
            "title": "模型分析完成",
            "description": (
                f"首要疑似{best_metadata['zh_name']}。"
            ),
            "confidence": normalized_best_confidence,
        }

    result = {
        "schema_version": "academy_model_result.v1",
        "model": {
            "academy_id": "npust_dn_lab",
            "model_name": (
                "vit_large_patch16_dinov3.lvd1689m"
            ),
            "model_version": "1.0.0",
        },
        "summary": summary,
        "result_blocks": [
            {
                "type": "diagnosis",
                "title": "診斷結果",
                "items": result_items,
            }
        ],

        # 正常診斷沒有補充資料需求時，仍須回傳空陣列。
        # 讓一站式平台可以固定讀取這個欄位。
        "need_more_info": [],

        "raw_output": {
            "model_label": best_class,
            "model_score": round(
                best_confidence / 100,
                4,
            ),
        },
    }

    if is_uncertain:
        result["need_more_info"] = [
            {
                "type": "image",
                "description": (
                    "請補拍患部近照、葉片背面及整株遠景，"
                    "並避免影像模糊或光線過暗。"
                ),
            }
        ]

    return result
