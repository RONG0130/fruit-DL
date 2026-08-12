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


DISEASE_METADATA = {
    "healthy": {
        "zh_name": "健康",
    },
    "anthracnose": {
        "zh_name": "炭疽病",
        "issue_type": "病害",
    },
    "algal_leaf_spot": {
        "zh_name": "藻斑病",
        "issue_type": "病害",
    },
    "rust": {
        "zh_name": "銹病",
        "issue_type": "病害",
    },
    "pest_whitefly": {
        "zh_name": "番荔枝粉蝨",
        "issue_type": "蟲害",
    },
    "pest_kanzawa_spider_mite": {
        "zh_name": "神澤氏葉蟎",
        "issue_type": "蟲害",
    },
    "pest_mealybug": {
        "zh_name": "粉介殼蟲",
        "issue_type": "蟲害",
    },
    "pest_tea_mite": {
        "zh_name": "茶葉蟎",
        "issue_type": "蟲害",
    },
}


def get_metadata(class_name: str) -> dict:
    """取得類別的繁體中文名稱與問題類型。"""

    return DISEASE_METADATA.get(
        class_name,
        {
            "zh_name": class_name,
        },
    )


def normalize_confidence(value: float) -> int:
    """依會議規格將信心度轉成 0～100 整數。"""

    return max(
        0,
        min(int(round(value)), 100),
    )


def build_api_result(
    merged_predictions: list[tuple[str, float]],
    image_count: int,
) -> dict:
    """將多張圖片合併預測轉成 API 結果。"""

    if not merged_predictions:
        raise ValueError("模型沒有回傳任何預測結果")

    best_class, best_confidence = (
        merged_predictions[0]
    )
    best_metadata = get_metadata(best_class)
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
        for class_name, confidence in merged_predictions:
            metadata = get_metadata(class_name)
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
                "title": "可能病因",
                "items": result_items,
            }
        ],
        "raw_output": {
            # raw_output 為規格允許的選填原始輸出，
            # 採用實作指南範例的最小格式。
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
