"""會議結果 JSON 規格測試。"""

from api.result_mapper import build_api_result


def find_attribute(item: dict, key: str) -> dict | None:
    """依 key 尋找結果項目的 attribute。"""

    for attribute in item.get(
        "attributes",
        [],
    ):
        if attribute["key"] == key:
            return attribute

    return None


def build_result(predictions):
    """建立單張圖片的測試結果。"""

    return build_api_result(
        merged_predictions=predictions,
        image_count=1,
    )


def test_confidence_is_integer_between_0_and_100():
    """summary 及 diagnosis items 的 confidence 必須為整數。"""

    result = build_result(
        [
            ("anthracnose", 99.65),
            ("rust", 0.19),
            ("healthy", 0.11),
        ]
    )

    assert result["summary"]["confidence"] == 100
    assert isinstance(
        result["summary"]["confidence"],
        int,
    )

    items = result["result_blocks"][0]["items"]

    for item in items:
        assert isinstance(item["confidence"], int)
        assert 0 <= item["confidence"] <= 100


def test_issue_type_key_uses_chinese_value():
    """issue_type 的 key 保持英文，value 必須使用規格中文。"""

    disease_result = build_result(
        [("anthracnose", 90.0)]
    )
    disease_item = disease_result[
        "result_blocks"
    ][0]["items"][0]
    disease_attribute = find_attribute(
        disease_item,
        "issue_type",
    )

    assert disease_attribute == {
        "key": "issue_type",
        "label": "問題類型",
        "value": "病害",
    }

    pest_result = build_result(
        [("pest_whitefly", 88.0)]
    )
    pest_item = pest_result[
        "result_blocks"
    ][0]["items"][0]
    pest_attribute = find_attribute(
        pest_item,
        "issue_type",
    )

    assert pest_attribute == {
        "key": "issue_type",
        "label": "問題類型",
        "value": "蟲害",
    }


def test_healthy_result_omits_issue_type_attribute():
    """健康不在允許的問題類型清單中，因此不輸出 issue_type。"""

    result = build_result(
        [("healthy", 96.0)]
    )
    item = result["result_blocks"][0]["items"][0]

    assert find_attribute(item, "issue_type") is None
    assert "attributes" not in item


def test_description_is_diagnosis_basis():
    """description 必須是判斷依據，不放規格外的防治建議。"""

    result = build_result(
        [("anthracnose", 90.0)]
    )
    item = result["result_blocks"][0]["items"][0]

    assert "影像分類模型" in item["description"]
    assert find_attribute(
        item,
        "management_advice",
    ) is None


def test_raw_output_matches_implementation_guide_example():
    """raw_output 收斂為規格範例使用的模型標籤與分數。"""

    result = build_result(
        [("anthracnose", 85.0)]
    )

    assert result["raw_output"] == {
        "model_label": "anthracnose",
        "model_score": 0.85,
    }


def test_model_and_result_keys_match_contract():
    """成功結果的固定欄位須符合實作指南。"""

    result = build_result(
        [("anthracnose", 85.0)]
    )

    assert set(result) == {
        "schema_version",
        "model",
        "summary",
        "result_blocks",
        "raw_output",
    }
    assert result["schema_version"] == (
        "academy_model_result.v1"
    )
    assert result["model"] == {
        "academy_id": "npust_dn_lab",
        "model_name": (
            "vit_large_patch16_dinov3.lvd1689m"
        ),
        "model_version": "1.0.0",
    }


def test_low_confidence_returns_uncertain_result():
    """低信心仍成功，但必須要求補充資訊而非直接宣稱病因。"""

    result = build_result(
        [
            ("anthracnose", 35.0),
            ("rust", 30.0),
            ("healthy", 20.0),
        ]
    )

    item = result["result_blocks"][0]["items"][0]

    assert result["summary"]["title"] == (
        "目前無法明確判斷"
    )
    assert item["label"] == "無法明確判定"
    assert item["confidence"] == 35
    assert "attributes" not in item
    assert result["need_more_info"][0]["type"] == (
        "image"
    )
