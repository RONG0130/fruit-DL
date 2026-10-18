"""會議結果 JSON 規格測試。"""

import pytest

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
            ("Anthracnose", 99.65),
            ("Rust", 0.19),
            ("Mealybug", 0.11),
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

    # 這批資料的最高值大於 1，代表整批已經是百分比。
    # 因此 0.19 與 0.11 不可被個別放大為 19% 與 11%。
    assert [
        item["confidence"]
        for item in items
    ] == [
        100,
        0,
        0,
    ]


def test_probability_scale_is_used_before_threshold():
    """0～1 機率必須先轉成百分比，再進行門檻判斷。"""

    high_result = build_result(
        [
            ("Anthracnose", 0.85),
            ("Rust", 0.10),
            ("Mealybug", 0.05),
        ]
    )

    assert high_result["summary"]["title"] == (
        "模型分析完成"
    )
    assert high_result["summary"]["confidence"] == 85
    assert high_result["need_more_info"] == []
    assert [
        item["confidence"]
        for item in high_result["result_blocks"][0]["items"]
    ] == [
        85,
        10,
        5,
    ]

    low_result = build_result(
        [("Anthracnose", 0.35)]
    )

    assert low_result["summary"]["title"] == (
        "目前無法明確判斷"
    )
    assert low_result["summary"]["confidence"] == 35
    assert len(low_result["need_more_info"]) > 0


def test_issue_type_key_uses_chinese_value():
    """issue_type 的 key 保持英文，value 必須使用規格中文。"""

    disease_result = build_result(
        [("Anthracnose", 90.0)]
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
        [("Sugar_apple_whitefly", 88.0)]
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


@pytest.mark.parametrize(
    ("class_name", "expected_scientific_name"),
    [
        (
            "Anthracnose",
            "Colletotrichum spp.",
        ),
        (
            "Algal_leaf_spot",
            "Cephaleuros spp.",
        ),
        (
            "Rust",
            "Pucciniales",
        ),
        (
            "Sugar_apple_whitefly",
            "Aleyrodidae",
        ),
        (
            "Kanzawa_spider_mite",
            "Tetranychus kanzawai",
        ),
        (
            "Mealybug",
            "Pseudococcidae",
        ),
        (
            "Tea_red_spider_mite",
            "Oligonychus coffeae",
        ),
    ],
)
def test_scientific_name_matches_annona_class(
    class_name: str,
    expected_scientific_name: str,
):
    """各釋迦病蟲害類別都必須輸出對應的正式學名。"""

    result = build_result(
        [(class_name, 90.0)]
    )
    item = result["result_blocks"][0]["items"][0]
    scientific_name_attribute = find_attribute(
        item,
        "scientific_name",
    )

    assert scientific_name_attribute == {
        "key": "scientific_name",
        "label": "學名",
        "value": expected_scientific_name,
    }


def test_description_is_diagnosis_basis():
    """description 必須是判斷依據，不放規格外的防治建議。"""

    result = build_result(
        [("Anthracnose", 90.0)]
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
        [("Anthracnose", 85.0)]
    )

    assert result["raw_output"] == {
        "model_label": "Anthracnose",
        "model_score": 0.85,
    }


@pytest.mark.parametrize(
    (
        "input_confidence",
        "expected_confidence",
        "expected_model_score",
    ),
    [
        (0.85, 85, 0.85),
        (85.0, 85, 0.85),
        (1.0, 100, 1.0),
        (100.0, 100, 1.0),
        (120.0, 100, 1.0),
    ],
)
def test_model_score_is_always_between_zero_and_one(
    input_confidence: float,
    expected_confidence: int,
    expected_model_score: float,
):
    """model_score 必須固定使用 0～1 的機率格式。"""

    result = build_result(
        [("Anthracnose", input_confidence)]
    )
    model_score = result["raw_output"][
        "model_score"
    ]

    assert result["summary"]["confidence"] == (
        expected_confidence
    )
    assert model_score == expected_model_score
    assert 0.0 <= model_score <= 1.0


def test_model_and_result_keys_match_contract():
    """成功結果的固定欄位須符合實作指南。"""

    result = build_result(
        [("Anthracnose", 85.0)]
    )

    assert set(result) == {
        "schema_version",
        "model",
        "summary",
        "result_blocks",
        "need_more_info",
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
            ("Anthracnose", 35.0),
            ("Rust", 30.0),
            ("Mealybug", 20.0),
        ]
    )

    item = result["result_blocks"][0]["items"][0]

    assert result["summary"]["title"] == (
        "目前無法明確判斷"
    )
    assert item["label"] == "無法明確判定"
    assert item["confidence"] == 35
    assert "attributes" not in item
    assert "need_more_info" in result
    assert isinstance(result["need_more_info"], list)
    assert len(result["need_more_info"]) > 0
    assert result["need_more_info"][0]["type"] == (
        "image"
    )


def test_normal_result_returns_empty_need_more_info():
    """正常診斷也必須回傳 need_more_info，內容為空陣列。"""

    result = build_result(
        [("Anthracnose", 90.0)]
    )

    assert "need_more_info" in result
    assert result["need_more_info"] == []
    assert isinstance(
        result["need_more_info"],
        list,
    )
