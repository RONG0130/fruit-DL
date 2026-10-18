#!/usr/bin/env python3
"""
植物病蟲害智慧辨識系統。

本版本保留學長姐原本的 Streamlit 畫面設計，
但不再由 Streamlit 直接載入 DINOv3 模型。

新的資料流程：
1. Streamlit 接收使用者上傳的 1～5 張圖片。
2. Streamlit 呼叫 POST /v1/tasks 建立任務。
3. Streamlit 呼叫 GET /v1/tasks/{task_id} 輪詢狀態。
4. FastAPI 在背景下載圖片並執行模型。
5. Streamlit 取得 SUCCEEDED 結果後顯示診斷內容。
6. GET /v1/health 用來確認 FastAPI 是否正常。
"""

import hashlib
import html
import os
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from streamlit_api_client import (
    APIClientError,
    PlantDiagnosisAPIClient,
)
from streamlit_plant import (
    PLANT_OPTIONS,
    PLANT_CLASSES,
    selection_fingerprint,
    clear_task_state,
)


# ============================================================
# 基本設定
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# Streamlit 暫存上傳圖片的位置。
# 本機圖片伺服器會從這個資料夾提供圖片。
UPLOAD_DIRECTORY = (
    PROJECT_ROOT
    / "runtime"
    / "streamlit_uploads"
)

# FastAPI 網址。
# 正式部署時可以透過環境變數改成農科院提供的 HTTPS 網址。
API_BASE_URL = os.getenv(
    "FRUIT_API_BASE_URL",
    "http://127.0.0.1:8000",
)

# 呼叫 FastAPI 使用的 Bearer Token。
# 正式環境不可將 Token 寫死在程式碼或提交至 GitHub。
API_TOKEN = os.getenv(
    "ACADEMY_API_TOKEN",
    "",
).strip()

# 本機圖片伺服器網址。
# 正式串接時應替換成物件儲存空間的 Signed URL。
IMAGE_BASE_URL = os.getenv(
    "STREAMLIT_IMAGE_BASE_URL",
    "http://127.0.0.1:9000",
)

# 依會議規格，平台最長輪詢時間為 30 分鐘。
TASK_TIMEOUT_SECONDS = 1800

# 建立任務後至少等待 1 秒，才開始第一次 GET。
INITIAL_POLL_DELAY_SECONDS = 1.0

# 後續每 2 秒查詢一次，符合會議要求的 1.5～3 秒。
POLL_INTERVAL_SECONDS = 2.0

# 設定成 35 分鐘，替至少 30 分鐘的規格保留時間誤差。
SIGNED_URL_VALIDITY_MINUTES = 35

# Streamlit 可以上傳的副檔名與 API MIME Type 對照。
ALLOWED_SUFFIX_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

# API 目前固定回傳前三名。
MAX_DISPLAY_RESULTS = 3


# ============================================================
# Streamlit 頁面設定
# ============================================================

st.set_page_config(
    page_title="植物病蟲害智慧辨識系統",
    page_icon="🌿",
    layout="wide",
)


# ============================================================
# 全站 CSS 樣式
# ============================================================

st.markdown(
    """
    <style>
        /* 主畫面背景 */
        .stApp {
            background-color: #768f5f;
        }

        /* 主內容寬度 */
        [data-testid="stMain"] .block-container {
            max-width: 850px;
            margin: 0 auto;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 2rem;
        }

        /* Streamlit Header */
        [data-testid="stHeader"] {
            background: transparent;
        }

        /* 側邊欄背景 */
        [data-testid="stSidebar"] {
            background-color: #52663f;
        }

        /* 側邊欄文字 */
        [data-testid="stSidebar"] * {
            color: #ffffff;
        }

        /* 側欄是白字，但選擇框是淺色底；明確設定選中值的深色文字。 */
        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background-color: #e8f1e3 !important;
            border-color: #a4b99a !important;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
            color: #263822 !important;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] svg {
            fill: #263822 !important;
        }

        /* 下拉清單是掛在側欄外的浮層，不能只靠側欄選擇器。 */
        [data-testid="stSelectboxVirtualDropdown"],
        [data-testid="stSelectboxVirtualDropdown"] [role="option"] {
            background-color: #e8f1e3 !important;
            color: #263822 !important;
        }

        [data-testid="stSelectboxVirtualDropdown"] [role="option"] * {
            color: #263822 !important;
        }

        /* 已選項以左側細線標示，不使用反白底色。 */
        [data-testid="stSelectboxVirtualDropdown"] [role="option"][aria-selected="true"] {
            border-left: 3px solid #3b4f32;
        }

        /* 側邊欄 Expander */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid #768f5f;
            border-radius: 10px;
            background-color: #52663f;
            overflow: hidden;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] details > summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] details > summary:hover,
        [data-testid="stSidebar"] [data-testid="stExpander"] details > summary:focus {
            background-color: #52663f !important;
        }

        /* Expander 標題 */
        details > summary {
            color: #ffffff;
            font-weight: 600;
        }

        /* 自訂資訊色塊 */
        .green-result-box {
            background-color: #52663f;
            color: #ffffff;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            font-weight: 600;
            font-size: 1.05rem;
            margin-bottom: 0.8rem;
        }

        /* 建議措施色塊 */
        .recommendation-box {
            background-color: #52663f;
            color: #ffffff;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            white-space: pre-line;
            font-size: 0.93rem;
            line-height: 1.7;
        }

        /* 圖片資訊 */
        .image-info-box {
            background-color: #52663f;
            color: #ffffff;
            border-radius: 10px;
            margin-top: 0.5rem;
            padding: 0.7rem 1rem;
            line-height: 1.7;
        }

        /* 修正提示框文字溢出 */
        .stAlert {
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }

        /* 修正上傳元件文字溢出 */
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] div {
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
        }

        /* Metric delta */
        [data-testid="stMetricDelta"] {
            background-color: #d8ffd3;
            border-radius: 999px;
            padding: 0.15rem 0.4rem;
            width: fit-content;
        }

        [data-testid="stMetricDelta"] > div {
            color: #3b4f32;
            font-weight: 550;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session State 初始化
# ============================================================

def initialize_session_state() -> None:
    """
    初始化 Streamlit Session State。

    Streamlit 每次操作元件都會重新執行整支程式，
    因此任務結果必須保存在 Session State。
    """

    default_values = {
        "upload_fingerprint": None,
        "latest_task_id": None,
        "latest_result": None,
        "latest_task_response": None,
        "latest_error": None,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# FastAPI Client
# ============================================================

@st.cache_resource
def get_api_client() -> PlantDiagnosisAPIClient:
    """
    建立 FastAPI Client。

    這裡快取的是 API Client，不是 DINOv3 模型。
    模型只會由 FastAPI 載入。
    """

    return PlantDiagnosisAPIClient(
        base_url=API_BASE_URL,
        api_token=API_TOKEN,
    )


api_client = get_api_client()


# ============================================================
# 圖片與任務輔助函式
# ============================================================

def calculate_upload_fingerprint(
    uploaded_files,
) -> str:
    """
    計算上傳圖片的 SHA-256 指紋。

    用途：
    使用者換了一組圖片時，自動清除上一組辨識結果。
    """

    hasher = hashlib.sha256()

    for uploaded_file in uploaded_files:
        hasher.update(
            uploaded_file.name.encode("utf-8")
        )
        hasher.update(
            uploaded_file.getvalue()
        )

    return hasher.hexdigest()


def validate_uploaded_images(
    uploaded_files,
) -> None:
    """在傳給 API 前先確認上傳內容確實是圖片。"""

    for index, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        suffix = (
            Path(uploaded_file.name)
            .suffix
            .lower()
        )

        if suffix not in ALLOWED_SUFFIX_TO_MIME:
            raise ValueError(
                f"第 {index} 張圖片格式不支援"
            )

        try:
            image_bytes = uploaded_file.getvalue()

            with Image.open(
                BytesIO(image_bytes)
            ) as image:
                image.verify()

        except (
            UnidentifiedImageError,
            OSError,
        ) as error:
            raise ValueError(
                f"第 {index} 張圖片無法解析或已損壞"
            ) from error


def save_uploaded_files(
    uploaded_files,
    batch_id: str,
) -> tuple[list[Path], list[dict]]:
    """
    保存圖片到本機圖片伺服器目錄。

    Returns:
        hosted_paths：
            實際保存的本機檔案。

        image_inputs：
            POST /v1/tasks 需要的 images 資料。
    """

    UPLOAD_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    hosted_paths: list[Path] = []
    image_inputs: list[dict] = []

    expires_at = (
        datetime.now(timezone.utc)
        + timedelta(
            minutes=SIGNED_URL_VALIDITY_MINUTES
        )
    ).isoformat()

    for index, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        suffix = (
            Path(uploaded_file.name)
            .suffix
            .lower()
        )

        mime_type = ALLOWED_SUFFIX_TO_MIME.get(
            suffix
        )

        if mime_type is None:
            raise ValueError(
                f"{uploaded_file.name} 格式不支援"
            )

        # 使用 UUID 產生檔名，避免不同使用者互相覆蓋。
        hosted_name = (
            f"{batch_id}_{index}{suffix}"
        )

        hosted_path = (
            UPLOAD_DIRECTORY / hosted_name
        )

        hosted_path.write_bytes(
            uploaded_file.getvalue()
        )

        image_url = (
            f"{IMAGE_BASE_URL}/"
            f"{quote(hosted_name)}"
        )

        image_inputs.append(
            {
                "image_url": image_url,
                "expires_at": expires_at,
                "mime_type": mime_type,
            }
        )

        hosted_paths.append(hosted_path)

    return hosted_paths, image_inputs


def remove_hosted_files(
    hosted_paths: list[Path],
) -> None:
    """任務結束後清除 Streamlit 暫存圖片。"""

    for hosted_path in hosted_paths:
        try:
            hosted_path.unlink(
                missing_ok=True
            )

        except OSError:
            # 清理失敗不影響已完成的模型結果。
            pass


def get_attribute_value(
    item: dict,
    key: str,
) -> str | None:
    """從 API item.attributes 取得指定 key 的 value。"""

    for attribute in item.get(
        "attributes",
        [],
    ):
        if attribute.get("key") == key:
            return attribute.get("value")

    return None


def get_diagnosis_items(
    result: dict,
) -> list[dict]:
    """從 result_blocks 找出 diagnosis items。"""

    for block in result.get(
        "result_blocks",
        [],
    ):
        if block.get("type") == "diagnosis":
            return block.get("items", [])

    return []


def convert_api_result_to_predictions(
    result: dict,
    top_k: int,
) -> list[tuple[str, float]]:
    """
    將 API 結果轉成學長姐原本畫面使用的格式。

    原本畫面格式：
        [
            ("algal_leaf_spot", 99),
            ("rust", 1),
        ]
    """

    diagnosis_items = get_diagnosis_items(
        result
    )

    predictions = []

    for item in diagnosis_items[:top_k]:
        class_name = get_attribute_value(
            item,
            "class_name",
        )

        # 正常情況會有 class_name。
        # 若沒有，暫時以中文 label 作為顯示名稱。
        if not class_name:
            class_name = item.get(
                "label",
                "unknown",
            )

        confidence = float(
            item.get("confidence", 0)
        )

        predictions.append(
            (
                class_name,
                confidence,
            )
        )

    return predictions


def run_diagnosis(
    uploaded_files,
    plant: str,
    user_description: str,
    location: str,
    environment: str,
) -> None:
    """
    建立 FastAPI 任務並輪詢到完成。

    使用的 API：
    1. GET /v1/health
    2. POST /v1/tasks
    3. GET /v1/tasks/{task_id}
    """

    if not api_client.health_check():
        st.session_state.latest_result = None
        st.session_state.latest_task_response = None
        st.session_state.latest_error = (
            "FastAPI 目前無法使用，請確認服務已啟動，"
            "且 ACADEMY_API_TOKEN 設定正確。"
        )
        return

    batch_id = uuid4().hex

    # 每一個新案件使用新的 platform_task_id。
    platform_task_id = (
        f"streamlit_{batch_id}"
    )

    hosted_paths: list[Path] = []
    task_created = False
    terminal_reached = False

    try:
        validate_uploaded_images(
            uploaded_files
        )

        hosted_paths, image_inputs = (
            save_uploaded_files(
                uploaded_files=uploaded_files,
                batch_id=batch_id,
            )
        )

        # 呼叫 POST /v1/tasks。
        created_task = api_client.create_task(
            platform_task_id=platform_task_id,
            images=image_inputs,
            plant=plant,
            context={
                "user_description": (
                    user_description or None
                ),
                "location": location or None,
                "environment": environment or None,
            },
        )

        task_id = created_task["task_id"]
        task_created = True

        st.session_state.latest_task_id = (
            task_id
        )
        st.session_state.latest_result = None
        st.session_state.latest_task_response = (
            created_task
        )
        st.session_state.latest_error = None

        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        status_placeholder.info(
            "任務已建立，正在等待模型處理"
        )

        deadline = (
            time.monotonic()
            + TASK_TIMEOUT_SECONDS
        )

        # 會議要求建立任務後至少等待 1 秒，
        # 才開始第一次 GET。
        time.sleep(
            INITIAL_POLL_DELAY_SECONDS
        )

        while time.monotonic() < deadline:
            # 呼叫 GET /v1/tasks/{task_id}。
            task_data = api_client.get_task(
                task_id
            )
            st.session_state.latest_task_response = (
                task_data
            )

            status = task_data["status"]

            progress = int(
                task_data.get(
                    "progress",
                    0,
                )
            )

            # 防止異常資料造成 Streamlit progress 報錯。
            progress = max(
                0,
                min(progress, 100),
            )

            if status == "PENDING":
                status_placeholder.info(
                    "任務正在排隊，請稍候"
                )

                progress_bar.progress(
                    progress
                )

            elif status == "PROCESSING":
                status_placeholder.info(
                    f"AI 正在分析圖片：{progress}%"
                )

                progress_bar.progress(
                    progress
                )

            elif status == "SUCCEEDED":
                terminal_reached = True

                progress_bar.progress(100)

                status_placeholder.success(
                    "AI 分析完成"
                )

                st.session_state.latest_result = (
                    task_data["result"]
                )

                st.session_state.latest_error = None
                return

            elif status == "FAILED":
                terminal_reached = True

                error = (
                    task_data.get("error")
                    or {}
                )

                st.session_state.latest_result = None

                st.session_state.latest_error = (
                    f"{error.get('code', 'UNKNOWN_ERROR')}："
                    f"{error.get('message', '辨識失敗')}"
                )
                return

            else:
                raise APIClientError(
                    code="UNKNOWN_TASK_STATUS",
                    message=(
                        f"收到未知任務狀態：{status}"
                    ),
                )

            # 後續每 2 秒查詢一次。
            time.sleep(
                POLL_INTERVAL_SECONDS
            )

        raise APIClientError(
            code="TASK_TIMEOUT",
            message="等待模型結果超過 30 分鐘",
        )

    except (
        APIClientError,
        ValueError,
    ) as error:
        st.session_state.latest_result = None

        if isinstance(
            error,
            APIClientError,
        ):
            message = (
                f"{error.code}："
                f"{error.message}"
            )
        else:
            message = str(error)

        st.session_state.latest_error = (
            message
        )

    finally:
        # 任務已完成、已失敗，或 POST 根本沒有成功時，
        # 可以刪除本機暫存圖片。
        #
        # 如果任務已建立但前端輪詢逾時，
        # Worker 可能仍在等待下載圖片，所以暫時保留。
        if terminal_reached or not task_created:
            remove_hosted_files(
                hosted_paths
            )


# ============================================================
# 結果顯示
# ============================================================

def display_api_result(
    result: dict,
    task_response: dict,
    image_count: int,
    top_k: int,
    confidence_threshold: int,
) -> None:
    """使用學長姐原本的畫面風格顯示 API 結果。"""

    summary = result.get(
        "summary",
        {},
    )

    diagnosis_items = get_diagnosis_items(
        result
    )

    # 低信心或影像資訊不足時，API 會在此欄位提供
    # 需要使用者補拍或補充的內容。必須在本函式內取得，
    # 才能於下方顯示而不會發生變數作用域錯誤。
    need_more_info = result.get(
        "need_more_info",
        [],
    )

    predictions = (
        convert_api_result_to_predictions(
            result=result,
            top_k=top_k,
        )
    )

    if not predictions:
        st.error(
            "API 已完成任務，但沒有回傳 diagnosis 結果。"
        )
        return

    best_class, best_probability = (
        predictions[0]
    )

    best_item = (
        diagnosis_items[0]
        if diagnosis_items
        else {}
    )

    best_class_zh = best_item.get("label", best_class)

    if best_class == best_class_zh:
        display_name = best_class_zh
    else:
        display_name = (
            f"{best_class_zh} ({best_class})"
        )

    if best_class_zh == "無法明確判定":
        result_title = "資訊不足"
    elif (
        best_probability
        >= confidence_threshold
    ):
        result_title = "診斷結果"
    else:
        result_title = (
            "可能診斷（信心度較低）"
        )

    st.subheader("診斷結果")

    st.markdown(
        f"""
        <div class="green-result-box">
            {html.escape(result_title)}：
            {html.escape(display_name)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    first_metric, second_metric = st.columns(2)

    with first_metric:
        st.metric(
            label="診斷信心度",
            value=(
                f"{best_probability:.0f}%"
            ),
            delta=(
                f"{best_probability - confidence_threshold:.0f}% "
                "vs 閾值"
            ),
        )

    with second_metric:
        st.metric(
            label="分析圖片數量",
            value=image_count,
        )

    summary_description = (
        summary.get("description")
    )

    if summary_description:
        st.info(
            summary_description
        )

    if need_more_info:
        st.markdown("---")
        st.markdown("### 建議補充資訊")

        for information in need_more_info:
            description = information.get(
                "description"
            )

            if description:
                st.warning(description)

    st.markdown("---")
    st.markdown("### 判斷依據")

    # 正式規格中的 description 是模型判斷依據，
    # 不可再把它當作防治或管理建議。
    diagnosis_basis = best_item.get(
        "description",
        "API 未提供判斷依據。",
    )

    st.markdown(
        f"""
        <div class="recommendation-box">
            {html.escape(diagnosis_basis)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # 詳細分析表格
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("詳細分析")

    table_rows = []

    for rank, item in enumerate(
        diagnosis_items[:top_k],
        start=1,
    ):
        issue_type = get_attribute_value(
            item,
            "issue_type",
        )

        table_rows.append(
            {
                "排名": rank,
                "類別": item.get(
                    "label",
                    "未知",
                ),
                "問題類型": (
                    issue_type or "不適用"
                ),
                "信心度 (%)": float(
                    item.get(
                        "confidence",
                        0,
                    )
                ),
            }
        )

    dataframe = pd.DataFrame(
        table_rows
    )

    styled_dataframe = (
        dataframe.style
        .hide(axis="index")
        .format(
            {
                "信心度 (%)": "{:.0f}",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "",
                    "props": [
                        ("width", "100%"),
                        (
                            "border-collapse",
                            "separate",
                        ),
                        (
                            "border-spacing",
                            "0",
                        ),
                        (
                            "border-radius",
                            "6px",
                        ),
                        (
                            "overflow",
                            "hidden",
                        ),
                        (
                            "font-size",
                            "0.90rem",
                        ),
                    ],
                },
                {
                    "selector": "thead th",
                    "props": [
                        (
                            "background-color",
                            "#ebf1e5",
                        ),
                        ("color", "#000000"),
                        (
                            "font-weight",
                            "500",
                        ),
                        (
                            "text-align",
                            "right",
                        ),
                        (
                            "padding",
                            "0.6rem 1rem",
                        ),
                    ],
                },
                {
                    "selector": "tbody td",
                    "props": [
                        (
                            "background-color",
                            "#52663f",
                        ),
                        ("color", "#ffffff"),
                        (
                            "text-align",
                            "right",
                        ),
                        (
                            "padding",
                            "0.55rem 1rem",
                        ),
                        (
                            "border-top",
                            "1px solid #768f5f",
                        ),
                    ],
                },
            ]
        )
    )

    st.markdown(
        styled_dataframe.to_html(),
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # 信心度長條圖
    # --------------------------------------------------------

    chart_dataframe = dataframe[
        [
            "類別",
            "信心度 (%)",
        ]
    ]

    chart = (
        alt.Chart(chart_dataframe)
        .mark_bar(
            color="#3b4f32"
        )
        .encode(
            x=alt.X(
                "類別:N",
                sort="-y",
                axis=alt.Axis(
                    title=None,
                    labelAngle=0,
                    labelFontSize=14,
                ),
            ),
            y=alt.Y(
                "信心度 (%):Q",
                scale=alt.Scale(
                    domain=[0, 100]
                ),
                axis=alt.Axis(
                    title=None
                ),
            ),
            tooltip=[
                "類別:N",
                alt.Tooltip(
                    "信心度 (%):Q",
                    format=".0f",
                ),
            ],
        )
        .properties(
            height=260,
            background="#52663f",
            padding={
                "left": 20,
                "right": 25,
                "top": 10,
                "bottom": 8,
            },
        )
        .configure_view(
            strokeWidth=0
        )
        .configure_axis(
            grid=True,
            gridColor="#768f5f",
            gridOpacity=0.6,
            labelColor="#ffffff",
            tickColor="#ffffff",
        )
    )

    st.altair_chart(
        chart,
        width="stretch",
    )

    # --------------------------------------------------------
    # 完整 API 回應
    # --------------------------------------------------------

    with st.expander(
        "完整 API 回應（開發測試）"
    ):
        # 顯示 GET /v1/tasks/{task_id} 的完整外層回應，
        # 包含 task_id、status 與 result，方便逐欄對照規格書。
        st.json(task_response)


# ============================================================
# 側邊欄
# ============================================================

with st.sidebar:
    st.header("系統資訊")

    # 放在案件表單之外，切換植物可立即更新側欄與清除舊結果。
    selected_plant = st.selectbox(
        "辨識植物",
        options=list(PLANT_OPTIONS),
        format_func=lambda code: PLANT_OPTIONS[code],
        index=0,
        key="selected_plant",
    )

    if api_client.health_check():
        st.success("FastAPI：正常")
    else:
        st.error(
            "FastAPI：無法使用（連線或認證失敗）"
        )

    st.caption(
        f"API 網址：{API_BASE_URL}"
    )

    with st.expander(
        "支援辨識類別",
        expanded=False,
    ):
        for index, (class_name, chinese_name) in enumerate(
            PLANT_CLASSES[selected_plant], start=1
        ):
            st.write(
                f"{index}. "
                f"{chinese_name} "
                f"({class_name})"
            )

    st.markdown("---")
    st.subheader("顯示設定")

    # API 目前只回傳前三名，
    # 所以前端最多顯示 3 筆。
    top_k = st.slider(
        "顯示前 K 個結果",
        min_value=1,
        max_value=MAX_DISPLAY_RESULTS,
        value=3,
    )

    confidence_threshold = st.slider(
        "信心度閾值 (%)",
        min_value=0,
        max_value=100,
        value=50,
        help="低於此值時會顯示信心度較低提醒",
    )

    st.markdown("---")

    st.info(
        "本機測試使用圖片伺服器模擬 Signed URL。"
        "正式環境將由一站式平台提供圖片網址。"
    )

    spy_image_path = (
        PROJECT_ROOT / "spy.PNG"
    )

    if spy_image_path.exists():
        st.image(
            str(spy_image_path),
            width="stretch",
        )


# ============================================================
# 主畫面
# ============================================================

st.title("🌿 植物病蟲害智慧辨識系統")

st.caption(
    "Streamlit → FastAPI → DINOv3 非同步辨識"
)

st.info(
    f"目前選擇：{PLANT_OPTIONS[selected_plant]}。"
    "請上傳該植物同一植株的照片。"
)

uploaded_files = st.file_uploader(
    "上傳同一植株的 1～5 張照片",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    accept_multiple_files=True,
    help=(
        "請上傳同一植株、同一病徵問題的不同角度照片。"
    ),
    label_visibility="hidden",
)


# ============================================================
# 已選擇圖片
# ============================================================

if uploaded_files:
    if len(uploaded_files) > 5:
        st.error(
            "每次任務最多只能上傳 5 張圖片。"
        )
        st.stop()

    fingerprint = selection_fingerprint(
        calculate_upload_fingerprint(uploaded_files),
        selected_plant,
    )

    # 使用者換了一組圖片時，清除上一組任務結果。
    if (
        st.session_state.upload_fingerprint
        != fingerprint
    ):
        st.session_state.upload_fingerprint = (
            fingerprint
        )
        clear_task_state(st.session_state)

    st.subheader("上傳的圖片")

    preview_columns = st.columns(
        min(
            len(uploaded_files),
            3,
        )
    )

    for index, uploaded_file in enumerate(
        uploaded_files
    ):
        try:
            image = Image.open(
                BytesIO(
                    uploaded_file.getvalue()
                )
            )

        except (
            UnidentifiedImageError,
            OSError,
        ):
            st.error(
                f"{uploaded_file.name} 無法解析"
            )
            continue

        with preview_columns[
            index % len(preview_columns)
        ]:
            st.image(
                image,
                caption=uploaded_file.name,
                width="stretch",
            )

            safe_file_name = html.escape(
                uploaded_file.name
            )

            safe_image_format = html.escape(
                str(image.format)
            )

            safe_image_mode = html.escape(
                str(image.mode)
            )

            st.markdown(
                f"""
                <div class="image-info-box">
                    檔案：{safe_file_name}<br>
                    尺寸：{image.size[0]} × {image.size[1]} px<br>
                    格式：{safe_image_format}<br>
                    色彩：{safe_image_mode}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # 案件補充資料
    # --------------------------------------------------------

    st.markdown("---")
    st.subheader("案件資訊")

    # 將案件資料與送出按鈕放在同一個表單。
    # 使用者輸入文字或切換欄位時不會重新執行整個頁面，
    # 只有按下「開始 AI 辨識」後才會一次提交所有資料。
    with st.form("diagnosis_form"):
        user_description = st.text_area(
            "症狀描述",
            placeholder=(
                "例如：葉片表面出現黑色斑點"
            ),
        )

        first_input, second_input = st.columns(2)

        with first_input:
            location = st.text_input(
                "地點",
                placeholder="例如：屏東縣",
            )

        with second_input:
            environment = st.text_input(
                "種植環境",
                placeholder="例如：露天、溫室",
            )

        submitted = st.form_submit_button(
            "開始 AI 辨識",
            type="primary",
            width="stretch",
        )

    # --------------------------------------------------------
    # 開始辨識
    # --------------------------------------------------------

    if submitted:
        run_diagnosis(
            uploaded_files=uploaded_files,
            plant=selected_plant,
            user_description=user_description,
            location=location,
            environment=environment,
        )

    # --------------------------------------------------------
    # 任務資訊與錯誤
    # --------------------------------------------------------

    if st.session_state.latest_task_id:
        st.caption(
            "任務編號："
            f"{st.session_state.latest_task_id}"
        )

    if st.session_state.latest_error:
        st.error(
            st.session_state.latest_error
        )

    # --------------------------------------------------------
    # 顯示成功結果
    # --------------------------------------------------------

    if st.session_state.latest_result:
        st.markdown("---")

        display_api_result(
            result=(
                st.session_state.latest_result
            ),
            task_response=(
                st.session_state.latest_task_response
                or {
                    "task_id": (
                        st.session_state.latest_task_id
                    ),
                    "status": "SUCCEEDED",
                    "result": (
                        st.session_state.latest_result
                    ),
                }
            ),
            image_count=len(uploaded_files),
            top_k=top_k,
            confidence_threshold=(
                confidence_threshold
            ),
        )

else:
    st.markdown(
        """
        <p style="
            text-align: center;
            color: #ffffff;
            background-color: #3b4f32;
            border-radius: 10px;
            padding: 0.6rem 1rem;
        ">
            請上傳 1～5 張圖片開始診斷
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("使用說明"):
        st.markdown(
            """
            ### 使用方法

            1. 上傳同一植株的 1～5 張照片。
            2. 可以填寫症狀、地點及種植環境。
            3. 按下「開始 AI 辨識」。
            4. Streamlit 會建立 FastAPI 任務。
            5. 系統會定期查詢任務狀態。
            6. 完成後顯示診斷、信心度及建議。

            ### 拍攝建議

            - 確保光線充足。
            - 對焦在主要病徵。
            - 避免照片過度模糊。
            - 可從不同角度拍攝同一植株。
            - 不要在同一任務混入不同植株的照片。

            ### 支援格式

            - JPEG
            - PNG
            """
        )


# ============================================================
# 頁尾
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align: center;
        color: #000000;
        padding: 1rem;
    ">
        <p>植物病蟲害智慧辨識系統</p>
        <p>使用 DINOv3 深度學習模型</p>
        <p>NPUST DN-LAB 2026</p>
    </div>
    """,
    unsafe_allow_html=True,
)
