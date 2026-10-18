"""
真實模型背景 Worker。

處理順序：
1. 從 SQLite 取得任務。
2. 下載並驗證圖片。
3. 使用 DINOv3 執行推論。
4. 合併多張圖片的機率。
5. 保存 API 格式結果。
"""

import asyncio
import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from predict import PlantDiseasePredictor

from api import store
from api.image_downloader import (
    ImageDownloadError,
    download_images,
)
from api.result_mapper import build_api_result
from api.model_registry import ModelManager


LOGGER = logging.getLogger(__name__)

CLEANUP_INTERVAL_SECONDS = max(
    1,
    int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300")),
)


def merge_predictions(
    per_image_predictions: list[
        list[tuple[str, float]]
    ],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """
    將多張圖片的機率取平均。

    例如兩張圖片對炭疽病分別預測 80% 和 60%，
    合併後就是 70%。
    """

    if not per_image_predictions:
        raise ValueError("沒有可合併的圖片預測")

    probability_sums: dict[str, float] = (
        defaultdict(float)
    )

    for predictions in per_image_predictions:
        if not predictions:
            raise ValueError("其中一張圖片沒有預測結果")

        for class_name, probability in predictions:
            probability_sums[class_name] += probability

    image_count = len(per_image_predictions)

    averaged_predictions = [
        (
            class_name,
            probability_sum / image_count,
        )
        for class_name, probability_sum
        in probability_sums.items()
    ]

    averaged_predictions.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return averaged_predictions[:top_k]


def predict_single_image(
    predictor: PlantDiseasePredictor,
    image_path: Path,
) -> list[tuple[str, float]]:
    """同步執行單張圖片推論並回傳所有類別機率。"""

    return predictor.predict(
        image=image_path,
        # 多張圖片合併時需要所有類別的機率，
        # 不能在平均之前只取前三名。
        top_k=predictor.num_classes,
    )


@asynccontextmanager
async def predictor_for_task(predictor, plant: str):
    """正式服務由管理器選模；舊測試可直接傳輕量 predictor。"""
    if isinstance(predictor, ModelManager):
        async with predictor.use(plant) as selected:
            yield selected
    else:
        yield predictor


async def process_task(
    task_id: str,
    predictor: PlantDiseasePredictor | ModelManager,
) -> None:
    """下載圖片、執行模型並保存結果。"""

    task = await store.get_task(task_id)

    if task is None:
        return

    if task["status"] in {
        "SUCCEEDED",
        "FAILED",
    }:
        return

    try:
        await store.mark_processing(
            task_id,
            progress=5,
        )

        request_data = task["request"]
        image_inputs = request_data["images"]

        # 每個任務使用獨立暫存資料夾。
        # 離開 with 區塊後會自動刪除圖片。
        with TemporaryDirectory(
            prefix=f"{task_id}_"
        ) as temporary_directory:
            async def update_download_progress(
                completed_count: int,
                total_count: int,
            ) -> None:
                """依已下載圖片張數更新 5～35 的進度。"""

                progress = 5 + int(
                    completed_count
                    / total_count
                    * 30
                )

                await store.mark_processing(
                    task_id,
                    progress=progress,
                )

            image_paths = await download_images(
                images=image_inputs,
                target_directory=Path(
                    temporary_directory
                ),
                progress_callback=(
                    update_download_progress
                ),
            )

            # 所有圖片皆已下載及驗證完成，準備開始推論。
            await store.mark_processing(
                task_id,
                progress=40,
            )

            all_class_predictions = []
            image_count = len(image_paths)

            plant = request_data.get("task", {}).get("plant", "annona")
            # 整個多圖推論期間持有同一模型，其他請求不能切換 GPU。
            async with predictor_for_task(predictor, plant) as selected:
                for image_index, image_path in enumerate(image_paths, start=1):
                    predictions = await asyncio.to_thread(
                        predict_single_image, selected, image_path
                    )
                    all_class_predictions.append(predictions)
                    inference_progress = 40 + int(image_index / image_count * 45)
                    await store.mark_processing(task_id, progress=inference_progress)

            # 所有圖片推論完成，開始合併平均機率。
            await store.mark_processing(
                task_id,
                progress=90,
            )

            merged_predictions = merge_predictions(
                per_image_predictions=(
                    all_class_predictions
                ),
                top_k=3,
            )

            result = build_api_result(
                merged_predictions=(
                    merged_predictions
                ),
                image_count=len(image_paths),
                plant=plant,
            )

            # API 結果格式已建立，準備寫入 SQLite。
            await store.mark_processing(
                task_id,
                progress=95,
            )

            await store.mark_succeeded(
                task_id,
                result,
            )

    except asyncio.CancelledError:
        # API 關閉時保留未完成狀態，
        # 下次啟動會重新排入佇列。
        raise

    except ImageDownloadError as error:
        LOGGER.warning(
            "任務 %s 圖片處理失敗：%s",
            task_id,
            error.message,
        )

        await store.mark_failed(
            task_id,
            {
                # 對外錯誤碼必須符合平台契約。
                # error.code 仍保留在應用程式內部供日誌除錯，
                # 不直接暴露規格以外的錯誤碼。
                "code": "IMAGE_DOWNLOAD_FAILED",
                "message": error.message,
            },
        )

    except Exception:
        LOGGER.exception(
            "任務 %s 模型推論失敗",
            task_id,
        )

        await store.mark_failed(
            task_id,
            {
                "code": "INTERNAL_ERROR",
                "message": "模型服務異常",
            },
        )


async def worker_loop(
    task_queue: asyncio.Queue,
    predictor: PlantDiseasePredictor | ModelManager,
) -> None:
    """單一 Worker 依序處理任務。"""

    while True:
        task_id = await task_queue.get()

        try:
            await process_task(
                task_id=task_id,
                predictor=predictor,
            )
        finally:
            task_queue.task_done()


async def cleanup_loop() -> None:
    """定期刪除超過保存期限的完成任務。"""

    while True:
        await asyncio.sleep(
            CLEANUP_INTERVAL_SECONDS
        )

        deleted_count = (
            await store.delete_expired_tasks()
        )

        if deleted_count:
            LOGGER.info(
                "已刪除 %s 筆過期任務",
                deleted_count,
            )
