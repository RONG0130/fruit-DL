"""
API 圖片下載與驗證工具。

負責：
1. 從平台提供的 URL 下載圖片。
2. 限制單張圖片大小。
3. 使用 Pillow 確認檔案確實是 JPEG 或 PNG。
4. 防止錯誤檔案進入模型。
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

import httpx
from PIL import Image, UnidentifiedImageError


# 單張圖片最大容量，預設 10 MB。
MAX_IMAGE_BYTES = int(
    os.getenv(
        "MAX_IMAGE_BYTES",
        str(10 * 1024 * 1024),
    )
)

# 圖片最大像素數，例如 5000 × 5000 = 2,500 萬像素。
MAX_IMAGE_PIXELS = int(
    os.getenv(
        "MAX_IMAGE_PIXELS",
        "25000000",
    )
)

ALLOWED_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


class ImageDownloadError(Exception):
    """可提供給背景 Worker 使用的圖片錯誤。"""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def validate_downloaded_image(
    image_path: Path,
    expected_mime_type: str,
    image_number: int,
) -> None:
    """使用 Pillow 驗證圖片內容、格式及像素數。"""

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            actual_mime_type = Image.MIME.get(
                image.format
            )

            if width * height > MAX_IMAGE_PIXELS:
                raise ImageDownloadError(
                    code="IMAGE_TOO_LARGE",
                    message=(
                        f"第 {image_number} 張圖片像素過大，"
                        f"目前尺寸為 {width} × {height}"
                    ),
                )

            # verify() 會檢查圖片資料是否損壞。
            image.verify()

    except ImageDownloadError:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ) as error:
        raise ImageDownloadError(
            code="INVALID_IMAGE",
            message=f"第 {image_number} 張圖片無法解析或已損壞",
        ) from error

    if actual_mime_type not in ALLOWED_MIME_TYPES:
        raise ImageDownloadError(
            code="UNSUPPORTED_IMAGE_TYPE",
            message=(
                f"第 {image_number} 張圖片不是支援的 "
                "JPEG 或 PNG"
            ),
        )

    if actual_mime_type != expected_mime_type:
        raise ImageDownloadError(
            code="MIME_TYPE_MISMATCH",
            message=(
                f"第 {image_number} 張圖片實際格式"
                f"為 {actual_mime_type}，"
                f"但請求填寫的是 {expected_mime_type}"
            ),
        )


async def download_images(
    images: list[dict],
    target_directory: Path,
    progress_callback: (
        Callable[[int, int], Awaitable[None]]
        | None
    ) = None,
) -> list[Path]:
    """
    下載所有輸入圖片。

    Args:
        images:
            從 SQLite 讀出的圖片資料。

        target_directory:
            本次任務使用的暫存資料夾。

        progress_callback:
            每完成一張圖片後呼叫，參數依序為
            已完成張數及總張數。未提供時不回報進度。

    Returns:
        已下載且驗證成功的圖片路徑。
    """

    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    downloaded_paths: list[Path] = []

    timeout = httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=30.0,
        pool=5.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={
            "User-Agent": "fruit-dl-api/1.0",
        },
    ) as client:
        for index, image_info in enumerate(
            images,
            start=1,
        ):
            expires_at_value = image_info.get(
                "expires_at"
            )

            if isinstance(
                expires_at_value,
                datetime,
            ):
                expires_at = expires_at_value
            else:
                try:
                    expires_at = datetime.fromisoformat(
                        str(expires_at_value)
                    )
                except (
                    TypeError,
                    ValueError,
                ) as error:
                    raise ImageDownloadError(
                        code="IMAGE_DOWNLOAD_FAILED",
                        message=(
                            f"第 {index} 張圖片的連結"
                            "到期時間無法解析"
                        ),
                    ) from error

            if (
                expires_at.tzinfo is None
                or expires_at
                <= datetime.now(timezone.utc)
            ):
                raise ImageDownloadError(
                    code="IMAGE_DOWNLOAD_FAILED",
                    message=(
                        f"第 {index} 張圖片連結已過期"
                    ),
                )

            image_url = str(
                image_info["image_url"]
            )

            expected_mime_type = (
                image_info["mime_type"]
                .lower()
                .strip()
            )

            suffix = ALLOWED_MIME_TYPES[
                expected_mime_type
            ]

            destination = (
                target_directory
                / f"image_{index}{suffix}"
            )

            try:
                async with client.stream(
                    "GET",
                    image_url,
                ) as response:
                    response.raise_for_status()

                    # 若伺服器有提供 Content-Length，
                    # 可以在下載前先拒絕過大的檔案。
                    content_length = response.headers.get(
                        "content-length"
                    )

                    if content_length:
                        try:
                            announced_size = int(
                                content_length
                            )
                        except ValueError:
                            announced_size = 0

                        if announced_size > MAX_IMAGE_BYTES:
                            raise ImageDownloadError(
                                code="IMAGE_TOO_LARGE",
                                message=(
                                    f"第 {index} 張圖片超過"
                                    "單張 10 MB 限制"
                                ),
                            )

                    downloaded_size = 0

                    with destination.open("wb") as file:
                        async for chunk in response.aiter_bytes():
                            downloaded_size += len(chunk)

                            if downloaded_size > MAX_IMAGE_BYTES:
                                raise ImageDownloadError(
                                    code="IMAGE_TOO_LARGE",
                                    message=(
                                        f"第 {index} 張圖片超過"
                                        "單張 10 MB 限制"
                                    ),
                                )

                            file.write(chunk)

            except ImageDownloadError:
                raise

            except httpx.TimeoutException as error:
                raise ImageDownloadError(
                    code="IMAGE_DOWNLOAD_TIMEOUT",
                    message=f"第 {index} 張圖片下載逾時",
                ) from error

            except httpx.HTTPStatusError as error:
                raise ImageDownloadError(
                    code="IMAGE_DOWNLOAD_FAILED",
                    message=(
                        f"第 {index} 張圖片下載失敗，"
                        f"HTTP 狀態碼為 "
                        f"{error.response.status_code}"
                    ),
                ) from error

            except httpx.HTTPError as error:
                raise ImageDownloadError(
                    code="IMAGE_DOWNLOAD_FAILED",
                    message=f"第 {index} 張圖片下載失敗",
                ) from error

            validate_downloaded_image(
                image_path=destination,
                expected_mime_type=expected_mime_type,
                image_number=index,
            )

            downloaded_paths.append(destination)

            if progress_callback is not None:
                await progress_callback(
                    index,
                    len(images),
                )

    return downloaded_paths
