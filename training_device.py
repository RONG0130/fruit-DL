"""訓練裝置選擇與檢查。"""


def resolve_device_name(
    requested_device: str,
    cuda_available: bool,
) -> str:
    """回傳訓練裝置，CUDA 不可用時禁止靜默退回 CPU。"""

    if requested_device == "cuda":
        if not cuda_available:
            raise RuntimeError(
                "CUDA 無法使用。請確認已安裝 CUDA 版 PyTorch，並執行 "
                "python -c \"import torch; print(torch.__version__, "
                "torch.version.cuda, torch.cuda.is_available())\" 檢查環境。"
            )
        return "cuda"

    if requested_device == "cpu":
        return "cpu"

    raise ValueError(
        f"不支援的訓練裝置：{requested_device}"
    )
