import json
import sys

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import train_versioned


def test_reports_use_best_checkpoint_and_include_classes_without_samples(tmp_path):
    # 最後模型全判 alpha，最佳模型則將第二張圖片誤判成 beta。
    # 期待的 50% 可抓出報告誤用最後模型的錯誤。
    model = torch.nn.Linear(2, 3, bias=False)
    model.weight.data.zero_()
    best_weights = torch.tensor([[2., 0.], [0., 2.], [-2., -2.]])
    classes = ["alpha", "beta", "healthy"]
    torch.save(
        {"model_state_dict": {"weight": best_weights}, "classes": classes, "epoch": 1},
        tmp_path / "best_model.pth",
    )
    loader = DataLoader(
        TensorDataset(torch.eye(2), torch.tensor([0, 0])), batch_size=2,
    )

    generate = getattr(train_versioned, "generate_validation_reports", None)
    assert callable(generate), "訓練入口缺少自動產生驗證報告功能"
    result = generate(model, loader, torch.device("cpu"), tmp_path, classes)

    assert result["accuracy"] == 50.0
    assert result["sample_count"] == 2
    assert result["checkpoint_epoch"] == 1
    report = (tmp_path / "evaluation" / "validation_report.txt").read_text("utf-8")
    assert "Val Accuracy: 50.00%" in report
    assert "healthy" in report
    with Image.open(tmp_path / "evaluation" / "confusion_matrix.png") as image:
        assert image.format == "PNG"
        assert image.width > 100 and image.height > 100
    assert torch.equal(model.weight, best_weights)


def test_training_completion_automatically_generates_reports(tmp_path, monkeypatch):
    # 僅替代大型預訓練模型與 epoch 運算；完整執行入口、保存與報告流程。
    for split in ("train", "val", "test"):
        for label in ("alpha", "beta"):
            (tmp_path / "dataset" / split / label).mkdir(parents=True)
    model = torch.nn.Linear(2, 2, bias=False)
    loader = DataLoader(
        TensorDataset(torch.eye(2), torch.tensor([0, 1])), batch_size=2,
    )
    monkeypatch.setattr(train_versioned.timm, "create_model", lambda *a, **k: model)
    monkeypatch.setattr(train_versioned, "resolve_data_config", lambda *a, **k: {})
    monkeypatch.setattr(train_versioned, "get_data_transforms", lambda config: (None, None))
    monkeypatch.setattr(
        train_versioned, "create_dataloaders",
        lambda *a, **k: (loader, loader, ["alpha", "beta"]),
    )
    monkeypatch.setattr(train_versioned, "create_model", lambda *a: model)

    def train_epoch(*args):
        epoch = args[-1]
        model.weight.data.copy_(torch.eye(2) if epoch == 1 else -torch.eye(2))
        return 0.1, 100.0

    monkeypatch.setattr(train_versioned, "train_one_epoch", train_epoch)
    monkeypatch.setattr(
        train_versioned, "validate",
        lambda *args: (0.1, 100.0 if args[-1] == 1 else 0.0),
    )
    monkeypatch.setattr(sys, "argv", [
        "train_versioned.py", "--plant", "tea", "--device", "cpu",
        "--data-dir", str(tmp_path / "dataset"),
        "--runs-root", str(tmp_path / "runs"), "--epochs", "2",
        "--config", str(tmp_path / "missing.yaml"),
    ])

    train_versioned.main()

    run_dir = next((tmp_path / "runs" / "tea").iterdir())
    assert (run_dir / "evaluation" / "confusion_matrix.png").is_file()
    assert (run_dir / "evaluation" / "validation_report.txt").is_file()
    summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert summary["evaluation"]["accuracy"] == 100.0
    assert summary["evaluation"]["checkpoint_epoch"] == 1
    assert json.loads((run_dir / "run_status.json").read_text("utf-8"))["status"] == "SUCCEEDED"
