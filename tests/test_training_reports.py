import json
import sys

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import evaluate
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
    Image.new("RGB", (2, 2)).save(
        tmp_path / "dataset" / "test" / "alpha" / "sample.png"
    )
    model = torch.nn.Linear(2, 2, bias=False)
    loader = DataLoader(
        TensorDataset(torch.eye(2), torch.tensor([0, 1])), batch_size=2,
    )
    test_loader = DataLoader(
        TensorDataset(torch.eye(2), torch.tensor([1, 0])), batch_size=2,
    )
    monkeypatch.setattr(train_versioned.timm, "create_model", lambda *a, **k: model)
    monkeypatch.setattr(train_versioned, "resolve_data_config", lambda *a, **k: {})
    monkeypatch.setattr(train_versioned, "get_data_transforms", lambda config: (None, None))
    observed_resampling = []

    def create_loaders(*args, **kwargs):
        observed_resampling.append(kwargs["config"]["dataset"]["resample"].copy())
        return loader, loader, ["alpha", "beta"]

    monkeypatch.setattr(train_versioned, "create_dataloaders", create_loaders)
    monkeypatch.setattr(train_versioned, "create_model", lambda *a: model)
    test_splits = []

    def create_test_loader(data_dir, split, transform, batch_size, num_workers, classes):
        test_splits.append(split)
        return test_loader, classes

    monkeypatch.setattr(evaluate, "create_eval_loader", create_test_loader)

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
        "--samples-per-class", "3",
    ])

    train_versioned.main()

    run_dir = next((tmp_path / "runs" / "tea").iterdir())
    assert (run_dir / "evaluation" / "confusion_matrix.png").is_file()
    assert (run_dir / "evaluation" / "validation_report.txt").is_file()
    assert (run_dir / "evaluation" / "val_summary.json").is_file()
    assert (run_dir / "evaluation" / "test_report.txt").is_file()
    assert (run_dir / "evaluation" / "test_summary.json").is_file()
    assert (run_dir / "evaluation" / "test_confusion_matrix.png").is_file()
    with Image.open(run_dir / "evaluation" / "loss_curve.png") as image:
        assert image.format == "PNG"
        assert image.width > 100
    assert test_splits == ["test"]
    summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert summary["evaluation"]["accuracy"] == 100.0
    assert summary["evaluation"]["checkpoint_epoch"] == 1
    assert summary["test_evaluation"]["accuracy"] == 0.0
    assert summary["test_evaluation"]["checkpoint_epoch"] == 1
    assert summary["loss_curve_path"] == "evaluation/loss_curve.png"
    assert observed_resampling == [{
        "enabled": True, "method": "fixed_count", "target": 3, "per_class": True,
    }]
    effective = (run_dir / "effective_config.yaml").read_text("utf-8")
    assert "target: 3" in effective
    assert json.loads((run_dir / "run_status.json").read_text("utf-8"))["status"] == "SUCCEEDED"


def test_training_accepts_a_test_split_with_one_missing_class(tmp_path):
    for split in ("train", "val"):
        for label in ("alpha", "beta"):
            (tmp_path / split / label).mkdir(parents=True)
    (tmp_path / "test" / "alpha").mkdir(parents=True)
    Image.new("RGB", (2, 2)).save(tmp_path / "test" / "alpha" / "sample.png")

    path, classes = train_versioned.validate_dataset(tmp_path)

    assert path == tmp_path.resolve()
    assert classes == ["alpha", "beta"]


def test_training_rejects_an_empty_test_split_before_training(tmp_path):
    for split in ("train", "val", "test"):
        for label in ("alpha", "beta"):
            (tmp_path / split / label).mkdir(parents=True)

    with pytest.raises(ValueError, match="test 集沒有任何圖片"):
        train_versioned.validate_dataset(tmp_path)
