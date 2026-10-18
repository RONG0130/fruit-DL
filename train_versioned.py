#!/usr/bin/env python3
"""
不覆蓋歷史紀錄的版本化模型訓練入口。

輸出內容：
- best_model.pth：驗證準確率最高的模型
- last_model.pth：最後一個 epoch 的模型
- checkpoint_epoch_N.pth：依 save-every 選擇性保存
- classes.json：類別索引
- effective_config.yaml：實際訓練設定
- metrics.csv：每個 epoch 的指標
- summary.json：訓練結果摘要
- run_status.json：RUNNING / SUCCEEDED / FAILED
- evaluation/validation_report.txt：最佳模型的驗證集分類報告
- evaluation/confusion_matrix.png：最佳模型的驗證集混淆矩陣
"""

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import timm
from timm.data import resolve_data_config

from train import (
    create_dataloaders,
    create_model,
    get_data_transforms,
    load_config,
    set_seed,
    train_one_epoch,
    validate,
)
from training_device import resolve_device_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="版本化植物病蟲害模型訓練"
    )
    parser.add_argument(
        "--plant",
        required=True,
        help="作物英文識別名稱，例如 tea、annona、legume",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="包含 train、val、test 的資料集目錄",
    )
    parser.add_argument(
        "--run-name",
        default="baseline",
        help="本次訓練名稱，預設 baseline",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="所有訓練紀錄的根目錄，預設 runs",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="訓練設定檔，預設 config.yaml",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="timm 模型名稱",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help=(
            "每隔幾個 epoch 保存額外 checkpoint；"
            "0 表示只保存 best 與 last"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help=(
            "訓練裝置，預設 cuda；CUDA 不可用時會停止，"
            "不會自動改用 CPU"
        ),
    )
    return parser.parse_args()


def safe_name(value):
    normalized = value.strip().lower()

    if not re.fullmatch(r"[a-z0-9_-]+", normalized):
        raise ValueError(
            "plant 與 run-name 只能包含英文字母、數字、"
            "底線及連字號"
        )

    return normalized


def create_run_directory(runs_root, plant, run_name):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    plant_name = safe_name(plant)
    safe_run_name = safe_name(run_name)

    run_dir = (
        Path(runs_root)
        / plant_name
        / f"{timestamp}_{safe_run_name}"
    )

    # exist_ok=False 是避免覆蓋的核心。
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path, data):
    Path(path).write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_classes(classes, run_dir):
    class_mapping = {
        str(index): class_name
        for index, class_name in enumerate(classes)
    }
    write_json(run_dir / "classes.json", class_mapping)


def save_effective_config(run_dir, settings):
    config_path = run_dir / "effective_config.yaml"

    config_path.write_text(
        yaml.safe_dump(
            settings,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def append_metrics(
    metrics_path,
    epoch,
    train_loss,
    train_accuracy,
    val_loss,
    val_accuracy,
):
    file_exists = metrics_path.exists()

    with metrics_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "val_loss",
                    "val_accuracy",
                ]
            )

        writer.writerow(
            [
                epoch,
                round(train_loss, 6),
                round(train_accuracy, 4),
                round(val_loss, 6),
                round(val_accuracy, 4),
            ]
        )


def build_checkpoint(
    model,
    classes,
    model_name,
    plant,
    epoch,
    val_accuracy,
):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "accuracy": val_accuracy,
        "classes": classes,
        "model_name": model_name,
        "plant": plant,
    }


def save_checkpoints(
    checkpoint,
    run_dir,
    epoch,
    is_best,
    save_every,
):
    # 只會覆寫同一個 run 內的 last_model。
    torch.save(
        checkpoint,
        run_dir / "last_model.pth",
    )

    if is_best:
        torch.save(
            checkpoint,
            run_dir / "best_model.pth",
        )

    if save_every > 0 and epoch % save_every == 0:
        torch.save(
            checkpoint,
            run_dir / f"checkpoint_epoch_{epoch}.pth",
        )


def validate_dataset(data_dir):
    data_path = Path(data_dir).resolve()
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    for required_dir in (train_dir, val_dir, test_dir):
        if not required_dir.is_dir():
            raise FileNotFoundError(
                f"資料集缺少必要目錄：{required_dir}"
            )

    train_classes = sorted(
        path.name
        for path in train_dir.iterdir()
        if path.is_dir()
    )
    val_classes = sorted(
        path.name
        for path in val_dir.iterdir()
        if path.is_dir()
    )
    test_classes = sorted(
        path.name
        for path in test_dir.iterdir()
        if path.is_dir()
    )

    if not train_classes:
        raise ValueError("訓練集沒有任何類別")

    if train_classes != val_classes:
        raise ValueError(
            "train 與 val 的類別不一致："
            f"{train_classes} != {val_classes}"
        )

    if train_classes != test_classes:
        raise ValueError(
            "train 與 test 的類別不一致："
            f"{train_classes} != {test_classes}"
        )

    return data_path, train_classes


def generate_validation_reports(model, val_loader, device, run_dir, classes):
    """以最佳權重重新評估驗證集，直接輸出文字與 PNG，無需圖形視窗。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.metrics import classification_report, confusion_matrix

    from evaluate import evaluate

    checkpoint = torch.load(
        run_dir / "best_model.pth",
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["classes"] != list(classes):
        raise ValueError("最佳模型的類別順序與驗證集不一致")
    checkpoint_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.eval()

    val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, device)
    labels = list(range(len(classes)))
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    output_dir = run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "validation_report.txt"
    report_path.write_text(
        "Validation Summary\n"
        f"Checkpoint: best_model.pth (epoch {checkpoint_epoch})\n"
        f"Val Loss: {val_loss:.4f}\n"
        f"Val Accuracy: {val_accuracy:.2f}%\n\n"
        "Classification Report\n" + report,
        encoding="utf-8",
    )

    # 使用 Agg 畫布直接存 PNG，不建立 Tk 視窗。
    figure = Figure(figsize=(10, 8))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    heatmap = axes.imshow(matrix, interpolation="nearest", cmap="Greens")
    figure.colorbar(heatmap, ax=axes)
    axes.set(
        xticks=labels, yticks=labels,
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted label", ylabel="True label",
        title="Confusion Matrix (Validation, Best Model)",
    )
    for tick in axes.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
        tick.set_fontsize(8)
    axes.tick_params(axis="y", labelsize=8)
    threshold = matrix.max() / 2.0
    for row in labels:
        for column in labels:
            axes.text(
                column, row, str(matrix[row, column]),
                ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", bbox_inches="tight")
    figure.clear()
    return {
        "split": "val",
        "checkpoint": "best_model.pth",
        "checkpoint_epoch": checkpoint_epoch,
        "sample_count": len(y_true),
        "loss": float(val_loss),
        "accuracy": float(val_accuracy),
        "report_path": "evaluation/validation_report.txt",
        "confusion_matrix_path": "evaluation/confusion_matrix.png",
    }


def main():
    args = parse_args()

    data_dir, dataset_classes = validate_dataset(
        args.data_dir
    )

    config = load_config(args.config)
    train_config = config.get("train", {})

    model_name = (
        args.model_name
        or config.get("model", {}).get(
            "name",
            "vit_large_patch16_dinov3.lvd1689m",
        )
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else train_config.get("batch_size", 8)
    )
    epochs = (
        args.epochs
        if args.epochs is not None
        else train_config.get("epochs", 30)
    )
    learning_rate = (
        args.lr
        if args.lr is not None
        else train_config.get("lr", 0.0001)
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else train_config.get("weight_decay", 0.01)
    )

    if epochs <= 0:
        raise ValueError("epochs 必須大於 0")

    if batch_size <= 0:
        raise ValueError("batch-size 必須大於 0")

    if args.save_every < 0:
        raise ValueError("save-every 不可小於 0")

    run_dir = create_run_directory(
        runs_root=args.runs_root,
        plant=args.plant,
        run_name=args.run_name,
    )

    started_at = datetime.now(timezone.utc)

    effective_config = {
        "plant": safe_name(args.plant),
        "run_name": safe_name(args.run_name),
        "data_dir": str(data_dir),
        "dataset_classes": dataset_classes,
        "model": {
            "name": model_name,
        },
        "train": {
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "save_every": args.save_every,
            "device": args.device,
        },
        "source_config": args.config,
        "command": sys.argv,
        "started_at": started_at.isoformat(),
    }

    save_effective_config(
        run_dir,
        effective_config,
    )
    write_json(
        run_dir / "run_status.json",
        {
            "status": "RUNNING",
            "started_at": started_at.isoformat(),
        },
    )

    print(f"本次訓練目錄：{run_dir}")
    print(f"資料類別：{dataset_classes}")

    try:
        set_seed(args.seed)

        device_name = resolve_device_name(
            requested_device=args.device,
            cuda_available=torch.cuda.is_available(),
        )
        device = torch.device(device_name)
        print(f"使用裝置：{device}")

        temp_model = timm.create_model(
            model_name,
            pretrained=False,
        )
        data_config = resolve_data_config(
            {},
            model=temp_model,
        )
        del temp_model

        train_transform, val_transform = (
            get_data_transforms(data_config)
        )

        train_loader, val_loader, classes = (
            create_dataloaders(
                str(data_dir),
                train_transform,
                val_transform,
                batch_size,
                args.num_workers,
                config=config,
            )
        )

        if list(classes) != dataset_classes:
            raise ValueError(
                "DataLoader 類別順序與資料集不一致："
                f"{classes} != {dataset_classes}"
            )

        save_classes(classes, run_dir)

        model = create_model(
            model_name,
            len(classes),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        amp_enabled = device.type == "cuda"
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )

        metrics_path = run_dir / "metrics.csv"
        best_accuracy = float("-inf")
        best_epoch = None

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
            )

            val_loss, val_accuracy = validate(
                model,
                val_loader,
                criterion,
                device,
                epoch,
            )

            append_metrics(
                metrics_path=metrics_path,
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )

            is_best = val_accuracy > best_accuracy

            if is_best:
                best_accuracy = val_accuracy
                best_epoch = epoch

            checkpoint = build_checkpoint(
                model=model,
                classes=classes,
                model_name=model_name,
                plant=safe_name(args.plant),
                epoch=epoch,
                val_accuracy=val_accuracy,
            )

            save_checkpoints(
                checkpoint=checkpoint,
                run_dir=run_dir,
                epoch=epoch,
                is_best=is_best,
                save_every=args.save_every,
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss={train_loss:.4f} | "
                f"Train Acc={train_accuracy:.2f}% | "
                f"Val Loss={val_loss:.4f} | "
                f"Val Acc={val_accuracy:.2f}%"
            )

        # 先釋放訓練用的最佳化器狀態，避免評估時多占用 GPU 記憶體。
        del optimizer, scaler, checkpoint
        print("正在使用最佳模型產生驗證報告...")
        evaluation_result = generate_validation_reports(
            model, val_loader, device, run_dir, classes,
        )
        finished_at = datetime.now(timezone.utc)

        summary = {
            "status": "SUCCEEDED",
            "plant": safe_name(args.plant),
            "run_dir": str(run_dir.resolve()),
            "best_epoch": best_epoch,
            "best_val_accuracy": round(
                best_accuracy,
                4,
            ),
            "epochs": epochs,
            "classes": classes,
            "evaluation": evaluation_result,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        }

        write_json(
            run_dir / "summary.json",
            summary,
        )
        write_json(
            run_dir / "run_status.json",
            {
                "status": "SUCCEEDED",
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
            },
        )

        print("訓練完成")
        print(f"最佳 epoch：{best_epoch}")
        print(f"最佳驗證準確率：{best_accuracy:.2f}%")
        print(f"模型位置：{run_dir / 'best_model.pth'}")
        print(f"驗證報告：{run_dir / evaluation_result['report_path']}")
        print(f"混淆矩陣：{run_dir / evaluation_result['confusion_matrix_path']}")

    except Exception as error:
        failed_at = datetime.now(timezone.utc)

        write_json(
            run_dir / "run_status.json",
            {
                "status": "FAILED",
                "started_at": started_at.isoformat(),
                "failed_at": failed_at.isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
