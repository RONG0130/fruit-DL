#!/usr/bin/env python3
"""
不覆蓋歷史紀錄的版本化模型訓練入口。

輸出內容：
- best_model.pth：驗證集 Macro Avg Recall 最高的模型
- last_model.pth：最後一個 epoch 的模型
- checkpoint_epoch_N.pth：依 save-every 選擇性保存
- classes.json：類別索引
- effective_config.yaml：實際訓練設定
- metrics.csv：每個 epoch 的指標
- summary.json：訓練結果摘要
- run_status.json：RUNNING / SUCCEEDED / FAILED
- evaluation/validation_report.txt：最佳模型的驗證集分類報告
- evaluation/confusion_matrix.png：最佳模型的驗證集混淆矩陣
- evaluation/test_report.txt：最佳模型的獨立測試集分類報告
- evaluation/loss_curve.png：逐 epoch 的訓練與驗證 loss
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
from torch.utils.data import Subset

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
        "--samples-per-class",
        type=int,
        default=None,
        help="每個類別每個 epoch 的固定取樣張數；覆蓋 config 的重採樣設定",
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
    val_macro_recall,
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
                    "val_macro_recall",
                ]
            )

        writer.writerow(
            [
                epoch,
                round(train_loss, 6),
                round(train_accuracy, 4),
                round(val_loss, 6),
                round(val_accuracy, 4),
                round(val_macro_recall, 4),
            ]
        )


def build_checkpoint(
    model,
    classes,
    model_name,
    plant,
    epoch,
    val_accuracy,
    val_macro_recall,
):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "accuracy": val_accuracy,
        "macro_avg_recall": val_macro_recall,
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

    unknown_test_classes = sorted(set(test_classes) - set(train_classes))
    if unknown_test_classes:
        raise ValueError(
            "test 包含訓練模型未定義的類別："
            f"{unknown_test_classes}"
        )
    from evaluate import ExpectedClassImageFolder
    if not ExpectedClassImageFolder(test_dir, train_classes).samples:
        raise ValueError(f"test 集沒有任何圖片：{test_dir}")

    return data_path, train_classes


def compact_evaluation_result(summary):
    """Keep stable relative paths and headline metrics in the run summary."""
    return {
        key: summary[key]
        for key in (
            "split", "checkpoint_epoch", "sample_count", "loss", "accuracy",
            "macro_avg_recall",
        )
    } | {
        "checkpoint": Path(summary["checkpoint"]).name,
        "report_path": f"evaluation/{summary['report_path']}",
        "confusion_matrix_path": f"evaluation/{summary['confusion_matrix_path']}",
        "summary_path": f"evaluation/{summary['summary_path']}",
    }


def generate_validation_reports(model, val_loader, device, run_dir, classes,
                                plant=None, data_dir=None):
    """Restore the best checkpoint and write its validation report."""
    from evaluate import evaluate, write_evaluation_report

    checkpoint = torch.load(
        run_dir / "best_model.pth",
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["classes"] != list(classes):
        raise ValueError("最佳模型的類別順序與驗證集不一致")
    checkpoint_epoch = checkpoint["epoch"]
    plant = plant or checkpoint.get("plant")
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.eval()

    val_loss, val_accuracy, y_true, y_pred = evaluate(model, val_loader, device)
    summary, _ = write_evaluation_report(
        split="val", loss=val_loss, accuracy=val_accuracy,
        y_true=y_true, y_pred=y_pred, classes=classes,
        output_dir=run_dir / "evaluation",
        checkpoint_path=run_dir / "best_model.pth",
        checkpoint_epoch=checkpoint_epoch, plant=plant, data_dir=data_dir,
        report_name="validation_report.txt", matrix_name="confusion_matrix.png",
    )
    return compact_evaluation_result(summary)


def main():
    args = parse_args()

    data_dir, dataset_classes = validate_dataset(
        args.data_dir
    )

    config = load_config(args.config)
    if args.samples_per_class is not None:
        if args.samples_per_class <= 0:
            raise ValueError("samples-per-class 必須大於 0")
        config.setdefault("dataset", {})["resample"] = {
            "enabled": True,
            "method": "fixed_count",
            "target": args.samples_per_class,
            "per_class": True,
        }
    resample_config = config.get("dataset", {}).get("resample", {})
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
        "dataset": {
            "resample": dict(resample_config),
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
            "selection_metric": "val_macro_avg_recall",
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

    stage = "training"
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
                seed=args.seed,
            )
        )

        if list(classes) != dataset_classes:
            raise ValueError(
                "DataLoader 類別順序與資料集不一致："
                f"{classes} != {dataset_classes}"
            )

        source_dataset = (
            train_loader.dataset.dataset
            if isinstance(train_loader.dataset, Subset)
            else train_loader.dataset
        )
        source_train_images = len(source_dataset)
        train_samples_per_epoch = len(train_loader.sampler)
        print(
            f"訓練資料：原圖 {source_train_images} 張，"
            f"每個 epoch 取樣 {train_samples_per_epoch} 次"
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
        best_macro_recall = float("-inf")
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

            validation_result = validate(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                len(classes),
            )
            if len(validation_result) == 3:
                val_loss, val_accuracy, val_macro_recall = validation_result
            else:
                # Preserve compatibility with callers that replace validate with a two-value stub.
                val_loss, val_accuracy = validation_result
                from evaluate import evaluate, macro_avg_recall
                _, _, val_true, val_pred = evaluate(model, val_loader, device)
                val_macro_recall = macro_avg_recall(val_true, val_pred, len(classes))
                if val_macro_recall is None:
                    raise ValueError("驗證集有類別沒有圖片，無法選擇最佳模型")

            append_metrics(
                metrics_path=metrics_path,
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_macro_recall=val_macro_recall,
            )

            is_best = val_macro_recall > best_macro_recall

            if is_best:
                best_macro_recall = val_macro_recall
                best_accuracy = val_accuracy
                best_epoch = epoch

            checkpoint = build_checkpoint(
                model=model,
                classes=classes,
                model_name=model_name,
                plant=safe_name(args.plant),
                epoch=epoch,
                val_accuracy=val_accuracy,
                val_macro_recall=val_macro_recall,
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
                f"Val Acc={val_accuracy:.2f}% | "
                f"Val Macro Recall={val_macro_recall:.2f}%"
            )

        # 先釋放訓練用的最佳化器狀態，避免評估時多占用 GPU 記憶體。
        del optimizer, scaler, checkpoint
        print("正在使用最佳模型產生驗證報告...")
        stage = "validation_evaluation"
        evaluation_result = generate_validation_reports(
            model, val_loader, device, run_dir, classes,
            plant=safe_name(args.plant), data_dir=data_dir,
        )
        from evaluate import (
            create_eval_loader, evaluate, plot_loss_curve,
            write_evaluation_report,
        )

        stage = "loss_curve"
        loss_curve_path = run_dir / "evaluation" / "loss_curve.png"
        plot_loss_curve(metrics_path, loss_curve_path, best_epoch)

        stage = "test_evaluation"
        print("正在使用最佳模型產生獨立測試集報告...")
        test_loader, _ = create_eval_loader(
            data_dir, "test", val_transform, batch_size, args.num_workers,
            classes,
        )
        test_loss, test_accuracy, test_true, test_pred = evaluate(
            model, test_loader, device,
        )
        test_summary, _ = write_evaluation_report(
            split="test", loss=test_loss, accuracy=test_accuracy,
            y_true=test_true, y_pred=test_pred, classes=classes,
            output_dir=run_dir / "evaluation",
            checkpoint_path=run_dir / "best_model.pth",
            checkpoint_epoch=evaluation_result["checkpoint_epoch"],
            plant=safe_name(args.plant), data_dir=data_dir,
        )
        test_evaluation_result = compact_evaluation_result(test_summary)
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
            "best_val_macro_recall": round(best_macro_recall, 4),
            "selection_metric": "val_macro_avg_recall",
            "epochs": epochs,
            "classes": classes,
            "sampling": {
                "resample": dict(resample_config),
                "source_train_images": source_train_images,
                "train_samples_per_epoch": train_samples_per_epoch,
            },
            "evaluation": evaluation_result,
            "test_evaluation": test_evaluation_result,
            "loss_curve_path": "evaluation/loss_curve.png",
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
        print(f"最佳驗證 Macro Avg Recall：{best_macro_recall:.2f}%")
        print(f"模型位置：{run_dir / 'best_model.pth'}")
        print(f"驗證報告：{run_dir / evaluation_result['report_path']}")
        print(f"混淆矩陣：{run_dir / evaluation_result['confusion_matrix_path']}")
        print(f"測試報告：{run_dir / test_evaluation_result['report_path']}")
        print(f"Loss 曲線：{loss_curve_path}")

    except Exception as error:
        failed_at = datetime.now(timezone.utc)

        write_json(
            run_dir / "run_status.json",
            {
                "status": "FAILED",
                "started_at": started_at.isoformat(),
                "failed_at": failed_at.isoformat(),
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
