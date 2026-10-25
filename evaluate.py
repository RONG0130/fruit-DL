#!/usr/bin/env python3
"""Evaluate one plant model on its validation or independent test split."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


MIN_CLASS_SUPPORT_WARNING = 20


def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="評估單一作物模型")
    parser.add_argument("--plant", required=True, help="作物識別名稱，例如 annona、tea")
    parser.add_argument("--data-dir", required=True, help="該作物資料集根目錄，含 train/val/test")
    parser.add_argument("--model-path", required=True, help="該作物 run 的 best_model.pth")
    parser.add_argument("--model-name", default=None, help="timm 模型名稱；預設讀 checkpoint 或 config")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None, help="預設為模型所在目錄下的 evaluation/")
    return parser.parse_args()


def get_eval_transform(model_name: str):
    temporary_model = timm.create_model(model_name, pretrained=False)
    data_config = resolve_data_config({}, model=temporary_model)
    del temporary_model
    input_size = data_config["input_size"][-1]
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])


class ExpectedClassImageFolder(Dataset):
    """ImageFolder-style dataset with a fixed checkpoint label mapping."""

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

    def __init__(self, root, classes, transform=None):
        self.root = Path(root)
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.transform = transform
        found_dirs = {
            path.name for path in self.root.iterdir() if path.is_dir()
        }
        unknown = sorted(found_dirs - set(self.classes))
        if unknown:
            raise ValueError(f"資料集包含模型未定義的類別：{unknown}")
        self.samples = []
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in self.extensions:
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_eval_loader(data_dir, split, eval_transform, batch_size, num_workers, classes):
    split_dir = Path(data_dir) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"找不到 {split} 資料夾：{split_dir}")
    dataset = ExpectedClassImageFolder(split_dir, classes, transform=eval_transform)
    if not dataset.samples:
        raise ValueError(f"{split} 集沒有任何圖片：{split_dir}")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.classes


def load_model(model_name, num_classes, model_path, device, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def evaluate(model, data_loader, device):
    criterion = nn.CrossEntropyLoss()
    all_labels, all_predictions = [], []
    running_loss = 0.0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(1)
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
    if total == 0:
        raise ValueError("評估資料集沒有圖片")
    return (
        running_loss / total,
        accuracy_score(all_labels, all_predictions) * 100.0,
        np.asarray(all_labels),
        np.asarray(all_predictions),
    )


def macro_avg_recall(y_true, y_pred, class_count):
    """Return percent, or None when any expected class has no actual samples."""
    labels = list(range(class_count))
    supports = np.bincount(np.asarray(y_true, dtype=int), minlength=class_count)
    if len(supports) != class_count or np.any(supports == 0):
        return None
    return float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0) * 100.0)


def plot_confusion_matrix(matrix, class_names, output_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(10, 8))
    heatmap = axes.imshow(matrix, interpolation="nearest", cmap="Greens")
    figure.colorbar(heatmap, ax=axes)
    labels = list(range(len(class_names)))
    axes.set(
        xticks=labels, yticks=labels,
        xticklabels=class_names, yticklabels=class_names,
        ylabel="True label", xlabel="Predicted label", title=title,
    )
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    axes.tick_params(axis="y", labelsize=8)
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in labels:
        for column in labels:
            axes.text(
                column, row, str(matrix[row, column]), ha="center", va="center",
                color="white" if matrix[row, column] > threshold else "black", fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_loss_curve(metrics_path, output_path, best_epoch=None):
    """Plot the recorded training history without running model inference."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with Path(metrics_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"訓練指標沒有 epoch 紀錄：{metrics_path}")

    epochs = [int(row["epoch"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    val_losses = [float(row["val_loss"]) for row in rows]
    figure = Figure(figsize=(10, 6))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(epochs, train_losses, label="Training loss")
    axes.plot(epochs, val_losses, label="Validation loss")
    if best_epoch is not None:
        axes.axvline(best_epoch, color="gray", linestyle="--", alpha=0.8,
                     label=f"Best checkpoint (epoch {best_epoch})")
    axes.set(xlabel="Epoch", ylabel="Loss", title="Training and validation loss")
    axes.legend()
    axes.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    figure.clear()


def write_evaluation_report(
    *, split, loss, accuracy, y_true, y_pred, classes, output_dir,
    checkpoint_path, checkpoint_epoch, plant=None, data_dir=None,
    report_name=None, matrix_name=None, summary_name=None,
):
    """Write matching metrics and artifacts for either validation or test."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)
    report_name = report_name or f"{split}_report.txt"
    matrix_name = matrix_name or f"{split}_confusion_matrix.png"
    summary_name = summary_name or f"{split}_summary.json"
    labels = list(range(len(classes)))
    supports = np.bincount(y_true, minlength=len(classes))
    macro_recall = macro_avg_recall(y_true, y_pred, len(classes))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes,
        digits=4, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    low_support = {
        classes[index]: int(count)
        for index, count in enumerate(supports)
        if count < MIN_CLASS_SUPPORT_WARNING
    }
    missing_classes = [
        classes[index] for index, count in enumerate(supports) if count == 0
    ]

    split_title = "Validation" if split == "val" else "Test"
    metric_prefix = "Val" if split == "val" else "Test"
    primary = "N/A (class has no actual samples)" if macro_recall is None else f"{macro_recall:.2f}%"
    summary_lines = [
        f"Primary Metric — Macro Avg Recall: {primary}",
        "Per-class accuracy is Recall: correctly classified samples / actual samples.",
        f"{split_title} Summary",
        f"Plant: {plant or 'unknown'}",
        f"Run: {checkpoint_path.parent.name}",
        f"Data Dir: {Path(data_dir).resolve() if data_dir is not None else 'unknown'}",
        f"Checkpoint: {checkpoint_path.name} (epoch {checkpoint_epoch if checkpoint_epoch is not None else 'unknown'})",
        f"{metric_prefix} Loss: {loss:.4f}",
        f"{metric_prefix} Accuracy: {accuracy:.2f}%",
        f"Sample Count: {len(y_true)}",
        "",
        "Per-class Correct / Actual",
        *[
            f"{name}: {int(matrix[index, index])}/{int(supports[index])}"
            for index, name in enumerate(classes)
        ],
        "",
        "Classification Report",
        report.rstrip(),
        "",
        "Notes: Classification report uses zero_division=0. The primary metric is unavailable if any class has zero support.",
    ]
    if low_support:
        summary_lines.append(
            "Low support warning (<20 samples): "
            + ", ".join(f"{name}={count}" for name, count in low_support.items())
        )
    if missing_classes:
        summary_lines.append(
            f"Missing {split} classes: " + ", ".join(missing_classes)
        )

    report_text = "\n".join(summary_lines) + "\n"
    report_path = output_dir / report_name
    report_path.write_text(report_text, encoding="utf-8")
    plot_confusion_matrix(matrix, classes, output_dir / matrix_name,
                          f"Confusion Matrix ({split_title})")
    summary = {
        "plant": plant,
        "split": split,
        "data_dir": str(Path(data_dir).resolve()) if data_dir is not None else None,
        "run": checkpoint_path.parent.name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "classes": list(classes),
        "sample_count": int(len(y_true)),
        "loss": float(loss),
        "accuracy": float(accuracy),
        "macro_avg_recall": macro_recall,
        "support_by_class": {name: int(supports[index]) for index, name in enumerate(classes)},
        "correct_by_class": {name: int(matrix[index, index]) for index, name in enumerate(classes)},
        "low_support_classes": low_support,
        "missing_classes": missing_classes,
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=classes,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
        "report_path": report_name,
        "confusion_matrix_path": matrix_name,
        "summary_path": summary_name,
    }
    (output_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return summary, report_text


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.batch_size <= 0:
        raise ValueError("batch-size 必須大於 0")

    model_path = Path(args.model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = list(checkpoint.get("classes", []))
    if not classes:
        classes_path = model_path.parent / "classes.json"
        if not classes_path.is_file():
            raise ValueError("checkpoint 未包含 classes，且找不到 classes.json")
        mapping = json.loads(classes_path.read_text(encoding="utf-8"))
        classes = [mapping[str(index)] for index in range(len(mapping))]
    else:
        classes_path = model_path.parent / "classes.json"
        if classes_path.is_file():
            mapping = json.loads(classes_path.read_text(encoding="utf-8"))
            saved_classes = [mapping[str(index)] for index in range(len(mapping))]
            if saved_classes != classes:
                raise ValueError("checkpoint 與 classes.json 的類別順序不一致")

    checkpoint_plant = checkpoint.get("plant")
    if checkpoint_plant and checkpoint_plant != args.plant:
        raise ValueError(f"模型作物為 {checkpoint_plant}，與指定作物 {args.plant} 不一致")
    model_name = (
        args.model_name
        or checkpoint.get("model_name")
        or load_config(args.config).get("model", {}).get("name", "vit_large_patch16_dinov3.lvd1689m")
    )
    transform = get_eval_transform(model_name)
    loader, _ = create_eval_loader(
        args.data_dir, args.split, transform, args.batch_size, args.num_workers, classes,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_epoch = checkpoint.get("epoch")
    model, checkpoint = load_model(
        model_name, len(classes), model_path, device, checkpoint=checkpoint,
    )
    del checkpoint
    loss, accuracy, y_true, y_pred = evaluate(model, loader, device)
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent / "evaluation"
    summary, report_text = write_evaluation_report(
        split=args.split, loss=loss, accuracy=accuracy,
        y_true=y_true, y_pred=y_pred, classes=classes,
        output_dir=output_dir, checkpoint_path=model_path,
        checkpoint_epoch=checkpoint_epoch, plant=args.plant,
        data_dir=args.data_dir,
    )
    metrics_path = model_path.parent / "metrics.csv"
    if metrics_path.is_file():
        plot_loss_curve(metrics_path, output_dir / "loss_curve.png", checkpoint_epoch)
    print(report_text)
    print(f"Report: {output_dir / summary['report_path']}")
    print(f"Confusion matrix: {output_dir / summary['confusion_matrix_path']}")
    if metrics_path.is_file():
        print(f"Loss curve: {output_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
