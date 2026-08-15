#!/usr/bin/env python3
"""
evaluate.py
使用訓練好的模型對驗證集 (val) 做完整評估：
- Validation Loss / Accuracy
- 每一類 Precision / Recall / F1-score
- Confusion Matrix (存成圖片)

用法：
python evaluate.py \
  --data-dir dataset \
  --model-path output_v2/best_model.pth \
  --model-name convnext_large.fb_in1k \
  --batch-size 32 \
  --output-dir output_v2
"""

import argparse
import os
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
import numpy as np
import matplotlib.pyplot as plt


def load_config(config_path="config.yaml"):
    """載入 YAML 設定檔"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description='評估模型在驗證集上的表現')

    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='資料集根目錄 (內含 train/ 和 val/，預設: dataset)')
    parser.add_argument('--model-path', type=str, required=True,
                        help='訓練好的模型權重 (例如: output_v2/best_model.pth)')
    parser.add_argument('--model-name', type=str, default=None,
                        help='timm 模型名稱 (若未指定則讀取 config.yaml)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='評估時的 batch size (預設: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader num_workers (預設: 4)')
    parser.add_argument('--output-dir', type=str, default='output_v2',
                        help='輸出報告與圖檔的目錄 (預設: output_v2)')

    return parser.parse_args()


def get_val_transform(model_name: str):
    """根據 timm 模型設定建立驗證用 transform（要跟 train 時一致）"""
    temp_model = timm.create_model(model_name, pretrained=False)
    data_config = resolve_data_config({}, model=temp_model)
    del temp_model

    input_size = data_config['input_size'][-1]
    mean = data_config['mean']
    std = data_config['std']

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return val_transform


def create_val_loader(data_dir, val_transform, batch_size, num_workers):

    # 直接把傳入路徑當作評估資料夾
    val_dir = data_dir

    val_dataset = datasets.ImageFolder(
        val_dir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return val_loader, val_dataset.classes


def load_model(model_name, num_classes, model_path, device):
    """建立模型並載入權重"""
    print(f"正在建立模型: {model_name}")
    model = timm.create_model(
        model_name,
        pretrained=False,         # 評估不需要再載入 ImageNet 預訓練
        num_classes=num_classes
    )

    print(f"從檔案載入權重: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def evaluate(model, val_loader, device):
    """在整個 val set 上跑推論，回傳 loss / acc / y_true / y_pred"""
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = outputs.max(1)

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / total
    acc = accuracy_score(all_labels, all_preds) * 100.0

    return avg_loss, acc, np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(cm, class_names, output_path, cmap="Greens"):
    """畫出混淆矩陣並存圖"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # 在每個格子上寫數字
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7
            )

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Confusion Matrix 已儲存至: {output_path}")

def main():
    args = parse_args()
    config = load_config()

    # 優先權：命令列參數 > 設定檔 > 預設值
    model_name = args.model_name or config.get('model', {}).get('name', 'vit_large_patch16_dinov3.lvd1689m')

    data_dir = args.data_dir
    model_path = args.model_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")

    # 1. 建立驗證用 transform & DataLoader
    print("\n正在準備驗證資料轉換與 DataLoader...")
    val_transform = get_val_transform(model_name)
    val_loader, class_names = create_val_loader(
        data_dir,
        val_transform,
        args.batch_size,
        args.num_workers
    )
    num_classes = len(class_names)
    print(f"驗證集大小: {len(val_loader.dataset)}")
    print(f"類別數量: {num_classes}")
    print(f"類別名稱: {class_names}")

    # 2. 載入模型
    model = load_model(model_name, num_classes, model_path, device)

    # 3. 在 val set 上評估
    print("\n開始在驗證集上評估...")
    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, device)

    print("\n=== Validation Summary ===")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.2f}%")

    # 4. 計算 Classification Report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print("\n=== Classification Report ===")
    print(report)

    # 存成文字檔
    report_path = output_dir / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Validation Summary\n")
        f.write(f"Val Loss: {val_loss:.4f}\n")
        f.write(f"Val Accuracy: {val_acc:.2f}%\n\n")
        f.write("Classification Report\n")
        f.write(report)
    print(f"✓ Classification Report 已儲存至: {report_path}")

    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    print("\n評估完成！")


if __name__ == '__main__':
    main()
