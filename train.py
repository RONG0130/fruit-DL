#!/usr/bin/env python3
"""
植物病蟲害辨識模型訓練程式
基於 DINOv3 (ViT Large) 進行遷移學習
"""

import argparse
import json
import os
import random
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, Subset
from torchvision import datasets, transforms
import timm
from timm.data import resolve_data_config
from tqdm import tqdm


def load_config(config_path="config.yaml"):
    """載入 YAML 設定檔"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description='訓練植物病蟲害辨識模型')

    # 資料路徑
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='資料集根目錄 (預設: dataset)')

    # 超參數
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小 (若未指定則讀取 config.yaml)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='訓練週期數 (若未指定則讀取 config.yaml)')
    parser.add_argument('--lr', type=float, default=None,
                        help='學習率 (若未指定則讀取 config.yaml)')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='權重衰減 (若未指定則讀取 config.yaml)')

    # 模型設定
    parser.add_argument('--model-name', type=str, default=None,
                        help='timm 模型名稱 (若未指定則讀取 config.yaml)')

    # 輸出設定
    parser.add_argument('--output-dir', type=str, default='output',
                        help='輸出目錄 (預設: output)')

    # 其他設定
    parser.add_argument('--num-workers', type=int, default=4,
                        help='資料載入執行緒數 (預設: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42)')

    return parser.parse_args()


def set_seed(seed):
    """設定隨機種子以確保可重現性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_transforms(config):
    """
    根據模型配置建立資料轉換
    Args:
        config: timm 模型的資料配置
    Returns:
        train_transform, val_transform
    """
    input_size = config['input_size'][-1]  # 取得 input size (例如 224)
    mean = config['mean']
    std = config['std']

    # 訓練集：包含資料增強
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 驗證集：僅 resize 和 normalize
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


class FixedCountSampler(Sampler):
    """每個 epoch 為各組取得固定張數，少數類別以有放回抽樣補足。"""

    def __init__(self, groups, target, seed=42):
        if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
            raise ValueError("resample target 必須是大於 0 的整數")
        if not groups or any(not group for group in groups):
            raise ValueError("無法對沒有圖片的類別重採樣")
        self.groups = [tuple(group) for group in groups]
        self.target = target
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.groups) * self.target

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        selected = []
        for group in self.groups:
            if len(group) >= self.target:
                selected.extend(rng.sample(group, self.target))
            else:
                selected.extend(group)
                selected.extend(rng.choices(group, k=self.target - len(group)))
        rng.shuffle(selected)
        return iter(selected)


def create_dataloaders(data_dir, train_transform, val_transform, batch_size, num_workers, config=None, seed=42):
    """建立訓練和驗證資料載入器，並實作資料重採樣 (Resample) 功能"""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 建立 ImageFolder 資料集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    classes = train_dataset.classes
    sampler = None

    # 實作 Resample 邏輯 (僅針對訓練集)
    if config and config.get('dataset', {}).get('resample', {}).get('enabled', False):
        resample_cfg = config['dataset']['resample']
        method = resample_cfg.get('method', 'fixed_count')
        target = resample_cfg.get('target', 300)
        per_class = resample_cfg.get('per_class', True)

        if method == 'fixed_count':
            if per_class:
                groups = [[] for _ in classes]
            else:
                groups = [[]]
            for idx, (_, label) in enumerate(train_dataset.samples):
                groups[label if per_class else 0].append(idx)
            sampler = FixedCountSampler(groups, target, seed=seed)
            group_names = classes if per_class else ['全部訓練資料']
            for name, group in zip(group_names, groups):
                print(
                    f"Resample {name}: 原圖={len(group)}, 每輪取樣={target}, "
                    f"重複抽樣={max(0, target - len(group))}"
                )
            print(f"每輪訓練取樣總數：{len(sampler)}")
        elif method == 'ratio':
            indices = []
            if per_class:
                groups = [[] for _ in classes]
                for idx, (_, label) in enumerate(train_dataset.samples):
                    groups[label].append(idx)
            else:
                groups = [list(range(len(train_dataset)))]
            for group in groups:
                count = min(int(len(group) * target), len(group))
                if count > 0:
                    indices.extend(random.sample(group, count))
            if indices:
                train_dataset = Subset(train_dataset, indices)
                print(f"Resample ratio={target}: 每輪訓練取樣總數={len(train_dataset)}")
        else:
            raise ValueError(f"不支援的 resample method：{method}")

    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, classes


def create_model(model_name, num_classes):
    """建立並初始化模型"""
    print(f"正在載入模型: {model_name}")
    print(f"類別數量: {num_classes}")

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )

    return model


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # 使用混合精度訓練
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 反向傳播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 統計
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新進度條
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch, num_classes=None):
    """驗證模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # 使用混合精度
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # 統計
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if num_classes is not None:
                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predicted.cpu().tolist())

            # 更新進度條
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    if num_classes is not None:
        from sklearn.metrics import recall_score

        labels = list(range(num_classes))
        supports = torch.bincount(
            torch.tensor(all_labels, dtype=torch.long),
            minlength=num_classes,
        ).tolist()
        if any(support == 0 for support in supports):
            raise ValueError("驗證集有類別沒有圖片，無法計算 Macro Avg Recall")
        macro_recall = recall_score(
            all_labels,
            all_predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ) * 100.0
        return epoch_loss, epoch_acc, macro_recall

    return epoch_loss, epoch_acc


def save_checkpoint(model, classes, output_dir, epoch, acc, is_best=False):
    """儲存模型檢查點"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': acc,
        'classes': classes
    }

    # 儲存當前 epoch 的模型
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    # 如果是最佳模型，額外儲存一份
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ 儲存最佳模型: {best_path} (準確率: {acc:.2f}%)")


def save_classes_json(classes, output_dir):
    """儲存類別映射表為 JSON 檔案"""
    classes_dict = {i: class_name for i, class_name in enumerate(classes)}

    json_path = os.path.join(output_dir, 'classes.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(classes_dict, f, ensure_ascii=False, indent=2)

    print(f"✓ 類別映射表已儲存至: {json_path}")


def main():
    args = parse_args()
    config = load_config()

    # 優先權：命令列參數 > 設定檔 > 預設值
    train_cfg = config.get('train', {})
    model_name = args.model_name or config.get('model', {}).get('name', 'vit_large_patch16_dinov3.lvd1689m')
    batch_size = args.batch_size or train_cfg.get('batch_size', 8)
    epochs = args.epochs or train_cfg.get('epochs', 30)
    lr = args.lr or train_cfg.get('lr', 1e-4)
    weight_decay = args.weight_decay or train_cfg.get('weight_decay', 0.01)

    # 設定隨機種子
    set_seed(args.seed)

    # 建立輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 檢查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")

    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用 CPU 訓練會非常慢！")

    # 建立臨時模型以獲取資料配置
    print(f"\n正在準備模型 [{model_name}] 的資料轉換...")
    temp_model = timm.create_model(model_name, pretrained=False)
    data_config = resolve_data_config({}, model=temp_model)
    print(f"模型輸入尺寸: {data_config['input_size']}")
    print(f"標準化參數 - Mean: {data_config['mean']}, Std: {data_config['std']}")
    del temp_model

    # 建立資料轉換
    train_transform, val_transform = get_data_transforms(data_config)

    # 建立資料載入器
    print(f"\n正在載入資料集: {args.data_dir}")
    train_loader, val_loader, classes = create_dataloaders(
        args.data_dir,
        train_transform,
        val_transform,
        batch_size,
        args.num_workers,
        config=config,
        seed=args.seed,
    )

    print(
        f"訓練資料集: {len(train_loader.dataset)} 張，"
        f"每個 epoch 取樣: {len(train_loader.sampler)} 次"
    )
    print(f"驗證集大小: {len(val_loader.dataset)}")
    print(f"類別數量: {len(classes)}")
    print(f"類別名稱: {classes}")

    # 儲存類別映射表
    save_classes_json(classes, args.output_dir)

    # 建立模型
    print("\n正在建立模型...")
    model = create_model(model_name, len(classes))
    model = model.to(device)

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 建立混合精度訓練的 GradScaler
    scaler = torch.amp.GradScaler('cuda')

    # 訓練迴圈
    print(f"\n開始訓練 (共 {epochs} 個 epochs)")
    print(f"參數設定: Batch Size={batch_size}, LR={lr}, Weight Decay={weight_decay}")
    print("=" * 80)

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # 訓練
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # 驗證
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # 印出統計資訊
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # 儲存檢查點
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint(model, classes, args.output_dir, epoch, val_acc, is_best)

        print("=" * 80)

    print(f"\n訓練完成！")
    print(f"最佳驗證準確率: {best_acc:.2f}%")
    print(f"模型已儲存至: {args.output_dir}")


if __name__ == '__main__':
    main()
