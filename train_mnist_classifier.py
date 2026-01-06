"""
ResNet MNIST分类器训练脚本
使用与DDPM相同的数据加载方式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt

# 使用你的数据加载方式
from src.data.mnist import get_mnist_loader_and_transform


# ==================== ResNet模型定义 ====================

class BasicBlock(nn.Module):
    """ResNet基础残差块"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18(nn.Module):
    """ResNet-18 for MNIST"""
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


# ==================== 训练和评估函数 ====================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
            
            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, class_correct, class_total


# ==================== 主函数 ====================

def main():
    # ==================== 配置参数 ====================
    EPOCHS = 20  # 训练轮数
    LEARNING_RATE = 0.001  # 学习率
    WEIGHT_DECAY = 1e-4  # 权重衰减
    PATH_TO_READY_MODEL = None  # 预训练模型路径（如果有）
    PATH_TO_SAVE_MODEL = "resnet18_mnist_classifier.pth"  # 模型保存路径
    
    # 创建必要的文件夹
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("projects/diffusion/stable-diffusion-from-scratch/loss", exist_ok=True)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"{'='*70}")
    print(f"ResNet-18 MNIST Classifier Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"{'='*70}\n")
    
    # ==================== 加载数据 ====================
    print("Loading MNIST dataset...")
    data = get_mnist_loader_and_transform()
    
    print(f"✓ Train samples: {len(data.train_dataset)}")
    print(f"✓ Val samples: {len(data.val_dataset)}")
    print(f"✓ Image shape: {data.train_dataset[0][0].shape}")
    print(f"✓ In channels: {data.in_channels}")
    print(f"✓ Out channels: {data.out_channels}\n")
    
    # ==================== 创建模型 ====================
    print("Creating ResNet-18 model...")
    model = ResNet18(num_classes=10).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}\n")
    
    # ==================== 加载预训练模型（如果有） ====================
    if PATH_TO_READY_MODEL is not None:
        print(f"Loading pretrained model from {PATH_TO_READY_MODEL}...")
        checkpoint = torch.load(PATH_TO_READY_MODEL, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded\n")
    
    # ==================== 定义损失函数和优化器 ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # ==================== 训练循环 ====================
    print(f"{'='*70}")
    print("Starting Training")
    print(f"{'='*70}\n")
    
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*70}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, data.train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, class_correct, class_total = evaluate(
            model, data.val_loader, criterion, device
        )
        
        # 学习率调度
        scheduler.step(val_acc)
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 打印结果
        print(f"\n{'─'*70}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 打印每个类别的准确率
        print(f"\n  Per-class Accuracy:")
        for i in range(10):
            class_acc = 100. * class_correct[i] / class_total[i]
            bar = '█' * int(class_acc / 2)
            print(f"    Digit {i}: {class_acc:5.2f}% {bar}")
        print(f"{'─'*70}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join("checkpoints", PATH_TO_SAVE_MODEL)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, save_path)
            
            print(f"\n✓ Saved best model to {save_path}")
            print(f"  Best Val Accuracy: {best_acc:.2f}%")
    
    # ==================== 保存训练曲线 ====================
    print(f"\n{'='*70}")
    print("Saving training curves...")
    print(f"{'='*70}")
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves (Final: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy", linewidth=2)
    plt.plot(val_accs, label="Val Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curves (Final: Train={train_accs[-1]:.2f}%, Val={val_accs[-1]:.2f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    img_path = os.path.join("projects/diffusion/stable-diffusion-from-scratch/loss", 
                            f"classifier_curves_{timestamp}.png")
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to: {img_path}")
    
    # ==================== 训练完成 ====================
    print(f"\n{'='*70}")
    print("Training Completed!")
    print(f"{'='*70}")
    print(f"Best Val Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {os.path.join('checkpoints', PATH_TO_SAVE_MODEL)}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()