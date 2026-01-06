
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt

from src.data.mnist import get_mnist_loader_and_transform


# ==================== 简单的分类器架构 ====================

class WeakClassifier(nn.Module):

    def __init__(self, num_classes=10):
        super(WeakClassifier, self).__init__()
        
        # 更简单的卷积层
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # 16 通道（vs 32+）
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 32 通道（vs 64+）
        self.bn2 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 更强的dropout
        self.dropout1 = nn.Dropout(0.4)  # 更高的dropout
        self.dropout2 = nn.Dropout(0.5)
        
        # 更小的全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  # 更小的隐藏层
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 28x28 -> 14x14
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 14x14 -> 7x7
        x = self.dropout1(x)
        
        # Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


class VeryWeakClassifier(nn.Module):
    """
    更弱的分类器（如果需要更低准确率）
    目标: 90-95% 准确率
    """
    def __init__(self, num_classes=10):
        super(VeryWeakClassifier, self).__init__()
        
        # 只有一个卷积层
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # 简单的全连接
        self.fc1 = nn.Linear(16 * 7 * 7, 32)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # 28x28 -> 14x14
        x = self.pool(x)  # 14x14 -> 7x7
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
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
    EPOCHS = 1  # 更少的epoch（故意不充分训练）
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 5e-4  # 更强的权重衰减
    BATCH_SIZE = 128
    
    MODEL_TYPE = "very_weak"  # "weak" or "very_weak"
    PATH_TO_SAVE_MODEL = "weak_classifier.pth"
    
    # 创建必要的文件夹
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("projects/diffusion/stable-diffusion-from-scratch/loss", exist_ok=True)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"{'='*70}")
    print(f"Training Weak MNIST Classifier for Guidance")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Target Accuracy: 95-98%")
    print(f"{'='*70}\n")
    
    # ==================== 加载数据 ====================
    print("Loading MNIST dataset...")
    data = get_mnist_loader_and_transform()
    
    print(f"✓ Train samples: {len(data.train_dataset)}")
    print(f"✓ Val samples: {len(data.val_dataset)}\n")
    
    # ==================== 创建模型 ====================
    print(f"Creating {MODEL_TYPE} classifier...")
    if MODEL_TYPE == "weak":
        model = WeakClassifier(num_classes=10).to(device)
    elif MODEL_TYPE == "very_weak":
        model = VeryWeakClassifier(num_classes=10).to(device)
    else:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")
    print(f"  (比较: ResNet-18 有 ~11M 参数)\n")
    
    # ==================== 定义损失函数和优化器 ====================
    # 使用Label Smoothing进一步降低过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 不使用学习率调度器，保持简单
    
    # ==================== 训练循环 ====================
    print(f"{'='*70}")
    print("Starting Training")
    print(f"{'='*70}\n")
    
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 早停机制：达到目标准确率就停止
    target_acc_min = 95.0
    target_acc_max = 98.0
    
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
        
        # 打印每个类别的准确率
        print(f"\n  Per-class Accuracy:")
        for i in range(10):
            class_acc = 100. * class_correct[i] / class_total[i]
            bar = '█' * int(class_acc / 2)
            print(f"    Digit {i}: {class_acc:5.2f}% {bar}")
        print(f"{'─'*70}")
        
        # 保存模型（每个epoch都保存，因为我们要的是"弱"分类器）
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
            'model_type': MODEL_TYPE,
        }, save_path)
        
        print(f"\n✓ Saved model to {save_path}")
        print(f"  Current Val Accuracy: {val_acc:.2f}%")
        
        # 早停：如果达到目标准确率范围，就停止
        if target_acc_min <= val_acc <= target_acc_max:
            print(f"\n🎯 达到目标准确率范围 ({target_acc_min}% - {target_acc_max}%)!")
            print(f"   当前准确率: {val_acc:.2f}%")
            print(f"   停止训练，这个模型适合用于引导生成。")
            break
        elif val_acc > target_acc_max:
            print(f"\n⚠ 准确率超过目标上限 ({val_acc:.2f}% > {target_acc_max}%)")
            print(f"   这个模型可能太强了，建议使用 'very_weak' 模型类型")
    
    # ==================== 保存训练曲线 ====================
    print(f"\n{'='*70}")
    print("Saving training curves...")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves (Final: Val={val_losses[-1]:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Accuracy", linewidth=2)
    plt.plot(val_accs, label="Val Accuracy", linewidth=2)
    plt.axhline(y=target_acc_min, color='g', linestyle='--', alpha=0.5, label='Target Min')
    plt.axhline(y=target_acc_max, color='r', linestyle='--', alpha=0.5, label='Target Max')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy Curves (Final: Val={val_accs[-1]:.2f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    img_path = os.path.join("projects/diffusion/stable-diffusion-from-scratch/loss", 
                            f"weak_classifier_curves_{timestamp}.png")
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to: {img_path}")
    
    # ==================== 训练完成 ====================
    print(f"\n{'='*70}")
    print("Training Completed!")
    print(f"{'='*70}")
    print(f"Final Val Accuracy: {val_accs[-1]:.2f}%")
    print(f"Model saved to: {os.path.join('checkpoints', PATH_TO_SAVE_MODEL)}")
    
    if target_acc_min <= val_accs[-1] <= target_acc_max:
        print(f"✅ 成功！准确率在目标范围内，适合用于引导生成。")
    elif val_accs[-1] < target_acc_min:
        print(f"⚠ 准确率偏低，可能需要训练更多epoch或使用'weak'模型")
    else:
        print(f"⚠ 准确率偏高，建议使用'very_weak'模型或减少训练epoch")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()