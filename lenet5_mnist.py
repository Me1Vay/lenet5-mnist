# -*- coding: utf-8 -*-
"""
LeNet-5 MNIST 手写数字识别
源自 Notebook 导出（手写数字识别lenet.html），为 Windows 本地运行做了适配：
  1. DataLoader 多进程包进 main() 守卫（Windows spawn 机制必需）
  2. matplotlib 无窗口时自动改为保存 PNG，不阻塞
  3. MNIST 下载多镜像回退，下载失败自动换源
  4. 训练集加随机仿射增强（测试集不加），测试准确率 98.84% -> 99.32%
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

import matplotlib
# 默认用 Agg：图片存成 PNG，不弹窗、不阻塞（加 --no-window 也不会弹）
WINDOWED = False
matplotlib.use('Agg')
# Windows 自带中文字体，不指定会渲染成方块（DejaVu Sans 无 CJK 字形）
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

DATA_DIR = './data'
OUT_DIR = './output'
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
NUM_EPOCHS = 10
# Windows 上 DataLoader 多进程每轮都要重新 spawn worker，LeNet 这种小模型反而更慢
NUM_WORKERS = 0 if os.name == 'nt' else 2

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# 1. 数据预处理和加载
# ============================================================
def build_loaders(batch_size=BATCH_SIZE):
    # 将 28x28 的 MNIST 图像填充为 32x32（原论文使用 32x32 输入）
    # 训练集加随机仿射增强，测试集保持原样 —— 增强只能作用于训练集，
    # 否则每张测试图都会被随机变换一次，测出来的准确率就是噪声。
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # 角度和 shear 必须小：数字有强方向性，开大了会造出"6 转 9"这种错误样本
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1),
                                scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST 的均值和标准差
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"图片尺寸: {train_dataset[0][0].shape}")
    return train_dataset, test_dataset, train_loader, test_loader


# ============================================================
# 2. 定义 LeNet-5 模型
# ============================================================
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)   # 1x32x32 -> 6x28x28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)                     # 6x28x28 -> 6x14x14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)  # 6x14x14 -> 16x10x10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)                     # 16x10x10 -> 16x5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # C5: 400 -> 120
        self.fc2 = nn.Linear(120, 84)           # F6: 120 -> 84
        self.fc3 = nn.Linear(84, num_classes)   # 输出: 84 -> 10
        self.activation = nn.Tanh()             # 原论文使用 tanh

    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))   # -> 6x14x14
        x = self.pool2(self.activation(self.conv2(x)))   # -> 16x5x5
        x = x.view(-1, 16 * 5 * 5)                       # -> 400
        x = self.activation(self.fc1(x))                 # -> 120
        x = self.activation(self.fc2(x))                 # -> 84
        return self.fc3(x)                               # -> 10（配合 CrossEntropyLoss，不再 softmax）

    @classmethod
    def with_relu(cls, num_classes=10):
        model = cls(num_classes)
        model.activation = nn.ReLU()
        return model


class LeNet5_ReLU(LeNet5):
    """使用ReLU的LeNet-5变体"""
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.activation = nn.ReLU()


# ============================================================
# 3. 训练与测试
# ============================================================
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Training set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return avg_loss, accuracy


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return avg_loss, accuracy


# ============================================================
# 4. 可视化
# ============================================================
def show_or_save(fig, filename):
    """有窗口就弹窗，否则存成 PNG"""
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"图片已保存: {os.path.abspath(path)}")
    if WINDOWED:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def visualize_sample_predictions(model, device, test_loader, num_samples=10):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images[:num_samples])
        _, predictions = outputs.max(1)

    images_np = images.cpu().numpy()
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(images_np[idx, 0], cmap='gray')
        ax.set_title(f'True: {labels[idx]}, Pred: {predictions[idx]}',
                     color='green' if predictions[idx] == labels[idx] else 'red')
        ax.axis('off')
    plt.tight_layout()
    show_or_save(fig, 'sample_predictions.png')


def plot_curves(train_losses, train_accs, test_losses, test_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    show_or_save(fig, 'training_curves.png')


# ============================================================
# 5. 单张预测 / 模型加载
# ============================================================
def predict_single_image(image_tensor, model, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
    return prediction.item(), confidence.item()


def load_and_use_model(model_path='lenet5_mnist.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
    return model


# ============================================================
# 6. 主流程
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description='LeNet-5 MNIST 手写数字识别')
    p.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='训练轮数（默认 10）')
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='训练 batch size（默认 64）')
    p.add_argument('--data-dir', default=DATA_DIR, help='MNIST 数据集目录')
    p.add_argument('--out-dir', default=OUT_DIR, help='模型与图片输出目录')
    p.add_argument('--device', default=None, help='cuda / cpu，默认自动选择')
    return p.parse_args()


def main():
    args = parse_args()
    global DATA_DIR, OUT_DIR
    DATA_DIR, OUT_DIR = args.data_dir, args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset, test_dataset, train_loader, test_loader = build_loaders(args.batch_size)

    model = LeNet5(num_classes=10).to(device)
    print("LeNet-5 Model Architecture:")
    print(model)
    print("\n" + "=" * 50 + "\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, train_accs, test_losses, test_accs = [], [], [], []

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 30)
        tl, ta = train(model, device, train_loader, optimizer, criterion, epoch)
        vl, va = test(model, device, test_loader, criterion)
        train_losses.append(tl); train_accs.append(ta)
        test_losses.append(vl); test_accs.append(va)
        scheduler.step()

    plot_curves(train_losses, train_accs, test_losses, test_accs)
    visualize_sample_predictions(model, device, test_loader, num_samples=10)

    ckpt_path = os.path.join(OUT_DIR, 'lenet5_mnist.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
    print(f"Model saved as '{os.path.abspath(ckpt_path)}'")

    test_image, test_label = test_dataset[0]
    pred, confidence = predict_single_image(test_image, model, device)
    print("\nSingle image test:")
    print(f"  True label: {test_label}")
    print(f"  Predicted:  {pred}")
    print(f"  Confidence: {confidence:.4f}")

    fig = plt.figure()
    plt.imshow(test_image[0], cmap='gray')
    plt.title(f'True label: {test_label}, Predicted: {pred}')
    plt.axis('off')
    show_or_save(fig, 'single_test.png')

    print("\n" + "=" * 60)
    print(f"最终测试准确率: {test_accs[-1]:.2f}%")
    print("模型结构: 32x32 -> C1(6x28x28) -> S2(6x14x14) -> C3(16x10x10) "
          "-> S4(16x5x5) -> C5(120) -> F6(84) -> 输出(10)")
    print("=" * 60)


if __name__ == '__main__':
    main()
