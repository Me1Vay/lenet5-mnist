# LeNet-5 复现与图像分类实践

从零复现经典 **LeNet-5**（LeCun et al., 1998）并在 MNIST 上做**消融实验**，配套一个可在浏览器手写的识别 Demo；另含一组**小样本迁移学习**实验（猫狗分类 80 张）。

> 全部在 **CPU** 上训练完毕（无 GPU 依赖），单组实验 3~6 分钟。

## 实验结果

四组消融实验（固定 `seed=42`，验证集从训练集切分 5000 条，测试集 10000 张仅在最终报告时使用）：

| 编号 | 配置 | 数据增强 | 激活 | 池化 | 优化器/调度 | 轮数 | 验证集 | 测试集 |
|---|---|---|---|---|---|---|---|---|
| E0 | 基线（论文原版结构） | 否 | tanh | avg | Adam/StepLR | 11 | 98.94% | **98.89%** |
| E1 | E0 + 数据增强 | 是 | tanh | avg | Adam/StepLR | 15 | 99.30% | **99.32%** |
| **E2** | **E1 + AdamW/Cosine/标签平滑** | 是 | tanh | avg | AdamW/Cosine | 15 | **99.40%** | **99.34%** |
| E3 | E2 + ReLU/MaxPool（结构变体） | 是 | relu | max | AdamW/Cosine | 15 | 99.36% | 99.33% |

**三条结论：**

1. **数据增强单独贡献 +0.43pp**，一项就跨过 99.3% 的线；
2. E2 仅比 E1 高 0.02pp —— 换优化器/调度器收益微乎其微，**不值得动代码**；
3. E3 比 E2 还低 0.01pp —— 在经典结构上**单独**换 ReLU/MaxPool 没有收益，所以模型定义一行没改，
   "复现经典 LeNet-5"这个说法站得住。
   （但 `lenet5_improved.py` 那条线加了 **BatchNorm** 后达到 **99.46%**，见下。）

### 落盘权重复核实测

仓库 `output/` 下的权重可直接加载复现（下表为全量 10000 张测试集实测）：

| 文件 | 架构 | 参数量 | 测试集 |
|---|---|---|---|
| `lenet5_mnist_base.pth` | LeNet5（tanh/avg），改动前 | 61,706 | 98.84% |
| `lenet5_mnist.pth` | LeNet5 + 增强 + AdamW/Cosine/标签平滑（= 上表 E2） | 61,706 | **99.34%** |
| `lenet5_mnist_v2.pth` | **LeNet5BN**（BN + ReLU + MaxPool），另一条线 | 62,614 | **99.46%** |

`lenet5_mnist.pth` 的 `meta` 字段完整记录了训练配置，可自证身份：

```python
import torch
torch.load('output/lenet5_mnist.pth', map_location='cpu')['meta']
# {'act': 'tanh', 'pool': 'avg', 'augment': True, 'optim': 'adamw', 'sched': 'cosine',
#  'label_smoothing': 0.05, 'lr': 0.001, 'weight_decay': 0.0001, 'batch_size': 64,
#  'epochs_run': 15, 'best_epoch': 13, 'val_acc': 99.4, 'test_acc': 99.34,
#  'seed': 42, 'train_size': 55000, 'val_size': 5000, 'seconds': 378.5}
```

## 快速开始

```bash
pip install torch torchvision matplotlib gradio
```

**手写识别 Demo**（Gradio，CPU 实时推理）：

```bash
python web_demo.py
# 打开 http://127.0.0.1:7860
```

**复现实验**：

```bash
python lenet5_mnist.py              # 默认 10 epoch
python lenet5_improved.py           # 增强版（BN + ReLU/MaxPool + 25 epoch），跑完打印普通/TTA 两档精度
```

MNIST 数据集会在首次运行时自动下载（多镜像回退）。

> ⚠️ **注意**：`lenet5_mnist.py` 当前默认仍是 `Adam + StepLR`、10 epoch，**不等于上表的 E2**，
> 而它的保存路径正是 `output/lenet5_mnist.pth` —— 直接跑会覆盖掉仓库里那份 99.34% 的权重。
> 重跑前请先备份，或加 `--out-dir` 输出到别处。复现 E2 需要改成
> `AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR` + `CrossEntropyLoss(label_smoothing=0.05)` 并 `--epochs 15`。

## 项目结构

```
lenet5-mnist/
├─ lenet5_mnist.py        # 主训练脚本：LeNet5 定义 + 训练/评估 + 消融入口
├─ lenet5_improved.py     # 增强版：LeNet5BN（BatchNorm）+ ReLU/MaxPool + TTA
├─ web_demo.py            # Gradio 手写识别 Demo
├─ pets_catdog/           # 小样本迁移学习（猫狗分类 80 张）
│  ├─ pets_catdog.ipynb   # 实验过程：类别塌缩 → 冻结 backbone 仅训分类头
│  ├─ predict.py          # 推理脚本
│  ├─ infer_snippet.py    # 可复用推理片段
│  └─ MINIMAL_PATCH.md    # 从"全量微调"到"只训分类头"的最小改动说明
├─ original/              # 改动前的原始脚本（对照用）+ DIFF.txt
├─ output/                # 权重与图表
└─ CHANGES.md             # 这版改了什么、为什么只改一处
```

## 两个值得说的实现细节

### 1. 手写 Demo 的输入适配（`web_demo.py` 的 `preprocess()`）

很多人做手写识别 Demo 直接 `resize` 就喂给模型，线上识别率会明显掉。这里做了完整的分布对齐：

1. Gradio 画布是 **RGBA 透明背景**，先把 alpha 合成到白底，否则透明区会被当成黑色；
2. 用**四周边框**估计背景亮度，决定是否需要反相（目标：笔画亮、背景暗）；
3. 裁剪到笔画实际范围 → 等比缩放使最长边为 20px → **居中贴进 28×28 黑底**，
   复现 MNIST 原始数据分布（MNIST 数字正是 20×20 居中放在 28×28 里）。

### 2. 数据增强只能加在训练集

```python
train_transform = ... + transforms.RandomAffine(degrees=10, translate=(0.1,0.1),
                                                 scale=(0.9,1.1), shear=5)
test_transform  = ...   # 保持原样
```

测试集加增强等于每张图被随机变换一次，测出来的准确率是**噪声**。
另外角度与 shear 必须小：数字有强方向性，角度开大了会造出"6 转 9"这类错误样本，反而拉低准确率。

## 迁移学习：小样本下的类别塌缩

`pets_catdog/` 记录了一个真实故障：**80 张训练样本**下做全量参数更新，模型几乎全部预测多数类，
准确率卡在 60%。

定位与修复：**冻结 backbone，只训练分类头（4098 个参数）**，并调整学习率调度，
验证集 P/R/F1 达到 1.000。最小改动过程记录在 `pets_catdog/MINIMAL_PATCH.md`。

## 环境

- Python 3.13 + PyTorch 2.14（CPU 版）+ torchvision + matplotlib + gradio
- 无 GPU 亦可运行；全部结果均为 CPU 训练所得

## 图表

`output/ablation.png`（消融对比）、`output/training_curves.png`、`output/training_curves_v2.png`、
`output/sample_predictions.png`、`output/single_test.png`
