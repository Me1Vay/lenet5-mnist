# LeNet-5 改了什么

## 结论

这次**代码改动只有一处**：训练集的 transform 加随机仿射增强（测试集不加）。
模型结构、训练循环、优化器、保存逻辑全部保持原样。这一处改动把测试准确率从
**98.84%** 推到 **99.32%**。

之后又单独做了一轮调参（AdamW + CosineAnnealingLR + 标签平滑 0.05），把测试集推到
**99.34%**，并覆盖保存在 `output/lenet5_mnist.pth`。所以这个文件里装的是**调参后**的版本，
不是"只加增强"那版。

> **四个数字别混**（2026-09-28 全量 10000 张测试集复核）：
>
> | 数字 | 指什么 | 能复现吗 |
> |---|---|---|
> | **98.84%** | `output/lenet5_mnist_base.pth`：改动前留存权重，实测 | ✅ |
> | 98.89% | 消融表 E0（基线）那**一次独立运行**的读数，权重未落盘 | ❌ 权重已丢 |
> | 99.32% | 只加数据增强（消融表 E1） | ❌ 权重未落盘 |
> | **99.34%** | `output/lenet5_mnist.pth`：增强 + AdamW/Cosine/标签平滑（消融表 E2），实测 | ✅ |
>
> 简历口径用 **98.89% → 99.34%（+0.45pp，对应 E0→E2）** 与消融表一致；
> `base.pth` 的 98.84% 是另一次运行的留存权重，两者都真实，但**不是同一次训练**，被追问时要说清。

## 唯一的改动

```diff
-    transform = transforms.Compose([
-        transforms.Resize((32, 32)),
-        transforms.ToTensor(),
-        transforms.Normalize((0.1307,), (0.3081,)),
-    ])
-    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
-    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
+    # 训练集加随机仿射增强，测试集保持原样
+    train_transform = transforms.Compose([
+        transforms.Resize((32, 32)),
+        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1),
+                                scale=(0.9, 1.1), shear=5),
+        transforms.ToTensor(),
+        transforms.Normalize((0.1307,), (0.3081,)),
+    ])
+    test_transform = transforms.Compose([
+        transforms.Resize((32, 32)),
+        transforms.ToTensor(),
+        transforms.Normalize((0.1307,), (0.3081,)),
+    ])
+    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=train_transform)
+    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
```

另外改了一处显示问题：`matplotlib` 指定了中文字体（`Microsoft YaHei`），
否则 `training_curves.png` 里的中文标题会渲染成方框。

## 为什么只改这一处

四组消融实验（seed 42，每组 15 epoch，均从同一批训练数据出发）：

| 组 | 配置 | 验证集 | 测试集 | 相对基线 |
|---|---|---|---|---|
| E0 | 基线（原版，无增强） | 98.94% | 98.89% | — |
| **E1** | **E0 + 数据增强** | **99.30%** | **99.32%** | **+0.43** |
| E2 | E1 + AdamW/Cosine/标签平滑 | 99.40% | 99.34% | +0.45 |
| E3 | E2 + ReLU/MaxPool（结构变体） | 99.36% | 99.33% | +0.44 |

三条结论：

1. **数据增强单独贡献了 +0.43 个百分点**，一项就跨过了 99.3% 的线
2. E2 只比 E1 高 0.02 —— 换优化器和调度器的收益微乎其微，不值得动代码
3. E3 比 E2 还低 0.01 —— **把 tanh/AvgPool 换成 ReLU/MaxPool 在 MNIST 上没有任何收益**，
   所以模型定义一行都不用改，"复现经典 LeNet-5"这个说法也站得住

## 注意事项

`RandomAffine` 的角度和 shear 必须小（本项目用 ±10° / 5°）。数字有强方向性，
角度开大了会造出"6 转 9"这类错误样本，反而拉低准确率。

测试集**不能**加增强 —— 否则每张测试图会被随机变换一次，测出来的数字是噪声。

## 复现

```bash
C:\Python313\python.exe lenet5_mnist.py                 # 默认 10 epoch
```

实验曲线与数据：`output/ablation.png`、`output/ablation.md`

> ✅ **默认配置已对齐 E2**（2026-09-29 修订）：`lenet5_mnist.py` 现在默认就是
> `AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR` + `CrossEntropyLoss(label_smoothing=0.05)`、
> 15 epoch，并从训练集切 5000 条验证集、按验证集挑最优 epoch（测试集只在最后评估一次）。
> 直接 `python lenet5_mnist.py` 即可复现 E2。
>
> 同时加了**防覆盖保护**：保存前检查同名 `.pth` 是否存在，默认**拒绝覆盖**（加 `--force` 才覆盖，
> 或用 `--out-dir` 另存）。保存的权重带 `meta` 字段记录完整配置。
> 兜底副本仍保留在 `output/lenet5_mnist_E2_99.34.pth`。

## 备份

| 文件 | 是什么 | 全量测试集实测（2026-09-28） |
|---|---|---|
| `output/lenet5_mnist_base.pth` | 改动前那版（无增强） | **98.84%** |
| `output/lenet5_mnist.pth` | 当前版本：增强 + AdamW/Cosine/标签平滑（E2） | **99.34%** |
| `output/lenet5_mnist_E2_99.34.pth` | 上面这个的兜底副本，防止重跑脚本被覆盖 | **99.34%** |
| `output/lenet5_mnist_v2.pth` | 另一条线：`lenet5_improved.py` 的 LeNet5BN（BN+ReLU+MaxPool） | **99.46%** |

> `v2` 换了结构（加了 BatchNorm），**不是"论文原版 LeNet-5"**。所以简历里
> "复现经典 LeNet-5 … 最优配置 99.34%" 指的是上表的 E2，口径没问题，别和 v2 的 99.46% 搞混。
