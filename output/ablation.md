# LeNet-5 MNIST 消融实验

> 一、消融实验读数（2026-09-17 跑）
> 二、落盘权重全量复核实测（2026-09-28 用 10000 张测试集重测）
> 三、复现方式（**含一个会覆盖权重的坑，务必看**）

## 一、消融实验读数（2026-09-17）

| 编号 | 配置 | 增强 | 激活 | 池化 | 优化器/调度 | 实跑轮数 | 验证集 | 测试集 | 耗时 |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 基线 · 论文原版 | 否 | tanh | avg | adam/step | 11 | 98.94% | **98.89%** | 209s |
| E1 | E0 + 数据增强 | 是 | tanh | avg | adam/step | 15 | 99.30% | **99.32%** | 376s |
| E2 | E1 + AdamW/Cosine/标签平滑 | 是 | tanh | avg | adamw/cosine | 15 | 99.40% | **99.34%** | 378s |
| E3 | E2 + ReLU/MaxPool（结构变体） | 是 | relu | max | adamw/cosine | 15 | 99.36% | **99.33%** | 387s |

- 基线 E0 测试集准确率 98.89%
- 最优配置 **E2**（E1 + AdamW/Cosine/标签平滑），测试集 99.34%，相对基线 +0.45 个百分点
- 全部实验固定 seed=42，验证集 5000 条（从训练集切分），测试集仅用于最终报告

## 二、落盘权重复核实测（2026-09-28，全量 10000 张）

`output/` 下实际存在 4 个权重（含 9-28 新增的兜底副本）。重测结果与权重自带记录逐一对上：

| 文件 | 架构 | 参数量 | 权重自带记录 | 实测 |
|---|---|---|---|---|
| `lenet5_mnist_base.pth` | LeNet5（tanh / avg） | 61,706 | — | **98.84%** |
| `lenet5_mnist.pth` | LeNet5（tanh / avg）+ 增强 + AdamW/Cosine/标签平滑 | 61,706 | `test_acc=99.34` | **99.34%** |
| `lenet5_mnist_E2_99.34.pth` | 同上，兜底副本（防重跑覆盖） | 61,706 | 同左 | **99.34%** |
| `lenet5_mnist_v2.pth` | **LeNet5BN**（BN + ReLU + MaxPool） | 62,614 | `test_acc=99.46` | **99.46%** |

- `lenet5_mnist.pth` 的 `meta` 字段完整记录了配置，可直接查：
  `act=tanh, pool=avg, augment=true, optim=adamw, sched=cosine, label_smoothing=0.05,`
  `lr=1e-3, weight_decay=1e-4, batch_size=64, epochs_run=15, best_epoch=13,`
  `val_acc=99.4, test_acc=99.34, seed=42, seconds=378.5` —— 即上表的 **E2**，实测复现一致。
- **E0 / E1 / E3 的权重都没有落盘**（当时只留了 E2）。所以 E0=98.89% 无法用现存权重复现，
  它是独立一次跑的读数；`lenet5_mnist_base.pth` 则是另一次"改动前"留存权重，实测 98.84%。
  **两个基线数都真实，但不是同一次运行**，对外引用要说清是哪一个。
- `lenet5_mnist_v2.pth` 来自 `lenet5_improved.py`（BN + ReLU/MaxPool + 25 epoch + Cosine + TTA），
  是**换过结构的另一条线，不是"论文原版 LeNet-5"**。它 99.46% 高于 E3 的 99.33%，
  说明：**在经典结构上单独换 ReLU/MaxPool 没有收益（E3），但再加 BatchNorm 才有增益（v2）**。

## 三、复现方式

```bash
C:\Python313\python.exe lenet5_mnist.py        # 默认 10 epoch
```

> ✅ **已修复**（2026-09-29）：`lenet5_mnist.py` 的默认配置现已对齐 E2，并加了防覆盖保护。
> 下面这段风险记录保留，作为"默认配置与实验配置脱节"这个坑的存档。

> ⚠️ **历史上的风险（现已修复）**：旧版脚本默认是
> `optim.Adam(lr=1e-3) + StepLR(step_size=5, gamma=0.5) + CrossEntropyLoss（无标签平滑）`、
> `NUM_EPOCHS=10`，**不等于 E2**，而它保存的路径正是 `output/lenet5_mnist.pth` ——
> 直接跑会用低配置覆盖掉 99.34% 的权重（`web_demo.py` 引用的就是它）。
> 现已在保存前加**存在性检查，默认拒绝覆盖**（加 `--force` 才覆盖，或 `--out-dir` 另存）。
> 兜底副本：`output/lenet5_mnist_E2_99.34.pth`。

**当前复现方式**：直接跑默认配置即可 ——

```bash
python lenet5_mnist.py                 # 默认 = E2，跑完打印验证集与测试集准确率
```

脚本默认值：`AdamW(lr=1e-3, weight_decay=1e-4)` + `CosineAnnealingLR` +
`CrossEntropyLoss(label_smoothing=0.05)`、15 epoch、seed 42、训练集切 5000 条做验证集、
按验证集挑最优 epoch 保存（测试集只在最后评估一次）。
训练完保存的 `.pth` 带 `meta` 字段，记录完整配置与实际成绩，可直接核对。

曲线与图片：`output/ablation.png`、`output/training_curves.png`、
`output/training_curves_v2.png`、`output/sample_predictions.png`、`output/single_test.png`
