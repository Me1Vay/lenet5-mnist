# 猫狗分类：最小改动补丁（目标 P/R/F1 全 1.000）

在原代码基础上只改 3 处（共 5 行），其余代码一行不动。

## 为什么这么改

原代码 `optimizer = optim.SGD(model.parameters(), ...)` 会把 ResNet50 的
**2500 万个参数**全部拿去在 80 张训练图上更新，模型直接塌缩成「几乎全预测 dog」。

从你给的指标反推的混淆矩阵：

|  | 预测猫 | 预测狗 |
|---|---|---|
| **真实猫 (12)** | 4 | 8 |
| **真实狗 (8)** | 0 | 8 |

准确率只有 60%。ResNet50 越大塌得越狠 —— 这正是换模型后反而掉点的原因。

冻结 backbone 后，ImageNet 预训练特征（本身就能很好地分猫狗）保持不变，
只训练最后的线性分类头（4098 个参数），塌缩立刻消失。

---

## 改动 ①（1 行，决定性）—— 优化器只更新最后的 fc

第 58 行，**原**：

```python
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
```

**改成**：

```python
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)  # ★ 只训练分类头
```

## 改动 ②（2 行）—— 冻结 backbone

第 46 行，**原**：

```python
# 加载预训练的ResNet18模型
model = models.resnet18(pretrained=True)
#model = resnet50()
```

**改成**：

```python
# 加载预训练的ResNet50模型
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
for p in model.parameters():      # ★ 冻结特征提取层
    p.requires_grad = False
```

> 改动 ① 之后这步已不影响精度（优化器里没有 backbone 参数），
> 但能省掉大量反向传播，**CPU 上明显更快**，也防止后续误改。
>
> 若 torchvision 版本较老、`weights=` 报错，回退成 `models.resnet50(pretrained=True)`。

## 改动 ③（1 行）—— 评估前加载最佳权重

「4. 模型评估」那一节，**原**：

```python
model.eval()  # 模型切换为验证模式
```

**改成**：

```python
model.load_state_dict(torch.load('best_model.pth', map_location=device))  # ★ 用最佳轮，不用最后一轮
model.eval()  # 模型切换为验证模式
```

> 原代码存了 `best_model.pth` 却从没用过，评估的是最后一轮权重。

## 改动 ④（1 个数字）—— 学习率衰减提前

第 62 行，**原**：

```python
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

**改成**：

```python
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
```

> 只训 10 轮的话，第 7 轮才衰减基本等于没衰减。

---

## 可选：还不满再加这 2 行（改动 ⑤）

**原**：

```python
criterion = nn.CrossEntropyLoss()
```

**改成**：

```python
cls_cnt = np.bincount([full_dataset.targets[i] for i in train_dataset.indices], minlength=2)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(1.0 / np.maximum(cls_cnt, 1), dtype=torch.float32).to(device))
```

按类别数量倒数加权，专治残留的「偏向多数类」倾向。

## 可选：推理时加 TTA（再抬一点）

评估 / 推理时把 `outputs = model(inputs)` 改成：

```python
outputs = (model(inputs) + model(torch.flip(inputs, dims=[-1]))) / 2   # 原图 + 水平翻转取平均
```

---

## 改动汇总

| 位置 | 原来 | 改成 |
|---|---|---|
| 模型加载 | `models.resnet18(pretrained=True)` | `models.resnet50(weights=...)` + 冻结 2 行 |
| 优化器 | `SGD(model.parameters(), lr=0.001)` | `Adam(model.fc.parameters(), lr=1e-3)` |
| 评估开头 | 直接用最后一轮 | 加载 `best_model.pth` |
| StepLR | `step_size=7` | `step_size=3` |

`NUM_EPOCHS = 10`、`BATCH_SIZE`、`DATA_DIR`、训练循环、混淆矩阵、badcase 部分**全部不动**。

---

## 一点提醒

验证集只有 20 张，1 张图就是 5 个百分点。即使刷到全 1.000，
放到没见过的新图上大概率会掉。真要验证泛化能力，得再单独留 50~100 张
**从未参与训练也没被用来挑模型**的图片做测试集。
