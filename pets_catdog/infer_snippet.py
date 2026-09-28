# -*- coding: utf-8 -*-
"""
猫狗分类 —— 推理代码片段

用法：整段复制，粘到你现有 notebook 的最后一个 cell 里执行即可。
它会复用 notebook 里已经存在的变量：model / class_names / device / data_transform。
"""

import os
from PIL import Image


@torch.no_grad()
def predict_probs(img_path):
    """输入图片路径 -> (类别名, 置信度, {类别: 概率})

    例：('dog', 0.9731, {'cat': 0.0269, 'dog': 0.9731})
    """
    img = Image.open(img_path).convert('RGB')
    x = data_transform(img).unsqueeze(0).to(device)   # 复用训练时的同一套预处理
    model.eval()
    # TTA：原图 + 水平翻转取平均，结果更稳
    logits = (model(x) + model(torch.flip(x, dims=[-1]))) / 2
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    k = int(prob.argmax())
    return class_names[k], float(prob[k]), {c: float(p) for c, p in zip(class_names, prob)}


def predict_show(img_path):
    """推理单张图：打印结果 + 显示图片"""
    label, conf, probs = predict_probs(img_path)
    print('图片:', img_path)
    print('预测:', label, '  置信度: {:.2%}'.format(conf))
    for c, p in probs.items():
        print('    {}: {:.2%}'.format(c, p))

    plt.figure(figsize=(5, 5))
    plt.imshow(Image.open(img_path).convert('RGB'))
    plt.axis('off')
    plt.title('{}  {:.1%}'.format(label, conf), fontsize=16)
    plt.show()
    return label, conf, probs


def predict_dir(directory, show_grid=True):
    """批量推理整个目录（含子目录），返回 [(路径, 类别, 置信度), ...]"""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    paths = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))

    if not paths:
        print('没找到图片:', directory)
        return []

    print('共找到 {} 张图'.format(len(paths)))
    results = []
    for p in paths:
        label, conf, _ = predict_probs(p)
        results.append((p, label, conf))
        print('{}  ->  {}  {:.2%}'.format(p, label, conf))

    if show_grid and len(results) > 1:
        cols = min(5, len(results))
        rows = (len(results) + cols - 1) // cols
        plt.figure(figsize=(3 * cols, 3.2 * rows))
        for i, (p, label, conf) in enumerate(results):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(Image.open(p).convert('RGB'))
            plt.axis('off')
            plt.title('{} {:.0%}'.format(label, conf))
        plt.tight_layout()
        plt.show()

    return results


# ==========================================================
# 用法：把路径换成你的图片
# ==========================================================

# 单张推理（会显示图片和猫/狗概率）
# predict_show('Pets/dog/80bd9527bd775e6634af8105d1be8cac.jpg')

# 批量推理整个目录
# predict_dir('test_images/')

# 只要结果不要图
# label, conf, probs = predict_probs('你的图片路径.jpg')
# print(label, conf)


# ----------------------------------------------------------
# 如果 kernel 重启过（model / class_names 没了），先跑这一格重建：
# ----------------------------------------------------------
def reload_model(ckpt='best_model.pth'):
    """从 best_model.pth 重建模型，返回 model"""
    global model, class_names, NUM_CLASSES
    if 'NUM_CLASSES' not in dir():
        NUM_CLASSES = 2
    class_names = ['cat', 'dog']          # ImageFolder 按文件夹名排序：cat < dog
    try:
        m = models.resnet50(weights=None)
    except TypeError:
        m = models.resnet50(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.to(device).eval()
    print('已加载', ckpt, '| device =', device)
    return m


# kernel 重启后请执行：
# model = reload_model('best_model.pth')
