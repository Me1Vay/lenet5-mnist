# -*- coding: utf-8 -*-
"""
猫狗分类独立推理脚本（不依赖 notebook）

用法：
    python predict.py Pets/dog/xxx.jpg
    python predict.py test_images/            # 整个目录批量推理
    python predict.py a.jpg --ckpt best_model.pth --classes cat,dog

注意：--classes 的顺序必须和训练时 ImageFolder 的类别顺序一致。
ImageFolder 按文件夹名排序，所以默认是 cat,dog。
"""
import argparse
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model(ckpt='best_model.pth', num_classes=2, device=None):
    """加载 best_model.pth，网络结构必须和训练时一致（ResNet50 + 新 fc）"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        m = models.resnet50(weights=None)
    except TypeError:
        m = models.resnet50(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict(state)
    m.to(device).eval()
    return m, device


@torch.no_grad()
def predict(path, model, device, class_names):
    """返回 (类别名, 置信度, 各类别概率 list)"""
    img = Image.open(path).convert('RGB')
    x = tf(img).unsqueeze(0).to(device)
    logits = (model(x) + model(torch.flip(x, dims=[-1]))) / 2  # TTA
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    k = int(prob.argmax())
    return class_names[k], float(prob[k]), prob


def collect_images(target):
    if os.path.isdir(target):
        paths = []
        for root, _, files in os.walk(target):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXTS):
                    paths.append(os.path.join(root, f))
        return paths
    return [target]


def main():
    ap = argparse.ArgumentParser(description='猫狗分类推理')
    ap.add_argument('img', help='图片路径或目录')
    ap.add_argument('--ckpt', default='best_model.pth', help='模型权重，默认 best_model.pth')
    ap.add_argument('--classes', default='cat,dog', help='类别名顺序，逗号分隔')
    args = ap.parse_args()

    class_names = [c.strip() for c in args.classes.split(',')]
    model, device = load_model(args.ckpt, len(class_names))
    print('device =', device)

    paths = collect_images(args.img)
    if not paths:
        print('没找到图片:', args.img)
        return

    for p in paths:
        label, conf, prob = predict(p, model, device, class_names)
        detail = '  '.join('{}={:.3f}'.format(n, float(v)) for n, v in zip(class_names, prob))
        print('{}  ->  {}  ({:.3f})   [{}]'.format(p, label, conf, detail))


if __name__ == '__main__':
    main()
