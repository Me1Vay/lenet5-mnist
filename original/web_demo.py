# -*- coding: utf-8 -*-
"""
手写数字识别 Web Demo
- 前端：Gradio Sketchpad 画板（手写 0-9）
- 后端：用训练好的 LeNet-5 模型 (output/lenet5_mnist.pth) 实时预测
- 关键适配：
  1. Gradio 6 的 Sketchpad 继承 ImageEditor，回调收到的是
     {'background', 'layers', 'composite'} 这个 dict（值为 numpy 数组），不是 PIL 图像。
  2. 画布是 RGBA、背景透明，需要 alpha 合成到白底再处理，否则透明区会被当成黑色。
  3. 统一成 MNIST 风格：笔画为亮 -> 裁剪笔画范围 -> 等比缩放 -> 居中放入 32x32 黑底。
启动：python web_demo.py  →  打开 http://127.0.0.1:7860
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lenet5_mnist import LeNet5  # 复用模型定义

DEVICE = torch.device('cpu')
CKPT = os.path.join(HERE, 'output', 'lenet5_mnist.pth')

model = LeNet5(num_classes=10).to(DEVICE)
state = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(state['model_state_dict'])
model.eval()
print("模型已加载:", CKPT)

import torchvision.transforms as T
tf = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,)),
])


def to_pil(img):
    """把 Gradio 6 Sketchpad/ImageEditor 的返回（dict / ndarray / PIL / 路径）统一成 PIL Image"""
    if img is None:
        return None
    if isinstance(img, dict):                     # Gradio 6: {'background','layers','composite'}
        cand = img.get('composite')
        if cand is None:
            layers = [l for l in (img.get('layers') or []) if l is not None]
            cand = layers[-1] if layers else img.get('background')
        img = cand
    if img is None:
        return None
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            return Image.fromarray(arr.astype('uint8'), mode='L')
        if arr.shape[2] == 4:                     # RGBA -> 透明区合成到白底
            a = arr[..., 3:4].astype(np.float32) / 255.0
            rgb = arr[..., :3].astype(np.float32)
            comp = rgb * a + 255.0 * (1.0 - a)
            return Image.fromarray(comp.clip(0, 255).astype('uint8'), mode='RGB')
        return Image.fromarray(arr.astype('uint8'), mode='RGB')
    return img


def preprocess(img):
    """统一成 MNIST 风格：灰度 -> 笔画为亮 -> 裁剪 -> 居中 -> 32x32 黑底"""
    img = to_pil(img)
    if img is None:
        return None
    g = np.array(img.convert('L'), dtype=np.float32)
    # 用四周边框估背景亮度，决定是否需要反相（目标：笔画亮、背景暗）
    bg = np.mean([g[:5, :].mean(), g[-5:, :].mean(),
                  g[:, :5].mean(), g[:, -5:].mean()])
    if bg > 127:                                  # 白底黑笔 -> 反相
        g = 255.0 - g
    # 裁剪到笔画范围
    mask = g > 30
    if mask.any():
        ys, xs = np.where(mask)
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # 等比缩放到最长边 20px，居中放到 28x28 黑底 —— 与 MNIST 原始分布一致
    # （训练时是 28x28 -> Resize(32)，这里保持同样的比例关系）
    h, w = g.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = Image.fromarray(g.clip(0, 255).astype('uint8'), mode='L').resize(
        (nw, nh), Image.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(small, ((28 - nw) // 2, (28 - nh) // 2))
    return canvas


def predict(img):
    if img is None:
        return "请先在左侧画板写个数字～", {}, None
    pil = preprocess(img)
    if pil is None or not np.array(pil).any():
        return "画板是空的，先写个数字再点「识别」～", {}, None

    x = tf(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1)[0].cpu().numpy()
    pred = int(prob.argmax())
    conf = float(prob[pred])
    bars = {str(i): float(prob[i]) for i in range(10)}
    preview = pil.resize((160, 160), Image.NEAREST)   # 放大显示模型实际输入
    return f"预测数字: {pred}   置信度: {conf * 100:.1f}%", bars, preview


with gr.Blocks(title="手写数字识别 · LeNet-5") as demo:
    gr.Markdown(
        "# ✏️ 手写数字识别 (LeNet-5)\n"
        "在左侧画板用鼠标写 0-9，点「识别」查看结果。"
        "后端模型在 MNIST 上训练到约 **98.8%** 准确率。"
    )
    with gr.Row():
        with gr.Column():
            sketch = gr.Sketchpad(label="在此手写数字", height=320, width=320,
                                  canvas_size=(280, 280), fixed_canvas=True)
            with gr.Row():
                btn = gr.Button("识别", variant="primary")
                clear = gr.Button("清除")
        with gr.Column():
            out_text = gr.Textbox(label="识别结果")
            out_bar = gr.Label(label="各类别概率", num_top_classes=10)
            out_img = gr.Image(label="模型实际看到的输入 (28×28→32×32)", height=180)

    btn.click(predict, inputs=sketch, outputs=[out_text, out_bar, out_img])
    clear.click(lambda: (None, None, None, None), None,
                [sketch, out_text, out_bar, out_img], queue=False)

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=7860, share=False,
                inbrowser=False, theme=gr.themes.Soft())
