"""
AEGIS-CD — 变化检测结果图（change map）生成与拼图
==================================================
用本地 web_demo/weights（E4 权重）+ web_demo/samples（真实样例）在 CPU 上推理，
对每个数据集挑选 3 个不同变化比例的样例，生成
「T1 / T2 / 真值 / 预测 / 彩色变化图（TP白/FP红/FN绿）」五列拼图。

用法（从仓库根目录运行）:
    python Visualization/generate_change_maps.py
"""

import os
import sys
import glob
import warnings

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from models.model import BaseNet
from models.data import Transforms as T

DATASETS = ["LEVIR-CD-256", "WHU-CD-256", "SYSU-CD-256", "CDD-CD-256"]
CKPT = {
    "LEVIR-CD-256": "LEVIR-CD-256_best_model_F1=0.9170.pth",
    "WHU-CD-256": "WHU-CD-256_best_model_F1=0.9453.pth",
    "SYSU-CD-256": "SYSU-CD-256_best_model_F1=0.8126.pth",
    "CDD-CD-256": "CDD-CD-256_best_model_F1=0.9711.pth",
}
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "web_demo", "weights")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "web_demo", "samples")
OUT_DIR = os.path.join(PROJECT_ROOT, "Visualization", "change_maps")

MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
TRANSFORM = T.Compose([
    T.Scale(256, 256),
    T.Normalize(mean=MEAN, std=STD),
    T.ToTensor(color_order="fixed"),
])

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _strip_thop(sd):
    return {
        k: v for k, v in sd.items()
        if not (k.endswith(".total_ops") or k.endswith(".total_params")
                or k in ("total_ops", "total_params"))
    }


def build_model(ds):
    ckpt = torch.load(os.path.join(WEIGHTS_DIR, CKPT[ds]),
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = _strip_thop(sd)
    model = BaseNet()  # 默认 == E4_IndDiff_SCDS
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer(model, a_bgr, b_bgr):
    img = np.concatenate([a_bgr, b_bgr], axis=2)  # [H, W, 6] BGR
    dummy = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    x, _ = TRANSFORM(img, dummy)
    x = x.unsqueeze(0)
    pre, post = x[:, 0:3], x[:, 3:6]
    out = model(pre, post)[0]          # 主头全分辨率 sigmoid 概率
    prob = out[0, 0].numpy()
    pred = (prob > 0.5).astype(np.uint8)
    return pred


def color_map(pred, gt):
    cmap = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    cmap[(pred == 1) & (gt == 1)] = [255, 255, 255]   # TP 白
    cmap[(pred == 1) & (gt == 0)] = [255, 0, 0]       # FP 红
    cmap[(pred == 0) & (gt == 0)] = [0, 0, 0]         # TN 黑
    cmap[(pred == 0) & (gt == 1)] = [0, 255, 0]       # FN 绿
    return cmap


def f1_iou(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)
    return f1, iou


def list_samples(ds):
    sdir = os.path.join(SAMPLES_DIR, ds)
    stems = sorted(f[:-len("_A.png")] for f in os.listdir(sdir) if f.endswith("_A.png"))
    out = []
    for stem in stems:
        a = cv2.imread(os.path.join(sdir, f"{stem}_A.png"), cv2.IMREAD_COLOR)
        b = cv2.imread(os.path.join(sdir, f"{stem}_B.png"), cv2.IMREAD_COLOR)
        lb = cv2.imread(os.path.join(sdir, f"{stem}_label.png"), cv2.IMREAD_GRAYSCALE)
        gt = (lb > 127).astype(np.uint8)
        ratio = float(gt.mean())
        out.append((stem, a, b, gt, ratio))
    return out


def compose(ds, samples, model):
    cols = ["T1 (pre-change)", "T2 (post-change)", "Ground Truth", "Prediction", "Change Map"]
    n = len(samples)
    fig, axes = plt.subplots(n, 5, figsize=(5 * 2.0, n * 2.0 + 0.6))
    if n == 1:
        axes = axes[None, :]

    for i, (stem, a, b, gt, ratio) in enumerate(samples):
        pred = infer(model, a, b)
        cm = color_map(pred, gt)
        f1, iou = f1_iou(pred, gt)
        a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        b_rgb = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        gt_bin = (gt * 255).astype(np.uint8)
        pred_bin = (pred * 255).astype(np.uint8)
        row = [a_rgb, b_rgb, gt_bin, pred_bin, cm]
        cmaps = [None, None, "gray", "gray", None]
        for j, im in enumerate(row):
            ax = axes[i, j]
            if cmaps[j]:
                ax.imshow(im, cmap=cmaps[j], vmin=0, vmax=255)
            else:
                ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"{stem}\n(change {ratio*100:.1f}%)",
                              fontsize=7, rotation=0, ha="right", va="center", labelpad=28)
            if i == 0:
                ax.set_title(cols[j], fontsize=9)
        axes[i, 4].text(0.02, 0.06, f"F1={f1:.3f}  IoU={iou:.3f}",
                        transform=axes[i, 4].transAxes, fontsize=7.5,
                        color="yellow", bbox=dict(facecolor="black", alpha=0.5, pad=1))

    fig.suptitle(f"{ds} — Change Detection", fontsize=12, y=0.99)
    fig.text(0.5, 0.005, "Change Map: white = TP, red = FP, green = FN (black = unchanged)",
             ha="center", fontsize=7.5, color="gray")
    fig.tight_layout(rect=[0.08, 0.03, 1, 0.97])
    out = os.path.join(OUT_DIR, f"{ds}_change_maps.png")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {ds}: 已生成 {out}（{n} 个样例）")


def main():
    for ds in DATASETS:
        print(f"== {ds} ==")
        model = build_model(ds)
        samples = list_samples(ds)
        # 按变化比例排序，取高/中/低三个，展示不同难度
        samples_sorted = sorted(samples, key=lambda s: -s[4])
        idxs = [0, len(samples_sorted) // 2, len(samples_sorted) - 1]
        picked = [samples_sorted[k] for k in idxs]
        compose(ds, picked, model)


if __name__ == "__main__":
    main()
