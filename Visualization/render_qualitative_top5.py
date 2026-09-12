"""
AEGIS-CD — 定性对比图渲染（每数据集挑 5 组，过滤变化比例下限）
================================================================
从 shortlist CSV 读取候选样本，过滤 GT 变化比例 >= 下限，按推荐分数取前 5，
渲染成 PNG：5 行 × [T1, T2, GT, AEGIS-CD, 14 对比方法] 共 18 列。

在服务器运行（aegiscd 环境）:
    /home/hzeng/envs/aegiscd/bin/python3.10 Visualization/render_qualitative_top5.py
"""

import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DATA_ROOT = "/home/hzeng/project/ZH/data/CD"
BASE = "/home/hzeng/project/ZH/AIC/Visualization"
QUAL = f"{BASE}/qualitative"
OUT = f"{BASE}/qualitative_top5"

MIN_CHANGE_RATIO = 1.0    # 变化比例下限（百分比）
MAX_CHANGE_RATIO = 50.0   # 变化比例上限（排除 100% 全变等退化样本）

DATASETS = ["LEVIR-CD-256", "WHU-CD-256", "SYSU-CD-256", "CDD-CD-256"]
PREFIX = {
    "LEVIR-CD-256": "levir_cd_256",
    "WHU-CD-256": "whu_cd_256",
    "SYSU-CD-256": "sysu_cd_256",
    "CDD-CD-256": "cdd_cd_256",
}
COMPARISON = ["RS-Mamba", "DSIFN", "SNUNet", "Change3D", "BIT", "ELGC-Net",
              "BiFA", "MaskCD", "ChangeRD", "CAIFNet", "WDMF-Net"]
ALL_METHODS = ["AEGIS-CD"] + COMPARISON
COLS = ["T1", "T2", "GT"] + ALL_METHODS

IMG_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def resolve(d, name):
    p = os.path.join(d, name)
    if os.path.exists(p):
        return p
    stem = os.path.splitext(name)[0]
    for ext in IMG_EXTS:
        p = os.path.join(d, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_gray(p):
    return np.asarray(Image.open(p).convert("L")) > 127


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


def error_overlay(pred, gt):
    rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    rgb[(pred) & (gt)] = [255, 255, 255]      # TP 白
    rgb[(pred) & (~gt)] = [255, 0, 0]         # FP 红/虚警
    rgb[(~pred) & (gt)] = [0, 255, 0]         # FN 绿/漏检
    return rgb


def changemap_path(ds, method, sample):
    stem = os.path.splitext(sample)[0]
    if method == "AEGIS-CD":
        return resolve(f"{BASE}/aegis_cd_changemaps/{ds}", f"{stem}.png")
    return resolve(f"{BASE}/comparison_changemaps/{method}/{ds}", f"{stem}.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    for ds in DATASETS:
        csv_path = f"{QUAL}/{ds}/{PREFIX[ds]}_qualitative_shortlist.csv"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))

        # 过滤变化比例区间 + 按原推荐分数顺序取前 5
        picked = [r for r in rows
                  if MIN_CHANGE_RATIO <= float(r["GT_change_ratio_percent"]) <= MAX_CHANGE_RATIO][:5]
        if len(picked) < 5:  # 兜底：过滤后不足 5 则放宽
            picked = rows[:5]
        print(f"{ds}: 过滤后取 {len(picked)} 组 -> "
              + ", ".join(f"{r['sample']}({r['GT_change_ratio_percent']}%)"
                          for r in picked))

        n = len(picked)
        # 精确计算 figsize，使每个子图正好为正方形（CELL×CELL），无上下留白
        ncols = len(COLS)
        CELL = 0.72
        WSPACE = 0.05    # 列间小缝
        HSPACE = 0.02    # 行间（极小的缝）
        LEFT, RIGHT, TOP, BOTTOM = 0.13, 0.995, 0.92, 0.035
        fig_w = CELL * (ncols + (ncols - 1) * WSPACE) / (RIGHT - LEFT)
        fig_h = CELL * (n + (n - 1) * HSPACE) / (TOP - BOTTOM)
        fig, axes = plt.subplots(n, ncols, figsize=(fig_w, fig_h))
        if n == 1:
            axes = axes[None, :]

        for i, r in enumerate(picked):
            sample = r["sample"]
            ratio = float(r["GT_change_ratio_percent"])
            gt_path = resolve(f"{DATA_ROOT}/{ds}/label", sample)
            a_path = resolve(f"{DATA_ROOT}/{ds}/A", sample)
            b_path = resolve(f"{DATA_ROOT}/{ds}/B", sample)
            gt = load_gray(gt_path)
            t1 = load_rgb(a_path)
            t2 = load_rgb(b_path)
            gt_img = np.stack([np.where(gt, 255, 0).astype(np.uint8)] * 3, axis=-1)

            imgs = [t1, t2, gt_img]
            for m in ALL_METHODS:
                cp = changemap_path(ds, m, sample)
                pred = load_gray(cp) if cp else np.zeros_like(gt)
                imgs.append(error_overlay(pred, gt))

            for j, im in enumerate(imgs):
                ax = axes[i, j]
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])
                if COLS[j] == "AEGIS-CD":
                    for s in ax.spines.values():
                        s.set_color("#2563eb")
                        s.set_linewidth(2)
                if i == 0:
                    ax.set_title(COLS[j], fontsize=6.5,
                                 fontweight="bold" if COLS[j] == "AEGIS-CD" else "normal",
                                 color="#2563eb" if COLS[j] == "AEGIS-CD" else "black")
            stem = os.path.splitext(sample)[0]
            axes[i, 0].set_ylabel(f"{stem}\n{ratio:.2f}%", fontsize=6,
                                  rotation=0, ha="right", va="center", labelpad=6)

        fig.text(0.5, 0.02,
                 "AEGIS-CD (blue box) vs. baselines.  Overlay: white=TP, red=FP, green=FN",
                 ha="center", fontsize=7, color="gray")
        fig.subplots_adjust(left=LEFT, bottom=BOTTOM, top=TOP, right=RIGHT,
                             wspace=WSPACE, hspace=HSPACE)
        out_path = f"{OUT}/{ds}_qualitative_top5.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  [ok] 已保存 {out_path}")

    print("DONE")


if __name__ == "__main__":
    main()
