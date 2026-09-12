"""
AEGIS-CD — 训练/评估可视化生成脚本
====================================
从 trainValLog.txt 解析训练曲线与测试指标，生成：
  1. loss 曲线图（TrLoss / VaLoss / F1）
  2. 混淆矩阵（由 OA / Recall / Precision 反推 TP/TN/FP/FN）
  3. 模型结果对比图（四数据集指标 + 消融 + 参数量/FLOPs）

用法（从仓库根目录运行）:
    python Visualization/generate_visualizations.py --results-dir Visualization/logs/E4_IndDiff_SCDS --what loss
    python Visualization/generate_visualizations.py --results-dir Visualization/logs/baseline --what all

results-dir 下每个数据集一个子目录，内含 trainValLog.txt：
    <results-dir>/<DATASET>/trainValLog.txt

说明：
  - loss 曲线是训练过程曲线，非「结果」，可在 baseline 跑完前先出（E4 与 baseline 同架构）。
  - 混淆矩阵 / 对比图属于「结果」，须等 baseline 训练完成后用 baseline 日志生成。
"""

import os
import re
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------

DATASETS = ["LEVIR-CD-256", "WHU-CD-256", "SYSU-CD-256", "CDD-CD-256"]
IMG = 256  # 输入尺寸 256×256

# 每个数据集的展示配色（遥感蓝绿为主）
DS_COLOR = {
    "LEVIR-CD-256": "#1f77b4",
    "WHU-CD-256": "#2ca02c",
    "SYSU-CD-256": "#ff7f0e",
    "CDD-CD-256": "#d62728",
}

plt.rcParams.update({
    "font.size": 10,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------

def parse_log(path):
    """解析 trainValLog.txt，返回结构化 dict。测试指标缺失时 test=None。"""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.splitlines()

    d = {
        "epochs": [], "trloss": [], "f1train": [],
        "val_epochs": [], "valloss": [], "oa_val": [], "iou_val": [],
        "f1val": [], "r_val": [], "p_val": [],
        "test": None, "test_count": None, "params": None, "flops": None,
        "best_f1": None, "best_epoch": None,
    }

    for ln in lines:
        # 每 epoch 摘要行：TrLoss + F1(train)
        m = re.search(r"Epoch\s+(\d+)/\d+\s+\|\s+TrLoss=([\d.]+).*?F1\(train\)=([\d.]+)", ln)
        if m:
            d["epochs"].append(int(m.group(1)))
            d["trloss"].append(float(m.group(2)))
            d["f1train"].append(float(m.group(3)))

        # 表格式行：Epoch  TrLoss  VaLoss  OA(val)  IoU(val)  F1(val)  R(val)  P(val)  BestF1
        if re.match(r"^\d+\s+\t", ln):
            parts = [p.strip() for p in ln.split("\t")]
            if len(parts) >= 9 and parts[0].isdigit():
                d["val_epochs"].append(int(parts[0]))
                d["valloss"].append(float(parts[1]))
                d["oa_val"].append(float(parts[3]))
                d["iou_val"].append(float(parts[4]))
                d["f1val"].append(float(parts[5]))
                d["r_val"].append(float(parts[6]))
                d["p_val"].append(float(parts[7]))

        # 测试集结果行
        m = re.search(r"TEST RESULTS\s*\|\s*OA=([\d.]+)\s+IoU=([\d.]+)\s+F1=([\d.]+)\s+R=([\d.]+)\s+P=([\d.]+)", ln)
        if m:
            d["test"] = {
                "OA": float(m.group(1)), "IoU": float(m.group(2)),
                "F1": float(m.group(3)), "R": float(m.group(4)), "P": float(m.group(5)),
            }

        # best F1 @ epoch
        m = re.search(r"BestF1=([\d.]+)\s+@\s+epoch\s+(\d+)", ln)
        if m:
            d["best_f1"] = float(m.group(1))
            d["best_epoch"] = int(m.group(2))

        # 数据集规模
        m = re.search(r"Train / Val / Test\s*:\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", ln)
        if m:
            d["test_count"] = int(m.group(3))

        # 参数量 / FLOPs
        m = re.search(r"Params \(total / trainable\)\s*:\s*([\d.]+)\s*M", ln)
        if m:
            d["params"] = float(m.group(1))
        m = re.search(r"THOP ops\s*:\s*([\d.]+)\s*G", ln)
        if m:
            d["flops"] = float(m.group(1))

    return d


def reconstruct_confusion_matrix(test, n_pixels):
    """由 OA / Recall / Precision 反推混淆矩阵（近似，误差来自 4 位小数舍入）。

    返回按 cm2score 约定的 [[TN, FP], [FN, TP]]（行=真实，列=预测，0=未变化，1=变化）。
    """
    OA, R, P = test["OA"], test["R"], test["P"]
    eps = 1e-12
    denom = 1.0 - 2.0 * R + R / (P + eps)
    if abs(denom) < 1e-12:
        denom = 1e-12
    pos = n_pixels * (1.0 - OA) / denom          # 真实变化像素 = TP + FN
    TP = R * pos
    FN = pos - TP
    FP = TP * (1.0 - P) / (P + eps)
    TN = OA * n_pixels - TP
    return {
        "TN": max(TN, 0.0), "FP": max(FP, 0.0),
        "FN": max(FN, 0.0), "TP": max(TP, 0.0),
    }


# ----------------------------------------------------------------------
# 绘图：loss 曲线
# ----------------------------------------------------------------------

def plot_loss_curves(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 2×2 汇总图
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ds in zip(axes.ravel(), DATASETS):
        if ds not in data:
            ax.set_title(f"{ds} (no log)", fontsize=11)
            ax.axis("off")
            continue
        d = data[ds]
        ax.plot(d["epochs"], d["trloss"], color=DS_COLOR[ds], lw=1.4, label="Train loss")
        ax.plot(d["val_epochs"], d["valloss"], color="#333333", lw=1.4, marker="o", ms=3, label="Val loss")
        if d["best_epoch"]:
            ax.axvline(d["best_epoch"], color="red", ls="--", lw=0.8, alpha=0.7)
            ax.annotate(f"best@{d['best_epoch']}", xy=(d["best_epoch"], ax.get_ylim()[1]),
                        xytext=(4, -2), textcoords="offset points", fontsize=8, color="red")
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("AEGIS-CD  Training / Validation Loss", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curves_2x2.png"))
    plt.close(fig)

    # 2×2 F1 汇总图
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, ds in zip(axes.ravel(), DATASETS):
        if ds not in data:
            ax.set_title(f"{ds} (no log)", fontsize=11)
            ax.axis("off")
            continue
        d = data[ds]
        ax.plot(d["val_epochs"], d["f1val"], color=DS_COLOR[ds], lw=1.6, marker="o", ms=3, label="F1 (val)")
        ax.plot(d["epochs"], d["f1train"], color="#999999", lw=1.0, label="F1 (train)")
        if d["best_epoch"]:
            ax.axvline(d["best_epoch"], color="red", ls="--", lw=0.8, alpha=0.7)
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
        ax.set_ylim(0.75, 1.0)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("AEGIS-CD  F1 vs Epoch", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_curves_2x2.png"))
    plt.close(fig)

    # 每个数据集单独一张（方便贴幻灯片）
    for ds in DATASETS:
        if ds not in data:
            continue
        d = data[ds]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(d["epochs"], d["trloss"], color=DS_COLOR[ds], lw=1.6, label="Train loss")
        ax1.plot(d["val_epochs"], d["valloss"], color="#333333", lw=1.6, marker="o", ms=3, label="Val loss")
        if d["best_epoch"]:
            ax1.axvline(d["best_epoch"], color="red", ls="--", lw=0.9, alpha=0.7, label=f"best@{d['best_epoch']}")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss", color=DS_COLOR[ds])
        ax1.set_title(f"{ds} — Loss & F1", fontsize=12)
        ax2 = ax1.twinx()
        ax2.plot(d["val_epochs"], d["f1val"], color="#d62728", lw=1.6, marker="s", ms=3, label="F1 (val)")
        ax2.set_ylabel("F1 (val)", color="#d62728")
        ax2.set_ylim(0.7, 1.0)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{ds}_loss_f1.png"))
        plt.close(fig)

    print(f"[loss] 已输出到 {out_dir}")


# ----------------------------------------------------------------------
# 绘图：混淆矩阵（结果 —— baseline 完成后生成）
# ----------------------------------------------------------------------

def plot_confusion_matrices(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def _cm_axes(ax, ds, cm, n):
        mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]], dtype=float)
        pct = mat / mat.sum() * 100.0
        im = ax.imshow(mat, cmap="Blues", norm=Normalize(vmin=0, vmax=mat.sum()))
        labels = [["Unchanged\n(TN)", "Changed\n(FP)"], ["Unchanged\n(FN)", "Changed\n(TP)"]]
        ticks = [["Actual Unchanged", "Actual Changed"], ["Pred. Unchanged", "Pred. Changed"]]
        for i in range(2):
            for j in range(2):
                txt = f"{mat[i, j]:,.0f}\n({pct[i, j]:.2f}%)"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if mat[i, j] > mat.sum() * 0.5 else "black",
                        fontsize=9, fontweight="bold")
                ax.text(j, i - 0.38, labels[i][j], ha="center", va="top", fontsize=6.5, color="gray")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Unchanged", "Changed"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Unchanged", "Changed"])
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.set_title(ds, fontsize=11)
        return im

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, ds in zip(axes.ravel(), DATASETS):
        if ds not in data or data[ds]["test"] is None:
            ax.set_title(f"{ds} (pending)", fontsize=11); ax.axis("off"); continue
        d = data[ds]
        n = d["test_count"] * IMG * IMG
        cm = reconstruct_confusion_matrix(d["test"], n)
        _cm_axes(ax, ds, cm, n)
    fig.suptitle("AEGIS-CD  Confusion Matrices (Test)", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrices_2x2.png"))
    plt.close(fig)

    for ds in DATASETS:
        if ds not in data or data[ds]["test"] is None:
            continue
        d = data[ds]
        n = d["test_count"] * IMG * IMG
        cm = reconstruct_confusion_matrix(d["test"], n)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        _cm_axes(ax, ds, cm, n)
        fig.suptitle(f"{ds} — Confusion Matrix (Test)", fontsize=12, y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{ds}_confusion_matrix.png"))
        plt.close(fig)

    print(f"[cm] 已输出到 {out_dir}")


# ----------------------------------------------------------------------
# 绘图：结果对比图（结果 —— baseline 完成后生成）
# ----------------------------------------------------------------------

# 消融数据来自 AGENTS.md 第 9 节（Run13 主实验的移除式消融，测试集 F1 %）
ABLATION = {
    "Full model":        {"LEVIR": 91.46, "SYSU": 83.88},
    "w/o EAOM":          {"LEVIR": 91.34, "SYSU": 80.85},
    "w/o RepDW":         {"LEVIR": 91.57, "SYSU": 82.44},
    "w/o EdgeGate":      {"LEVIR": 91.60, "SYSU": 82.93},
    "w/o Indep. Head":   {"LEVIR": 91.52, "SYSU": 83.16},
}


def plot_comparison(data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ---- 四数据集指标对比（分组柱状图）----
    metrics = ["F1", "IoU", "R", "P"]
    avail = [ds for ds in DATASETS if ds in data and data[ds]["test"] is not None]
    if avail:
        vals = {m: [data[ds]["test"][m] * 100 for ds in avail] for m in metrics}
        x = np.arange(len(avail))
        w = 0.19
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for i, m in enumerate(metrics):
            bars = ax.bar(x + (i - 1.5) * w, vals[m], w, label=m, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"][i])
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                        f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(avail, fontsize=9)
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 105)
        ax.legend(ncol=4, fontsize=8)
        ax.set_title("AEGIS-CD  Test Metrics by Dataset", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "dataset_metrics.png"))
        plt.close(fig)

    # ---- 消融对比 ----
    names = list(ABLATION.keys())
    levir = [ABLATION[k]["LEVIR"] for k in names]
    sysu = [ABLATION[k]["SYSU"] for k in names]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - w / 2, levir, w, label="LEVIR", color="#1f77b4")
    b2 = ax.bar(x + w / 2, sysu, w, label="SYSU", color="#ff7f0e")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("Test F1 (%)")
    ax.set_ylim(78, 95)
    ax.legend(fontsize=9)
    ax.set_title("Ablation Study (Test F1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation.png"))
    plt.close(fig)

    # ---- 参数量 / FLOPs ----
    params = data[avail[0]]["params"] if avail and data[avail[0]]["params"] else None
    flops = data[avail[0]]["flops"] if avail and data[avail[0]]["flops"] else None
    if params and flops:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(["Parameters (M)", "FLOPs (G)"], [params, flops], color=["#1f77b4", "#2ca02c"], width=0.5)
        ax.text(0, params + 0.05, f"{params:.2f} M", ha="center", fontsize=11, fontweight="bold")
        ax.text(1, flops + 0.05, f"{flops:.2f} G", ha="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("Value")
        ax.set_title("AEGIS-CD  Model Efficiency", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "efficiency.png"))
        plt.close(fig)

    print(f"[comparison] 已输出到 {out_dir}（可用数据集: {avail}）")


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="含 <DATASET>/trainValLog.txt 的目录")
    ap.add_argument("--out-dir", default="Visualization")
    ap.add_argument("--what", default="all", choices=["loss", "cm", "comparison", "all"])
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    args = ap.parse_args()

    data = {}
    for ds in args.datasets:
        path = os.path.join(args.results_dir, ds, "trainValLog.txt")
        if os.path.isfile(path):
            data[ds] = parse_log(path)
            t = data[ds]["test"]
            tstr = f"test F1={t['F1']:.4f}" if t else "未完成"
            print(f"  解析 {ds}: epochs={len(data[ds]['epochs'])}, {tstr}")
        else:
            print(f"  跳过 {ds}: 无日志 {path}")

    if args.what in ("loss", "all"):
        plot_loss_curves(data, os.path.join(args.out_dir, "loss_curves"))
    if args.what in ("cm", "all"):
        plot_confusion_matrices(data, os.path.join(args.out_dir, "confusion_matrices"))
    if args.what in ("comparison", "all"):
        plot_comparison(data, os.path.join(args.out_dir, "comparison"))


if __name__ == "__main__":
    main()
