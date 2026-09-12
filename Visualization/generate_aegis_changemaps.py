"""
AEGIS-CD — 测试集二值 changemap 生成（在服务器 GPU 0 上运行）
============================================================
为每个数据集加载 best_model，对 test 集推理，保存二值变化图
（0/255 灰度 PNG），文件名与对比方法一致（<stem>.png）。

用法（服务器，aegiscd 环境）:
    python Visualization/generate_aegis_changemaps.py
"""

import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = "/home/hzeng/project/ZH/AIC"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning)

from models.model import BaseNet
from models.data import dataset as D
from models.data import Transforms as T

DATASETS = ["LEVIR-CD-256", "WHU-CD-256", "SYSU-CD-256", "CDD-CD-256"]
DATA_ROOT = "/home/hzeng/project/ZH/data/CD"
CKPT_DIR = "/home/hzeng/project/ZH/AIC/saved_models/all/Run13/E4_IndDiff_SCDS"
OUT_ROOT = "/home/hzeng/project/ZH/AIC/Visualization/aegis_cd_changemaps"

MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
VAL_TRANSFORM = T.Compose([
    T.Scale(256, 256),
    T.Normalize(mean=MEAN, std=STD),
    T.ToTensor(color_order="fixed"),
])


def strip_thop(sd):
    return {
        k: v for k, v in sd.items()
        if not (k.endswith(".total_ops") or k.endswith(".total_params")
                or k in ("total_ops", "total_params"))
    }


def main():
    device = torch.device("cuda:0")
    for ds in DATASETS:
        ckpt_path = os.path.join(CKPT_DIR, ds, "best_model_F1=*.pth")
        import glob
        ckpt_files = sorted(glob.glob(ckpt_path))
        assert ckpt_files, f"no checkpoint for {ds}"
        ckpt = torch.load(ckpt_files[-1], map_location="cpu", weights_only=False)
        sd = strip_thop(ckpt.get("state_dict", ckpt))

        model = BaseNet()
        model.load_state_dict(sd, strict=True)
        model.to(device).eval()

        dataset_root = os.path.join(DATA_ROOT, ds)
        test_data = D.Dataset("test", file_root=dataset_root,
                              transform=VAL_TRANSFORM, list_name="test")
        loader = torch.utils.data.DataLoader(
            test_data, batch_size=64, shuffle=False, num_workers=4,
            pin_memory=True)

        out_dir = os.path.join(OUT_ROOT, ds)
        os.makedirs(out_dir, exist_ok=True)

        offset = 0
        with torch.no_grad():
            for img, _ in loader:
                pre = img[:, 0:3].to(device).float()
                post = img[:, 3:6].to(device).float()
                prob = model(pre, post)[0]          # [B, 1, H, W] 全分辨率 sigmoid
                pred = (prob[:, 0] > 0.5).cpu().numpy().astype(np.uint8)
                for b in range(pred.shape[0]):
                    name = test_data.file_list[offset + b]
                    stem = os.path.splitext(name)[0]
                    Image.fromarray((pred[b] * 255).astype(np.uint8), mode="L").save(
                        os.path.join(out_dir, f"{stem}.png"))
                offset += pred.shape[0]

        n = len(os.listdir(out_dir))
        print(f"[ok] {ds}: {n} changemaps -> {out_dir}")
    print("DONE")


if __name__ == "__main__":
    main()
