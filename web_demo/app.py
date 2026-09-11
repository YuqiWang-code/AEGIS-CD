"""
AEGIS-CD — 变化检测演示系统 (Flask backend)
=============================================
Run13 E4_IndDiff_SCDS architecture (Siamese MobileNetV2 + HFEA + EAOM + RepDW
decoder + EdgeGate + scale-decoupled independent heads + native SCDS).

Serves a single-page browser UI and a JSON API.  Loads one model per dataset
(all four share the same architecture, only the trained weights differ).

Run from the project root (it chdir's there itself so the MobileNetV2
pretrained-weight relative path resolves):

    python web_demo/app.py            # http://127.0.0.1:5000
    python web_demo/app.py --port 8000

The server-side `--port` is optional.  Works identically on the training server
(conda env `aegiscd`) if you upload this folder and run it there.
"""

import os
import sys
import io
import time
import base64
import warnings

# Suppress the harmless pkg_resources deprecation notice from pytorch_wavelets'
# DTCWT sub-package (we only use the DWT, which does not touch pkg_resources).
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- resolve the project root and make `models.*` importable ---
_DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_DEMO_DIR)
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory, abort

# Use all physical cores for the CPU-bound DWT/grid_sample ops.
torch.set_num_threads(max(1, os.cpu_count() or 1))

from models.model import BaseNet
from models.data import Transforms as T
from models.utils.metric_tool import cm2score, get_confuse_matrix


# =========================================================
#  Configuration
# =========================================================

DATASETS = ["LEVIR-CD-256", "WHU-CD-256", "SYSU-CD-256", "CDD-CD-256"]

CKPT_FILES = {
    "LEVIR-CD-256": "best_model_F1=0.9170.pth",
    "WHU-CD-256": "best_model_F1=0.9453.pth",
    "SYSU-CD-256": "best_model_F1=0.8126.pth",
    "CDD-CD-256": "best_model_F1=0.9711.pth",
}

# Test-set reference metrics (from trainValLog.txt, val-selected, no leakage).
REFERENCE = {
    "LEVIR-CD-256": dict(F1=0.9146, IoU=0.8426, P=0.9190, R=0.9103, OA=0.9913),
    "WHU-CD-256": dict(F1=0.9424, IoU=0.8911, P=0.9593, R=0.9261, OA=0.9955),
    "SYSU-CD-256": dict(F1=0.8388, IoU=0.7223, P=0.8699, R=0.8098, OA=0.9266),
    "CDD-CD-256": dict(F1=0.9715, IoU=0.9445, P=0.9776, R=0.9654, OA=0.9927),
}

# Exact val preprocessing: BGR image pair -> 6ch normalized RGB tensor.
MEAN = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
STD = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
TRANSFORM = T.Compose([
    T.Scale(256, 256),
    T.Normalize(mean=MEAN, std=STD),
    T.ToTensor(color_order="fixed"),
])

WEIGHTS_DIR = os.path.join(_DEMO_DIR, "weights")
SAMPLES_DIR = os.path.join(_DEMO_DIR, "samples")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


# =========================================================
#  Model loading
# =========================================================

def _strip_thop(state_dict):
    """Remove thop.profile's injected total_ops/total_params buffers.

    The best_model checkpoints were saved with thop's FLOPs-counting buffers
    still registered, so a raw ``strict=True`` load would see ~188 unexpected
    keys.  They carry no learned parameters and are safe to drop.
    """
    return {
        k: v for k, v in state_dict.items()
        if not (k.endswith(".total_ops") or k.endswith(".total_params")
                or k in ("total_ops", "total_params"))
    }


def _build_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _strip_thop(state_dict)

    model = BaseNet()  # defaults == Run13 E4_IndDiff_SCDS
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


print("Loading AEGIS-CD models (E4_IndDiff_SCDS) ...")
MODELS = {}
for ds in DATASETS:
    path = os.path.join(WEIGHTS_DIR, f"{ds}_{CKPT_FILES[ds]}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    MODELS[ds] = _build_model(path)
    print(f"  [ok] {ds}")

_n_params = sum(p.numel() for p in next(iter(MODELS.values())).parameters())
print(f"Ready.  {len(MODELS)} models loaded, {_n_params / 1e6:.2f} M params each.")

# Warm-up: spin up CPU thread pools and JIT-fuse kernels so the first
# user prediction is not a cold start.
print("Warming up ...")
with torch.no_grad():
    _dummy = torch.zeros(1, 3, 256, 256)
    for _m in MODELS.values():
        _m(_dummy, _dummy)
print("Warm-up done.")


# =========================================================
#  Inference
# =========================================================

def _encode_png(img_bgr_or_rgb, is_bgr=False):
    """Encode a HxW(x3) uint8 numpy array as a base64 PNG data URL."""
    if is_bgr:
        img_bgr_or_rgb = cv2.cvtColor(img_bgr_or_rgb, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", img_bgr_or_rgb)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _run_inference(model, a_img, b_img):
    """a_img/b_img: BGR uint8 HxWx3.  Returns (prob, pred, infer_ms)."""
    img = np.concatenate([a_img, b_img], axis=2)  # [H, W, 6] BGR
    dummy = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    x, _ = TRANSFORM(img, dummy)
    x = x.unsqueeze(0)
    pre, post = x[:, 0:3], x[:, 3:6]
    with torch.no_grad():
        t0 = time.time()
        out = model(pre, post)[0]          # primary full-res sigmoid prob
        infer_ms = (time.time() - t0) * 1000.0
    prob = out[0, 0].numpy()
    pred = (prob > 0.5).astype(np.uint8)
    return prob, pred, infer_ms


def _render_outputs(a_bgr, b_bgr, prob, pred):
    """Build display images (RGB) + encoded PNG data URLs."""
    t1_rgb = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB)
    t2_rgb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB)

    # binary change mask (white = change)
    mask = (pred * 255).astype(np.uint8)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # probability heatmap (jet)
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # overlay: predicted change tinted red over T2
    red = np.zeros_like(t2_rgb)
    red[..., 0] = 255
    overlay = np.where(pred[..., None] > 0, (0.5 * t2_rgb + 0.5 * red), t2_rgb)
    overlay = overlay.astype(np.uint8)

    return {
        "t1": _encode_png(t1_rgb),
        "t2": _encode_png(t2_rgb),
        "mask": _encode_png(mask_rgb),
        "heat": _encode_png(heat_rgb),
        "overlay": _encode_png(overlay),
    }


def _metrics_from(prob, pred, gt=None):
    result = {
        "pred_change_ratio": round(float(pred.mean()), 4),
        "prob_mean": round(float(prob.mean()), 4),
    }
    if gt is not None:
        cm = get_confuse_matrix(2, [gt], [pred])
        s = cm2score(cm)
        result.update({
            "F1": round(float(s["F1"]), 4),
            "IoU": round(float(s["IoU"]), 4),
            "precision": round(float(s["precision"]), 4),
            "recall": round(float(s["recall"]), 4),
            "OA": round(float(s["OA"]), 4),
            "kappa": round(float(s["Kappa"]), 4),
        })
    return result


# =========================================================
#  Routes
# =========================================================

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/meta")
def meta():
    return jsonify({
        "model_name": "AEGIS-CD (Run13 E4_IndDiff_SCDS)",
        "description": (
            "轻量化遥感双时相变化检测：Siamese MobileNetV2 + HFEA + EAOM + "
            "RepDW 解码器 + EdgeGate + 尺度解耦独立头 (native SCDS)。"
        ),
        "params_m": round(_n_params / 1e6, 2),
        "datasets": [
            {
                "name": ds,
                "reference": REFERENCE[ds],
                "ckpt": CKPT_FILES[ds],
            }
            for ds in DATASETS
        ],
    })


@app.route("/api/samples")
def samples():
    items = []
    for ds in DATASETS:
        sdir = os.path.join(SAMPLES_DIR, ds)
        if not os.path.isdir(sdir):
            continue
        for f in sorted(os.listdir(sdir)):
            if not f.endswith("_A.png"):
                continue
            stem = f[:-len("_A.png")]
            a = f"{stem}_A.png"
            b = f"{stem}_B.png"
            label = f"{stem}_label.png"
            if not all(os.path.isfile(os.path.join(sdir, x)) for x in (b, label)):
                continue
            # change ratio from label for the gallery caption
            lb = cv2.imread(os.path.join(sdir, label), cv2.IMREAD_GRAYSCALE)
            ratio = float((lb > 127).mean()) if lb is not None else 0.0
            items.append({
                "id": f"{ds}/{stem}",
                "dataset": ds,
                "name": stem,
                "a": f"/samples/{ds}/{a}",
                "b": f"/samples/{ds}/{b}",
                "label": f"/samples/{ds}/{label}",
                "change_ratio": round(ratio, 4),
            })
    return jsonify({"samples": items})


@app.route("/samples/<path:filename>")
def sample_file(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    ds = data.get("dataset")
    name = data.get("name")
    if ds not in MODELS:
        return jsonify({"error": f"unknown dataset {ds!r}"}), 400

    sdir = os.path.join(SAMPLES_DIR, ds)
    a_path = os.path.join(sdir, f"{name}_A.png")
    b_path = os.path.join(sdir, f"{name}_B.png")
    l_path = os.path.join(sdir, f"{name}_label.png")
    if not all(os.path.isfile(p) for p in (a_path, b_path)):
        return jsonify({"error": f"sample {name!r} not found in {ds}"}), 404

    a_img = cv2.imread(a_path, cv2.IMREAD_COLOR)
    b_img = cv2.imread(b_path, cv2.IMREAD_COLOR)
    prob, pred, infer_ms = _run_inference(MODELS[ds], a_img, b_img)

    gt = None
    if os.path.isfile(l_path):
        lb = cv2.imread(l_path, cv2.IMREAD_GRAYSCALE)
        gt = (lb > 127).astype(np.uint8)

    out = _render_outputs(a_img, b_img, prob, pred)
    out["metrics"] = _metrics_from(prob, pred, gt)
    out["metrics"]["infer_ms"] = round(infer_ms, 2)
    out["dataset"] = ds
    out["name"] = name
    return jsonify(out)


@app.route("/api/upload_predict", methods=["POST"])
def upload_predict():
    if "t1" not in request.files or "t2" not in request.files:
        return jsonify({"error": "需要上传 t1 和 t2 两张图片"}), 400
    ds = request.form.get("dataset", DATASETS[0])
    if ds not in MODELS:
        return jsonify({"error": f"unknown dataset {ds!r}"}), 400

    def _read(field):
        f = request.files[field]
        buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"{field} 无法解析为图片")
        return img

    try:
        a_img = _read("t1")
        b_img = _read("t2")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prob, pred, infer_ms = _run_inference(MODELS[ds], a_img, b_img)
    out = _render_outputs(a_img, b_img, prob, pred)
    out["metrics"] = _metrics_from(prob, pred, gt=None)
    out["metrics"]["infer_ms"] = round(infer_ms, 2)
    out["dataset"] = ds
    out["name"] = "upload"
    return jsonify(out)


if __name__ == "__main__":
    port = 5000
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print(f"\n  AEGIS-CD 演示系统已启动 →  http://127.0.0.1:{port}\n")

    # Open the default browser once the server socket is bound (models are
    # already loaded above).  Pass --no-browser to disable (e.g. on the server).
    if "--no-browser" not in sys.argv:
        import threading
        import webbrowser
        threading.Timer(
            1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")
        ).start()

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
