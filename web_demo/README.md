# AEGIS-CD 变化检测演示系统

浏览器可打开的遥感双时相变化检测演示，用于比赛演示视频录屏。

- 模型：**AEGIS-CD（Run13 E4_IndDiff_SCDS）**
  Siamese MobileNetV2 + HFEA + EAOM + RepDW 解码器 + EdgeGate + 尺度解耦独立头（native SCDS）。
- 权重：`weights/` 下 4 个数据集（LEVIR / WHU / SYSU / CDD）的 `best_model_F1=*.pth`。
- 样例：`samples/` 下 4 个数据集各 8 对带真值标注的双时相影像（已按“变化比例”精选）。

## 目录结构

```
web_demo/
├── app.py            # Flask 后端：加载模型 + 推理 + JSON API
├── static/index.html # 前端（单文件，浏览器直接打开）
├── weights/          # 4 个 best_model checkpoint（15MB × 4）
├── samples/          # 32 对样例（A/B/label）
├── requirements.txt
└── README.md
```

## 快速开始

### 方式一：本地运行（推荐，录屏最稳）

> ⚠ **重要**：必须用 **`C:\Python314\python.exe`**（已装好 torch CPU + 全部依赖）。
> 不要用 Anaconda `(base)` 里的 `python`——它的 torch 是坏的（c10.dll 加载失败）。

**最简单**：直接双击 `web_demo\run_demo.bat`，6 秒后自动打开浏览器。

**或手动**，在项目根目录 `F:\Code_Repositories_2\CursorCode\AIC` 打开终端：

```bash
# 首次安装依赖（一次性）
C:\Python314\python.exe -m pip install -r web_demo/requirements.txt

# 启动
C:\Python314\python.exe web_demo/app.py
```

然后浏览器打开 **http://127.0.0.1:5000** 即可。

### 方式二：在训练服务器上运行 + 本机浏览器访问

服务器已有 `aegiscd` 环境，无需下载权重。把 `web_demo/` 上传到服务器后：

```bash
# 服务器端
conda activate aegiscd
cd /home/hzeng/project/ZH/AIC
python web_demo/app.py --port 8000        # 注意改 host 绑定见下

# 本机另开一个终端，做 SSH 端口转发：
ssh -L 5000:127.0.0.1:8000 hzeng@100.81.254.36
```

然后本机浏览器打开 **http://127.0.0.1:5000**。
（若要让服务器监听 0.0.0.0 供转发，把 `app.py` 末尾 `app.run(host="127.0.0.1", ...)`
改成 `host="0.0.0.0"`。）

## 使用说明（录屏脚本）

1. 打开页面后，顶部 4 个数据集按钮选择 **模型权重**（LEVIR / WHU / SYSU / CDD）。
2. 左侧“样例库”点任意缩略图 → 自动推理，右侧显示：
   - 时相 1 / 时相 2
   - 预测二值变化图（白=变化）
   - 概率热力图（红=高置信）
3. “T2 / 预测叠加对比”滑条 → 把预测变化区域叠加到 T2 上做对比。
4. 下方指标卡：F1、IoU、Precision、Recall、OA、变化比例、推理耗时。
5. “上传图片”标签 → 选择自己的两时相图片（T1/T2）→ 运行检测（无真值时只显示变化图与变化比例）。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api/meta` | 模型信息 + 4 数据集参考指标 |
| GET  | `/api/samples` | 样例列表（含缩略图 URL 与变化比例） |
| POST | `/api/predict` | `{dataset, name}` → 预测图 + 指标 |
| POST | `/api/upload_predict` | multipart `t1`,`t2`,`dataset` → 预测图 + 指标 |

## 备注

- 权重 checkpoint 保存时混入了 `thop.profile` 的 `*.total_ops/total_params` 缓冲，
  `app.py` 加载前会自动剥离再 `strict=True` 加载。
- 推理走与训练/测试**完全一致**的预处理（Scale → Normalize → ToTensor `fixed`）。
