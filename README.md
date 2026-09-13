# GitHub 更新方式（README.md / models / web_demo / Visualization / docs/整理/ / docs/参考文献/）

GitHub 仓库：[YuqiWang-code/AEGIS-CD](https://github.com/YuqiWang-code/AEGIS-CD)

以后更新根目录 `README.md`、本地 `models/`、`web_demo/`、`Visualization/`（论文/比赛图稿）、
`docs/整理/`（PaperList、架构图 pptx、比赛规则 pdf）或 `docs/参考文献/`（技术文档交叉引用指南）
后，在项目根目录执行：

```bash
git add README.md
git add models/
git add web_demo/
git add Visualization/
git add docs/整理/RS-CD【PaperList】.xlsx
git add docs/整理/AEGIS-CD_Diagrams.pptx
git add docs/整理/JSAI2025.pdf
git add docs/参考文献/AEGIS-CD_AIC技术文档_参考文献与交叉引用指南.md
git commit -m "Update AEGIS-CD"
git push
```

> 注：`docs/experiment_metrics.xlsx`（完整实验指标）与 `web_demo/weights/`（大文件，走
> GitHub Releases 分发），以及 `Visualization/` 下的 `__pycache__/`、`*.zip`、`logs/`
> 不上传 GitHub，仅保留本地。`web_demo/samples/`（96 张样例图）已随仓库提交。
> `docs/参考文献/` 只提交技术文档指南（.md），其下的论文 PDF 仅保留本地。

---

# AEGIS-CD 协作说明

本文件适用于仓库根目录及所有子目录。开始任务前先阅读本文件，再阅读相关代码。
用户在当前任务中的明确指令优先于本文件。

## 1. 项目定位

AEGIS-CD 是一个**轻量化遥感二时相影像变化检测**模型，用于全球校园人工智能
算法精英大赛的算法模型创新方向。研究重点是轻量化差异建模、跨尺度特征融合、
可重参数化解码和尺度适配监督。

- 参数量 **3.61 M**，THOP 注册算子 **2.83 G**
- 四个公开数据集（LEVIR / WHU / SYSU / CDD）统一 `256×256` 输入
- 单卡训练，PyTorch，CPU 亦可推理（`web_demo/`）

## 2. 环境

### 2.1 环境创建（服务器）

```bash
conda create -n aegiscd python=3.10 -y
conda activate aegiscd

# PyTorch cu128
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 其余依赖
pip install einops fvcore numpy pillow tqdm pyyaml thop scipy opencv-python matplotlib
pip install pytorch_wavelets PyWavelets
```

### 2.2 服务器信息

| 项 | 值 |
| --- | --- |
| 服务器 | `100.81.254.36`（用户名 `hzeng`；凭据见 `.vscode/sftp.json`） |
| 项目路径 | `/home/hzeng/project/ZH/AIC/` |
| 数据根 | `/home/hzeng/project/ZH/data/CD/` |
| GPU | 2 × NVIDIA GeForce RTX 5090（~32 GB/卡） |
| CUDA Driver | 13.1 |
| conda 环境 | `aegiscd` |

### 2.3 预训练权重

| 文件 | 路径 |
| --- | --- |
| MobileNetV2 ImageNet-1K | `pre-trained_weights/mobilenet_v2-b0353104.pth` |

`mobilenet_v2.py` 中的 `mobilenet_v2(pretrained=True)` 自动从 `pre-trained_weights/`
加载，构造模型时必需。

### 2.4 关键依赖

| 包 | 用途 |
| --- | --- |
| torch (cu128) | 深度学习框架 |
| einops | 张量重排 |
| thop | FLOPs 统计 |
| numpy / pillow / tqdm / scipy | 训练辅助 |
| opencv-python | 图像读取 |
| pytorch_wavelets | DWT 小波变换 |
| PyWavelets | DWT 底层依赖 |

> `pytorch_wavelets` 的 DTCWT 子包在 import 时使用 `pkg_resources`。
> `setuptools>=81` 会触发 `ModuleNotFoundError: pkg_resources`
> （DWT 本身不使用它，import 前加一条 `warnings.filterwarnings` 即可静音）。

## 3. 模型架构

### 3.1 数据流

```
T1 [B,3,256,256]  +  T2 [B,3,256,256]
        │
        ▼  共享权重（Siamese）
MobileNetV2  →  五级特征 idx=[1,3,6,13,17]，通道 16/24/32/96/320
        │  尺寸 128² / 64² / 32² / 16² / 8²
        ▼
HFEA (EncoderFusion) — 相邻三级层次融合，四级输出均为 64 通道
        │  尺寸 64² / 32² / 16² / 8²
        ▼
EAOM ×4（尺度独立）— 小波域差异编码
        │  DWConv 对齐 → |f1-warp(f2)| → 高斯抗锯齿 → Haar DWT
        │  HF 旋转不变能量 → Edge Oracle → 三方向 ratio → IDWT
        │  → 与对齐上下文做 sigmoid(q·k) 逐元素门控 → 拼接融合
        ▼
RepDW 解码器 ×4（尺度独立）— 自顶向下
        │  z4 = RepDW4(d4);  zi = RepDWi(di + Up2(zi+1))     ← 逐元素相加，非拼接
        ▼
EdgeGate（仅作用于最高分辨率的 z1）
        │  out = z1 + α · detail ⊙ edge ，α 初始化为 0
        ▼
4 × Conv1×1 → 独立预测头（decoder_out / 2 / 3 / 4）
        ▼
输出 4 个尺度的概率图
```

### 3.2 张量尺寸（`256×256` 输入，省略 batch 维）

| 位置 | 尺寸 |
| --- | --- |
| 五级编码器输出 | `16×128×128`, `24×64×64`, `32×32×32`, `96×16×16`, `320×8×8` |
| 每个时相的 HFEA 输出 | `64×64×64`, `64×32×32`, `64×16×16`, `64×8×8` |
| 四级 EAOM / 解码特征 | `64×64×64`, `64×32×32`, `64×16×16`, `64×8×8` |
| `BaseNet.forward()`，`native` | `(1×256×256, 1×32×32, 1×16×16, 1×8×8)` |
| `BaseNet.forward()`，`legacy` | 四个输出均为 `1×256×256` |

主头 logits 先双线性上采样 4 倍再取 sigmoid；`native` 模式下三个辅助头
在各自原生尺度取 sigmoid。

### 3.3 模块

实测参数量（`BaseNet()` 默认配置，合计 **3,611,629 ≈ 3.61 M**）：

| 模块 | 开关 | 作用 | 参数 |
| --- | --- | --- | --- |
| MobileNetV2 编码器 | 固定（共享权重） | 五级特征提取 | 2.2239 M |
| HFEA | 固定 | 相邻三级编码器特征层次融合 → 64 通道 | 0.6966 M |
| EAOM | `--diff-mode eaom` | 小波域差异编码（对齐 + 边缘 oracle + 跨注意力） | 0.1644 M × 4 = 0.6574 M |
| RepDW | `--decoder-mode rep_dw` | 可重参数化解码块，四尺度独立 | 0.0070 M × 4 = 0.0279 M（训练图） |
| EdgeGate | `--boundary-mode edgegate` | 主头前的加性边界细节注入 | 5,569 参数 |
| 独立预测头 | `--head-mode independent` | 四尺度各自独立的 `Conv1×1` | 65 × 4 = 260 |

RepDW 部署图（`switch_to_deploy()` 折叠后）单个约 5,824 参数，
四个合计约 0.023 M，比训练图更小。

**EAOM**（`models/eaom.py`）：先由拼接特征预测二维 offset，只 warp 第二个输入
（`f1` 保持原样）；对 `|f1 − warp(f2)|` 做固定 `3×3, σ=0.8` 高斯滤波后做一级 Haar DWT。
LL 使用平均/最大双池化共享 MLP 通道注意力；三个高频子带能量为
`sqrt(HL² + LH² + HH² + 1e-6)`，经深度卷积、通道注意力与边缘掩码残差调制后，
由三个独立 `Conv1×1 + tanh` 生成 `2·(1 + tanh(·))` ∈ [0,4] 的方向系数分别缩放原始子带。
IDWT 重建后与对齐二时相上下文做 `sigmoid(q·k)` **逐元素门控**（非 softmax 自注意力），
最后拼接融合。

> 需要区分的两点：Edge Oracle 的实际输入是**高斯滤波后的全分辨率差异**，不是高频能量；
> IDWT 前的高频堆叠顺序以代码 `[HL_en, LH_en, HH_en]` 为准。
> 固定的 DWT/IDWT 与能量范数不引入学习参数，但整个 EAOM 含多组可学习卷积，
> **不是零参数模块**，能量范数也不等于模型具备严格旋转不变性。

**RepDW**（`models/rep_decoder.py`）：训练时四支线性路径
`DW5×5+BN`、`DW3×3+BN`、`DW1×1+BN`、`Identity+BN` 求和，统一经 GELU、`PW1×1+BN`，
最后加回输入；末端 BN 的 gamma/beta 初始化为零，初始映射为恒等。
`eval()` 后调用 `switch_to_deploy()` 将卷积与 BN 折叠、核补零求和，
转换为单个带 bias 的 `DW5×5` + 带 bias 的 `PW1×1`（保留 GELU 与外层残差）。
四尺度各有一个独立实例；当前转换函数**只融合 RepDW**。

**EdgeGate / EGBR**（`models/edgegate.py`）：两支 `DW3×3+BN+ReLU` 分别接
`PW(C→1)+sigmoid` 与 `PW(C→C)`，输出 `x + α·detail⊙edge`，α 是初始化为零的单个标量。
这是加性残差细化；**当前没有单独的边界监督损失**。

## 4. 配置轴

| 参数 | 合法值 | 默认 |
| --- | --- | --- |
| `--diff-mode` | `eaom`, `none` | `eaom` |
| `--diff-sharing` | `shared`, `independent` | `independent` |
| `--supervision-mode` | `native`, `legacy` | `native` |
| `--decoder-mode` | `msa`, `rep_dw` | `rep_dw` |
| `--head-mode` | `shared`, `independent` | `independent` |
| `--boundary-mode` | `off`, `edgegate` | `edgegate` |
| `--ds-profile` | `legacy`, `primary`, `main_only` | `legacy` |
| `--dice-reduction` | `batch_global`, `per_image` | `batch_global` |
| `--color-order` | `legacy`, `fixed` | `fixed` |
| `--deterministic` | flag | 关闭 |

`--diff-mode none` 即纯 `|f1 − f2|` 差异，没有独立的 CFDM 对照路径。
`--ds-profile` 指**损失权重组**，`--supervision-mode` 指**输出/监督分辨率**，两者独立。

## 5. 数据

### 5.1 目录结构

结构：`<data_root>/<DATASET>/{A,B,label,list}/`，`list/{train,val,test}.txt` 每行一个文件名。

```
/home/hzeng/project/ZH/data/CD/<DATASET_NAME>/
├── A/              # 时相 1（pre-change）影像
├── B/              # 时相 2（post-change）影像
├── label/          # 二值变化标签（0 = 未变化，255 = 变化）
└── list/           # train.txt / val.txt / test.txt
```

`dataset.py` 按 `<root>/A|B|label/<name>` 读取，支持列表回退结构
`<dataset_root>/<split>/list/<split>.txt`。

### 5.2 各数据集详情

| 数据集 | 样本数 | 格式 | 命名 / list | 特殊处理 |
| --- | --- | --- | --- | --- |
| LEVIR-CD-256 | 10,192 | PNG | `train_100_10.png` / list 含 `.png` | 无 |
| LEVIR-CD+256 | 15,760 | PNG | `train_100_00.png` / list 含 `.png` | 无（当前实验未用） |
| SYSU-CD-256 | 20,000 | PNG | `00000.png`…`19999.png`；0–11999 train / 12000–15999 val / 16000–19999 test | 无 |
| WHU-CD-256 | 7,434 | PNG | `whucd_00001.png`…；list 目录共 26 个文件，**仅用 train/val/test** | 忽略半监督划分文件 |
| CDD-CD-256 | 15,998 | **JPG** | list 为**纯数字**（无扩展名、无 split 前缀） | 自动解析为 `{split}_{name}.jpg` |

CDD 的 `_resolve_filename()` 按顺序自动探测：① 原样（含扩展名则信任）→
② `{split}_{name}.{ext}` → ③ `{name}.{ext}`，对调用方透明。

### 5.3 场景特点与建模难点

四数据集共同特点：双时相配准良好但存在辐射差异（光照/色调/阴影/季节植被 → 伪变化）；
变化类通常少于未变化类（易偏向背景）；目标尺度跨度大；`256×256` 裁剪带来边界效应；
建筑重要但不是唯一变化类型（过强建筑先验会损害 SYSU/CDD）。

| 数据集 | 核心场景 | 典型分辨率 | 主要难点 |
| --- | --- | --- | --- |
| LEVIR-CD-256 | 得州建筑增长/拆除 | ~0.5 m | 正类稀少、小建筑召回、季节/光照伪变化 |
| SYSU-CD-256 | 香港及周边通用城市/郊区变化 | ~0.5 m | 类型最杂、标签较粗、植被/水体/光照干扰 |
| WHU-CD-256 | 基督城震后建筑重建 | ~0.2–0.3 m | 建筑密集且相似、正类稀少、单一大图来源 |
| CDD-CD-256 | 多源季节与灾害场景 | ~3 cm–1 m | 强伪变化、跨尺度、多场景 |

- **LEVIR**：建筑增长与拆除为主，独立变化实例多、尺度差异大；正类约占 4.65%。
  适合检验多尺度建筑表征与时相不变特征，但不能据此证明对通用变化有效。
- **SYSU**（重点难点）：变化类型最杂（城市扩张、建筑、道路、植被、填海、海域、
  场地整理），标注粗糙，植被/水体/阴影/光照产生强外观差异但不一定是真实语义变化。
  要求**通用变化表征**而非建筑专用边缘先验；分析时同时看 P/R/IoU 与分场景错误。
- **WHU**：变化高度集中于建筑新增/拆除/重建，建筑密集且外观相似；来自单一大图，
  空间邻近 patch 相关性强，划分和随机种子可能影响结果解释。
- **CDD**：多源、季节 + 灾害前后，分辨率跨度最大，伪变化干扰强；较高总体 F1
  不代表所有场景都解决，应防止多数易样本掩盖特定场景失败。

> 像素比例等统计值为引用口径，用于论文定量结论前应通过本地标签重新审计。

### 5.4 颜色协议

OpenCV 读入 BGR → 两时相拼为六通道 → `Normalize` 用与 BGR 对应的均值/标准差 →
`ToTensor(color_order='fixed')` 分别反转每个时相的三个通道为 RGB，保留时相顺序。
`legacy` 会反转整个六通道数组，导致**时相交换**，仅为历史兼容保留。
颜色协议必须与 checkpoint 一致。

## 6. 训练与评估协议

- 验证使用 `val.txt`；CLI 只允许 `--val-split val`。第 1 个 epoch 与之后每
  `--val-interval` 个 epoch 执行验证，按验证集 F1 的严格提升保存最佳 checkpoint。
- 训练结束后加载最佳验证 checkpoint，对 `test.txt` 做**一次**正式评估。
  禁止把测试集作为早停、超参搜索或消融分支继续与否的依据。
- 损失为 `binary_cross_entropy + Soft Dice`，模型已返回 sigmoid 概率，
  **不可**换成 `BCEWithLogitsLoss`。
- `native` 模式对标签执行 `adaptive_avg_pool2d` 到每个输出尺寸，形成占比软标签；
  不要二值化辅助标签，也不要把辅助预测先上采样到全分辨率再称为 native。
- 混淆矩阵按整个 split 累积后计算变化类指标；日志数值为 0–1，表格百分数为其 ×100。

默认训练参数（`models/scripts/train.py`）：

| 项 | 默认 | 说明 |
| --- | --- | --- |
| `--epochs` | 200 | |
| `--batch-size` | **48** | `train_scripts/baseline/` 使用 **64** |
| `--lr` / `--lr-mode` | `5e-4` / `poly` | Adam，`betas=(0.9,0.99)` |
| `--weight-decay` | `1e-4` | |
| `--val-interval` | 10 | |
| `--seed` | 2333 | |
| `--num-workers` | 4 | |

### 6.1 训练日志

每次训练启动时 `train.py` 打印完整配置头（Dataset / Params / FLOPs / Modules / GPU /
Optimizer / LR / Batch / Epochs / Val interval / Loss / Seed / Save dir …）并写入
`trainValLog.txt` 顶部。每 epoch 输出：

```
[14:35:22] Epoch 11/200 | TrLoss=0.1523 | VaLoss=0.1431 | F1(train)=0.8123 | F1(val)=0.8045 | best=0.8045 | elapsed=5m ETA=95m ★ BEST
```

`trainValLog.txt` 末尾为累计指标表（`Epoch / TrLoss / VaLoss / OA / IoU / F1 / R / P / BestF1`），
最后一行为 `TEST RESULTS | OA=... IoU=... F1=... R=... P=...`（加载最佳权重后对测试集评估）。

| 注意事项 | 说明 |
| --- | --- |
| FLOPs 统计 | 用 `thop.profile`（非 fvcore）。`fvcore` 不支持 `pytorch_wavelets` 的 DWT 算子，结果不准。 |
| GPU 显存 | RTX 5090 32 GB，batch 64 约占用 ~20 GB。 |
| PYTHONPATH | 无需设置，`train.py` 顶部自动把项目根目录加入 `sys.path`。 |

## 7. 目录结构

```
models/
├── model.py                BaseNet、HFEA、DecoderFusion、MSA_Module
├── eaom.py                 EAOM 小波域差异编码
├── rep_decoder.py          RepDWBlock（PlainDWBlock 为单支对照，未接入配置轴）
├── edgegate.py             EdgeGate 边界细化
├── backbone/mobilenet_v2.py
├── data/{dataset.py,Transforms.py}
├── scripts/{train.py,test.py}
└── utils/{losses.py,metric_tool.py,param.py,utils.py,extract_metrics.py}

train_scripts/
├── baseline/               四个数据集的 AEGIS-CD 主实验（common.sh 共享协议）
├── Ablation/First/         核心模块移除式消融（A/B/C/D × LEVIR/SYSU）
└── all/Run13/              历史实验脚本

saved_models/
├── baseline/<DATASET>/     主实验结果（trainValLog.txt / run_config.json / best_model_F1=*.pth）
├── Ablation/First/<配置>/<DATASET>/
└── all/Run13/              历史实验记录

web_demo/                   浏览器演示系统（Flask + 单页前端）
Visualization/              论文/比赛图稿 + 生成脚本
analyse/                    日志 → Excel 汇总脚本
docs/整理/                  PaperList 对比表、架构图 pptx
```

`models/utils/extract_metrics.py` 是遗留脚本，解析的是旧日志格式
（`Parameters:` / `Test OA=...`），**不适配当前 `TEST RESULTS` 格式**，使用前需先适配。

## 8. 常用命令

从仓库根目录执行。

```bash
# 语法检查
python -m compileall -q models

# 架构预检（仅覆盖 RepDW 等价性与可选数据路径探测，不等于完整前向验证）
python models/scripts/test.py --smoke --onGPU false

# 主实验：四个数据集，串行在物理 GPU 1
screen -dmS baseline bash train_scripts/baseline/run_all_gpu1.sh
tail -f saved_models/baseline/*/trainValLog.txt

# 单个数据集
bash train_scripts/baseline/train_LEVIR_CD_256.sh

# 单次训练（等价于 baseline 协议）
python models/scripts/train.py --dataset LEVIR-CD-256 \
  --data-root /home/hzeng/project/ZH/data/CD \
  --savedir ./saved_models/baseline --epochs 200 --batch-size 64 \
  --lr 5e-4 --weight-decay 1e-4 --lr-mode poly \
  --val-interval 10 --val-split val --num-workers 4 \
  --seed 2333 --deterministic --color-order fixed \
  --diff-mode eaom --diff-sharing independent --supervision-mode native \
  --decoder-mode rep_dw --head-mode independent --boundary-mode edgegate \
  --ds-profile legacy --dice-reduction batch_global

# 复杂度和参数量
python models/utils/param.py --device cpu

# 演示系统（本地 CPU 推理；必须用 C:\Python314\python.exe，见 web_demo/README.md）
python web_demo/app.py                      # http://127.0.0.1:5000
```

启动脚本是**崩溃安全**的：已存在 `.run_complete` 会跳过，存在
`last_checkpoint.pth.tar` 会自动 `--resume` 续训。

> 每个新实验使用独立 `--savedir`。再次运行会清除已有完成标记，
> 可能覆盖配置/日志并删除同目录旧的最佳权重。

## 9. 结果

主实验结果（测试集，按验证集 F1 选模，无测试集泄漏）：

| 数据集 | F1 (%) | IoU (%) | Precision (%) | Recall (%) | OA (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| LEVIR-CD-256 | 91.46 | 84.26 | 91.90 | 91.03 | 99.13 |
| WHU-CD-256 | 94.24 | 89.11 | 95.93 | 92.61 | 99.55 |
| SYSU-CD-256 | 83.88 | 72.23 | 86.99 | 80.98 | 92.66 |
| CDD-CD-256 | 97.15 | 94.45 | 97.76 | 96.54 | 99.27 |

核心模块移除式消融（LEVIR-CD + SYSU-CD，测试集 F1 %）：

| 消融设置 | SYSU F1 | SYSU Δ |
| --- | ---: | ---: |
| Full model | 83.88 | — |
| w/o EAOM（纯 \|diff\|） | 80.85 | −3.03 |
| w/o RepDW（→ MSA 解码器） | 82.44 | −1.44 |
| w/o EdgeGate | 82.93 | −0.95 |
| w/o Independent Head | 83.16 | −0.72 |

四个模块在 **SYSU（难点数据集）** 上均有明显正贡献

## 10. 提交与安全

- **不要**向 GitHub 提交密码、连接凭据、数据集、预测输出、大型 checkpoint
  或无关缓存。`.gitignore` 已忽略 `saved_models/`、`web_demo/weights/`、
  `docs/experiment_metrics.xlsx`；但不要依赖它自动排除所有输出。
- 提交前查看实际暂存内容，按任务只暂存需要的文件：

```bash
git status --short
git add README.md
# git add models/          # 若本次确实修改模型
git diff --cached --stat
git diff --cached
git commit -m "Update AEGIS-CD"
git push
```

- 除非用户明确要求，不进行强制推送或历史重写。
- 比赛材料遵守匿名要求：不出现学校名称/Logo 或指导教师等信息。
- MobileNetV2、借鉴的模块与数据集应保留真实来源；公开可读不等于已获授权，
  不作无依据的原创归属声明。
