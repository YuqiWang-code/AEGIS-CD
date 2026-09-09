# AEGIS-CD 仓库协作说明

本文件适用于仓库根目录及所有子目录。开始任务前先阅读本文件，再阅读相关代码。用户在当前任务中的明确指令优先于本文件。

本版依据公开仓库 `YuqiWang-code/AEGIS-CD` 的提交 `92878388cbd73f4a93f2dc253c0fd13a2e24cc0b` 整理，核对日期为 2026-09-09。后续提交如改变实现，以实际代码和对应实验记录为准，并同步更新本文件。

## 1. 项目定位与证据边界

AEGIS-CD 面向遥感二时相影像的二值变化检测，当前用于全球校园人工智能算法精英大赛的算法模型创新方向。研究重点是轻量化差异建模、跨尺度特征融合、可重参数化解码和尺度适配监督。工程工作应服务于算法语义、实验公平性和可复现性，避免与研究目标无关的大规模重构。

当前 `BaseNet` 默认配置对应仓库论文表中的 **Run13 E4_IndDiff_SCDS**。这表示代码与表格所标注的结构配置对应，不代表已用当前提交重新完成全部实验。

按问题类型采用下列证据：

1. 模型结构、输入输出、参数默认值：以 `models/` 的执行代码为准，特别是构造函数和 `forward()`。
2. 训练与评估协议：以当前训练/测试入口、`models/utils/losses.py`、实际运行的 `run_config.json` 和 checkpoint 元数据为准。
3. 实验性能：优先核查同一运行的日志、配置、checkpoint 与完成标记。当前公开快照没有这些原始实验文件，只有汇总表。
4. 文献与已有实验汇总：参考 `docs/整理/RS-CD【PaperList】.xlsx`，必须区分论文整理值、表内标注的实测值与当前提交复测值。
5. 代码注释、旧版 AGENTS.md、历史 Run 编号只用于理解背景。与执行代码冲突时，不据此恢复已移除的模块或推断结果。

**不要把未随仓库提交的本地文件写成已存在的事实。** 当前没有 `train_scripts/`、`analyse/`、`saved_models/`、`pre-trained_weights/`、README、依赖锁定文件、独立测试目录或根目录 LICENSE，也没有旧文档所述的 Run13 全套日志及服务器说明。

## 2. 当前文件与职责

| 路径 | 当前职责 |
| --- | --- |
| `models/model.py` | `BaseNet`、HFEA (`EncoderFusion`)、CFDM (`DiffModule`)、WFAM (`AlignedModule`)、MSA 对照解码器、共享/独立头与深监督输出 |
| `models/backbone/mobilenet_v2.py` | MobileNetV2、倒残差块、五级特征提取与本地预训练权重读取 |
| `models/eaom.py` | EAOM，对齐、差异小波分解、低频/高频增强与上下文融合 |
| `models/rep_decoder.py` | `RepDWBlock`、部署融合函数和未接入 BaseNet 配置轴的 `PlainDWBlock` |
| `models/edgegate.py` | `EdgeGate`，主头前的边界细节残差注入，注释也称 EGBR |
| `models/data/dataset.py` | 双时相影像与标签加载、列表读取、CDD 文件名解析 |
| `models/data/Transforms.py` | 训练增强、归一化、颜色通道处理、张量转换，也保留未启用的变换类 |
| `models/scripts/train.py` | 训练、验证选模、断点恢复、配置保存、训练结束后的测试 |
| `models/scripts/test.py` | checkpoint 协议恢复、独立评估、误差图输出、计时、RepDW 冒烟检查；存在第 9 节所述阻断问题 |
| `models/utils/losses.py` | BCE + Soft Dice、深监督权重及 native/legacy 标签处理的唯一实现来源 |
| `models/utils/metric_tool.py` | 累积混淆矩阵、变化类 F1/IoU/Precision/Recall、OA、Kappa |
| `models/utils/param.py` | 模型参数量和 THOP 注册算子计数、RepDW 融合前后对比 |
| `models/utils/extract_metrics.py` | 遗留日志提取脚本，默认读取 `saved_models/baseline/`，当前日志适配不足 |
| `models/utils/utils.py` | 可视化网格及历史反归一化辅助函数 |
| `docs/整理/RS-CD【PaperList】.xlsx` | 文献对比、已有 E1–E4 消融、待做的模块替换/关闭实验 |
| `.gitignore` | 忽略 `docs/experiment_metrics.xlsx` 和 `docs/整理/*`，但显式保留上述 PaperList 工作簿 |

## 3. 默认模型及张量约定

### 3.1 有效配置轴

| 参数 | 当前合法值 | 默认值 |
| --- | --- | --- |
| `diff_mode` / `--diff-mode` | `eaom`, `cfdm` | `eaom` |
| `diff_sharing` / `--diff-sharing` | `independent`, `shared` | `independent` |
| `supervision_mode` / `--supervision-mode` | `native`, `legacy` | `native` |
| `decoder_mode` / `--decoder-mode` | `rep_dw`, `msa` | `rep_dw` |
| `head_mode` / `--head-mode` | `independent`, `shared` | `independent` |
| `boundary_mode` / `--boundary-mode` | `edgegate`, `off` | `edgegate` |

旧 AGENTS.md 中的 SDTR、APID、LFDS、TCT、RepHFEA、BDSR、BAIC 及其配置轴均不在当前实现中。当前不支持 `return_aux`，也没有 `strip_training_only_modules()`。训练脚本顶部的 `--use-eaom` 示例已过时，应使用 `--diff-mode eaom`。

### 3.2 主干数据流

输入为两个 `[B,3,H,W]` 的浮点 RGB 张量。数据读取阶段将两时相拼为六通道，训练与测试入口再拆分为前后两幅影像。

1. **共享编码器**：同一个 MobileNetV2 先后处理两时相，返回 `features` 索引 `[1,3,6,13,17]` 的五级特征。通道数为 `[16,24,32,96,320]`，步长为 `[2,4,8,16,32]`。
2. **共享 HFEA**：同一个 `EncoderFusion` 分别处理两个特征金字塔。输出四级特征，通道均为 64，步长为 `[4,8,16,32]`。
3. **尺度独立 EAOM**：`diff1` 至 `diff4` 分别处理对应尺度的二时相特征。四个模块独立构造，参数不共享，也没有像分类头那样进行相同初值的深拷贝。
4. **自顶向下解码**：`z4=RepDW4(d4)`，依次计算 `zi=RepDWi(di+Up2(z{i+1}))`，得到四级 64 通道特征。这里是逐元素相加，不是拼接。
5. **边界细化**：EdgeGate 只作用于最高分辨率的 `z1`，不直接作用于另外三个辅助头特征。
6. **独立预测头**：四个 `Conv1×1(64,1)` 分别输出 logits。辅助头由主头深拷贝初始化，初值相同但参数存储独立。
7. **概率输出**：主头 logits 先双线性上采样 4 倍，再取 sigmoid。native 模式的三个辅助头在各自原生尺度取 sigmoid。

默认 `256×256` 输入的尺寸如下，表中省略 batch 维：

| 位置 | 尺寸 |
| --- | --- |
| 五级编码器输出 | `16×128×128`, `24×64×64`, `32×32×32`, `96×16×16`, `320×8×8` |
| 每个时相的 HFEA 输出 | `64×64×64`, `64×32×32`, `64×16×16`, `64×8×8` |
| 四级 EAOM / 解码特征 | `64×64×64`, `64×32×32`, `64×16×16`, `64×8×8` |
| `BaseNet.forward()` 的 native 返回值 | `(p1,p2,p3,p4)`，依次为 `1×256×256`, `1×32×32`, `1×16×16`, `1×8×8` |
| legacy 返回值 | 四个输出均为 `1×256×256` |

当前融合与输出按固定倍率插值，不能宣称已支持任意尺寸。复现默认实验使用 `256×256`；扩展输入尺寸前应验证邻级尺度匹配、小波分解及输出尺寸。

### 3.3 模块内部语义

**HFEA**：前三个输出尺度分别融合相邻三级编码器特征，各支路先变换到 32 通道并对齐空间尺寸；拼接后为 96 通道。最深输出只有两支路，拼接后为 64 通道。每个聚合模块使用 `Conv3×3 + BN + ReLU + Conv3×3 + BN` 得到 64 通道，再加上中心尺度原始特征的 `Conv1×1` 投影，最后 ReLU。

**EAOM**：先由拼接特征预测二维 offset，只 warp 第二个输入。对 `abs(f1-warp(f2))` 做固定 `3×3, σ=0.8` 高斯滤波，再进行一级 Haar DWT。LL 使用平均池化与最大池化的共享 MLP 通道注意力。三个高频子带的能量为 `sqrt(HL²+LH²+HH²+1e-6)`；能量经过深度卷积、通道注意力和边缘掩码残差调制，再通过三个独立 `Conv1×1 + tanh` 生成 `2*(1+tanh(...))` 方向系数，分别缩放原始子带。IDWT 重建后，与对齐二时相的上下文做逐元素 `sigmoid(q*k)` 门控，最后拼接融合。

必须保留以下区分：

- Edge Oracle 的实际输入是**高斯滤波后的全分辨率差异**，不是高频能量。其 C 通道 sigmoid 掩码经 `AvgPool2×2` 匹配小波尺度。
- EAOM 的 `sigmoid(q*k)` 是逐元素门控，不是空间 token 两两相关的 softmax 自注意力。
- IDWT 前的高频堆叠顺序以代码 `[HL_en,LH_en,HH_en]` 为准；不要仅根据方向名称调整库张量的索引。
- 当前没有 T1 prior injection。文件开头相关描述已过时。
- 固定 DWT/IDWT 和能量运算不引入学习参数，但整个 EAOM 含多组可学习卷积，不能称为“零参数模块”。能量范数也不等于整个模型具备严格旋转不变性。

**RepDW**：训练时四支线性路径 `DW5×5+BN`、`DW3×3+BN`、`DW1×1+BN` 和 `Identity+BN` 求和，之后统一经过 GELU、`PW1×1+BN`，最后加回输入。末端 BN 的 gamma/beta 初始化为零，初始映射为恒等。`eval()` 后将卷积与 BN 折叠、核补零并求和，转换为一个带 bias 的 DW5×5 和一个带 bias 的 PW1×1，保留 GELU 与外层残差。四尺度各有一个独立 RepDW。当前转换函数只融合 RepDW，不能称为“整个模型融合成单卷积”。

**EdgeGate / EGBR**：两支 `DW3×3 + BN + ReLU` 分别接 `PW(C,1)+sigmoid` 与 `PW(C,C)`，产生边界图和细节，输出 `x + α*detail*edge`。α 是初始化为零的单个可学习标量。这是加性残差细化；当前没有单独的边界监督损失。注释中的梯度必定不小于 1 或必不产生负面交互等表述不能作为数学保证。

**CFDM / MSA 对照路径**：CFDM 先对两时相各做共享 WaveletAttention，再预测 offset 并对齐第二时相，通过差异空间注意力与上下文通道注意力融合。MSA 解码器是四尺度共享的同一个模块，使用 `sigmoid(detach(conv(x)))` 分割前景/背景注意力分支，内部注意力在每头通道维计算相关性。默认 E4 不执行这些对照路径。

## 4. 深监督与实验变量

训练和独立评估共用 `models/utils/losses.py`。模型返回 sigmoid 概率，损失是 `binary_cross_entropy` 加 Soft Dice，不能直接换成 `BCEWithLogitsLoss` 而仍保留 sigmoid。

native 模式对标签执行 `adaptive_avg_pool2d` 到每个输出尺寸，形成占比软标签。不要二值化这些辅助标签，也不要把三个辅助预测先上采样到全分辨率再称为 native SCDS。

| `--ds-profile` | `(p1,p2,p3,p4)` 损失权重 |
| --- | --- |
| `legacy`，默认 | `(1.0,0.8,0.4,0.2)` |
| `primary` | `(1.0,0.5,0.25,0.125)` |
| `main_only` | `(1.0,0.0,0.0,0.0)` |

`--ds-profile legacy` 指**权重组**，`--supervision-mode legacy` 指**输出/监督分辨率**，两者是独立设置。当前默认组合是 `native + legacy权重`。默认 Dice 为 `batch_global`；`per_image` 是额外可选变量。零权重项不构造其辅助损失图，但当前 BaseNet 仍计算并返回辅助预测。

默认训练参数：200 epochs、batch size **48**、Adam `betas=(0.9,0.99)`、学习率 `5e-4`、weight decay `1e-4`、Poly 指数 0.9、验证间隔 10、seed 2333、4 workers、`color-order=fixed`。`--deterministic` 默认关闭，需要显式传入。首个 epoch 且 global iteration 小于 200 时使用代码中的 warm-up。不要把旧实验的 batch size 64 或确定性设置当作当前 CLI 默认值。

## 5. 数据与环境

从仓库根目录执行脚本。模型构造会读取相对路径 `pre-trained_weights/mobilenet_v2-b0353104.pth`；当前没有“不加载预训练权重”的 CLI 参数。该文件需要另行准备，缺失时训练、完整评估及复杂度统计都会在模型构造处失败。

由 imports 可识别的依赖包括 Python、PyTorch、NumPy、OpenCV、Pillow、einops、pytorch_wavelets、THOP、openpyxl，`models/utils/utils.py` 还使用 torchvision。仓库未锁定版本，不要虚构已经验证的 Python/CUDA/依赖版本组合。复现实验应保存实际版本和硬件信息。

数据根路径由 `--data-root` 指定，其下是 `--dataset` 对应目录。训练/测试脚本中的绝对路径只是原环境默认值，不代表其他机器上存在。

| 数据目录内路径 | 含义 |
| --- | --- |
| `A/<filename>` | 前时相影像 |
| `B/<filename>` | 后时相影像 |
| `label/<filename>` | 单通道变化标签 |
| `list/train.txt` | 训练列表 |
| `list/val.txt` | 验证列表 |
| `list/test.txt` | 测试列表 |

列表每行一个名称。加载器也支持 `<dataset_root>/<split>/list/<split>.txt` 的回退结构。CDD 裸编号会在 A 目录中尝试解析为 `{split}_{name}.ext` 或 `{name}.ext`；解析出的同名文件必须在 B 和 label 中存在。

默认训练增强依次为 Scale、RandomCropResize、RandomFlip、RandomExchange、Normalize、ToTensor。`AmpMix`、`GaussianNoise`、`Resize` 虽有类定义，但没有接入当前训练流水线。不得将它们写成默认方法贡献。

OpenCV 首先读入 BGR。Normalize 使用与 BGR 顺序对应的均值/标准差，fixed ToTensor 再分别反转每个时相的三个通道为 RGB，保留时相顺序。legacy 会反转整个六通道数组，导致时相交换。颜色协议必须与 checkpoint 一致。

## 6. 训练、恢复与评估纪律

- 验证使用 `val.txt`，CLI 仅允许 `--val-split val`。第 1 个 epoch 和之后每个验证间隔执行验证，按验证集 F1 的严格提升保存最佳 checkpoint。
- 训练结束后加载最佳验证 checkpoint，对 `test.txt` 做一次正式评估。禁止把测试集作为早停、超参搜索或消融分支继续与否的依据。
- 最佳文件名称为 `best_model_F1=<数值>.pth`，当前保存的是包含 `state_dict`、`protocol_signature`、配置与最佳轮次的字典，不能一律按裸 state_dict 读取。
- `last_checkpoint.pth.tar` 用临时文件加 `os.replace` 每 epoch 原子更新，包含优化器、进度、最佳值、训练 DataLoader generator、Python/NumPy/Torch/CUDA RNG 状态。它与最佳验证模型用途不同。
- 完整 checkpoint 的协议签名须严格匹配。旧 checkpoint 无签名时，代码仅执行历史兼容校验；需明确记录该限制，不得称为完整协议核验。只加载模型权重时重置优化器并从 epoch 0 开始。
- 测试结束并写入日志后才生成 `.run_complete`，然后删除滚动恢复 checkpoint。确认正式完成至少应同时核查完成标记、`TEST RESULTS`、配置和模型来源，禁止伪造标记或推测缺失结果。
- 当前训练入口**不自动检测完成标记并跳过任务，也不自动选择续训文件**。再次运行会清除已有完成标记，可能覆盖配置/日志并删除同目录旧最佳文件。因此每个新实验使用独立 `--savedir`，续训显式指定 `--resume`。
- 独立评估会从带元数据的 checkpoint 恢复模型/数据协议；显式 CLI 冲突时报错，模型权重使用 `strict=True`。不要以 `strict=False` 掩盖结构不匹配。
- 混淆矩阵按整个 split 累积后计算变化类指标。日志数值为 0–1，表格百分数为其乘 100；百分点差值不得写成相对提升百分比。
- `test.py` 的计时覆盖完整 `BaseNet.forward()`，仍包含辅助预测，排除数据加载、损失、指标和保存图像。GPU 模式有预热和同步，批量耗时除以影像对数量不能等同于单样本端到端延迟。
- 参数量需说明训练图/部署图、全部/可训练参数、是否包含未参与输出的尾层与辅助头。THOP 只统计其注册算子，DWT、grid_sample 等 functional 运算可能遗漏，不能直接声称为完整硬件 FLOPs。

## 7. 可用命令

先准备依赖、数据和预训练权重，再运行以下命令。示例路径需要替换为实际路径，CUDA 设备按当前可用资源明确选择，不继承历史服务器的 GPU 分配。

```bash
# 语法检查
python -m compileall -q models

# 当前仅检查 RepDW 与可选数据路径，不能替代完整网络前向测试
python models/scripts/test.py --smoke --onGPU false

# 新的默认结构实验；显式记录协议，独立保存目录
python models/scripts/train.py \
  --data-root /path/to/CD --dataset LEVIR-CD-256 \
  --savedir ./saved_models/e4_new \
  --epochs 200 --batch-size 48 --lr 5e-4 --weight-decay 1e-4 \
  --val-interval 10 --seed 2333 --deterministic --color-order fixed \
  --diff-mode eaom --diff-sharing independent \
  --decoder-mode rep_dw --head-mode independent --boundary-mode edgegate \
  --supervision-mode native --ds-profile legacy --dice-reduction batch_global

# 续训：保持上面全部训练协议不变，额外指定对应实验文件
# --resume ./saved_models/e4_new/LEVIR-CD-256/last_checkpoint.pth.tar

# 模型复杂度；仍依赖预训练权重和 THOP
python models/utils/param.py --device cpu
```

**完整独立评估须先修复第 9 节的缺失方法调用。** 修复后可按实际最佳 checkpoint 名称运行：

```bash
python models/scripts/test.py \
  --data-root /path/to/CD \
  --checkpoint /path/to/best_model_F1=0.xxxx.pth \
  --eval-split test --experiment-name e4_eval

# 可选：在相同训练态权重加载后进行 RepDW 融合评估
# 追加 --deploy-reparam
```

CFDM/MSA 是替换式对照，不能写成简单删除后的同一网络。以完整 E4 为参考，一次只修改一项：`--diff-mode cfdm`、`--decoder-mode msa`、`--boundary-mode off`、`--head-mode shared`。共享差异和 legacy 监督对应另两个配置轴，需同时记录其余固定参数。

## 8. 仓库中的实验记录

当前唯一提交的工作簿为 `docs/整理/RS-CD【PaperList】.xlsx`，包含 `Sheet1`、`消融表(已有)`、`消融表(待做)` 三张表。以下只是已有表格记录，当前仓库缺少对应日志与 checkpoint，不能称为本次重新复现的结果。

| 实验 | 差异模块共享 | 监督尺度 | LEVIR F1 (%) | WHU F1 (%) | SYSU F1 (%) | CDD F1 (%) | Macro F1 (%) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1_Control | shared | legacy | 91.09 | 94.25 | 82.77 | 96.91 | 91.25 |
| E2_IndDiff | independent | legacy | 91.36 | 94.14 | 82.93 | 97.15 | 91.40 |
| E3_SCDS | shared | native | 91.13 | 93.80 | 83.11 | 96.73 | 91.19 |
| E4_IndDiff_SCDS | independent | native | 91.46 | 94.24 | 83.88 | 97.15 | 91.68 |

Macro 是四个数据集 F1 的算术平均。E4 相对 E1 的平均差约 0.43 个百分点。这组 2×2 实验支持分析“独立差异 × native 监督”，不能单独证明 RepDW、EdgeGate 或独立预测头的增益。

`Sheet1` 的 AEGIS-CD 行另记参数量 3.61 M、FLOPs 2.83 G；该行未完整注明计数口径，使用前需通过当前 `param.py` 及同口径测量核实。待做表仅 Full model 行有值，其余四个替换/关闭实验均为空，不能写成已完成或填入估计值。旧版 E5–E12、44/48 完成状态等未获当前提交材料支撑，不在本文件继续认定。

对比表还存在需核对的录入项，例如 `Sheet1!Q5` 的 `76,75`、`Sheet1!H14` 的 `92,06`，以及 FC-Siam-Conc 的 CDD F1/IoU 顺序疑点。不要在未回查来源时自动改值，也不要将整理表所有行都称为同环境实测。

## 9. 当前已识别的问题与最小修复方向

这些是代码静态核对结果，不等于已实施修复。处理时保持模型与实验协议不变。

| 问题 | 影响与处理方向 |
| --- | --- |
| `test.py` 无条件调用 `model.strip_training_only_modules()`，BaseNet 未定义此方法 | 加载 checkpoint 后会触发 `AttributeError`，包括 `--deploy-reparam` 路径。先核实当前已无待剥离模块，删除失效调用或实现有明确语义的兼容接口，再跑完整评估。不要借修复之名裁掉当前损失仍依赖的辅助输出。 |
| `--smoke` 只调用 RepDW 检查和目录探测 | 不实例化 BaseNet，不检查 EAOM/HFEA/EdgeGate、完整梯度、checkpoint 加载或真实评估路径。不能以其 PASS 声称完整网络已通过验证。 |
| `resolve_explicit_modes()` 与 `validate_effective_model_protocol()` 是空实现 | 不存在额外的有效拓扑检查。当前主要依赖 CLI/签名处理、构造函数合法值校验和 strict 权重加载。 |
| `extract_metrics.py` 解析的是旧 `Parameters:`、`Test OA=...` 格式 | 不适配当前 `Params (total / trainable)` 与 `TEST RESULTS` 日志，也没有完成标记过滤；参数为空且指标解析成功时还有格式化风险。修复后再用，不沿用旧文档里不存在的 analyse 脚本。 |
| EAOM、EdgeGate 和若干入口文档字符串残留旧描述 | 以执行代码更新文档，特别是 T1 prior、零参数宣称、梯度保证、`--use-eaom`、Run13/Run14 标记混用。 |
| MobileNetV2 的最终 1280 通道层仍执行，但不在返回的五级特征中 | 会影响参数/计算量口径。改变或裁剪需作为明确改动记录并验证，不能在图或效率结果中默认为已删除。 |
| Kappa 已计算但训练/测试汇总未打印 | 仅重跑现有 test.py 也不会自动得到显示的 Kappa。需要修复评估阻断，并增加明确的输出/保存逻辑。 |
| 没有依赖锁定、权重分发说明与完整运行证据 | 完整复现和发布材料需要补齐实际版本、预训练权重来源、数据划分及实验文件。 |

## 10. 修改与验证要求

开始修改前运行 `git status --short`，保留用户已有工作。优先按 `rg` 定位调用链，避免仅据类名、注释或旧文档作判断。

- 修改模型后至少验证四输出尺寸、概率范围、有限数值和相关梯度。默认尺寸以 `256×256` 为准。
- 涉及共享关系时验证对象/存储是否独立。独立分类头还应验证初始化数值一致；独立 EAOM 不要求相同初始化。
- 修改颜色处理时验证固定模式下 RGB 和前后时相顺序。
- 修改 native 监督时验证池化软标签、loss 尺寸、权重和 Dice 归约，训练与评估共用同一实现。
- 修改 RepDW 时先 `eval()` 再融合。既测试恒等初始化，也测试非零残差分支，避免用恒等初始化掩盖错误。已有代码区分 CPU float64、CPU float32 与 CUDA 的数值容差。
- checkpoint 相关修改需覆盖元数据恢复、显式冲突拒绝、strict 加载及完整续训/仅权重初始化的区别。
- 报告实际执行过的检查及环境限制；缺少依赖、权重或数据时，不把静态检查说成训练/完整推理成功。

## 11. 比赛图稿、归因与提交

图稿应突出当前默认配置，注明输入/输出尺度、共享关系、相加/拼接、训练监督与推理输出。CFDM、WFAM、MSA 等对照路径单独标注，不与默认 EAOM/RepDW 串成一条流水线。图中不加入未实现的 Transformer 主干、SAM、频域额外损失或边界标签监督。

MobileNetV2、现有基线框架、借鉴的模块和数据集应保留真实来源。公开可读不等于已经获得任意使用授权；当前仓库未附根目录许可证，参赛前根据实际来源补齐引用与授权说明，不作无依据的原创归属声明。

用户提供的 `JSAI2025.pdf` 文件正文实际为 **2026 第八届全球校园人工智能算法精英大赛·算法创新赛** 规则。按该文件第 3 页，正式作品方案、演示视频、答辩材料应遵守匿名要求，不出现学校名称/Logo 或指导教师等信息；答辩 PPT 最终以 PDF 提交。该 PDF 是会话提供的材料，当前没有提交到仓库；后续规则变更需以新的正式文件核对。

不要向 GitHub 提交密码、连接凭据、数据集、预测输出、大型 checkpoint 或无关缓存。当前 `.gitignore` 没有覆盖所有这类输出，不能依赖它自动排除。提交前查看实际暂存内容，按任务只暂存需要的文件，例如：

```bash
git status --short
git add AGENTS.md
# 若本次确实修改模型，再按范围暂存相应文件
# git add models/
git diff --cached --stat
git diff --cached
git commit -m "Update AEGIS-CD collaboration guide"
git push
```

更新 PaperList 时显式暂存其真实路径，不使用当前不存在且被忽略的 `docs/experiment_metrics.xlsx` 代替。除非用户明确要求，不进行强制推送或历史重写。
