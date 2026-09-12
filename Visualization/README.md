# Visualization — AEGIS-CD 可视化结果

本目录存放 AEGIS-CD 的训练曲线、混淆矩阵、结果对比图、变化检测结果图（change map）
与**定性对比图**（AEGIS-CD vs 14 个对比方法）。全部可复现。

## 目录结构

```
Visualization/
├── generate_visualizations.py   # 从训练日志生成 loss/混淆矩阵/对比图
├── generate_change_maps.py      # 本地推理生成 change map 并拼图
├── generate_aegis_changemaps.py # 服务器生成 AEGIS-CD 测试集 changemap
├── watch_baseline.py            # 轮询服务器，baseline 全部完成后自动提示
├── logs/                        # 原始训练日志
│   ├── E4_IndDiff_SCDS/         # Run13 E4_IndDiff_SCDS（4 数据集完整）
│   └── baseline/                # baseline 重训（LEVIR 已完成，其余进行中）
├── loss_curves/                 # loss / F1 训练曲线（6 张）
├── confusion_matrices/          # 混淆矩阵（5 张）
├── comparison/                  # 结果对比图（3 张）
├── change_maps/                 # 变化检测结果拼图（4 张，本地推理）
└── qualitative/                 # 定性对比图（4 数据集，AEGIS-CD vs 对比方法）
```

## 当前状态（2026-09-12）

| 项 | 状态 | 说明 |
|----|------|------|
| loss 曲线 | ✅ | 6 张：4 数据集单独 + 2 张 2×2 汇总 |
| 混淆矩阵 | ✅ | 5 张：4 数据集单独 + 1 张 2×2 汇总 |
| 结果对比图 | ✅ | 3 张：四数据集指标 + 消融 + 效率 |
| change map | ✅ | 4 张：每数据集 3 样例 × 5 列（T1/T2/真值/预测/彩色变化图） |
| 定性对比图 | ✅ | 4 数据集，每集 15 方法（AEGIS-CD + 14 对比）|

> 所有图数据来自 `E4_IndDiff_SCDS` 的训练日志，但**图中不标注 E4 来源**（标题均为通用的
> AEGIS-CD）。若后续需换用 baseline 结果，重跑一次生成命令即可，无需改图。

## 定性对比图（qualitative/，核心成果）

从 **CCVL 项目服务器**（172.18.232.151）转移了 14 个已发表对比方法的 changemap，
在 AEGIS 服务器上生成 AEGIS-CD 自身的 changemap，复用 CCVL 的 `select_qualitative_*.py`
挑图脚本（`primary-method=AEGIS-CD`）产出。

**14 个对比方法**：RS-Mamba、CDMamba、ChangeMamba-tiny、DSIFN、SNUNet、Change3D、BIT、
ELGC-Net、BiFA、MaskCD、ChangeRD、CAIFNet、WDMF-Net、MambaFedCD。
（CCVL-tiny 因论文未登刊，不作为对比方法。）

**每个数据集产出**（服务器 `/home/hzeng/project/ZH/AIC/Visualization/qualitative/<DS>/`，
已同步到本地 `Visualization/qualitative/<DS>/`）：
- `*_qualitative_visual_grid.html` — **定性图**：每行一个候选样本，列 = T1 / T2 / GT + 15 个
  方法的误差叠加图（白=TP、红=FP、绿=FN），AEGIS-CD 蓝框标记为主方法。
- `*_qualitative_shortlist.csv` — 按推荐分数排序的候选样本（含逐样本逐方法 F1/IoU）。
- `*_per_image_metrics_*.csv` — 全测试集逐样本逐方法指标（服务器上）。
- `run_summary.json` — 运行摘要（methods=15、error_count=0）。

**服务器上的中转数据**：
- `comparison_changemaps/<METHOD>/<DS>/` — 14 个对比方法 changemap（674M）
- `aegis_cd_changemaps/<DS>/` — AEGIS-CD changemap（9792 张）

## 数据来源与口径

- **loss 曲线**：每 epoch 的 `TrLoss` + 每 10 epoch 的 `VaLoss`/`F1(val)`，直接解析
  `trainValLog.txt`（摘要行 + 表格式行）。
- **混淆矩阵**：日志只存了 OA / IoU / F1 / R / P，未存 TP/TN/FP/FN。脚本用
  `OA / Recall / Precision` + 测试集总像素数（`test_count × 256²`）**精确反推**
  （验证误差 0.0000%）。口径与 `metric_tool.cm2score` 的 `[[TN,FP],[FN,TP]]` 一致。
- **参数量 / FLOPs**：从日志 banner 的 `Params` 与 `THOP ops` 读取（3.61 M / 2.83 G）。
- **消融对比**：数据硬编码自 `AGENTS.md` 第 9 节（Run13 移除式消融，测试集 F1 %）。
- **change map**：本地加载 `web_demo/weights`（E4 权重）+ `web_demo/samples`（真实样例）做 CPU 推理，
  每数据集按变化比例取高/中/低 3 个样例，拼成 5 列（T1 / T2 / 真值 / 预测 / 彩色变化图）。
  彩色变化图：白=TP、红=FP、绿=FN、黑=TN。生成命令：`python Visualization/generate_change_maps.py`
- **定性对比 changemap**：AEGIS-CD 用 `/home/hzeng/envs/aegiscd/bin/python3.10` 在服务器 GPU 0 推理
  `generate_aegis_changemaps.py` 生成（命名 `<stem>.png`，与对比方法一致）。
  CDD 测试集 list 为纯数字，挑图前已解析为 `test_XXXXX.jpg` 以对齐 changemap 命名。
