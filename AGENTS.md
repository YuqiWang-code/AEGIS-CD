# AEGIS-CD 仓库协作说明

本文件适用于仓库根目录及所有子目录。后续开发者或智能体应先阅读本文件，再阅读任务直接相关的设计、脚本和代码。用户当次指令与本文件冲突时，以用户当次指令为准。

## 1. 项目与当前状态

AEGIS-CD 是轻量化遥感二时相变化检测项目。主模型位于 `models/`，采用共享权重的 Siamese MobileNetV2、多尺度时相交互和多尺度解码。

截至 2026-09-03：

- Run13 已完成 E1–E11 的四个数据集，共 44/48 个正式实验。
- E12_BAIC 的 LEVIR-CD、WHU-CD、SYSU-CD、CDD 均未完成；不得报告或推断 E12 结果。
- `saved_models/` 下的 `.pth` 和 `.pth.tar` 已按用户要求清除。日志、配置、完成标记和指标表是当前本地可用的实验记录。
- `docs/PROJECT_OVERVIEW.md` 主要记录截至 Run12 的历史。涉及 Run13 时，以当前代码、Run13 脚本、完成日志和 `docs/experiment_metrics.xlsx` 为准。

## 2. 事实来源与有效结果

判断项目现状时依次参考：

1. `models/` 与 `train_scripts/all/Run13/` 中的当前实现；
2. `saved_models/all/Run13/` 中已完成的日志；
3. `docs/experiment_metrics.xlsx`；
4. `docs/temporary/Run13_Design.md`；
5. `docs/PROJECT_OVERVIEW.md` 的历史背景。

Run13 实验只有同时存在 `.run_complete` 且 `trainValLog.txt` 包含 `TEST RESULTS` 才算完成。不得纳入 `_interrupted_*`、仅有训练/验证日志的目录或根据其他实验推测的数值。

`analyse/experiment_metrics.py` 已对 Run13 实施完成标记过滤。修改它时必须保留该约束，同时保持旧实验没有 `.run_complete` 时的历史兼容性。

## 3. 模型结构与 Run13 模块

核心入口是 `models/model.py` 的 `BaseNet`。当前主要配置轴包括：

- 分类头：`shared` / `independent`；
- 差异建模：`cfdm` / `eaom` / `sdtr`；
- 差异模块共享：`shared` / `independent`；
- 深监督尺寸：`legacy` / `native`；
- 编码器融合：`hfea` / `rephfea_pyr`；
- 边界模块：`off` / `edgegate` / `bdsr`；
- 一致性：`off` / `baic`；
- 幅相监督：`off` / `lf_shared`。

| 模块 | 文件 | 用途 |
| --- | --- | --- |
| SDTR | `models/temporal_relation.py` | 空间感知双向时相关系建模 |
| APID | `models/amp_phase.py` | 浅层幅相交互差异注入 |
| LFDS | `models/frequency_supervision.py` | 训练期低频差异监督 |
| TCT | `models/change_tokens.py` | 深层变化 token 交互 |
| RepHFEA | `models/rep_hfea.py` | 可重参数化编码器金字塔融合 |
| BDSR | `models/bdsr.py` | 边界引导解码细化，替代 EdgeGate |
| BAIC | `models/consistency.py` | 边界感知双向一致性约束 |
| RepDW | `models/rep_decoder.py` | 可部署融合的深度可分离解码块 |

Run13 消融链：E1 Control、E2 Independent Diff、E3 SCDS、E4 Independent Diff + SCDS、E5 SDTR、E6 APID、E7 SDTR + APID、E8 LFDS、E9 TCT、E10 RepHFEA、E11 BDSR、E12 BAIC。E12 仍待运行。

## 4. Run13 当前结果

以下是工作簿中已完成 Run13 实验的 Test F1（%），Macro 为四数据集算术平均：

| 实验 | LEVIR-CD | WHU-CD | SYSU-CD | CDD | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1_Control | 91.09 | 94.25 | 82.77 | 96.91 | 91.25 |
| E2_IndDiff | 91.36 | 94.14 | 82.93 | 97.15 | 91.40 |
| E3_SCDS | 91.13 | 93.80 | 83.11 | 96.73 | 91.19 |
| E4_IndDiff_SCDS | 91.46 | 94.24 | 83.88 | 97.15 | 91.68 |
| E5_SDTR | 91.43 | 94.46 | 79.26 | 97.10 | 90.56 |
| E6_APID | 91.66 | 94.21 | 82.18 | 97.14 | 91.30 |
| E7_SDTR_APID | 91.50 | 94.28 | 80.84 | 97.08 | 90.92 |
| E8_LFDS | 91.34 | 94.31 | 79.56 | 96.87 | 90.52 |
| E9_TCT | 91.18 | 94.17 | 80.45 | 96.87 | 90.67 |
| E10_RepHFEA | 90.89 | 94.43 | 80.89 | 96.90 | 90.78 |
| E11_BDSR | 91.01 | 94.13 | 79.76 | 96.97 | 90.47 |

可复核结论：

- Run13 当前 Macro 最佳为 E4 的 91.68。
- E4 相比 E1：LEVIR-CD +0.37、WHU-CD -0.01、SYSU-CD +1.11、CDD +0.24、Macro +0.43 个百分点。
- E6 的 LEVIR-CD 91.66 是当前整个工作簿中的该数据集最佳 Test F1，但 E6 不是 Macro 最佳。
- SDTR 及后续完整堆叠在 SYSU-CD 上明显下降；继续改动前应优先检查尺度、窗口、监督耦合和优化稳定性。
- E8–E11 没有超过 E4；E12 完成前不能声称 Run13 完整模型的最终性能。

工作簿中的历史单数据集最佳 Test F1：LEVIR-CD 91.66（Run13 E6_APID）、WHU-CD 94.85（baseline）、SYSU-CD 84.14（Module_Ablation/EAOM_MSCA_Prior）、CDD 97.35（sfif/Run2）。跨批次比较必须说明配置，不能只比较数字。

## 5. 训练与评估硬约束

Run13 共同协议定义在 `train_scripts/all/Run13/common.sh`：

- 200 epochs，batch size 64，学习率 `5e-4`，权重衰减 `1e-4`；
- Poly 学习率，验证间隔 10 epochs，种子 2333，确定性模式；
- `color-order=fixed`；
- 独立分类头、独立差异模块、RepDW 解码器；
- legacy 深监督权重 `(1.0, 0.8, 0.4, 0.2)`；
- Dice 归约 `batch_global`，尺度归一化 `plain`；
- 只用验证集选择最佳 checkpoint，测试集只在训练结束后评估一次。严禁用测试结果筛选模型、门控实验或决定是否继续训练。

实现修改必须保持：

- fixed 颜色顺序下 BAIC 方向与输入时相语义一致，不得反转；
- 独立头初始数值相同但存储独立，不能意外共享参数；
- native 深监督使用与输出尺寸对应的面积池化标签；
- LFDS 和边界辅助输出仅在需要时通过 `return_aux` 暴露；
- RepDW/RepHFEA 的训练态与部署态切换通过数值等价性 smoke test；
- 参数量与 FLOPs 口径一致；THOP 是 functional ops 估算，不得直接宣称为严格硬件 FLOPs。

训练代码每个 epoch 原子写入 `last_checkpoint.pth.tar`，保存模型、优化器、epoch、DataLoader generator 以及 Python/NumPy/PyTorch/CUDA RNG 状态，完成后清除续训 checkpoint。Run13 脚本自动续训。不要把续训 checkpoint 当作最佳模型，也不要伪造 `.run_complete`。

## 6. 服务器与数据

- 服务器项目目录：`/home/hzeng/project/ZH/AIC/`；Conda 环境：`aegiscd`。
- Run13 总脚本使用物理 GPU 1，GPU 0 留给其他用户。设置 `CUDA_VISIBLE_DEVICES=1` 后日志显示逻辑 `cuda:0` 属正常映射。
- 数据根目录和结构见 `docs/SERVER_DATASETS_ENV.md`。CDD 使用预处理后的 `train/val/test`，不能误用原始 `ChangeDetectionDataset/Real/subset`。
- `.vscode/sftp.json` 含连接配置；不得把密码或秘密复制到文档、日志、提交或回复中。
- SFTP 默认不同步 `saved_models/`。判断本地状态前，应确认服务器结果已经同步。

## 7. 修改、验证与指标更新

开始修改前先运行 `git status --short`，保留用户已有改动；再阅读相关模型、训练/测试入口、Run13 共同脚本和设计文档。改变实验语义时应同步更新代码、参数、smoke test 和文档。

最低验证要求：

- 修改过的 Python 文件通过语法检查；
- 相关测试覆盖张量形状、梯度、参数独立性和部署等价性；
- 服务器训练前通过 Run13 总脚本的 preflight；
- 只将有完成标记与最终测试段的 Run13 结果写入工作簿。

更新指标与导出快照：

```bash
python analyse/experiment_metrics.py
python analyse/export_models_and_metrics.py --name models_and_metrics_all_Run13.txt
```

输出为 `docs/experiment_metrics.xlsx` 和 `docs/temporary/models_and_metrics_all_Run13.txt`。更新工作簿后检查工作表仍为 `F1 Overview`、`Test Metrics`、`Model Info`、`All Data`，Excel 错误为 0，Run13 行数与完成数一致，且不含 `_interrupted` 或未完成 E12。

## 8. GitHub 更新

公开更新的核心文件是根目录 `AGENTS.md`、`models/` 和 `docs/experiment_metrics.xlsx`：

```bash
git add AGENTS.md
git add models/
git add docs/experiment_metrics.xlsx
git commit -m "Update AEGIS-CD"
git push
```

提交前检查暂存区，不要提交服务器凭据、大型 checkpoint、缓存文件或与任务无关的用户改动。
