# AEGIS-CD 技术文档参考文献与交叉引用指南

> 更新日期：2026-09-13  
> 使用场景：AIC 算法模型创新赛技术文档（Word）  
> 对齐对象：AEGIS-CD 最终模型技术路线  
> 当前模型主线：`T1/T2 → Siamese MobileNetV2 → HFEA → 4×EAOM → 4×RepDW Decoder → EdgeGate → 4 个独立预测头 → Native-scale Deep Supervision`
>
> 本文档的第一部分**严格按当前提供的 17 条参考文献原文与顺序整理**，不在此处擅自改写引用格式；后续部分用于说明每篇文献在技术文档中的推荐引用位置、主要贡献以及与 AEGIS-CD 的对应关系。

---

# 1. 参考文献（GB/T 7714，按当前整理版本）


[1] Daudt R C, Le Saux B, Boulch A. Fully convolutional siamese networks for change detection[C]//2018 25th IEEE international conference on image processing (ICIP). IEEE, 2018: 4063-4067.

[2] Lin T Y, Dollar P, Girshick R, et al. Proceedings of the IEEE conference on computer vision and pattern recognition[J]. Honolulu, HI,(2117–2125), 2017.

[3] Yang H, Chen Y, Wu W, et al. A lightweight siamese neural network for building change detection using remote sensing images[J]. Remote Sensing, 2023, 15(4): 928.

[4] Liu Y, Li S, Bruzzone L. A lightweight wavelet-aligned difference and mask-guided fusion network for change detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2026, 64: 1-16.

[5] Xie Z, Miao S, Jiang Y, et al. FSG-Net: Frequency-spatial synergistic gated network for high-resolution remote sensing change detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2026.

[6] Zhang W, Guo W, Li Y, et al. Change detection meets frequency learning: A coarse-to-fine dual-domain detection network[J]. IEEE Transactions on Geoscience and Remote Sensing, 2025, 63: 1-13.

[7] Wang P, Liu Y, Ma Q, et al. EGAFNet: Edge-Guided Adaptive Fusion Network with Spatial-Frequency Interaction for Remote Sensing Change Detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2026.

[8] Mallat S G. A theory for multiresolution signal decomposition: the wavelet representation[J]. IEEE transactions on pattern analysis and machine intelligence, 1989, 11(7): 674-693.

[9] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks[J]. Advances in neural information processing systems, 2015, 28.

[10] Ding X, Zhang X, Ma N, et al. Repvgg: Making vgg-style convnets great again[C]//2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2021: 13728-13737.

[11] Shi Q, Liu M, Li S, et al. A deeply supervised attention metric-based network and an open aerial image dataset for remote sensing change detection[J]. IEEE transactions on geoscience and remote sensing, 2021, 60: 1-16.

[12] Chen H, Shi Z. A spatial-temporal attention-based method and a new dataset for remote sensing image change detection[J]. Remote sensing, 2020, 12(10): 1662.

[13] Ji S, Wei S, Lu M. Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set[J]. IEEE Transactions on geoscience and remote sensing, 2018, 57(1): 574-586.

[14] Lebedev M A, Vizilter Y V, Vygolov O V, et al. Change detection in remote sensing images using conditional adversarial networks[J]. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2018, 42: 565-571.

[15] Zhai Y, Pan J, Zhang H, et al. Efficient adjacent feature harmonizer network with UAV-CD+ dataset for remote sensing change detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2024, 62: 1-18.

[16] Zhou W, Zhu Y, Lei J, et al. LSNet: Lightweight spatial boosting network for detecting salient objects in RGB-thermal images[J]. IEEE Transactions on Image Processing, 2023, 32: 1329-1340.

[17] Chen H, Pu F, Yang R, et al. RDP-Net: Region detail preserving network for change detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2022, 60: 1-10.


> **提交前核验提醒：** 第 [2] 条按当前提供文本原样保留，但其作者、年份与页码 `2117–2125` 对应的论文实际为 Lin 等人的 **Feature Pyramid Networks for Object Detection (FPN)**，发表于 CVPR 2017。当前条目把会议论文集名称写成了论文题名，并标成了 `[J]`，建议最终 Word 定稿前重新导出/核验 GB/T 7714 格式。
>
> **另一个重要提醒：** AEGIS-CD 最终编码骨干使用 MobileNetV2，但当前这 17 条中没有 MobileNetV2 原始论文。如果技术文档正文明确写到“采用 MobileNetV2 作为 backbone/encoder”，建议最终参考文献中仍补入 Sandler 等人的 MobileNetV2 原论文，以避免核心骨干缺少出处。

---

# 2. AEGIS-CD 模块与参考文献快速对应表

| AEGIS-CD 技术点 | 推荐引用 | 用途 |
|---|---|---|
| Siamese 双时相共享编码 | [1] [3] | 说明变化检测中的 Siamese 范式及轻量化发展 |
| 多层级/多尺度特征融合 | [2] [3] [4] [15] | 支撑 HFEA 的层级特征交互设计动机 |
| HFEA 相邻层级增强聚合 | [4] [15]，辅以 [2] [3] | 说明相邻层级、多尺度和轻量融合思路 |
| EAOM 频域/小波变化建模 | [4] [5] [6] [8] | 说明频率线索、小波分解和空间—频率协同 |
| EAOM 时相特征对齐 | [4] [9] | 支撑可微 warp / 特征域对齐思想 |
| EAOM 高频方向/边缘信息 | [5] [7] [8] | 支撑高频细节、边缘与变化轮廓建模 |
| EdgeGate / EGBR 边界细化 | [7] [17] | 说明边缘引导与区域细节保持的重要性 |
| RepDW 结构重参数化 | [10] | 训练多分支、部署单分支等价融合 |
| Native-scale Deep Supervision | [11] | 支撑多尺度/深监督训练思路 |
| 轻量部署与效率设计 | [3] [4] [15] [17]，可选 [16] | 说明参数量、FLOPs、部署效率研究背景 |
| LEVIR-CD | [12] | 数据集原始出处 |
| WHU-CD | [13] | 数据集来源 |
| SYSU-CD | [11] | 数据集原始出处 |
| CDD-CD | [14] | 数据集原始出处 |

---

# 3. 按技术文档章节推荐引用

## 3.1 “项目背景 / 国内外研究现状”

优先引用：

- **Siamese 变化检测基础：** [1]
- **轻量变化检测：** [3] [4] [15] [17]
- **频域/小波变化检测：** [4] [5] [6] [7]
- **多尺度特征融合：** [2] [15]
- **边界与细节保持：** [7] [17]

这一部分主要回答“已有方法做到了什么、还存在什么问题”，不要在这里详细解释 AEGIS-CD 内部算子。

## 3.2 “总体网络结构”

推荐引用：

- Siamese 双时相框架：[1] [3]
- 多尺度层次特征：[2] [15]
- 轻量化路线：[3] [4] [15] [17]

建议在介绍完相关工作后明确：AEGIS-CD 并非对上述单一模型的复现，而是围绕轻量变化检测目标重新设计 HFEA、EAOM、RepDW 和 EdgeGate 等模块。

## 3.3 “HFEA：层级特征增强聚合”

优先级：

1. [15] EAFH-Net
2. [4] WDMF-Net
3. [2] FPN
4. [3] LightCDNet

HFEA 的论述重点应是：不同 backbone 层级分别具有空间细节与高层语义，直接只取单尺度会造成信息利用不足，因此通过投影、尺度调整与相邻层级融合，构建更适合后续变化建模的四尺度特征。

## 3.4 “EAOM：对齐—抗混叠—小波—边缘调制—门控融合”

建议分机制引用：

- **时相对齐：** [4] [9]
- **小波理论：** [8]
- **小波/频率变化检测：** [4] [5] [6]
- **空间—频率协同：** [5] [6] [7]
- **高频/边缘细节：** [5] [7] [8]

这里最重要的是避免写成“直接采用某论文模块”。推荐表述为：已有研究证明特征对齐、频域差异与边缘信息对遥感变化检测有效，AEGIS-CD 在此基础上针对轻量部署重新组合并设计 EAOM。

## 3.5 “RepDW 解码器与结构重参数化”

核心引用仅需：

- [10] RepVGG

说明训练阶段保留多个线性分支以增强优化和表示能力，部署阶段将卷积与 BN 等价折叠为单一深度卷积，从而不把训练时的多分支开销带入推理阶段。

## 3.6 “EdgeGate / EGBR 边界细化”

推荐：

- [7] EGAFNet
- [17] RDP-Net

用于说明二值变化检测不仅需要区域级正确分类，还要避免建筑边缘、小变化目标和细窄结构在连续下采样/解码中被过度平滑。

## 3.7 “深监督与损失函数”

推荐：

- [11] DSAMNet

可用于支撑多尺度预测分支参与训练的必要性。需要注意：AEGIS-CD 使用的是自身设计的 native-scale supervision，辅助标签按输出原生分辨率构造，不应描述成直接复制 DSAMNet 的监督策略。

## 3.8 “实验数据集”

必须按数据集分别引用：

- LEVIR-CD → [12]
- WHU-CD → [13]
- SYSU-CD → [11]
- CDD-CD → [14]

数据集原始论文最好在第一次介绍该数据集名称时就引用，而不是只统一放在实验章节最后。

---

# 4. 逐篇文献用途与贡献索引


## [1] FC-Siam：Siamese 变化检测基础

**当前参考文献：**  
Daudt R C, Le Saux B, Boulch A. Fully convolutional siamese networks for change detection[C]//2018 25th IEEE international conference on image processing (ICIP). IEEE, 2018: 4063-4067.

**推荐放在技术文档：** 研究背景/相关工作；总体网络结构（Siamese 双时相编码器）

**这篇文献主要做了什么：**  
提出全卷积 Siamese 变化检测框架，用共享或成对特征提取的方式处理双时相图像，是深度学习遥感变化检测中经典的 Siamese 范式之一。

**对 AEGIS-CD 能提供什么支撑：**  
AEGIS-CD 同样以 T1/T2 双时相影像为输入并采用共享权重编码器。该文献最适合用于说明为何选择 Siamese 主干，而不是说明 EAOM、HFEA 等具体创新模块。

**正文可参考的引用方式：**  
> “早期研究已证明，全卷积 Siamese 架构能够有效建模双时相影像之间的变化信息[1]。在此基础上，本文进一步面向轻量化遥感变化检测构建共享权重双分支编码框架。”

**引用优先级：** ★★★★★ 必引

---

## [2] FPN：多尺度金字塔与层级融合基础

**当前参考文献：**  
Lin T Y, Dollar P, Girshick R, et al. Proceedings of the IEEE conference on computer vision and pattern recognition[J]. Honolulu, HI,(2117–2125), 2017.

**推荐放在技术文档：** 相关工作中的多尺度特征融合；HFEA 设计动机

**这篇文献主要做了什么：**  
该条目实际对应 Lin 等人的 Feature Pyramid Networks（FPN）。FPN利用卷积网络天然的层级结构，通过自顶向下路径与横向连接构建多尺度语义特征金字塔。

**对 AEGIS-CD 能提供什么支撑：**  
AEGIS-CD 的 HFEA 不是 FPN 的直接复现，但同样利用不同层级特征并通过尺度调整实现跨层融合，因此可作为“层级多尺度特征融合”的经典基础工作。

**正文可参考的引用方式：**  
> “为充分利用不同编码层级的空间细节与高层语义信息，多尺度特征金字塔与跨层融合已被广泛采用[2]。本文据此进一步设计适配轻量变化检测的层级特征增强聚合模块。”

**引用优先级：** ★★★★☆ 推荐；提交前核验条目

---

## [3] LightCDNet：轻量 Siamese 变化检测

**当前参考文献：**  
Yang H, Chen Y, Wu W, et al. A lightweight siamese neural network for building change detection using remote sensing images[J]. Remote Sensing, 2023, 15(4): 928.

**推荐放在技术文档：** 研究现状；轻量化设计；Backbone/Siamese；HFEA 相关工作

**这篇文献主要做了什么：**  
面向建筑变化检测提出轻量 Siamese 网络，强调轻量骨干、多层特征利用与高效变化建模，是与 AEGIS-CD 部署目标高度接近的工作。

**对 AEGIS-CD 能提供什么支撑：**  
可用于支撑“轻量 Siamese + 多尺度特征融合”技术路线，并与 AEGIS-CD 的共享轻量编码器、HFEA 形成直接对比。

**正文可参考的引用方式：**  
> “针对遥感变化检测模型参数量与计算量较高的问题，已有研究开始采用轻量 Siamese 架构实现精度与效率的平衡[3]。”

**引用优先级：** ★★★★★ 必引

---

## [4] WDMF-Net：小波对齐差异与掩膜融合

**当前参考文献：**  
Liu Y, Li S, Bruzzone L. A lightweight wavelet-aligned difference and mask-guided fusion network for change detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2026, 64: 1-16.

**推荐放在技术文档：** 相关工作；HFEA；EAOM；特征对齐；频域/小波设计

**这篇文献主要做了什么：**  
该工作围绕轻量变化检测引入相邻层级聚合、时相特征对齐、小波域差异建模与融合机制，兼顾效率与变化表征。

**对 AEGIS-CD 能提供什么支撑：**  
与 AEGIS-CD 的 HFEA + EAOM 技术链最接近。可用于说明相邻特征融合、对齐、小波差异建模在轻量变化检测中的有效性；正文应强调 AEGIS-CD 的 EAOM/RepDW/EdgeGate 为自己的具体实现，而非直接复现。

**正文可参考的引用方式：**  
> “近期轻量变化检测研究进一步将特征对齐与小波差异建模结合，以增强复杂场景下的时相差异表征能力[4]。受此类研究启发，本文设计了面向多尺度特征的 EAOM。”

**引用优先级：** ★★★★★ 核心必引

---

## [5] FSG-Net：频率—空间协同门控

**当前参考文献：**  
Xie Z, Miao S, Jiang Y, et al. FSG-Net: Frequency-spatial synergistic gated network for high-resolution remote sensing change detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2026.

**推荐放在技术文档：** 相关工作中的频域变化检测；EAOM 设计动机

**这篇文献主要做了什么：**  
通过频率信息与空间信息的协同建模，并利用门控机制完成特征选择和融合，服务于高分辨率遥感变化检测。

**对 AEGIS-CD 能提供什么支撑：**  
可支撑 EAOM 中“小波高低频信息 + 空间上下文 + 门控融合”的研究背景。不要写成 AEGIS-CD 直接采用 FSG-Net 模块。

**正文可参考的引用方式：**  
> “频率域信息能够补充空间域难以显式刻画的结构变化，频率—空间协同和门控融合已被用于提升高分辨率变化检测性能[5]。”

**引用优先级：** ★★★★★ 核心推荐

---

## [6] 双域频率学习变化检测

**当前参考文献：**  
Zhang W, Guo W, Li Y, et al. Change detection meets frequency learning: A coarse-to-fine dual-domain detection network[J]. IEEE Transactions on Geoscience and Remote Sensing, 2025, 63: 1-13.

**推荐放在技术文档：** 研究现状；频域方法综述；EAOM

**这篇文献主要做了什么：**  
从空间域与频率域联合建模变化信息，采用由粗到细的双域检测思想，说明频率线索对于遥感变化检测具有独立价值。

**对 AEGIS-CD 能提供什么支撑：**  
最适合放在“为什么 EAOM 不只做空间差分，而要显式引入频域信息”的论证位置。

**正文可参考的引用方式：**  
> “仅依赖空间域差分容易受到纹理、光照和局部结构变化的干扰，因此已有工作尝试联合空间域与频率域进行变化建模[6]。”

**引用优先级：** ★★★★★ 核心推荐

---

## [7] EGAFNet：边缘引导 + 空间频率交互

**当前参考文献：**  
Wang P, Liu Y, Ma Q, et al. EGAFNet: Edge-Guided Adaptive Fusion Network with Spatial-Frequency Interaction for Remote Sensing Change Detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2026.

**推荐放在技术文档：** 相关工作；EAOM；EdgeGate/EGBR；边界细化

**这篇文献主要做了什么：**  
将边缘引导与空间—频率交互结合，用边缘信息帮助恢复变化区域轮廓和细粒度结构。

**对 AEGIS-CD 能提供什么支撑：**  
与 AEGIS-CD 的 Edge Oracle / EdgeGate 边界细节注入以及 EAOM 的空间—频率信息协同高度相关，是解释“为什么还需要边界增强”的重要引用。

**正文可参考的引用方式：**  
> “变化区域边界往往比内部区域更易出现定位误差，边缘引导的空间—频率联合建模能够改善轮廓与细粒度结构恢复[7]。”

**引用优先级：** ★★★★★ 核心必引

---

## [8] Mallat 小波多分辨率理论

**当前参考文献：**  
Mallat S G. A theory for multiresolution signal decomposition: the wavelet representation[J]. IEEE transactions on pattern analysis and machine intelligence, 1989, 11(7): 674-693.

**推荐放在技术文档：** 方法原理；EAOM 中 Haar DWT/IDWT 数学基础

**这篇文献主要做了什么：**  
建立经典的小波多分辨率分析理论，为通过低频与不同方向高频子带描述图像结构提供理论基础。

**对 AEGIS-CD 能提供什么支撑：**  
AEGIS-CD 的 EAOM 显式使用 Haar DWT/IDWT 及 LL/HL/LH/HH 子带，可在公式、模块原理或方法基础部分引用。

**正文可参考的引用方式：**  
> “依据小波多分辨率分析理论[8]，图像可分解为低频近似分量与多个方向的高频细节分量，因此本文在 EAOM 中显式建模不同频带的时相差异。”

**引用优先级：** ★★★★★ 方法基础必引

---

## [9] Spatial Transformer Networks：可微空间变换

**当前参考文献：**  
Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks[J]. Advances in neural information processing systems, 2015, 28.

**推荐放在技术文档：** 方法原理；EAOM 的轻量时相对齐

**这篇文献主要做了什么：**  
提出可嵌入神经网络并端到端训练的可微空间变换机制，使网络能够学习几何变换并对特征进行自适应重采样。

**对 AEGIS-CD 能提供什么支撑：**  
EAOM 通过预测位移并使用 grid_sample 对第二时相特征进行可微 warp，对齐思想可由该工作提供基础支撑。

**正文可参考的引用方式：**  
> “为缓解双时相影像中局部空间偏移对差异计算的干扰，本文借鉴可微空间变换思想[9]，在特征域进行轻量级自适应对齐。”

**引用优先级：** ★★★★☆ 推荐

---

## [10] RepVGG：结构重参数化

**当前参考文献：**  
Ding X, Zhang X, Ma N, et al. Repvgg: Making vgg-style convnets great again[C]//2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2021: 13728-13737.

**推荐放在技术文档：** 方法原理；RepDW Decoder；模型部署与推理效率

**这篇文献主要做了什么：**  
提出训练阶段多分支、推理阶段等价折叠为单分支卷积的结构重参数化范式，在不保留额外推理分支的情况下提升训练表征能力。

**对 AEGIS-CD 能提供什么支撑：**  
AEGIS-CD 的 RepDW 正是训练期多分支 DW 卷积 + BN、部署期等价融合的思路。应明确写“借鉴 structural re-parameterization 思想”，避免称为 RepVGG 原模块。

**正文可参考的引用方式：**  
> “借鉴结构重参数化思想[10]，本文在训练阶段使用多分支深度卷积增强表征能力，并在部署阶段将其等价融合为单分支卷积，以兼顾训练能力与推理效率。”

**引用优先级：** ★★★★★ RepDW 必引

---

## [11] DSAMNet / SYSU-CD：深监督与数据集

**当前参考文献：**  
Shi Q, Liu M, Li S, et al. A deeply supervised attention metric-based network and an open aerial image dataset for remote sensing change detection[J]. IEEE transactions on geoscience and remote sensing, 2021, 60: 1-16.

**推荐放在技术文档：** 相关工作；训练策略/深监督；实验数据集 SYSU-CD

**这篇文献主要做了什么：**  
提出深监督注意力度量式变化检测网络，同时公开 SYSU-CD 数据集。其深监督设计和公开数据集都与 AEGIS-CD 技术文档直接相关。

**对 AEGIS-CD 能提供什么支撑：**  
一篇文献承担两个引用角色：方法部分支撑多尺度/深监督，实验部分作为 SYSU-CD 原始出处。AEGIS-CD 的 native-scale supervision 是自己的具体监督方式。

**正文可参考的引用方式：**  
> “为增强不同尺度预测分支的学习能力，变化检测网络中常引入深监督机制[11]。同时，本文实验采用的 SYSU-CD 数据集亦来源于该工作[11]。”

**引用优先级：** ★★★★★ 必引

---

## [12] STANet / LEVIR-CD

**当前参考文献：**  
Chen H, Shi Z. A spatial-temporal attention-based method and a new dataset for remote sensing image change detection[J]. Remote sensing, 2020, 12(10): 1662.

**推荐放在技术文档：** 研究现状；实验数据集 LEVIR-CD；对比实验说明

**这篇文献主要做了什么：**  
提出空间—时间注意力变化检测方法并发布 LEVIR-CD 数据集，LEVIR-CD 已成为建筑变化检测的重要基准之一。

**对 AEGIS-CD 能提供什么支撑：**  
主要在“实验数据集”部分作为 LEVIR-CD 原始出处；若相关工作讨论时空注意力，也可再次引用。

**正文可参考的引用方式：**  
> “LEVIR-CD 由 Chen 和 Shi 提出[12]，包含大规模双时相高分辨率遥感影像，主要用于建筑变化检测评测。”

**引用优先级：** ★★★★★ 数据集必引

---

## [13] WHU Building / WHU-CD 数据来源

**当前参考文献：**  
Ji S, Wei S, Lu M. Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set[J]. IEEE Transactions on geoscience and remote sensing, 2018, 57(1): 574-586.

**推荐放在技术文档：** 实验设置；数据集介绍 WHU-CD

**这篇文献主要做了什么：**  
构建并公开多源建筑提取航空/卫星影像数据，为后续 WHU 建筑变化检测数据的使用提供基础来源。

**对 AEGIS-CD 能提供什么支撑：**  
主要作为 WHU-CD/WHU Building 数据来源引用，不需要强行和 AEGIS-CD 方法模块建立联系。

**正文可参考的引用方式：**  
> “WHU-CD 数据来源于公开的 WHU 建筑遥感影像数据[13]，具有高空间分辨率和明显的建筑变化区域。”

**引用优先级：** ★★★★★ 数据集必引

---

## [14] CDD / Season-varying CD 数据来源

**当前参考文献：**  
Lebedev M A, Vizilter Y V, Vygolov O V, et al. Change detection in remote sensing images using conditional adversarial networks[J]. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2018, 42: 565-571.

**推荐放在技术文档：** 实验设置；数据集介绍 CDD-CD

**这篇文献主要做了什么：**  
利用条件对抗网络研究遥感变化检测，并形成常用的季节变化 CDD 数据来源，具有明显季节、光照和外观差异。

**对 AEGIS-CD 能提供什么支撑：**  
主要用于说明 CDD-CD 数据集来源以及它对模型抗伪变化/季节变化能力的测试价值。

**正文可参考的引用方式：**  
> “CDD 数据包含较明显的季节与外观变化，可用于考察模型在复杂非变化扰动下的鲁棒性，其原始来源见文献[14]。”

**引用优先级：** ★★★★★ 数据集必引

---

## [15] EAFH-Net：相邻特征协调与超轻量变化检测

**当前参考文献：**  
Zhai Y, Pan J, Zhang H, et al. Efficient adjacent feature harmonizer network with UAV-CD+ dataset for remote sensing change detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2024, 62: 1-18.

**推荐放在技术文档：** 相关工作；HFEA；轻量化设计；多尺度邻层融合

**这篇文献主要做了什么：**  
提出面向遥感变化检测的高效相邻特征协调网络，使用轻量骨干，并设计多尺度邻层特征融合与空间/通道协调机制；同时发布 UAV-CD+ 数据集。

**对 AEGIS-CD 能提供什么支撑：**  
与 HFEA 的“相邻层级特征融合”和轻量化目标非常契合，可作为 HFEA 设计动机的重要直接相关工作。

**正文可参考的引用方式：**  
> “相邻层级特征之间的互补信息对于轻量变化检测尤为重要，EAFH-Net 通过邻层多尺度融合提升了跨层特征交互能力[15]。本文进一步针对自身编码特征设计 HFEA。”

**引用优先级：** ★★★★★ HFEA 强烈推荐

---

## [16] LSNet：轻量空间增强网络（跨任务参考）

**当前参考文献：**  
Zhou W, Zhu Y, Lei J, et al. LSNet: Lightweight spatial boosting network for detecting salient objects in RGB-thermal images[J]. IEEE Transactions on Image Processing, 2023, 32: 1329-1340.

**推荐放在技术文档：** 轻量化相关工作；设计讨论；可选引用

**这篇文献主要做了什么：**  
该工作面向 RGB-T 显著目标检测，采用轻量化网络与空间增强思想，以降低参数和计算成本并提升细节表征。

**对 AEGIS-CD 能提供什么支撑：**  
它不是遥感变化检测论文，因此与 AEGIS-CD 的任务相关性弱于 LightCDNet、EAFH-Net、RDP-Net。可用于补充“轻量网络设计”的跨任务依据，但篇幅有限时可不引。

**正文可参考的引用方式：**  
> “轻量骨干结合空间增强机制在其他密集预测任务中也显示出较好的效率—精度平衡[16]。”

**引用优先级：** ★★☆☆☆ 可选

---

## [17] RDP-Net：区域细节保持变化检测

**当前参考文献：**  
Chen H, Pu F, Yang R, et al. RDP-Net: Region detail preserving network for change detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2022, 60: 1-10.

**推荐放在技术文档：** 相关工作；轻量化；EdgeGate/边界细节；实验对比

**这篇文献主要做了什么：**  
针对变化检测中下采样导致细节丢失和模型过重的问题，强调区域细节保持；其工作还使用边缘相关约束提升边界与小区域关注，并面向紧凑部署设计轻量模型。

**对 AEGIS-CD 能提供什么支撑：**  
可用于支撑 AEGIS-CD 的 EdgeGate/EGBR 边界增强、轻量部署目标以及“变化区域小目标/边缘容易丢失”的问题陈述。

**正文可参考的引用方式：**  
> “轻量变化检测网络在多次下采样后容易损失小区域与边界细节，RDP-Net 从区域细节保持角度对这一问题进行了研究[17]。因此，本文在高分辨率解码阶段进一步引入 EdgeGate。”

**引用优先级：** ★★★★☆ 推荐

---


# 5. 写 Word 技术文档时的推荐交叉引用组合

## 5.1 如果一句话介绍 AEGIS-CD 的整体技术路线

可组合引用 **[1][3][4][10][15]**：

- [1]：Siamese 变化检测基础；
- [3]：轻量 Siamese 变化检测；
- [4]：轻量 + 对齐 + 小波差异；
- [10]：结构重参数化；
- [15]：相邻层级融合与超轻量设计。

不建议在一句话后堆 8～10 篇文献；只选真正支撑该句内容的 2～5 篇即可。

## 5.2 写 HFEA 时

推荐引用顺序：

**[15] → [4] → [2] → [3]**

其中 [15] 与 HFEA 的“相邻特征融合”最直接，[4] 同时覆盖轻量网络和邻层聚合，[2] 是经典多尺度层级融合基础，[3] 提供轻量 Siamese 背景。

## 5.3 写 EAOM 时

推荐引用组合：

**[4][5][6][8][9]**

如果强调边界/高频，再增加：

**[7]**

对应关系：

- 对齐：[4][9]
- 小波理论：[8]
- 小波/频域变化建模：[4][5][6]
- 空间—频率协同与门控：[5][6][7]
- 边缘细节：[7]

## 5.4 写 RepDW 时

只需重点引用：

**[10]**

这样比堆多篇重参数化论文更清晰，也更容易向评委说明“AEGIS-CD 借鉴的是结构重参数化思想，而 RepDW 是针对当前轻量解码器设计的具体实现”。

## 5.5 写 EdgeGate 时

推荐：

**[7][17]**

- [7] 强调 edge-guided + spatial-frequency interaction；
- [17] 强调变化检测中的 region/detail preservation。

## 5.6 写实验数据集时

建议直接写成：

- LEVIR-CD[12]
- WHU-CD[13]
- SYSU-CD[11]
- CDD-CD[14]

不要用一篇综述替代数据集原始论文。

---

# 6. 引用优先级建议

## A 级：建议正文一定出现

[1]、[3]、[4]、[5]、[6]、[7]、[8]、[10]、[11]、[12]、[13]、[14]、[15]

这些文献能够直接覆盖 AEGIS-CD 的 Siamese 主线、轻量变化检测、HFEA、EAOM、频域/小波、边缘增强、RepDW、深监督和四个数据集。

## B 级：根据正文表述引用

[2]、[9]、[17]

- [2]：当正文解释多尺度层级特征融合/FPN 思想时引用；
- [9]：当正文解释可微特征对齐、warp、grid sampling 时引用；
- [17]：当正文强调轻量模型中的边界/小目标/区域细节保持时引用。

## C 级：可选

[16]

LSNet 的轻量设计有参考价值，但它属于 RGB-T 显著目标检测而非遥感变化检测。如果篇幅紧张，它是 17 篇里最容易删去或只放在扩展相关工作中的一篇。

---

# 7. 最后检查清单

在把这些文献粘贴进 AIC 技术文档前，建议检查：

- [ ] 正文第一次出现关键外部方法时有对应编号；
- [ ] HFEA 没有写成直接复制 EAFH-Net/WDMF-Net；
- [ ] EAOM 没有写成直接复制 FSG-Net、EGAFNet 或 WDMF-Net；
- [ ] RepDW 明确是“借鉴结构重参数化思想”，而不是声称直接使用 RepVGG 模块；
- [ ] 四个数据集都引用各自原始出处；
- [ ] 第 [2] 条 FPN 的 GB/T 7714 条目已重新核验；
- [ ] 如果正文明确写 MobileNetV2 backbone，已补充 MobileNetV2 原论文；
- [ ] 全文参考文献编号与正文第一次引用顺序一致；
- [ ] Word 中正文 `[n]` 与文末 `[n]` 一一对应；
- [ ] 所有最终引用格式统一为同一套 GB/T 7714 顺序编码制。

---

> **文档定位：** 这份 Markdown 不是只列“有哪些论文”，而是作为 AEGIS-CD 技术文档写作时的“参考文献导航表”。写到某个模型模块时，可以直接通过上面的模块对应表和逐篇索引定位应该引用哪些文献、该文献能支撑什么论述。
