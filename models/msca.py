"""
MSCA — Multi-Scale Convolutional Attention
============================================

Four independent MSCA instances refine the four difference scales before
DecoderFusion.

Inputs
------
- context_f1: pre-Prior encoder-fusion feature from T1 [B, C, H, W]
- context_f2: pre-Prior encoder-fusion feature from T2 [B, C, H, W]
- diff:       current difference feature             [B, C, H, W]

Pipeline
--------
1. Independently align T1/T2 context with separate 1x1 projections.
2. Generate a soft fusion weight from concatenated aligned context.
3. Apply four parallel depthwise branches: 1x1, 3x3, 5x5, 7x7.
4. Selectively fuse the four branches with temperature-scaled SK softmax.
5. Apply channel attention and parallel 3x3 + 5x5 spatial attention.
6. Refine the difference feature with an explicit identity residual:

       f_refined = diff + f_modulated * residual_strength

Key design decisions
--------------------
- Four-scale independent MSCA instances.
- Soft branch weighting; no hard Top-k routing.
- Pre-Prior T1/T2 context is used for semantic guidance.
- Explicit identity residual is retained.
- No auxiliary supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCA(nn.Module):
    """Multi-Scale Convolutional Attention — semantic-guided difference refinement."""

    def __init__(self, channels=64, reduction=4):
        super(MSCA, self).__init__()
        self.channels = channels

        # Phase 7 优化 B: Context 独立对齐（保留 T1/T2 独立性）
        self.ctx_align_t1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.ctx_align_t2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.ctx_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),  # 生成融合权重
        )

        # ---- Step 2: Multi-scale DWConv branches ----
        self.branch_1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.branch_5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, 1, 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.branch_7 = nn.Sequential(
            nn.Conv2d(channels, channels, 7, 1, 3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Phase 7 优化 A: 可学习 Temperature
        # Phase 9 修复: 初始值 30.0 → 3.0。τ=30 使 softmax(attn/τ) 接近均匀分布，
        # SK 门控在训练早期退化为四分支平均，学不到"选择性"。
        self.temperature = nn.Parameter(torch.ones(1) * 3.0)

        # ---- Step 3: Selective Kernel Gating (4 branches) ----
        self.sk_gap = nn.AdaptiveAvgPool2d(1)
        self.sk_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 4, 1, bias=False),
        )
        self.sk_softmax = nn.Softmax(dim=1)

        # ---- Step 4: Dual-Attention ----
        # Channel attention (SE-style)
        self.ch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # Phase 6 微调 A: 空间注意力自适应感受野（3×3 + 5×5）
        self.sp_attn_3x3 = nn.Conv2d(channels, 1, 3, 1, 1, bias=False)
        self.sp_attn_5x5 = nn.Conv2d(channels, 1, 5, 1, 2, bias=False)

        # Phase 7 优化 C: 可学习残差强度（per-pixel adaptive residual）
        self.residual_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def forward(self, context_f1, context_f2, diff):
        """
        Args:
            context_f1: pre-Prior T1 context feature [B, C, H, W]
            context_f2: pre-Prior T2 context feature [B, C, H, W]
            diff:       current difference feature   [B, C, H, W]

        Returns:
            refined difference feature [B, C, H, W]
        """
        # Phase 7 优化 B: Context 独立对齐（保留 T1/T2 差异信息）
        ctx_t1 = self.ctx_align_t1(context_f1)
        ctx_t2 = self.ctx_align_t2(context_f2)
        fusion_weight = self.ctx_fusion(torch.cat([ctx_t1, ctx_t2], dim=1))
        ctx = ctx_t1 * fusion_weight + ctx_t2 * (1 - fusion_weight)

        # ---- Step 2: Multi-scale branches ----
        u1 = self.branch_1(ctx)                              # [B, C, H, W]
        u3 = self.branch_3(ctx)
        u5 = self.branch_5(ctx)
        u7 = self.branch_7(ctx)
        u_sum = u1 + u3 + u5 + u7                            # [B, C, H, W]

        # ---- Step 3: Selective Kernel Gating ----
        gap = self.sk_gap(u_sum)                             # [B, C, 1, 1]
        attn_raw = self.sk_mlp(gap)                          # [B, 4C, 1, 1]
        # Temperature-scaled softmax over 4 branches
        attn_stack = attn_raw.view(attn_raw.size(0), 4, self.channels, 1, 1)
        attn_stack = self.sk_softmax(attn_stack / self.temperature)
        a1, a3, a5, a7 = (attn_stack[:, 0], attn_stack[:, 1],
                          attn_stack[:, 2], attn_stack[:, 3])

        f_multi = u1 * a1 + u3 * a3 + u5 * a5 + u7 * a7     # [B, C, H, W]

        # ---- Step 4: Dual-Attention modulation ----
        ch_w = self.ch_attn(f_multi)                         # [B, C, 1, 1]
        f_ch = f_multi * ch_w                                # channel attention

        # Phase 6 微调 A: 3×3 + 5×5 空间注意力并行
        sp_w = torch.sigmoid(self.sp_attn_3x3(f_ch) + self.sp_attn_5x5(f_ch))  # [B, 1, H, W]

        # Phase 7 优化 C: 可学习残差强度（自适应残差）
        f_modulated = diff * ch_w * sp_w                     # [B, C, H, W]
        residual_strength = self.residual_gate(f_modulated)
        # Phase 8 修复: 恢复显式恒等残差 `+ diff`（原实现去掉恒等项，
        # diff 会被 ch_w*sp_w 与 residual_strength 双重衰减，梯度流不稳）
        f_refined = diff + f_modulated * residual_strength   # true identity + adaptive modulation

        return f_refined
