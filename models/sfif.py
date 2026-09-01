"""
SFIF — Spatial-Frequency Interactive Fusion  (v3, DAWIM removed)
==================================================================
Drop-in replacement for MSA_Module in DecoderFusion.

Architecture (v3, pure DWT):
  A. Spatial path: 5×5 DWConv + PWConv (unchanged)
  B. Frequency path: Haar DWT → LL/HF enhancement
     - LL enhancement
     - HF rotation-invariant energy + ratio modulation
     - f4 (H≤16): LL-only (no HF)
  C. SE-Gate: spatial + freq → GAP→FC→Sigmoid → channel-wise gate
  D. Output projection + residual

4 independent instances per decoder scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class SFIF(nn.Module):
    """Spatial-Frequency Interactive Fusion v3 — MSA_Module replacement."""

    def __init__(self, channels=64):
        super(SFIF, self).__init__()
        C = channels

        # ---- A. Spatial path ----
        # Phase 6 优化 D: 3×3 + 5×5 并行（增强多尺度感受野）
        self.spatial_3x3 = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU(),
        )
        self.spatial_5x5 = nn.Sequential(
            nn.Conv2d(C, C, 5, 1, 2, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU(),
            nn.Conv2d(C, C, 5, 1, 2, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU(),
        )

        # ---- B. Frequency path (DWT + DAWIM) ----
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(wave='haar')

        # LL + HF enhancement convs
        self.ll_enhance = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU(),
        )
        # Phase 7 优化 A: 共享增强器 + 方向调制（替代三个独立分支）
        self.hf_shared_enhance = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(inplace=True),
        )
        self.direction_modulation = nn.Conv2d(3, 3, 1, bias=False)  # 3 方向 → 3 方向

        self.freq_norm = nn.Sequential(nn.BatchNorm2d(C), nn.GELU())

        # ---- C. SE-Gate (spatial + freq → channel-wise) ----
        # Phase 6 优化 B: 非对称 Gating（输出 2C，独立权重）
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            nn.Conv2d(C * 2, C // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, C * 2, 1),  # 输出 2C (was C)
        )

        # Phase 7 优化 C: 自适应空间分支权重
        self.spatial_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, 2, 1),  # 输出 2 个权重（3×3 和 5×5）
        )

        # ---- D. Output ----
        self.out_proj = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU(),
        )

    def forward(self, x):
        C = x.size(1)

        # ---- A. Spatial ----
        # Phase 7 优化 C: 自适应空间分支权重
        feat_3x3 = self.spatial_3x3(x)
        feat_5x5 = self.spatial_5x5(x)
        weights = F.softmax(self.spatial_gate(x), dim=1)  # [B, 2, 1, 1]
        w_3x3, w_5x5 = weights[:, 0:1], weights[:, 1:2]
        spatial_feat = feat_3x3 * w_3x3 + feat_5x5 * w_5x5

        # ---- B. Frequency (DWT) ----
        x_L, x_H = self.dwt(x)
        hf_bands = x_H[0]
        HL, LH, HH = hf_bands[:,:,0], hf_bands[:,:,1], hf_bands[:,:,2]

        # LL enhancement
        ll_en = self.ll_enhance(x_L)

        # Phase 7 优化 A: 共享增强器 + 方向调制
        hf_stack = torch.stack([HL, LH, HH], dim=2)  # [B, C, 3, H/2, W/2]
        B, C_val, _, H2, W2 = hf_stack.shape

        # 共享增强（循环处理三个方向）
        hf_enhanced = []
        for i in range(3):
            hf_enhanced.append(self.hf_shared_enhance(hf_stack[:, :, i]))
        hf_enhanced = torch.stack(hf_enhanced, dim=2)  # [B, C, 3, H/2, W/2]

        # 方向调制（学习三个方向的相对权重）
        # 将 [B, C, 3, H/2, W/2] 转换为 [B, 3, H/2, W/2]
        hf_mean = hf_stack.mean(dim=1)  # [B, 3, H/2, W/2]
        direction_weights = F.softmax(
            self.direction_modulation(hf_mean), dim=1
        )  # [B, 3, H/2, W/2]
        # 广播相乘：[B, C, 3, H/2, W/2] * [B, 1, 3, H/2, W/2]
        hf_en = hf_enhanced * direction_weights.unsqueeze(1)  # 在 C 维度扩展

        # Phase 7 优化 D: 频域激活（IDWT 后加非线性）
        freq_feat = self.idwt((ll_en, [hf_en]))
        freq_feat = self.freq_norm(freq_feat)  # 已有 GELU

        # ---- C. SE-Gate ----
        # Phase 7 优化 B: 软互斥 Gating（归一化避免信息丢失）
        cat = torch.cat([spatial_feat, freq_feat], dim=1)
        gates = self.se_fc(self.se_pool(cat))  # [B, 2C, 1, 1]
        gate_sp, gate_fq = gates.chunk(2, dim=1)
        gate_sp = torch.sigmoid(gate_sp)
        gate_fq = torch.sigmoid(gate_fq)
        # 软互斥：归一化到 [0, 1] 区间，和为 1
        gate_sum = gate_sp + gate_fq + 1e-6
        gate_sp_norm = gate_sp / gate_sum
        gate_fq_norm = gate_fq / gate_sum
        fused = spatial_feat * gate_sp_norm + freq_feat * gate_fq_norm

        # ---- D. Output ----
        return self.out_proj(fused) + x
