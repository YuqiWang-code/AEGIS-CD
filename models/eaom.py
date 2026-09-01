"""
EAOM — Edge-Aware Oracle Module  (v2, DAWIM removed)
======================================================
Zero-parameter frequency-domain replacement for CFDM (DiffModule).

Key features:
1. **HF rotation-invariant energy** — Euclidean norm sqrt(LH²+HL²+HH²)
   replaces channel-wise concatenation (0 extra params).
2. **Residual gradient gating** — ``hf *= (1 + edge_mask)`` prevents
   gradient vanishing when sigmoid → 0 during early training.
3. **Dual-pool LL attention** — AvgPool + MaxPool with shared MLP
   captures sparse small-object change peaks (0 extra params).
4. **DWConv alignment** — depthwise-separable first conv in offset_conv
   saves parameters for the fusion stage.
5. **Anti-aliasing blur** — fixed Gaussian kernel before DWT suppresses
   Haar checkerboard artifacts from sub-pixel mis-registration.
6. **T1 prior injection** — Hadamard product of T1 features with IDWT
   output filters seasonal spectral drift from real structural change.

I/O contract — identical to DiffModule
    Input:  f1 [B, C, H, W],  f2 [B, C, H, W]
    Output: [B, C, H, W]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


# =========================================================
#  Utility: fixed Gaussian anti-aliasing kernel
# =========================================================

def make_gaussian_kernel(kernel_size=3, sigma=0.8, channels=64):
    """Return a fixed (non-learnable) 2D Gaussian kernel for depthwise conv."""
    ax = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
    g1d = torch.exp(-ax ** 2 / (2.0 * sigma ** 2))
    g2d = g1d[:, None] * g1d[None, :]
    g2d = g2d / g2d.sum()
    # [C, 1, k, k]  for depthwise → each channel blurred independently
    return g2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


# =========================================================
#  EAOM
# =========================================================

class EAOM(nn.Module):
    """Edge-Aware Oracle Module — drop-in replacement for ``DiffModule``."""

    def __init__(self, channels=64, expansion=1.0):
        super(EAOM, self).__init__()
        inner = max(int(channels * expansion), channels)

        # ---- Wavelet transforms (0 learnt parameters) ----
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(wave='haar')

        # ---- 1. Lightweight alignment (DWConv → pointwise → offset) ----
        self.offset_conv = nn.Sequential(
            # depthwise — saves ~channels² params vs regular 3×3
            nn.Conv2d(channels * 2, channels * 2, 3, 1, 1,
                      groups=channels * 2, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            # pointwise compression
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # offset prediction
            nn.Conv2d(channels, 2, 3, 1, 1),
        )

        # ---- 2. Edge Oracle on rotation-invariant HF energy ----
        # Phase 6 优化 A: per-channel edge mask (not 1-ch broadcast)
        self.edge_oracle = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1,
                      groups=channels, bias=False),   # DWConv
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),  # C-ch mask (was 1)
            nn.Sigmoid(),
        )

        # ---- 3. Dual-pool channel attention on LL ----
        self.ll_attn_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ll_attn_maxpool = nn.AdaptiveMaxPool2d(1)
        self.ll_attn_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
        )
        self.ll_attn_sigmoid = nn.Sigmoid()

        # Phase 6 优化 B: HF 通道注意力（与 LL 对等）
        self.hf_attn_avgpool = nn.AdaptiveAvgPool2d(1)
        self.hf_attn_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Phase 7 优化 A: 三方向独立 ratio 调制（替代统一 ratio）
        # Phase 8 修复: 移除 nn.Sigmoid（tanh 已提供 [-1,1]→[0,4] 映射，
        # 原双重激活把 ratio 锁死在 [2, 3.52]，丧失衰减能力）
        self.hf_ratio_LH = nn.Conv2d(channels, channels, 1, bias=False)
        self.hf_ratio_HL = nn.Conv2d(channels, channels, 1, bias=False)
        self.hf_ratio_HH = nn.Conv2d(channels, channels, 1, bias=False)

        # Phase 7 优化 C: Diff-Context 交叉注意力
        self.cross_attn_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.cross_attn_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.cross_attn_v = nn.Conv2d(channels, channels, 1, bias=False)

        # ---- 4. Sub-band enhancement (lightweight) ----
        self.ll_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.hf_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1,
                      groups=channels, bias=False),   # DWConv on energy
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # ---- 5. Context & fusion ----
        self.ctx_conv = nn.Conv2d(channels * 2, channels, 1)
        # Phase 6 优化 C: 移除 T1 门控，fuse 改为 2C→C (was 3C→C)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),  # 2C → C
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # ---- 6. Anti-aliasing Gaussian blur (0 learnable params) ----
        gauss_k = make_gaussian_kernel(3, 0.8, channels)
        self.register_buffer('gaussian_kernel', gauss_k)

    # ------------------------------------------------------------------
    def flow_warp(self, x, flow):
        """Backward-warp *x* according to *flow* (N,2,H,W)."""
        n, c, h, w = x.size()
        device = x.device
        norm = torch.tensor([[[[w, h]]]], device=device, dtype=x.dtype)
        col = torch.linspace(-1.0, 1.0, h, device=device).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w, device=device).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        return F.grid_sample(x, grid, align_corners=True)

    # ------------------------------------------------------------------
    def _anti_alias(self, x):
        """Apply fixed Gaussian blur to suppress Haar checkerboard artifacts."""
        return F.conv2d(x, self.gaussian_kernel, padding=1, groups=x.size(1))

    # ------------------------------------------------------------------
    def forward(self, f1, f2):
        """
        Args:
            f1: pre-change features  [B, C, H, W]
            f2: post-change features [B, C, H, W]
        Returns:
            enhanced difference features [B, C, H, W]
        """
        B, C, H, W = f1.shape

        # ---- 1. Lightweight alignment (DWConv-based) ----
        cat = torch.cat([f1, f2], dim=1)
        offset = self.offset_conv(cat)
        f2_aligned = self.flow_warp(f2, offset)

        # ---- 2. |diff| → anti-alias → DWT ----
        diff = torch.abs(f1 - f2_aligned)
        diff = self._anti_alias(diff)                # Gaussian pre-blur
        diff_L, diff_H = self.dwt(diff)              # LL + H tuple
        hf = diff_H[0]                                # [B, C, 3, H/2, W/2]

        # Extract directional sub-bands
        HL = hf[:, :, 0, :, :]   # horizontal  edges
        LH = hf[:, :, 1, :, :]   # vertical    edges
        HH = hf[:, :, 2, :, :]   # diagonal    edges

        # ---- 3. Rotation-invariant HF energy (0 params) ----
        E_edge = torch.sqrt(LH ** 2 + HL ** 2 + HH ** 2 + 1e-6)  # [B, C, H/2, W/2]

        # Phase 7 优化 B: 高分辨率 Edge Oracle（在原始分辨率计算）
        edge_mask_hr = self.edge_oracle(diff)                    # [B, C, H, W]
        edge_mask = F.avg_pool2d(edge_mask_hr, 2, 2)             # [B, C, H/2, W/2]

        # ---- 5. Dual-pool channel attention on LL ----
        ll_avg = self.ll_attn_mlp(self.ll_attn_avgpool(diff_L))
        ll_max = self.ll_attn_mlp(self.ll_attn_maxpool(diff_L))
        ch_attn = self.ll_attn_sigmoid(ll_avg + ll_max)          # [B, C, 1, 1]

        # ---- 6. LL enhancement ----
        LL_en = diff_L * ch_attn
        LL_en = self.ll_enhance(LL_en)                           # [B, C, H/2, W/2]

        # ---- 7. HF enhancement on rotation-invariant energy ----
        E_edge_en = self.hf_enhance(E_edge)                      # [B, C, H/2, W/2]
        # Phase 6 优化 B: HF 通道注意力
        hf_ch_attn = self.hf_attn_mlp(self.hf_attn_avgpool(E_edge))  # [B, C, 1, 1]
        # Residual gating: 优化 A 的 per-channel edge_mask
        E_edge_en = E_edge_en * hf_ch_attn * (1.0 + edge_mask)

        # Phase 7 优化 A + D: 三方向独立 ratio + 软约束（Tanh 映射到 [0, 4]）
        ratio_LH = 2.0 * (1 + torch.tanh(self.hf_ratio_LH(E_edge_en)))  # [0, 4]
        ratio_HL = 2.0 * (1 + torch.tanh(self.hf_ratio_HL(E_edge_en)))
        ratio_HH = 2.0 * (1 + torch.tanh(self.hf_ratio_HH(E_edge_en)))
        LH_en = LH * ratio_LH
        HL_en = HL * ratio_HL
        HH_en = HH * ratio_HH

        # Stack for IDWT: [B, C, 3, H/2, W/2]
        hf_en = torch.stack([HL_en, LH_en, HH_en], dim=2)

        # ---- 8. IDWT reconstruction ----
        diff_recon = self.idwt((LL_en, [hf_en]))                 # [B, C, H, W]

        # Phase 6 优化 C: 移除 T1 prior injection
        # ---- 9. Context from aligned pair ----
        ctx = self.ctx_conv(torch.cat([f1, f2_aligned], dim=1))

        # Phase 7 优化 C: Diff-Context 交叉注意力
        q = self.cross_attn_q(diff_recon)  # [B, C, H, W]
        k = self.cross_attn_k(ctx)
        v = self.cross_attn_v(ctx)
        attn = torch.sigmoid(q * k)  # 简化点积注意力
        diff_enhanced = diff_recon + attn * v  # 残差连接

        # ---- 10. Fuse: diff + context ----
        output = self.fuse(torch.cat([diff_enhanced, ctx], dim=1))  # 2C → C
        return output
