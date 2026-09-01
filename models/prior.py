"""
HFC Prior Injector v3 — Multi-Component Physical Prior Injection
==================================================================
Upgrades over v2:
  - 4 physical prior components replace single FFT high-pass
  - P1: Illumination-invariant reflectance (log-chromaticity projection)
  - P2: Structure tensor geometry (corner/edge/flat decomposition)
  - P3: Shadow-aware confidence map (B/G ratio + brightness)
  - P4: Multi-scale texture (Gabor filter bank + LBP)
  - Asymmetric per-channel injection retained from v2

Insertion point: immediately after HFEA, before DiffModule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorGenerator(nn.Module):
    """Generate 4 physical prior components from RGB images."""

    def __init__(self, out_channels=64):
        super().__init__()
        C = out_channels

        # P1: Illumination-invariant reflectance (3→C)
        self.p1_conv = nn.Sequential(nn.Conv2d(3, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU())

        # P2: Structure tensor geometry (3→C) — corner/edge/flat
        self.p2_conv = nn.Sequential(nn.Conv2d(3, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU())

        # Phase 6 优化 D: 删除 P3 阴影先验（贡献最小）
        # P4: Multi-scale texture (5→C) — Gabor×4 + variance×1
        self.p4_conv = nn.Sequential(nn.Conv2d(5, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU())

        # Phase 6 优化 D: Component blending weights（3 分量：P1, P2, P4）
        self.comp_weights = nn.Parameter(torch.ones(3) / 3.0)

        # Precomputed directed Gabor kernels (4 orientations, 7×7, σ=2.0, λ=4.0)
        ksize, sigma, lam = 7, 2.0, 4.0
        xs = torch.arange(-ksize // 2 + 1, ksize // 2 + 1, dtype=torch.float32)
        ys = torch.arange(-ksize // 2 + 1, ksize // 2 + 1, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        gauss = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sigma ** 2))
        gauss = gauss / (gauss.sum() + 1e-6)
        for deg in [0, 45, 90, 135]:
            rad = torch.tensor(deg, dtype=torch.float32) * torch.pi / 180.0
            # Gabor: Gaussian × cos(2π × (x cosθ + y sinθ) / λ)
            wave = torch.cos(2 * torch.pi * (gx * torch.cos(rad) + gy * torch.sin(rad)) / lam)
            kern = (gauss * wave).view(1, 1, ksize, ksize)
            self.register_buffer(f'gabor_kernel_{deg}', kern)

    @property
    def gabor_kernels(self):
        return [getattr(self, f'gabor_kernel_{t}') for t in ['0', '45', '90', '135']]

    # ------------------------------------------------------------------
    def _illumination_invariant(self, rgb):
        """P1: log-chromaticity projection removes illumination direction."""
        eps = 1e-6
        R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        c1 = torch.log(R / (G + eps) + eps)
        c2 = torch.log(B / (G + eps) + eps)
        c3 = torch.log((R + B) / (G + eps) + eps)
        return torch.cat([c1, c2, c3], dim=1)

    def _structure_tensor(self, rgb):
        """P2: gradient structure tensor with eigenvalue decomposition."""
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        # Sobel gradients
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=rgb.device,
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=rgb.device,
                                dtype=torch.float32).view(1, 1, 3, 3)
        Ix = F.conv2d(gray, sobel_x, padding=1)
        Iy = F.conv2d(gray, sobel_y, padding=1)
        Ix2, Iy2, Ixy = Ix * Ix, Iy * Iy, Ix * Iy
        # Gaussian smooth (3×3 avg)
        k = torch.ones(1, 1, 3, 3, device=rgb.device) / 9.0
        Ix2_s = F.conv2d(Ix2, k, padding=1)
        Iy2_s = F.conv2d(Iy2, k, padding=1)
        Ixy_s = F.conv2d(Ixy, k, padding=1)
        # Eigenvalues
        trace = Ix2_s + Iy2_s + 1e-6
        det = Ix2_s * Iy2_s - Ixy_s * Ixy_s + 1e-6
        disc = torch.clamp(trace * trace - 4 * det, min=0)
        lam1 = (trace + torch.sqrt(disc)) / 2
        lam2 = (trace - torch.sqrt(disc)) / 2
        corner = lam1 * lam2 / (lam1 + lam2 + 1e-6)
        edge = lam1 - lam2
        flat = 1.0 / (1.0 + lam1 + lam2)
        return torch.cat([corner, edge, flat], dim=1)

    # Phase 6 优化 D: _shadow_aware 方法已删除

    def _gabor_lbp_texture(self, rgb):
        """P4: 4 Gabor orientations + local variance + zero-pad to 13ch."""
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        B, _, H, W = gray.shape
        eps = 1e-6
        textures = []
        for theta in [0, 45, 90, 135]:
            kern = self.gabor_kernels[theta // 45]  # precomputed buffer
            textures.append(F.conv2d(gray, kern, padding=kern.shape[-1] // 2))
        # Local variance
        local_mean = F.avg_pool2d(gray, 3, 1, 1)
        local_var = F.avg_pool2d((gray - local_mean) ** 2, 3, 1, 1)
        textures.append(local_var)
        # Phase 6 优化 B: 直接返回 5 通道（删除 zero-pad）
        return torch.cat(textures, dim=1)  # [B, 5, H, W]

    # ------------------------------------------------------------------
    def forward(self, t1_rgb, t2_rgb):
        """Generate 3 prior components (Phase 6: P3 删除) and their temporal differences."""
        # P1: illumination-invariant
        iif_t1 = self._illumination_invariant(t1_rgb)
        iif_t2 = self._illumination_invariant(t2_rgb)
        prior1 = self.p1_conv(torch.abs(iif_t1 - iif_t2))

        # P2: structure tensor
        st_t1 = self._structure_tensor(t1_rgb)
        st_t2 = self._structure_tensor(t2_rgb)
        prior2 = self.p2_conv(torch.abs(st_t1 - st_t2))

        # Phase 6 优化 D: P3 shadow-aware 已删除

        # P4: multi-scale texture
        tex_t1 = self._gabor_lbp_texture(t1_rgb)
        tex_t2 = self._gabor_lbp_texture(t2_rgb)
        prior4 = self.p4_conv(torch.abs(tex_t1 - tex_t2))

        # Phase 6 优化 D: Weighted blend (3 分量)
        w = F.softmax(self.comp_weights, dim=0)
        prior_blend = w[0] * prior1 + w[1] * prior2 + w[2] * prior4

        return prior_blend  # 不再返回 prior_shadow


class HFCPriorInjector(nn.Module):
    """Multi-component physical prior injector — asymmetric per-channel."""

    def __init__(self, channels=64):
        super().__init__()
        C = channels
        self.generator = PriorGenerator(C)
        self.fuse1 = nn.Sequential(nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU())
        self.fuse2 = nn.Sequential(nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C), nn.GELU())
        # Phase 6 优化 C: α 降至 0.08（折中 0.05 和 0.1，减轻分布偏移）
        self.alpha1 = nn.Parameter(torch.ones(1, C, 1, 1) * 0.08)
        self.alpha2 = nn.Parameter(torch.ones(1, C, 1, 1) * 0.08)
        # Phase 6 优化 D: beta1/beta2 已删除（P3 阴影先验删除）

    def forward(self, f1_list, f2_list, t1_rgb=None, t2_rgb=None):
        if t1_rgb is None:
            t1_rgb = f1_list[0].detach()[:, :3]
        if t2_rgb is None:
            t2_rgb = f2_list[0].detach()[:, :3]

        # De-normalize: images come in normalized (mean/std), revert to [0,1] for physics
        # ToTensor presents each temporal image to the model in RGB order.
        mean = torch.tensor([0.485, 0.456, 0.406], device=t1_rgb.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=t1_rgb.device).view(1, 3, 1, 1)
        t1_raw = (t1_rgb * std + mean).clamp(0, 1)
        t2_raw = (t2_rgb * std + mean).clamp(0, 1)

        # Phase 6 优化 D: generator 只返回 prior_blend（不再有 prior_shadow）
        prior_blend = self.generator(t1_raw, t2_raw)

        f1_enh, f2_enh = [], []
        for feat1, feat2 in zip(f1_list, f2_list):
            H, W = feat1.shape[2:]
            p_blend = F.interpolate(prior_blend, size=(H, W), mode='bilinear', align_corners=False)
            p1 = self.fuse1(p_blend)
            p2 = self.fuse2(p_blend)
            # Phase 6 优化 D: Additive prior only（删除 subtractive shadow prior）
            f1_enh.append(feat1 + self.alpha1 * p1.clamp(-1, 1))
            f2_enh.append(feat2 + self.alpha2 * p2.clamp(-1, 1))
        return tuple(f1_enh), tuple(f2_enh)
