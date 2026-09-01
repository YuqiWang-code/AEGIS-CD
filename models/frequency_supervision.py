"""
LFDS — Local Frequency Direct Supervision (Run13, training-only).

Splits the 64x64 feature into 4x4 non-overlapping patches, computes each
patch's dual-temporal spectral change over DC/low/mid/high bands, and predicts
a 16x16 change map supervised by area-averaged GT. Phase discrepancy is
amplitude-confidence weighted so near-zero coefficients cannot inject noise.

Removed at inference time (the head does not participate in the main forward).
"""

import torch
import torch.nn as nn


class LFDS(nn.Module):
    def __init__(self, channels=64, patch_size=4, out_channels=1,
                 phase_tau=0.1):
        super().__init__()
        C = channels
        self.patch_size = patch_size
        self.phase_tau = float(phase_tau)
        if self.phase_tau <= 0:
            raise ValueError('phase_tau must be > 0')
        self.num_bands = 4
        descriptor_channels = 3 * self.num_bands * C
        # Per channel: 3 discrepancy types × 4 frequency bands.
        self.head = nn.Sequential(
            nn.Conv2d(descriptor_channels, C, 1, groups=C, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, max(C // 4, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(C // 4, 1), out_channels, 1, bias=False),
        )

    @staticmethod
    def _band_masks(size, device, dtype):
        """Return disjoint DC/low/mid/high masks in fftshift order."""
        coords = torch.arange(size, device=device) - size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        radius2 = yy.square() + xx.square()
        masks = torch.stack([
            radius2 == 0,
            (radius2 > 0) & (radius2 <= 1),
            (radius2 > 1) & (radius2 <= 2),
            radius2 > 2,
        ])
        if not torch.all(masks.any(dim=(-2, -1))):
            raise ValueError(
                f'patch_size={size} is too small for four frequency bands'
            )
        return masks.to(dtype=dtype)

    def forward(self, f1, f2):
        # f1, f2: [B, C, H, W]
        if f1.shape != f2.shape:
            raise ValueError(
                f'LFDS requires matching feature shapes, got '
                f'{tuple(f1.shape)} and {tuple(f2.shape)}'
            )
        B, C, H, W = f1.shape
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            raise ValueError(
                f'LFDS patch_size={ps} must divide HxW={H}x{W}'
            )
        gh, gw = H // ps, W // ps

        p1 = f1.reshape(B, C, gh, ps, gw, ps).permute(0, 1, 2, 4, 3, 5).reshape(B, C, gh * gw, ps, ps)
        p2 = f2.reshape(B, C, gh, ps, gw, ps).permute(0, 1, 2, 4, 3, 5).reshape(B, C, gh * gw, ps, ps)

        F1 = torch.fft.fftshift(
            torch.fft.fft2(p1, dim=(-2, -1)), dim=(-2, -1)
        )
        F2 = torch.fft.fftshift(
            torch.fft.fft2(p2, dim=(-2, -1)), dim=(-2, -1)
        )
        A1, A2 = F1.abs(), F2.abs()
        phi1, phi2 = F1.angle(), F2.angle()

        D_A = torch.abs(torch.log1p(A1) - torch.log1p(A2))
        phase_confidence = torch.sqrt(A1 * A2 + 1e-8)
        phase_confidence = phase_confidence / (
            phase_confidence + self.phase_tau
        )
        phase_delta = phi1 - phi2
        D_phi_sin = phase_confidence * torch.abs(torch.sin(phase_delta))
        D_phi_cos = phase_confidence * (1.0 - torch.cos(phase_delta))

        masks = self._band_masks(ps, f1.device, f1.dtype)
        counts = masks.sum(dim=(-2, -1)).clamp_min(1.0)

        def band_mean(t):
            # [B,C,N,ps,ps] × [K,ps,ps] -> [B,C,N,K]
            summed = torch.einsum('bcnxy,kxy->bcnk', t, masks)
            return summed / counts.view(1, 1, 1, self.num_bands)

        z = torch.stack(
            [band_mean(D_A), band_mean(D_phi_sin), band_mean(D_phi_cos)],
            dim=1,
        )  # [B,3,C,gh*gw,4]
        # Interleave all 12 descriptors per feature channel for groups=C.
        z = z.permute(0, 2, 1, 4, 3)
        z = z.reshape(B, 3 * self.num_bands * C, gh, gw)
        return self.head(z)  # [B, out_channels, gh, gw]
