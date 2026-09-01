"""
APID-LF — Low-Frequency Amplitude-Phase Invariant Decoupling (Run13).

Shares only the low-frequency amplitude between T1/T2 features at the two
highest-resolution scales, leaving phase and mid/high-frequency amplitude
untouched, to suppress illumination/appearance pseudo-change.

Epoch-0 identity via a zero-initialised residual; remains part of inference
when enabled because it directly normalises the bi-temporal feature pair.
"""

import torch
import torch.nn as nn


class APID(nn.Module):
    def __init__(self, radius_ratio=0.125):
        super().__init__()
        self.radius_ratio = radius_ratio
        self.gamma = nn.Parameter(torch.zeros(1))

    def _low_freq_mask(self, H, W, device, dtype):
        # fftshift places the DC bin at integer index H//2, W//2 for even
        # feature sizes (64/32); using (H-1)/2 would offset the mask by 0.5 px.
        cy, cx = H // 2, W // 2
        ry = self.radius_ratio * H
        rx = self.radius_ratio * W
        y = torch.arange(H, device=device, dtype=dtype)
        x = torch.arange(W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        mask = ((yy - cy) ** 2 / (ry ** 2) + (xx - cx) ** 2 / (rx ** 2)) <= 1.0
        return mask.to(dtype=dtype)[None, None, :, :]  # [1, 1, H, W]

    def forward(self, f1, f2):
        if f1.shape != f2.shape or f1.ndim != 4:
            raise ValueError(
                f'APID requires matching BCHW tensors, got '
                f'{tuple(f1.shape)} and {tuple(f2.shape)}'
            )
        B, C, H, W = f1.shape
        F1 = torch.fft.fft2(f1, dim=(-2, -1))
        F2 = torch.fft.fft2(f2, dim=(-2, -1))

        # amplitude in shifted domain so the low-frequency mask is centred
        A1 = torch.fft.fftshift(F1.abs(), dim=(-2, -1))
        A2 = torch.fft.fftshift(F2.abs(), dim=(-2, -1))
        phase1 = F1.angle()
        phase2 = F2.angle()

        mask = self._low_freq_mask(H, W, f1.device, f1.dtype)
        Ac = torch.sqrt(A1 * A2 + 1e-8)

        A1_new = mask * Ac + (1.0 - mask) * A1
        A2_new = mask * Ac + (1.0 - mask) * A2
        A1_new = torch.fft.ifftshift(A1_new, dim=(-2, -1))
        A2_new = torch.fft.ifftshift(A2_new, dim=(-2, -1))

        F1_new = A1_new * torch.exp(1j * phase1)
        F2_new = A2_new * torch.exp(1j * phase2)
        f1_hat = torch.fft.ifft2(F1_new, dim=(-2, -1)).real
        f2_hat = torch.fft.ifft2(F2_new, dim=(-2, -1)).real

        scale = torch.tanh(self.gamma)
        return f1 + scale * (f1_hat - f1), f2 + scale * (f2_hat - f2)
