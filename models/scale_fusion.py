"""
Scale-Calibrated Residual Fusion (SCRF) — Run12.

Replaces fixed `z = f + upsample(f_next)` with learnable scalar modulation:
    alpha = 1 + tanh(gamma)
    z = f + alpha * upsample(f_next)

where gamma is initialized to 0, so alpha starts at 1 (identity warm-start).
"""

import torch
import torch.nn as nn


class SCRF(nn.Module):
    """Scale-Calibrated Residual Fusion.

    Args:
        num_scales: number of fusion points (default 3: f3←f4, f2←f3, f1←f2).
    """

    def __init__(self, num_scales=3):
        super(SCRF, self).__init__()
        # gamma_i initialized to 0 → alpha_i = 1 + tanh(0) = 1.0
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(num_scales)
        ])

    def calibrated_add(self, lateral, upsampled, gamma):
        """Calibrated residual addition.

        Args:
            lateral: [B, C, H, W], current-scale feature.
            upsampled: [B, C, H, W], upsampled top-down feature.
            gamma: scalar Parameter.

        Returns:
            [B, C, H, W], fused feature.
        """
        alpha = 1.0 + torch.tanh(gamma)
        return lateral + alpha * upsampled

    def forward(self, lateral, upsampled, scale_idx):
        """Apply calibrated fusion for a specific scale.

        Args:
            lateral: [B, C, H, W].
            upsampled: [B, C, H, W].
            scale_idx: 0 (f3←f4), 1 (f2←f3), or 2 (f1←f2).

        Returns:
            [B, C, H, W].
        """
        if scale_idx < 0 or scale_idx >= len(self.gammas):
            raise IndexError(
                f'scale_idx={scale_idx} out of range [0, {len(self.gammas)})'
            )
        if lateral.shape != upsampled.shape:
            raise ValueError(
                f'Shape mismatch: lateral {lateral.shape} vs upsampled {upsampled.shape}'
            )
        return self.calibrated_add(lateral, upsampled, self.gammas[scale_idx])


def get_scale_fusion(mode='plain'):
    """Factory function for scale fusion strategies.

    Args:
        mode: 'plain' (fixed addition) or 'scrf' (learnable calibration).

    Returns:
        SCRF instance if mode='scrf', else None (plain uses fixed `+`).
    """
    if mode == 'scrf':
        return SCRF(num_scales=3)
    elif mode == 'plain':
        return None
    else:
        raise ValueError(f"Unknown scale_fusion mode: {mode!r}")
