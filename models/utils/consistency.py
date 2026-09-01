"""
BAIC — Batch Amplitude-Invariance Consistency (Run13, training-only).

Mixes only the low-frequency amplitude across a random batch permutation
(same permutation and mix ratio for T1/T2), then enforces prediction
consistency on high-confidence pixels between clean and mixed inputs.

Inference cost: zero.
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from torch.distributions import Beta


def low_freq_mask(H, W, r=0.125, device='cpu', dtype=torch.float32):
    cy, cx = H // 2, W // 2
    ry, rx = r * H, r * W
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    mask = ((yy - cy) ** 2 / (ry ** 2) + (xx - cx) ** 2 / (rx ** 2)) <= 1.0
    return mask.to(dtype=dtype)


def amplitude_mix(x, r=0.125, mean=None, std=None):
    """Mix low-frequency amplitude across a random batch permutation.

    Args:
        x: [B, C, H, W] float tensor (e.g. C=6 bi-temporal RGB).
        mean/std: optional channel statistics.  When provided, mixing is
            performed in de-normalised RGB space and then normalised back.
    Returns:
        x_mixed: [B, C, H, W].
    """
    if x.ndim != 4:
        raise ValueError(f'amplitude_mix expects BCHW input, got {tuple(x.shape)}')
    B, C, H, W = x.shape
    if B < 2:
        # A permutation of a singleton batch cannot provide an augmentation.
        return x
    device, dtype = x.device, x.dtype
    pi = torch.randperm(B, device=device)

    work = x
    mean_t = std_t = None
    if (mean is None) != (std is None):
        raise ValueError('mean and std must either both be provided or both be None')
    if mean is not None:
        mean_t = torch.as_tensor(mean, device=device, dtype=dtype).view(1, C, 1, 1)
        std_t = torch.as_tensor(std, device=device, dtype=dtype).view(1, C, 1, 1)
        if mean_t.numel() != C or std_t.numel() != C:
            raise ValueError(
                f'mean/std must contain {C} values for a {C}-channel input'
            )
        work = x * std_t + mean_t

    spectrum = torch.fft.fft2(work, dim=(-2, -1))
    A = torch.fft.fftshift(spectrum.abs(), dim=(-2, -1))
    phase = spectrum.angle()

    lam = Beta(torch.tensor(0.5, device=device),
               torch.tensor(0.5, device=device)).sample()
    mask = low_freq_mask(H, W, r, device, dtype)  # [H, W]

    A_mixed = (1 - lam) * A + lam * A[pi]
    A_new = mask * A_mixed + (1 - mask) * A
    A_new = torch.fft.ifftshift(A_new, dim=(-2, -1))

    spectrum_new = A_new * torch.exp(1j * phase)
    mixed = torch.fft.ifft2(spectrum_new, dim=(-2, -1)).real
    if mean_t is not None:
        mixed = mixed.clamp(0.0, 1.0)
        mixed = (mixed - mean_t) / std_t
    return mixed


def consistency_loss(p_t, p_s, conf=0.9):
    """L_cons = sum(M |p_s - p_t|) / (sum(M) + eps), M = high-confidence teacher."""
    p_t = p_t.detach()
    M = ((p_t > conf) | (p_t < (1 - conf))).float()
    num = (M * torch.abs(p_s - p_t)).sum()
    den = M.sum() + 1e-6
    return num / den


@contextmanager
def freeze_batchnorm_running_stats(module):
    """Use stored BN statistics without freezing affine parameters/gradients.

    BAIC performs a second, augmented student forward for every clean batch.
    Temporarily switching only BatchNorm modules to evaluation behaviour keeps
    E12 from updating running_mean/running_var twice while all model weights,
    including BN gamma/beta, remain trainable.
    """
    states = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            states.append((child, child.training))
            child.train(False)
    try:
        yield
    finally:
        for child, was_training in states:
            child.train(was_training)
