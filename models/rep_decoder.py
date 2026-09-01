"""Lightweight convolutional decoder blocks for AEGIS-CD Phase 11.

``PlainDWBlock`` is the single-branch control. ``RepDWBlock`` uses four
linear depthwise branches while training and folds them into one depthwise
5x5 convolution for deployment.  Both blocks keep the same residual output
contract: ``y = x + delta``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainDWBlock(nn.Module):
    """Single-branch depthwise-separable residual decoder block."""

    def __init__(self, channels=64):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=5, padding=2,
                groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)

        # Exact identity at initialization: the correction branch is zero.
        nn.init.zeros_(self.pw_bn.weight)
        nn.init.zeros_(self.pw_bn.bias)

    def forward(self, x):
        delta = self.pw_bn(self.pw(self.act(self.dw(x))))
        return x + delta


class RepDWBlock(nn.Module):
    """Reparameterizable multi-branch depthwise residual decoder block.

    Training graph:
        DW5+BN + DW3+BN + DW1+BN + Identity+BN -> GELU -> PW1+BN -> +x

    Deploy graph:
        DW5(bias=True) -> GELU -> PW1(bias=True) -> +x

    Call :meth:`eval` before :meth:`switch_to_deploy`.  The conversion is
    exact up to floating-point round-off because every fused branch is linear
    before the shared GELU activation.
    """

    def __init__(self, channels=64, deploy=False):
        super().__init__()
        self.channels = channels
        self.deploy = bool(deploy)

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                channels, channels, kernel_size=5, padding=2,
                groups=channels, bias=True,
            )
            self.pw_reparam = nn.Conv2d(
                channels, channels, kernel_size=1, bias=True,
            )
        else:
            self.rbr_5x5 = self._conv_bn(channels, kernel_size=5, padding=2)
            self.rbr_3x3 = self._conv_bn(channels, kernel_size=3, padding=1)
            self.rbr_1x1 = self._conv_bn(channels, kernel_size=1, padding=0)
            self.rbr_identity = nn.BatchNorm2d(channels)

            self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.pw_bn = nn.BatchNorm2d(channels)

            # Exact identity at initialization: the correction branch is zero.
            nn.init.zeros_(self.pw_bn.weight)
            nn.init.zeros_(self.pw_bn.bias)

        self.act = nn.GELU()

    @staticmethod
    def _conv_bn(channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=kernel_size, padding=padding,
                groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

    @staticmethod
    def _fuse_conv_bn(branch):
        """Fold a ``Conv2d + BatchNorm2d`` branch into kernel and bias."""
        if isinstance(branch, (tuple, list)):
            conv, bn = branch
        else:
            conv, bn = branch[0], branch[1]
        kernel = conv.weight
        if conv.bias is None:
            conv_bias = torch.zeros_like(bn.running_mean)
        else:
            conv_bias = conv.bias

        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
        fused_bias = bn.bias + (conv_bias - bn.running_mean) * scale
        return fused_kernel, fused_bias

    def _identity_kernel(self, bn):
        """Represent identity+BN as a depthwise 5x5 kernel and bias."""
        kernel = torch.zeros(
            self.channels, 1, 5, 5,
            device=bn.weight.device, dtype=bn.weight.dtype,
        )
        kernel[:, 0, 2, 2] = 1.0
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
        fused_bias = bn.bias - bn.running_mean * scale
        return fused_kernel, fused_bias

    @staticmethod
    def _pad_kernel(kernel, target_size=5):
        """Zero-pad an odd square depthwise kernel to ``target_size``."""
        size = kernel.size(-1)
        if size == target_size:
            return kernel
        if size > target_size or (target_size - size) % 2 != 0:
            raise ValueError(
                f'Cannot symmetrically pad {size}x{size} kernel to '
                f'{target_size}x{target_size}'
            )
        pad = (target_size - size) // 2
        return F.pad(kernel, [pad, pad, pad, pad])

    def get_equivalent_kernel_bias(self):
        """Return the fused depthwise 5x5 kernel and bias."""
        if self.deploy:
            return self.rbr_reparam.weight, self.rbr_reparam.bias

        k5, b5 = self._fuse_conv_bn(self.rbr_5x5)
        k3, b3 = self._fuse_conv_bn(self.rbr_3x3)
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1)
        kid, bid = self._identity_kernel(self.rbr_identity)

        kernel = (
            k5 + self._pad_kernel(k3) + self._pad_kernel(k1) + kid
        )
        bias = b5 + b3 + b1 + bid
        return kernel, bias

    def switch_to_deploy(self):
        """Fuse training branches in-place and return ``self``.

        BatchNorm running statistics are meaningful only in evaluation mode,
        so converting a training-mode block is rejected explicitly.
        """
        if self.deploy:
            return self
        if self.training:
            raise RuntimeError('Call eval() before switch_to_deploy()')

        dw_kernel, dw_bias = self.get_equivalent_kernel_bias()
        pw_kernel, pw_bias = self._fuse_conv_bn((self.pw, self.pw_bn))

        dw_reparam = nn.Conv2d(
            self.channels, self.channels, kernel_size=5, padding=2,
            groups=self.channels, bias=True,
        ).to(device=dw_kernel.device, dtype=dw_kernel.dtype)
        dw_reparam.weight.data.copy_(dw_kernel)
        dw_reparam.bias.data.copy_(dw_bias)
        self.rbr_reparam = dw_reparam

        pw_reparam = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=True,
        ).to(device=pw_kernel.device, dtype=pw_kernel.dtype)
        pw_reparam.weight.data.copy_(pw_kernel)
        pw_reparam.bias.data.copy_(pw_bias)
        self.pw_reparam = pw_reparam

        del self.rbr_5x5
        del self.rbr_3x3
        del self.rbr_1x1
        del self.rbr_identity
        del self.pw
        del self.pw_bn
        self.deploy = True
        return self

    def forward(self, x):
        if self.deploy:
            z = self.rbr_reparam(x)
            delta = self.pw_reparam(self.act(z))
        else:
            z = (
                self.rbr_5x5(x)
                + self.rbr_3x3(x)
                + self.rbr_1x1(x)
                + self.rbr_identity(x)
            )
            delta = self.pw_bn(self.pw(self.act(z)))
        return x + delta


def switch_repdw_to_deploy(module):
    """Convert every unique :class:`RepDWBlock` below ``module`` in-place."""
    if module.training:
        raise RuntimeError('Call model.eval() before switch_repdw_to_deploy()')
    converted = 0
    for child in list(module.modules()):
        if isinstance(child, RepDWBlock) and not child.deploy:
            child.switch_to_deploy()
            converted += 1
    return converted
