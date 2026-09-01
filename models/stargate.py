"""
StarGate — Star-Operation Cross-Temporal Gating  (Phase 8)
============================================================
Drop-in replacement for MSCA in the difference-refinement path.

Design principle (CVPR 2024 "Rewrite the Stars", Ma et al.):
  The "Star Operation"  (W1^T x) * (W2^T x)  uses *element-wise
  multiplication* of two linear projections to reach an implicit
  high-dimensional feature space, at the cost of one extra element-wise
  multiply (no matrix multiply, no attention).  It captures *multiplicative*
  (AND-like) feature interactions that additive SE / SK attention cannot.

Why it fits change detection:
  A change pixel should fire only where (a) the temporal difference `diff` is
  strong AND (b) the cross-temporal context `ctx` is semantically meaningful.
  This is exactly a multiplicative gate:  out = diff + diff * f(diff ⊙ ctx).

I/O contract — identical to MSCA:
    Input:  context_f1 [B, C, H, W],  context_f2 [B, C, H, W],  diff [B, C, H, W]
    Output: refined diff [B, C, H, W]   (clean residual: out = diff + diff * gate)

Params per instance (~19.6K @ C=64), lighter than MSCA (~22K):
    ctx_fuse     128→64  1×1  = 8192
    branch_ctx   DW 5×5 (1600) + PW 1×1 (4096) = 5696
    branch_diff  DW 5×5 (1600) + PW 1×1 (4096) = 5696
    -------------------------------------------------------------------
    total ≈ 19.6K
"""

import torch
import torch.nn as nn


class StarGate(nn.Module):
    """Star-Operation cross-temporal multiplicative gating."""

    def __init__(self, channels=64):
        super(StarGate, self).__init__()
        C = channels

        # ---- 1. Cross-temporal context fusion (T1/T2 -> single ctx) ----
        self.ctx_fuse = nn.Conv2d(C * 2, C, 1, bias=False)

        # ---- 2. Two star branches: one sees context, one sees diff ----
        # Each is a depthwise (large receptive field) + pointwise linear map,
        # i.e. a factorised linear projection W^T x.
        self.branch_ctx = nn.Sequential(
            nn.Conv2d(C, C, 5, 1, 2, groups=C, bias=False),   # DW 5×5
            nn.BatchNorm2d(C),
            nn.ReLU6(inplace=True),
            nn.Conv2d(C, C, 1, bias=False),                    # PW 1×1
        )
        self.branch_diff = nn.Sequential(
            nn.Conv2d(C, C, 5, 1, 2, groups=C, bias=False),   # DW 5×5
            nn.BatchNorm2d(C),
            nn.ReLU6(inplace=True),
            nn.Conv2d(C, C, 1, bias=False),                    # PW 1×1
        )

        # ---- 3. Gate normalisation -> sigmoid ----
        self.gate = nn.Sequential(nn.BatchNorm2d(C), nn.Sigmoid())

    # ------------------------------------------------------------------
    def forward(self, context_f1, context_f2, diff):
        """Args/Returns mirror MSCA.forward."""
        # cross-temporal context (absolute semantics from T1 & T2)
        ctx = self.ctx_fuse(torch.cat([context_f1, context_f2], dim=1))

        # Star operation: element-wise multiply of two linear projections
        u = self.branch_ctx(ctx)          # [B, C, H, W]
        v = self.branch_diff(diff)        # [B, C, H, W]
        g = self.gate(u * v)              # multiplicative gate [B, C, H, W]

        # clean residual: identity `diff` is always preserved → stable gradient
        return diff + diff * g
