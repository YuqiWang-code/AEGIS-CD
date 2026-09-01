"""
EGBR — Edge-Guided Boundary Refinement  (Phase 9)
===================================================
Lightweight boundary-sharpening module inserted at the decoder *head*.

Motivation (SFEARNet, IEEE 2025 — Semantic Flow + Edge-Aware Refinement):
  Change-detection F1/IoU losses concentrate on the *boundary* of change
  regions.  The upstream modules (EAOM / MSCA / SFIF) operate on the 64-ch
  diff / decoder features holistically, but nothing explicitly sharpens the
  change-region boundary.  EGBR injects boundary detail at the final head,
  which is orthogonal to the diff-refinement chain — so it cannot suffer the
  chain-stacking negative interaction observed in Run6/Run7.

Design — additive residual detail injection (NOT a multiplicative gate):
    edge   = sigmoid( DWConv(BN(ReLU(f1))) -> 1x1(C->1) )   # boundary map
    detail = DWConv(BN(ReLU(f1))) -> 1x1(C->C)              # boundary detail
    out    = f1 + scale * detail * edge                     # residual injection

Key properties:
  1. **Additive** — `out = f1 + ...`, identity preserved.  Unlike StarGate's
     multiplicative `diff + diff*g`, an additive boundary-detail injection
     cannot zero out the feature (Run7 showed multiplicative gating hurts).
  2. **Warm-start identity** — `scale` is initialised to 0, so the module is
     an exact identity at epoch 0 and only *learns* to add boundary detail.
  3. **Orthogonal position** — applied only to the primary head feature f1
     (64×64), after DecoderFusion, before the final 1×1 conv.  Deep-supervision
     aux heads f2/f3/f4 are untouched.
  4. **Gradient flow** — d(out)/d(f1) = 1 + scale * edge * detail'(f1) ≥ 1 on
     the identity path, so gradients never vanish through the module.

Params ~5.3K (single instance, on f1): edge_conv (576 + 64) + detail (576 + 4096).

I/O contract:
    Input:  f1 [B, C, H, W]   (DecoderFusion's highest-resolution feature)
    Output: f1 [B, C, H, W]   (boundary-refined)
"""

import torch
import torch.nn as nn


class EdgeGate(nn.Module):
    """Edge-Guided Boundary Refinement (EGBR) — head-level boundary sharpening."""

    def __init__(self, channels=64):
        super(EdgeGate, self).__init__()
        C = channels

        # ---- boundary map: f1 -> [B, 1, H, W] in (0, 1) ----
        self.edge_conv = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),   # DW 3×3
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 1, 1, bias=False),                   # C -> 1
            nn.Sigmoid(),
        )

        # ---- boundary detail extractor: f1 -> [B, C, H, W] ----
        self.detail = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),   # DW 3×3
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1, bias=False),                   # PW 1×1
        )

        # learnable residual strength, init 0 → identity warm start
        self.scale = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def forward(self, f1):
        """Inject boundary detail into f1. Identity when scale == 0."""
        edge = self.edge_conv(f1)                 # [B, 1, H, W]
        detail = self.detail(f1)                  # [B, C, H, W]
        return f1 + self.scale * detail * edge    # additive residual
