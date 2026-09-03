"""
SDTR — Scale-Decoupled Temporal Relation.

Shallow:
    bidirectional local cosine correspondence.

Deep:
    bidirectional lightweight cross-attention with two topologies:

    - replace:
        historical Run13 behaviour; temporal relation directly produces
        the difference representation.

    - residual:
        Run14 behaviour; EAOM remains the base difference and temporal
        relation contributes a zero-initialised residual correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SDTR(nn.Module):
    VALID_MODES = ('auto', 'shallow', 'deep')
    VALID_DEEP_RELATION_MODES = ('replace', 'residual')

    def __init__(self, channels=64, window_shallow_hi=5, window_shallow_lo=3,
                num_heads=4, beta_init=0.0, mode='auto', window=None,
                temperature=0.2, deep_relation_mode='replace'):
        super().__init__()
        C = channels
        if mode not in self.VALID_MODES:
            raise ValueError(f'Unknown SDTR mode={mode!r}; expected {self.VALID_MODES}')
        if window is not None and (window < 1 or window % 2 != 1):
            raise ValueError('SDTR local window must be a positive odd integer')
        if temperature <= 0:
            raise ValueError('SDTR local-correlation temperature must be > 0')
        if deep_relation_mode not in self.VALID_DEEP_RELATION_MODES:
            raise ValueError(
                f'Unknown deep_relation_mode={deep_relation_mode!r}; '
                f'expected one of {self.VALID_DEEP_RELATION_MODES}'
            )
        if mode == 'shallow' and deep_relation_mode != 'replace':
            raise ValueError(
                'deep_relation_mode applies only to mode="deep" or mode="auto"'
            )
        self.mode = mode
        self.window = window
        self.window_hi = window_shallow_hi
        self.window_lo = window_shallow_lo
        self.num_heads = num_heads
        self.temperature = float(temperature)
        self.deep_relation_mode = deep_relation_mode

        if mode in ('auto', 'shallow'):
            # One shared metric projection makes cosine correspondence
            # temporally symmetric.  Dirac initialization starts from the
            # original feature metric rather than two unrelated random maps.
            self.metric_proj = nn.Conv2d(C, C, 1, bias=False)
            nn.init.dirac_(self.metric_proj.weight)
            self.phi = nn.Sequential(
                nn.Conv2d(2 * C + 1, C, 1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            )
            self.beta_shallow = nn.Parameter(torch.tensor(beta_init))

        if mode in ('auto', 'deep'):
            # ---- deep: lightweight bidirectional cross-attention ----
            if C % num_heads != 0:
                raise ValueError('channels must be divisible by num_heads')

            self.head_dim = C // num_heads

            self.wq = nn.Conv2d(C, C, 1, bias=False)
            self.wk = nn.Conv2d(C, C, 1, bias=False)
            self.wv = nn.Conv2d(C, C, 1, bias=False)

            self.out_proj = nn.Sequential(
                nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
                nn.Conv2d(C, C, 1, bias=False),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            )

            # Run14: optional residual temporal-relation topology.
            #
            # replace:
            #     identical to the historical Run13 deep SDTR.
            #
            # residual:
            #     keep EAOM as the reliable base difference and let deep temporal
            #     relation learn only a zero-initialised residual correction.
            if self.deep_relation_mode == 'residual':
                from models.eaom import EAOM

                self.deep_anchor = EAOM(C)
                self.alpha_deep = nn.Parameter(torch.tensor(0.0))

    def _window(self, f):
        if self.window is not None:
            return self.window
        return self.window_hi if f.shape[-1] >= 64 else self.window_lo

    def _local_align(self, q, k, v, window):
        """Memory-efficient local cosine alignment.

        Only ``[B,K,H,W]`` correlation/attention maps are stacked.  Feature
        shifts are accumulated one offset at a time, avoiding the former
        ``[B,C,K,H,W]`` unfold tensors that made batch-32 BAIC unsafe.
        """
        B, C, H, W = q.shape
        pad = window // 2
        offsets = [
            (dy, dx)
            for dy in range(-pad, pad + 1)
            for dx in range(-pad, pad + 1)
        ]

        q_norm = F.normalize(q, p=2, dim=1, eps=1e-6)
        k_norm = F.normalize(k, p=2, dim=1, eps=1e-6)
        k_padded = F.pad(k_norm, (pad, pad, pad, pad))
        valid_padded = F.pad(
            q.new_ones(1, 1, H, W), (pad, pad, pad, pad)
        )

        correlations = []
        for dy, dx in offsets:
            top = pad + dy
            left = pad + dx
            shifted_k = k_padded[:, :, top:top + H, left:left + W]
            valid = valid_padded[:, :, top:top + H, left:left + W].bool()
            corr = (q_norm * shifted_k).sum(dim=1, keepdim=True)
            corr = corr.masked_fill(~valid, torch.finfo(corr.dtype).min)
            correlations.append(corr)

        corr = torch.cat(correlations, dim=1)              # [B, K, H, W]
        corr_max = corr.max(dim=1, keepdim=True).values    # [B, 1, H, W]
        attn = F.softmax(corr / self.temperature, dim=1)

        v_padded = F.pad(v, (pad, pad, pad, pad))
        v_hat = torch.zeros_like(v)
        for index, (dy, dx) in enumerate(offsets):
            top = pad + dy
            left = pad + dx
            shifted_v = v_padded[:, :, top:top + H, left:left + W]
            v_hat = v_hat + attn[:, index:index + 1] * shifted_v
        return v_hat, corr_max

    def _shallow(self, f1, f2):
        p1 = self.metric_proj(f1)
        p2 = self.metric_proj(f2)

        f2_hat, c12 = self._local_align(p1, p2, f2, self._window(f1))
        f1_hat, c21 = self._local_align(p2, p1, f1, self._window(f2))

        D = 0.5 * (torch.abs(f1 - f2_hat) + torch.abs(f2 - f1_hat))
        P = 0.5 * (f1 * f2_hat + f2 * f1_hat)
        C = 0.5 * (c12 + c21)

        R = self.phi(torch.cat([D, P, C], dim=1))
        return D + self.beta_shallow * R

    def _cross_attend(self, f_a, f_b):
        """R = Attn(Q(f_a), K(f_b), V(f_b))."""
        B, C, H, W = f_a.shape
        N = H * W
        nh = self.num_heads
        hd = self.head_dim

        q = self.wq(f_a).reshape(B, nh, hd, N)
        k = self.wk(f_b).reshape(B, nh, hd, N)
        v = self.wv(f_b).reshape(B, nh, hd, N)

        attn = torch.einsum('bhnx,bhny->bhxy', q, k) * (hd ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhxy,bhny->bhnx', attn, v)   # [B, nh, hd, N]
        out = out.reshape(B, C, H, W)
        return out

    def _deep(self, f1, f2):
        # Bidirectional temporal relation.
        r12 = self._cross_attend(f1, f2)
        r21 = self._cross_attend(f2, f1)

        # Historical Run13 deep relation difference.
        d_relation = 0.5 * (
            torch.abs(f1 - r12)
            + torch.abs(f2 - r21)
        )

        relation_delta = self.out_proj(d_relation)

        # Historical Run13 behaviour: relation replaces the original
        # difference representation.
        if self.deep_relation_mode == 'replace':
            return relation_delta

        # Run14 residual behaviour:
        # keep EAOM as the base representation and use temporal relation
        # only as a learnable residual correction.
        base_diff = self.deep_anchor(f1, f2)

        return (
            base_diff
            + torch.tanh(self.alpha_deep) * relation_delta
        )

    def forward(self, f1, f2):
        if f1.shape != f2.shape:
            raise ValueError(
                f'SDTR requires matching feature shapes, got '
                f'{tuple(f1.shape)} and {tuple(f2.shape)}'
            )
        if self.mode == 'shallow':
            return self._shallow(f1, f2)
        if self.mode == 'deep':
            return self._deep(f1, f2)
        if f1.shape[-1] >= 32:
            return self._shallow(f1, f2)
        return self._deep(f1, f2)
