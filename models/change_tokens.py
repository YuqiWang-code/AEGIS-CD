"""
TCT — Temporal Change Tokens (Run13).

Builds a small set of learnable change tokens from T1/T2 deep features, then
re-injects global semantic change relations into the spatial diff.  Applied
only to the deep scales (16x16, 8x8).

I/O contract identical to MSCA:
    Input:  f1 [B, C, H, W], f2 [B, C, H, W], diff [B, C, H, W]
    Output: refined diff [B, C, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TokenTransformerBlock(nn.Module):
    """One pre-norm self-attention/MLP block shared by T1 and T2."""

    def __init__(self, channels, num_heads):
        super().__init__()
        C = channels
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        self.norm1 = nn.LayerNorm(C)
        self.q = nn.Linear(C, C)
        self.k = nn.Linear(C, C)
        self.v = nn.Linear(C, C)
        self.out = nn.Linear(C, C)
        self.norm2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(
            nn.Linear(C, C * 4),
            nn.GELU(),
            nn.Linear(C * 4, C),
        )

    def forward(self, x):
        B, T, C = x.shape
        nh, hd = self.num_heads, self.head_dim
        y = self.norm1(x)
        q = self.q(y).reshape(B, T, nh, hd).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B, T, nh, hd).permute(0, 2, 1, 3)
        v = self.v(y).reshape(B, T, nh, hd).permute(0, 2, 1, 3)
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * (hd ** -0.5)
        attn = F.softmax(attn, dim=-1)
        y = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        y = y.permute(0, 2, 1, 3).reshape(B, T, C)
        x = x + self.out(y)
        return x + self.mlp(self.norm2(x))


class TCT(nn.Module):
    def __init__(self, channels=64, token_num=8, num_heads=4, depth=2,
                 eta_init=0.0):
        super().__init__()
        C = channels
        self.token_num = token_num
        self.num_heads = num_heads
        assert C % num_heads == 0
        self.head_dim = C // num_heads
        self.depth = depth

        self.token_projector = nn.Conv2d(C, token_num, 1, bias=False)

        # Two distinct depth blocks; the same stack is applied to T1 and T2,
        # which is the temporal weight-sharing required by the design.
        self.token_blocks = nn.ModuleList([
            _TokenTransformerBlock(C, num_heads) for _ in range(depth)
        ])

        # re-injection: Q from spatial diff, K/V from change tokens
        self.spatial_q = nn.Conv2d(C, C, 1, bias=False)
        self.token_k = nn.Linear(C, C)
        self.token_v = nn.Linear(C, C)
        self.eta = nn.Parameter(torch.tensor(eta_init))

    def _tokenize(self, f):
        B, C, H, W = f.shape
        N = H * W
        w = self.token_projector(f).reshape(B, self.token_num, N)
        w = F.softmax(w, dim=-1)
        f_flat = f.reshape(B, C, N)
        return torch.einsum('btn,bcn->btc', w, f_flat)  # [B, T, C]

    def _token_transformer(self, tokens):
        x = tokens
        for block in self.token_blocks:
            x = block(x)
        return x

    def forward(self, f1, f2, diff):
        B, C, H, W = diff.shape
        N = H * W
        nh, hd = self.num_heads, self.head_dim

        t1 = self._token_transformer(self._tokenize(f1))
        t2 = self._token_transformer(self._tokenize(f2))
        t_delta = torch.abs(t1 - t2)  # [B, T, C]

        q = self.spatial_q(diff).reshape(B, C, N).permute(0, 2, 1)  # [B, N, C]
        k = self.token_k(t_delta)  # [B, T, C]
        v = self.token_v(t_delta)  # [B, T, C]

        q = q.reshape(B, N, nh, hd).permute(0, 2, 1, 3)            # [B, nh, N, hd]
        k = k.reshape(B, self.token_num, nh, hd).permute(0, 2, 1, 3)  # [B, nh, T, hd]
        v = v.reshape(B, self.token_num, nh, hd).permute(0, 2, 1, 3)

        attn = torch.einsum('bhnc,bhtc->bhnt', q, k) * (hd ** -0.5)  # [B, nh, N, T]
        attn = F.softmax(attn, dim=-1)
        g = torch.einsum('bhnt,bhtc->bhnc', attn, v)                # [B, nh, N, hd]
        g = g.permute(0, 2, 1, 3).reshape(B, N, C)
        g = g.permute(0, 2, 1).reshape(B, C, H, W)

        return diff + self.eta * g
