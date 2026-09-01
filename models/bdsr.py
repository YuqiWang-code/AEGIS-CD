"""
BDSR — Boundary-Decoupled Semantic Refinement (Run13).

Replaces EdgeGate with an explicit, supervised boundary branch:
    B    = sigmoid(H_b(F))            # boundary map  [B,1,H,W]
    D_b  = H_d(F - avgpool(F))        # high-frequency detail  [B,C,H,W]
    Fout = F + beta * D_b * B         # additive identity topology, beta init 0

Boundary label: Dilate(Y) - Erode(Y), downsampled to the feature resolution.
"""

import torch
import torch.nn as nn


class BDSR(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        C = channels
        self.boundary = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        self.detail = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 1, bias=False),
        )
        self.avg = nn.AvgPool2d(3, stride=1, padding=1)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, f):
        B = self.boundary(f)
        high_freq = f - self.avg(f)
        D_b = self.detail(high_freq)
        return f + self.beta * D_b * B, B
