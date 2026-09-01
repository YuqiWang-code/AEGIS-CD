"""
RepHFEA-Pyramid — reparameterized encoder fusion (Run13).

Same 4-stage neighbour fusion topology as HFEA, but:
- stage output channels grow as [64, 80, 96, 128] to retain richer deep
  semantics (the difference stage then adapts them back to 64ch);
- each stage uses a RepDW multi-branch block (train: DW5+DW3+DW1+identity;
  deploy: single DW5) instead of a full 3x3 aggregation.

I/O contract identical to EncoderFusion:
    Input:  x1..x5 backbone features  (16, 24, 32, 96, 320 ch)
    Output: s1..s4  ([64, 80, 96, 128] ch at 64/32/16/8 resolution)
"""

import torch.nn as nn

from models.rep_decoder import RepDWBlock


class RepHFEA(nn.Module):
    OUT_CHANNELS = [64, 80, 96, 128]

    def __init__(self, inc=None):
        super().__init__()
        if inc is None:
            inc = [16, 24, 32, 96, 320]
        self.inc = inc
        out = self.OUT_CHANNELS

        # stage1 (64x64): x1(stride2) + x2 + x3(up2)
        self.s1_c1 = self._down(inc[0], out[0])
        self.s1_c2 = self._proj(inc[1], out[0])
        self.s1_c3 = self._up(inc[2], out[0])
        self.s1_block = RepDWBlock(out[0])

        # stage2 (32x32): x2(stride2) + x3 + x4(up2)
        self.s2_c1 = self._down(inc[1], out[1])
        self.s2_c2 = self._proj(inc[2], out[1])
        self.s2_c3 = self._up(inc[3], out[1])
        self.s2_block = RepDWBlock(out[1])

        # stage3 (16x16): x3(stride2) + x4 + x5(up2)
        self.s3_c1 = self._down(inc[2], out[2])
        self.s3_c2 = self._proj(inc[3], out[2])
        self.s3_c3 = self._up(inc[4], out[2])
        self.s3_block = RepDWBlock(out[2])

        # stage4 (8x8): x4(stride2) + x5
        self.s4_c1 = self._down(inc[3], out[3])
        self.s4_c2 = self._proj(inc[4], out[3])
        self.s4_block = RepDWBlock(out[3])

    @staticmethod
    def _down(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _proj(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _up(cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(cin, cout, 1, stride=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2, x3, x4, x5):
        s1 = self.s1_c1(x1) + self.s1_c2(x2) + self.s1_c3(x3)
        s1 = self.s1_block(s1)

        s2 = self.s2_c1(x2) + self.s2_c2(x3) + self.s2_c3(x4)
        s2 = self.s2_block(s2)

        s3 = self.s3_c1(x3) + self.s3_c2(x4) + self.s3_c3(x5)
        s3 = self.s3_block(s3)

        s4 = self.s4_c1(x4) + self.s4_c2(x5)
        s4 = self.s4_block(s4)

        return s1, s2, s3, s4
