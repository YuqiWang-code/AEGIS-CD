"""AEGIS-CD change detector.

Siamese MobileNetV2 + HFEA + EAOM difference encoder + RepDW decoder + EdgeGate,
with scale-decoupled independent heads and native (SCDS) deep supervision.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import mobilenet_v2
from einops import rearrange
import numbers
from pytorch_wavelets import DWTForward, DWTInverse


def validate_checkpoint_head_mode(state_dict, head_mode):
    """Fail-fast when a checkpoint's prediction-head mode mismatches the CLI.

    A checkpoint with ``decoder_out2/3/4.*`` keys was trained with
    ``head_mode='independent'``; a checkpoint without them was trained with
    ``head_mode='shared'``.  Structure mismatches must fail loudly (never be
    silently swallowed with ``strict=False``) because head mode changes the
    checkpoint's key set.
    """
    has_independent = any(
        k.startswith(('decoder_out2.', 'decoder_out3.', 'decoder_out4.'))
        for k in state_dict
    )
    expected = (head_mode == 'independent')
    if has_independent != expected:
        ckpt_mode = 'independent' if has_independent else 'shared'
        raise ValueError(
            f'Checkpoint uses head_mode={ckpt_mode}, '
            f'but CLI requests head_mode={head_mode}.'
        )


# -------------------------------------------------------HFEA----------------------------------------------------------#
class EncoderFusion(nn.Module):
    def __init__(self, inc, midc=32, outc=64):
        super(EncoderFusion, self).__init__()

        if inc is None:
            inc = [16, 24, 32, 96, 320]
        self.inc = inc
        self.midc = midc
        self.outc = outc
        self.fusec = [midc * 3, midc * 3, midc * 3, midc * 2]


        # stage 1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.inc[0], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.inc[1], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.inc[2], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # stage 2
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(self.inc[1], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(self.inc[2], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(self.inc[3], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # stage 3
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(self.inc[2], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(self.inc[3], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(self.inc[4], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        # stage 4
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(self.inc[3], self.midc, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(self.inc[4], self.midc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.midc),
            nn.ReLU(inplace=True)
        )

        # aggregation
        self.aggregation_s1 = AggregationModule(self.fusec[0], self.inc[1], self.outc)
        self.aggregation_s2 = AggregationModule(self.fusec[1], self.inc[2], self.outc)
        self.aggregation_s3 = AggregationModule(self.fusec[2], self.inc[3], self.outc)
        self.aggregation_s4 = AggregationModule(self.fusec[3], self.inc[4], self.outc)

    def forward(self, x1, x2, x3, x4, x5):
        s1_f1 = self.conv1_1(x1)
        s1_f2 = self.conv1_2(x2)
        s1_f3 = self.conv1_3(x3)
        s1 = self.aggregation_s1(torch.cat([s1_f1, s1_f2, s1_f3], dim=1), x2)

        s2_f1 = self.conv2_1(x2)
        s2_f2 = self.conv2_2(x3)
        s2_f3 = self.conv2_3(x4)
        s2 = self.aggregation_s2(torch.cat([s2_f1, s2_f2, s2_f3], dim=1), x3)

        s3_f1 = self.conv3_1(x3)
        s3_f2 = self.conv3_2(x4)
        s3_f3 = self.conv3_3(x5)
        s3 = self.aggregation_s3(torch.cat([s3_f1, s3_f2, s3_f3], dim=1), x4)

        s4_f1 = self.conv4_1(x4)
        s4_f2 = self.conv4_2(x5)
        s4 = self.aggregation_s4(torch.cat([s4_f1, s4_f2], dim=1), x5)
        return s1, s2, s3, s4


# feature aggregation
class AggregationModule(nn.Module):
    def __init__(self, fusec, inc, outc):
        super(AggregationModule, self).__init__()
        self.fusec = fusec
        self.inc = inc
        self.outc = outc

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fusec, self.outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.outc, self.outc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.outc)
        )

        self.conv_identity = nn.Conv2d(self.inc, self.outc, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))
        return c_out


# -------------------------------------------------------MFFM----------------------------------------------------------#
class DecoderFusion(nn.Module):
    """Top-down multi-scale decoder (MSA or RepDW)."""

    VALID_DECODER_MODES = ('msa', 'rep_dw')

    def __init__(self, channels, decoder_mode='rep_dw'):
        super(DecoderFusion, self).__init__()
        if decoder_mode not in self.VALID_DECODER_MODES:
            raise ValueError(
                f'Unknown decoder_mode={decoder_mode!r}; '
                f'expected one of {self.VALID_DECODER_MODES}'
            )

        self.decoder_mode = decoder_mode

        if decoder_mode == 'rep_dw':
            from models.rep_decoder import RepDWBlock
            self.rep4 = RepDWBlock(channels)
            self.rep3 = RepDWBlock(channels)
            self.rep2 = RepDWBlock(channels)
            self.rep1 = RepDWBlock(channels)
        else:
            # One shared MSA instance across all four decoder scales.
            self.module = MSA_Module(channels)

    def forward(self, f1, f2, f3, f4):
        if self.decoder_mode == 'rep_dw':
            f4 = self.rep4(f4)
            f4_up = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')

            f3 = self.rep3(f3 + f4_up)
            f3_up = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')

            f2 = self.rep2(f2 + f3_up)
            f2_up = F.interpolate(f2, scale_factor=(2, 2), mode='bilinear')

            f1 = self.rep1(f1 + f2_up)
            return f1, f2, f3, f4
        else:
            f4 = self.module(f4)
            f4_up = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')

            f3 = f3 + f4_up
            f3 = self.module(f3)
            f3_up = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')

            f2 = f2 + f3_up
            f2 = self.module(f2)
            f2_up = F.interpolate(f2, scale_factor=(2, 2), mode='bilinear')

            f1 = f1 + f2_up
            f1 = self.module(f1)
            return f1, f2, f3, f4


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, channels):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(channels)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class FeedForward(nn.Module):
    def __init__(self, channels, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden = int(channels * ffn_expansion_factor)

        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    def __init__(self, channels, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.qkv2conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.qkv3conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)

        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape

        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))

        if mask is not None:
            q = q * mask
            k = k * mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return self.project_out(out)


class MSA_Head(nn.Module):
    def __init__(self, channels=64, num_heads=4, ffn_expansion_factor=4, bias=False):
        super(MSA_Head, self).__init__()
        self.norm1 = LayerNorm(channels)
        self.attn = Attention(channels, num_heads, bias)
        self.norm2 = LayerNorm(channels)
        self.ffn = FeedForward(channels, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        return x + self.ffn(self.norm2(x))


class MSA_Module(nn.Module):
    """Shared decoder MSA (foreground/background attention)."""

    def __init__(self, channels=64):
        super(MSA_Module, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.background = MSA_Head(channels)
        self.foreground = MSA_Head(channels)

        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip = x
        mask = self.conv(x)
        # Legacy control: the projection is frozen, while its input still
        # changes as upstream layers learn.
        mask = mask.detach()
        mask = torch.sigmoid(mask)
        xf = self.foreground(x, mask)      # Foreground
        xb = self.background(x, 1 - mask)  # Background
        x_fused = self.fuse(torch.cat([xb, xf], dim=1))

        return self.out(skip * x_fused)


# ----------------------------------------------------AEGIS-CD--------------------------------------------------------#
class BaseNet(nn.Module):
    VALID_HEAD_MODES = ('shared', 'independent')
    VALID_DIFF_MODES = ('eaom', 'none')
    VALID_DIFF_SHARING = ('shared', 'independent')
    VALID_SUPERVISION_MODES = ('legacy', 'native')
    VALID_BOUNDARY_MODES = ('off', 'edgegate')

    def __init__(self, diff_mode='eaom', diff_sharing='independent',
                 supervision_mode='native', decoder_mode='rep_dw',
                 head_mode='independent', boundary_mode='edgegate'):
        super(BaseNet, self).__init__()
        if decoder_mode not in DecoderFusion.VALID_DECODER_MODES:
            raise ValueError(
                f'Unknown decoder_mode={decoder_mode!r}; '
                f'expected one of {DecoderFusion.VALID_DECODER_MODES}'
            )
        if head_mode not in self.VALID_HEAD_MODES:
            raise ValueError(
                f'Unknown head_mode={head_mode!r}; '
                f'expected one of {self.VALID_HEAD_MODES}'
            )
        if diff_mode not in self.VALID_DIFF_MODES:
            raise ValueError(
                f'Unknown diff_mode={diff_mode!r}; '
                f'expected one of {self.VALID_DIFF_MODES}'
            )
        if diff_sharing not in self.VALID_DIFF_SHARING:
            raise ValueError(
                f'Unknown diff_sharing={diff_sharing!r}; '
                f'expected one of {self.VALID_DIFF_SHARING}'
            )
        if supervision_mode not in self.VALID_SUPERVISION_MODES:
            raise ValueError(
                f'Unknown supervision_mode={supervision_mode!r}; '
                f'expected one of {self.VALID_SUPERVISION_MODES}'
            )
        if boundary_mode not in self.VALID_BOUNDARY_MODES:
            raise ValueError(
                f'Unknown boundary_mode={boundary_mode!r}; '
                f'expected one of {self.VALID_BOUNDARY_MODES}'
            )

        self.diff_mode = diff_mode
        self.diff_sharing = diff_sharing
        self.supervision_mode = supervision_mode
        self.decoder_mode = decoder_mode
        self.head_mode = head_mode
        self.boundary_mode = boundary_mode
        self.use_edgegate = boundary_mode == 'edgegate'

        self.encoder = mobilenet_v2.mobilenet_v2(pretrained=True)
        self.encoder_fusion = EncoderFusion(
            inc=[16, 24, 32, 96, 320], midc=32, outc=64
        )

        def make_diff():
            from models.eaom import EAOM
            return EAOM(64)

        if diff_mode == 'none':
            # Pure |diff| baseline: no difference module at any scale.
            self.diff = None
            self.diff1 = self.diff2 = self.diff3 = self.diff4 = None
        elif diff_sharing == 'shared':
            self.diff = make_diff()
        else:
            self.diff1 = make_diff()
            self.diff2 = make_diff()
            self.diff3 = make_diff()
            self.diff4 = make_diff()

        if self.use_edgegate:
            from models.edgegate import EdgeGate
            self.edgegate = EdgeGate(64)

        self.decoder_fusion = DecoderFusion(64, decoder_mode=decoder_mode)

        # Primary prediction head (scale 0 / 64x64).
        self.decoder_out = nn.Conv2d(64, 1, kernel_size=1)

        # Scale-decoupled heads: independent mode deep-copies the primary head
        # so shared/independent are functionally identical at epoch 0.
        if self.head_mode == 'independent':
            self.decoder_out2 = copy.deepcopy(self.decoder_out)
            self.decoder_out3 = copy.deepcopy(self.decoder_out)
            self.decoder_out4 = copy.deepcopy(self.decoder_out)

    def switch_to_deploy(self):
        """Fuse every RepDW training block in-place for deployment."""
        from models.rep_decoder import switch_repdw_to_deploy
        return switch_repdw_to_deploy(self)

    def _compute_differences(self, f1, f2):
        if self.diff_mode == 'none':
            return tuple(torch.abs(a - b) for a, b in zip(f1, f2))
        if self.diff_sharing == 'shared':
            return tuple(self.diff(a, b) for a, b in zip(f1, f2))
        modules = (self.diff1, self.diff2, self.diff3, self.diff4)
        return tuple(module(a, b) for module, a, b in zip(modules, f1, f2))

    def forward(self, x1, x2):
        # feature extraction
        x1_0, x1_1, x1_2, x1_3, x1_4 = self.encoder(x1)
        x2_0, x2_1, x2_2, x2_3, x2_4 = self.encoder(x2)

        # feature enhancement
        f1 = self.encoder_fusion(x1_0, x1_1, x1_2, x1_3, x1_4)
        f2 = self.encoder_fusion(x2_0, x2_1, x2_2, x2_3, x2_4)

        # feature difference
        diff1, diff2, diff3, diff4 = self._compute_differences(f1, f2)

        # feature fusion
        f1, f2, f3, f4 = self.decoder_fusion(diff1, diff2, diff3, diff4)

        # EdgeGate: boundary refinement on the primary head feature f1 only
        if self.use_edgegate:
            f1 = self.edgegate(f1)

        # Prediction heads (f1 uses the primary head)
        f1 = self.decoder_out(f1)
        if self.head_mode == 'independent':
            f2 = self.decoder_out2(f2)
            f3 = self.decoder_out3(f3)
            f4 = self.decoder_out4(f4)
        else:
            f2 = self.decoder_out(f2)
            f3 = self.decoder_out(f3)
            f4 = self.decoder_out(f4)

        # The primary output is always full-resolution.  In native SCDS mode,
        # auxiliary predictions stay at 32/16/8 so their area-averaged targets
        # match the semantic resolution that produced them.
        f1_up = F.interpolate(
            f1, scale_factor=(4, 4), mode='bilinear', align_corners=False
        )
        f1_up = torch.sigmoid(f1_up)

        if self.supervision_mode == 'native':
            outputs = (
                f1_up,
                torch.sigmoid(f2),
                torch.sigmoid(f3),
                torch.sigmoid(f4),
            )
        else:
            outputs = (
                f1_up,
                torch.sigmoid(F.interpolate(
                    f2, scale_factor=(8, 8), mode='bilinear',
                    align_corners=False,
                )),
                torch.sigmoid(F.interpolate(
                    f3, scale_factor=(16, 16), mode='bilinear',
                    align_corners=False,
                )),
                torch.sigmoid(F.interpolate(
                    f4, scale_factor=(32, 32), mode='bilinear',
                    align_corners=False,
                )),
            )

        return outputs
