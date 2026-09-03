"""AEGIS-CD multi-phase change detector, including the Run13 architecture.

Historical switches remain checkpoint-compatible.  Run13 adds explicit modes
for scale-specific temporal relation, native deep supervision, APID-LF,
LFDS/TCT, RepHFEA-Pyramid, BDSR, and BAIC training consistency.
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
    silently swallowed with ``strict=False``) because head mode is a Run11
    experimental variable.
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


# -------------------------------------------------------WFAM----------------------------------------------------------#
class AlignedModule(nn.Module):
    def __init__(self, channels):
        super(AlignedModule, self).__init__()

        self.wavelet = WaveletAttention(channels)
        self.offset_conv = nn.Conv2d(channels * 2, 2, kernel_size=3, stride=1, padding=1)

    def flow_warp(self, x, flow):
        n, c, h, w = x.size()
        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(x, grid, align_corners=True)
        return output

    def forward(self, x, y):
        x = self.wavelet(x)
        y = self.wavelet(y)
        cat = torch.cat([x, y], 1)
        offset = self.offset_conv(cat)
        warp_y = self.flow_warp(y, offset)
        return x, warp_y


class ConvModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class WaveletAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWTForward(J=1, mode='zero', wave='haar')
        self.idwt = DWTInverse(wave='haar')

        self.high_HL = ConvModule(channels)
        self.high_LH = ConvModule(channels)
        self.high_HH = ConvModule(channels)
        self.LL_attn = LLAttention(channels)

        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)
        self.fc = nn.Linear(channels, channels, bias=True)

    def forward(self, x):
        x_L, x_H = self.dwt(x)
        x_HL = x_H[0][:, :, 0, :, :]
        x_LH = x_H[0][:, :, 1, :, :]
        x_HH = x_H[0][:, :, 2, :, :]

        x_HL_en = self.high_HL(x_HL)
        x_LH_en = self.high_LH(x_LH)
        x_HH_en = self.high_HH(x_HH)

        x_H_en = torch.stack([x_HL_en, x_LH_en, x_HH_en], dim=2)
        x_L_en = self.LL_attn(x_L)

        x_re = self.idwt((x_L_en, [x_H_en]))
        out = x_re + x
        return out


class LLAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)

        q = self.q(x_flat).reshape(B, H * W, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)

        kv = self.kv(x_flat).reshape(B, H * W, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


# ------------------------------------------------------CFDM-----------------------------------------------------------#
class DiffModule(nn.Module):
    def __init__(self, channels):
        super(DiffModule, self).__init__()

        self.align = AlignedModule(channels)
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1)
        self.attention = SC_Attention(channels)
        self.cbr = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))

    def forward(self, x, y):
        x, y = self.align(x, y)
        diff = torch.abs(x - y)
        con = self.conv(torch.cat([x, y], dim=1))
        attn = self.attention(diff, con)
        diff_en = diff * attn
        con_en = con * attn
        output = torch.cat([diff_en, con_en], dim=1)
        output = self.cbr(output)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv3 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(2, 1, kernel_size=5, padding=2, bias=False)
        self.conv7 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, max], dim=1)

        attn3 = self.conv3(cat)
        attn5 = self.conv5(cat)
        attn7 = self.conv7(cat)

        attn = attn3 + attn5 + attn7
        attn = self.sigmoid(attn)
        return attn


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_mlp(self.avg_pool(x))
        maxout = self.shared_mlp(self.max_pool(x))
        attn = self.sigmoid(avgout + maxout)
        return attn


class SC_Attention(nn.Module):
    def __init__(self, channels):
        super(SC_Attention, self).__init__()
        self.spatial = SpatialAttention(channels)
        self.channel = ChannelAttention(channels)

    def forward(self, diff, con):
        attn1 = self.spatial(diff)
        attn2 = self.channel(con)
        attn = attn1 * attn2
        return attn


# -------------------------------------------------------MFFM----------------------------------------------------------#
class DecoderFusion(nn.Module):
    """Top-down multi-scale decoder with legacy and Phase 11 modes."""

    VALID_DECODER_MODES = ('msa', 'plain_dw', 'rep_dw', 'rep_dw_shared')

    def __init__(self, channels, use_sfif=False, grmsa_mode='off',
                 decoder_mode='msa', scale_fusion_mode='plain'):
        super(DecoderFusion, self).__init__()
        if decoder_mode not in self.VALID_DECODER_MODES:
            raise ValueError(
                f'Unknown decoder_mode={decoder_mode!r}; '
                f'expected one of {self.VALID_DECODER_MODES}'
            )
        if grmsa_mode not in MSA_Module.VALID_MODES:
            raise ValueError(
                f'Unknown grmsa_mode={grmsa_mode!r}; '
                f'expected one of {MSA_Module.VALID_MODES}'
            )
        if use_sfif and grmsa_mode != 'off':
            raise ValueError('SFIF and GR-MSA are alternative DecoderFusion implementations')
        if use_sfif and decoder_mode != 'msa':
            raise ValueError('SFIF and --decoder-mode are alternative decoder implementations')
        if decoder_mode != 'msa' and grmsa_mode != 'off':
            raise ValueError('GR-MSA modes apply only when decoder_mode="msa"')

        # Run12: SCRF currently only applies to rep_dw (fail-fast for invalid combinations)
        if scale_fusion_mode == 'scrf' and decoder_mode != 'rep_dw':
            raise ValueError(
                f'SCRF (scale_fusion_mode="scrf") only applies to decoder_mode="rep_dw"; '
                f'got decoder_mode={decoder_mode!r}. '
                f'Use scale_fusion_mode="plain" for other decoder modes.'
            )

        self.use_sfif = use_sfif
        self.grmsa_mode = grmsa_mode
        self.decoder_mode = decoder_mode
        self.scale_fusion_mode = scale_fusion_mode

        # Run12: Scale-Calibrated Residual Fusion (SCRF)
        from models.scale_fusion import get_scale_fusion
        self.scrf = get_scale_fusion(scale_fusion_mode)  # None if 'plain', SCRF instance if 'scrf'

        if use_sfif:
            from models.sfif import SFIF
            self.sfif4 = SFIF(channels)
            self.sfif3 = SFIF(channels)
            self.sfif2 = SFIF(channels)
            self.sfif1 = SFIF(channels)
        elif decoder_mode == 'plain_dw':
            from models.rep_decoder import PlainDWBlock
            self.plain4 = PlainDWBlock(channels)
            self.plain3 = PlainDWBlock(channels)
            self.plain2 = PlainDWBlock(channels)
            self.plain1 = PlainDWBlock(channels)
        elif decoder_mode == 'rep_dw':
            from models.rep_decoder import RepDWBlock
            self.rep4 = RepDWBlock(channels)
            self.rep3 = RepDWBlock(channels)
            self.rep2 = RepDWBlock(channels)
            self.rep1 = RepDWBlock(channels)
        elif decoder_mode == 'rep_dw_shared':
            from models.rep_decoder import RepDWBlock
            self.rep_shared = RepDWBlock(channels)
        else:
            # One shared MSA instance is intentionally retained across all four
            # decoder scales for Run9/checkpoint compatibility.
            self.module = MSA_Module(channels, grmsa_mode=grmsa_mode)

    def forward(self, f1, f2, f3, f4):
        if self.use_sfif:
            f4 = self.sfif4(f4)
            f4_up = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')

            f3 = f3 + f4_up
            f3 = self.sfif3(f3)
            f3_up = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')

            f2 = f2 + f3_up
            f2 = self.sfif2(f2)
            f2_up = F.interpolate(f2, scale_factor=(2, 2), mode='bilinear')

            f1 = f1 + f2_up
            f1 = self.sfif1(f1)
            return f1, f2, f3, f4
        elif self.decoder_mode == 'plain_dw':
            f4 = self.plain4(f4)
            f4_up = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')

            f3 = self.plain3(f3 + f4_up)
            f3_up = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')

            f2 = self.plain2(f2 + f3_up)
            f2_up = F.interpolate(f2, scale_factor=(2, 2), mode='bilinear')

            f1 = self.plain1(f1 + f2_up)
            return f1, f2, f3, f4
        elif self.decoder_mode == 'rep_dw':
            f4 = self.rep4(f4)
            f4_up = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')

            # Run12: SCRF replaces fixed `+` with learnable scalar modulation
            if self.scrf is not None:
                f3 = self.rep3(self.scrf(f3, f4_up, scale_idx=0))
            else:
                f3 = self.rep3(f3 + f4_up)
            f3_up = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')

            if self.scrf is not None:
                f2 = self.rep2(self.scrf(f2, f3_up, scale_idx=1))
            else:
                f2 = self.rep2(f2 + f3_up)
            f2_up = F.interpolate(f2, scale_factor=(2, 2), mode='bilinear')

            if self.scrf is not None:
                f1 = self.rep1(self.scrf(f1, f2_up, scale_idx=2))
            else:
                f1 = self.rep1(f1 + f2_up)
            return f1, f2, f3, f4
        elif self.decoder_mode == 'rep_dw_shared':
            f4 = self.rep_shared(f4)
            f4_up = F.interpolate(f4, scale_factor=(2, 2), mode='bilinear')

            f3 = self.rep_shared(f3 + f4_up)
            f3_up = F.interpolate(f3, scale_factor=(2, 2), mode='bilinear')

            f2 = self.rep_shared(f2 + f3_up)
            f2_up = F.interpolate(f2, scale_factor=(2, 2), mode='bilinear')

            f1 = self.rep_shared(f1 + f2_up)
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
    """Shared decoder MSA with the orthogonal Run9 ablation modes.

    Modes form a 2x2 factorial design:
      off      : detached mask + legacy multiplicative output
      mask     : trainable mask + legacy multiplicative output
      residual : detached mask + additive identity output
      full     : trainable mask + additive identity output (GR-MSA)

    The residual path is precisely ``skip + scale * out(fuse([xb, xf]))``.
    ``scale`` is registered only for residual/full, so mask-only adds no
    parameters and residual/full add one shared scalar.
    """

    VALID_MODES = ('off', 'mask', 'residual', 'full')

    def __init__(self, channels=64, grmsa_mode='off'):
        super(MSA_Module, self).__init__()
        if grmsa_mode not in self.VALID_MODES:
            raise ValueError(
                f'Unknown grmsa_mode={grmsa_mode!r}; expected one of {self.VALID_MODES}'
            )

        self.grmsa_mode = grmsa_mode
        self.train_mask = grmsa_mode in ('mask', 'full')
        self.use_residual = grmsa_mode in ('residual', 'full')
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.background = MSA_Head(channels)
        self.foreground = MSA_Head(channels)

        self.fuse = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        if self.use_residual:
            self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        skip = x
        mask = self.conv(x)
        if not self.train_mask:
            # Legacy control: the projection is frozen, while its input still
            # changes as upstream layers learn; the resulting mask is not a
            # fixed image.
            mask = mask.detach()
        mask = torch.sigmoid(mask)
        xf = self.foreground(x, mask)      # Foreground
        xb = self.background(x, 1 - mask)  # Background
        x_fused = self.fuse(torch.cat([xb, xf], dim=1))

        if self.use_residual:
            delta = self.out(x_fused)
            return skip + self.scale * delta

        return self.out(skip * x_fused)


# ----------------------------------------------------AEGIS-CD--------------------------------------------------------#
class BaseNet(nn.Module):
    VALID_HEAD_MODES = ('shared', 'independent')
    VALID_DIFF_MODES = ('cfdm', 'eaom', 'sdtr')
    VALID_DIFF_SHARING = ('shared', 'independent')
    VALID_SDTR_SCOPES = ('all', 'shallow', 'deep')
    VALID_SUPERVISION_MODES = ('legacy', 'native')
    VALID_AMP_PHASE_MODES = ('off', 'lf_shared')
    VALID_ENCODER_FUSION_MODES = ('hfea', 'rephfea_pyr')
    VALID_BOUNDARY_MODES = ('off', 'edgegate', 'bdsr')
    VALID_CONSISTENCY_MODES = ('off', 'baic')

    def __init__(self, use_eaom=False, use_sfif=False, use_prior=False,
                 use_msca=False, use_stargate=False, use_edgegate=False,
                 grmsa_mode='off', decoder_mode='msa',
                 head_mode='shared', scale_fusion_mode='plain',
                 diff_mode=None, diff_sharing='shared',
                 supervision_mode='legacy', amp_phase_mode='off',
                 use_lfds=False, use_tct=False,
                 encoder_fusion_mode='hfea', boundary_mode=None,
                 consistency_mode='off',
                 sdtr_scope='all'):
        super(BaseNet, self).__init__()
        if decoder_mode not in DecoderFusion.VALID_DECODER_MODES:
            raise ValueError(
                f'Unknown decoder_mode={decoder_mode!r}; '
                f'expected one of {DecoderFusion.VALID_DECODER_MODES}'
            )
        if grmsa_mode not in MSA_Module.VALID_MODES:
            raise ValueError(
                f'Unknown grmsa_mode={grmsa_mode!r}; '
                f'expected one of {MSA_Module.VALID_MODES}'
            )
        if head_mode not in self.VALID_HEAD_MODES:
            raise ValueError(
                f'Unknown head_mode={head_mode!r}; '
                f'expected one of {self.VALID_HEAD_MODES}'
            )
        if diff_mode is None:
            # Historical API compatibility: --use-eaom selected EAOM and the
            # absence of the flag selected CFDM before Run13 introduced an
            # explicit difference mode.
            diff_mode = 'eaom' if use_eaom else 'cfdm'
        elif diff_mode not in self.VALID_DIFF_MODES:
            raise ValueError(
                f'Unknown diff_mode={diff_mode!r}; '
                f'expected one of {self.VALID_DIFF_MODES}'
            )
        if use_eaom and diff_mode != 'eaom':
            raise ValueError('--use-eaom conflicts with diff_mode != "eaom"')
        if diff_sharing not in self.VALID_DIFF_SHARING:
            raise ValueError(
                f'Unknown diff_sharing={diff_sharing!r}; '
                f'expected one of {self.VALID_DIFF_SHARING}'
            )
        if sdtr_scope not in self.VALID_SDTR_SCOPES:
            raise ValueError(
                f'Unknown sdtr_scope={sdtr_scope!r}; '
                f'expected one of {self.VALID_SDTR_SCOPES}'
            )

        if (
            diff_mode == 'sdtr'
            and sdtr_scope != 'all'
            and diff_sharing != 'independent'
        ):
            raise ValueError(
                'sdtr_scope="shallow" or "deep" requires '
                'diff_sharing="independent"'
            )

        if diff_mode != 'sdtr' and sdtr_scope != 'all':
            raise ValueError(
                'sdtr_scope only applies when diff_mode="sdtr"; '
                f'got diff_mode={diff_mode!r}, sdtr_scope={sdtr_scope!r}'
            )
        if supervision_mode not in self.VALID_SUPERVISION_MODES:
            raise ValueError(
                f'Unknown supervision_mode={supervision_mode!r}; '
                f'expected one of {self.VALID_SUPERVISION_MODES}'
            )
        if amp_phase_mode not in self.VALID_AMP_PHASE_MODES:
            raise ValueError(
                f'Unknown amp_phase_mode={amp_phase_mode!r}; '
                f'expected one of {self.VALID_AMP_PHASE_MODES}'
            )
        if encoder_fusion_mode not in self.VALID_ENCODER_FUSION_MODES:
            raise ValueError(
                f'Unknown encoder_fusion_mode={encoder_fusion_mode!r}; '
                f'expected one of {self.VALID_ENCODER_FUSION_MODES}'
            )
        if boundary_mode is None:
            boundary_mode = 'edgegate' if use_edgegate else 'off'
        elif boundary_mode not in self.VALID_BOUNDARY_MODES:
            raise ValueError(
                f'Unknown boundary_mode={boundary_mode!r}; '
                f'expected one of {self.VALID_BOUNDARY_MODES}'
            )
        if use_edgegate and boundary_mode != 'edgegate':
            raise ValueError('--use-edgegate conflicts with boundary_mode != "edgegate"')
        if consistency_mode not in self.VALID_CONSISTENCY_MODES:
            raise ValueError(
                f'Unknown consistency_mode={consistency_mode!r}; '
                f'expected one of {self.VALID_CONSISTENCY_MODES}'
            )
        if use_sfif and grmsa_mode != 'off':
            raise ValueError('SFIF and GR-MSA are mutually exclusive decoder modes')
        if use_sfif and decoder_mode != 'msa':
            raise ValueError('SFIF and --decoder-mode are mutually exclusive decoder modes')
        if decoder_mode != 'msa' and grmsa_mode != 'off':
            raise ValueError('GR-MSA modes apply only to decoder_mode="msa"')

        self.use_eaom = diff_mode == 'eaom'
        self.use_sfif = use_sfif
        self.use_prior = use_prior
        self.use_msca = use_msca
        self.use_stargate = use_stargate
        self.use_edgegate = boundary_mode == 'edgegate'
        self.use_bdsr = boundary_mode == 'bdsr'
        self.use_lfds = bool(use_lfds)
        self.use_tct = bool(use_tct)
        self.grmsa_mode = grmsa_mode
        self.decoder_mode = decoder_mode
        self.head_mode = head_mode
        self.diff_mode = diff_mode
        self.diff_sharing = diff_sharing
        self.sdtr_scope = sdtr_scope
        self.supervision_mode = supervision_mode
        self.amp_phase_mode = amp_phase_mode
        self.encoder_fusion_mode = encoder_fusion_mode
        self.boundary_mode = boundary_mode
        self.consistency_mode = consistency_mode
        self.encoder = mobilenet_v2.mobilenet_v2(pretrained=True)

        if encoder_fusion_mode == 'rephfea_pyr':
            from models.rep_hfea import RepHFEA
            self.encoder_fusion = RepHFEA(inc=[16, 24, 32, 96, 320])
            # Per-scale adapters preserve the richer encoder pyramid while
            # keeping every difference implementation at the established 64ch.
            self.diff_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(cin, 64, 1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.GELU(),
                )
                for cin in RepHFEA.OUT_CHANNELS
            ])
        else:
            self.encoder_fusion = EncoderFusion(
                inc=[16, 24, 32, 96, 320], midc=32, outc=64
            )
            self.diff_adapters = None

        def make_diff(scale_idx=None):

            # 1. 普通 EAOM
            if diff_mode == 'eaom':
                from models.eaom import EAOM
                return EAOM(64)

            # 2. SDTR family
            if diff_mode == 'sdtr':
                from models.eaom import EAOM
                from models.temporal_relation import SDTR

                # historical shared SDTR
                if diff_sharing == 'shared':
                    return SDTR(64)

                if scale_idx not in (0, 1, 2, 3):
                    raise ValueError(
                        'Independent SDTR requires scale_idx 0..3'
                    )

                shallow_scale = scale_idx in (0, 1)
                deep_scale = scale_idx in (2, 3)

                use_sdtr = (
                    self.sdtr_scope == 'all'
                    or (
                        self.sdtr_scope == 'shallow'
                        and shallow_scale
                    )
                    or (
                        self.sdtr_scope == 'deep'
                        and deep_scale
                    )
                )

                # This scale is outside the selected SDTR scope.
                if not use_sdtr:
                    return EAOM(64)

                if scale_idx == 0:
                    return SDTR(
                        64,
                        mode='shallow',
                        window=5,
                        temperature=0.2,
                    )

                if scale_idx == 1:
                    return SDTR(
                        64,
                        mode='shallow',
                        window=3,
                        temperature=0.2,
                    )

                return SDTR(
                    64,
                    mode='deep',
                )

            # 3. historical CFDM
            return DiffModule(64)

        if diff_sharing == 'shared':
            self.diff = make_diff()
        else:
            # Explicit attributes are intentional: checkpoint keys make the
            # Run13 sharing factor immediately auditable.
            self.diff1 = make_diff(0)
            self.diff2 = make_diff(1)
            self.diff3 = make_diff(2)
            self.diff4 = make_diff(3)

        if amp_phase_mode == 'lf_shared':
            from models.amp_phase import APID
            self.apid1 = APID(radius_ratio=0.125)
            self.apid2 = APID(radius_ratio=0.125)

        if self.use_lfds:
            from models.frequency_supervision import LFDS
            self.lfds = LFDS(channels=64, patch_size=4)

        if self.use_tct:
            from models.change_tokens import TCT
            self.tct3 = TCT(channels=64, token_num=8, num_heads=4, depth=2)
            self.tct4 = TCT(channels=64, token_num=8, num_heads=4, depth=2)

        if use_prior:
            from models.prior import HFCPriorInjector
            self.prior = HFCPriorInjector(64)

        if use_msca:
            from models.msca import MSCA
            self.msca4 = MSCA(64)
            self.msca3 = MSCA(64)
            self.msca2 = MSCA(64)
            self.msca1 = MSCA(64)

        if use_stargate:
            from models.stargate import StarGate
            self.stargate4 = StarGate(64)
            self.stargate3 = StarGate(64)
            self.stargate2 = StarGate(64)
            self.stargate1 = StarGate(64)

        if self.use_edgegate:
            from models.edgegate import EdgeGate
            self.edgegate = EdgeGate(64)
        elif self.use_bdsr:
            from models.bdsr import BDSR
            self.bdsr = BDSR(64)

        self.decoder_fusion = DecoderFusion(
            64, use_sfif=use_sfif, grmsa_mode=grmsa_mode,
            decoder_mode=decoder_mode, scale_fusion_mode=scale_fusion_mode,
        )

        # Primary prediction head.
        # Keep the original attribute name for backward compatibility with
        # Run10 / historical checkpoints.
        self.decoder_out = nn.Conv2d(64, 1, kernel_size=1)

        # Run11: scale-decoupled prediction heads.
        # IMPORTANT: initialise auxiliary heads as exact copies of the primary
        # head, so shared/independent modes are functionally identical at
        # epoch 0; the only difference is whether their parameters remain
        # tied afterwards.
        if self.head_mode == 'independent':
            self.decoder_out2 = copy.deepcopy(self.decoder_out)
            self.decoder_out3 = copy.deepcopy(self.decoder_out)
            self.decoder_out4 = copy.deepcopy(self.decoder_out)

    def switch_to_deploy(self):
        """Fuse every RepDW training block in-place for deployment."""
        from models.rep_decoder import switch_repdw_to_deploy
        return switch_repdw_to_deploy(self)

    def strip_training_only_modules(self):
        """Remove auxiliary heads that are never used by inference.

        Call this only after loading a training checkpoint.  The main forward
        already skips LFDS unless ``return_aux=True``; deleting it also removes
        its parameters from a serialized deployment model.
        """
        stripped = []
        if hasattr(self, 'lfds'):
            del self.lfds
            self.use_lfds = False
            stripped.append('lfds')
        return tuple(stripped)

    def _adapt_encoder_features(self, features):
        if self.diff_adapters is None:
            return tuple(features)
        return tuple(
            adapter(feature)
            for adapter, feature in zip(self.diff_adapters, features)
        )

    def _compute_differences(self, f1, f2):
        if self.diff_sharing == 'shared':
            return tuple(self.diff(a, b) for a, b in zip(f1, f2))
        modules = (self.diff1, self.diff2, self.diff3, self.diff4)
        return tuple(module(a, b) for module, a, b in zip(modules, f1, f2))

    def forward(self, x1, x2, return_aux=False):
        # feature extraction
        x1_0, x1_1, x1_2, x1_3, x1_4 = self.encoder(x1)
        x2_0, x2_1, x2_2, x2_3, x2_4 = self.encoder(x2)

        # feature enhancement
        f1 = self.encoder_fusion(x1_0, x1_1, x1_2, x1_3, x1_4)
        f2 = self.encoder_fusion(x2_0, x2_1, x2_2, x2_3, x2_4)
        f1 = self._adapt_encoder_features(f1)
        f2 = self._adapt_encoder_features(f2)

        # APID-LF acts only on the 64x64 and 32x32 feature pairs.  Separate
        # instances provide the scale-specific zero-initialised gamma_s.
        if self.amp_phase_mode == 'lf_shared':
            f1_1, f2_1 = self.apid1(f1[0], f2[0])
            f1_2, f2_2 = self.apid2(f1[1], f2[1])
            f1 = (f1_1, f1_2, f1[2], f1[3])
            f2 = (f2_1, f2_2, f2[2], f2[3])

        # LFDS is computed lazily only when the training/validation caller
        # explicitly asks for auxiliary predictions.
        lfds_inputs = (f1[0], f2[0])

        # Save original HFEA context for MSCA/StarGate (before Prior modifies it)
        if self.use_msca or self.use_stargate:
            ctx_f1 = f1
            ctx_f2 = f2

        # HFC prior injection (between HFEA and DiffModule)
        if self.use_prior:
            f1, f2 = self.prior(f1, f2, t1_rgb=x1, t2_rgb=x2)

        # feature difference
        diff1, diff2, diff3, diff4 = self._compute_differences(f1, f2)

        # TCT complements the local/deep temporal relation with global change
        # tokens only at the semantic 16x16 and 8x8 scales.
        if self.use_tct:
            diff3 = self.tct3(f1[2], f2[2], diff3)
            diff4 = self.tct4(f1[3], f2[3], diff4)

        # MSCA: use ORIGINAL HFEA context (pre-Prior), not Prior-modified
        if self.use_msca:
            diff1 = self.msca1(ctx_f1[0], ctx_f2[0], diff1)
            diff2 = self.msca2(ctx_f1[1], ctx_f2[1], diff2)
            diff3 = self.msca3(ctx_f1[2], ctx_f2[2], diff3)
            diff4 = self.msca4(ctx_f1[3], ctx_f2[3], diff4)

        # StarGate: multiplicative cross-temporal gating (may follow MSCA)
        if self.use_stargate:
            diff1 = self.stargate1(ctx_f1[0], ctx_f2[0], diff1)
            diff2 = self.stargate2(ctx_f1[1], ctx_f2[1], diff2)
            diff3 = self.stargate3(ctx_f1[2], ctx_f2[2], diff3)
            diff4 = self.stargate4(ctx_f1[3], ctx_f2[3], diff4)

        # feature fusion
        f1, f2, f3, f4 = self.decoder_fusion(diff1, diff2, diff3, diff4)

        # EdgeGate: boundary refinement on the primary head feature f1 only
        if self.use_edgegate:
            f1 = self.edgegate(f1)
        boundary_pred = None
        if self.use_bdsr:
            f1, boundary_pred = self.bdsr(f1)

        # ------------------------------------------------------------------
        # Prediction heads
        # ------------------------------------------------------------------
        # f1 always uses the historical primary head.
        f1 = self.decoder_out(f1)

        if self.head_mode == 'independent':
            f2 = self.decoder_out2(f2)
            f3 = self.decoder_out3(f3)
            f4 = self.decoder_out4(f4)
        else:
            # Historical Run10 behaviour: one classifier shared by all scales.
            f2 = self.decoder_out(f2)
            f3 = self.decoder_out(f3)
            f4 = self.decoder_out(f4)

        # The primary output always remains full-resolution.  In native SCDS
        # mode, auxiliary predictions stay at 32/16/8 so their area-averaged
        # targets match the semantic resolution that produced them.
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

        if not return_aux:
            return outputs

        aux = {}
        if self.use_lfds:
            aux['frequency'] = torch.sigmoid(self.lfds(*lfds_inputs))
        if boundary_pred is not None:
            aux['boundary'] = boundary_pred
        return outputs, aux
