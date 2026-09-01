"""Report AEGIS-CD parameter counts and FLOPs for training/deploy graphs."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from thop import profile

from models.model import BaseNet


def complexity(model, inputs):
    params = sum(p.numel() for p in model.parameters())
    flops, _profiled_params = profile(model, inputs=inputs, verbose=False)
    return params, flops


def estimate_sdtr_relation_macs(model, channels=64):
    """Analytic MACs for SDTR matching/attention omitted by THOP.

    Projections/convolutions remain in THOP. This counts only shallow cosine
    matching + value accumulation and deep QK/AV products for batch 1.
    """
    if getattr(model, 'diff_mode', None) != 'sdtr':
        return 0
    total = 0
    for size, window in ((64, 5), (32, 3)):
        positions = size * size
        candidates = window * window
        total += 4 * channels * positions * candidates
    for size in (16, 8):
        positions = size * size
        total += 4 * channels * positions * positions
    return total


def print_complexity(label, model, inputs):
    params, flops = complexity(model, inputs)
    relation_macs = estimate_sdtr_relation_macs(model)
    print(
        f'{label}: Params={params / 1e6:.6f} M  '
        f'THOP_registered_ops={flops / 1e9:.6f} G  '
        f'SDTR_relation_MACs={relation_macs / 1e9:.6f} G'
    )
    print('  NOTE: FFT and other functional ops are not included; do not cite THOP as total FLOPs.')


def main():
    parser = argparse.ArgumentParser(description='AEGIS-CD complexity report')
    parser.add_argument(
        '--decoder-mode', default='rep_dw',
        choices=['msa', 'plain_dw', 'rep_dw', 'rep_dw_shared'],
    )
    parser.add_argument('--grmsa-mode', default='off',
                        choices=['off', 'mask', 'residual', 'full'])
    parser.add_argument('--head-mode', default='shared',
                        choices=['shared', 'independent'])
    parser.add_argument('--scale-fusion', default='plain',
                        choices=['plain', 'scrf'],
                        help='Scale fusion mode (Run12)')
    parser.add_argument('--use-eaom', action='store_true')
    parser.add_argument('--use-edgegate', action='store_true')
    parser.add_argument('--use-sfif', action='store_true')
    parser.add_argument('--diff-mode', default=None,
                        choices=['cfdm', 'eaom', 'sdtr'])
    parser.add_argument('--diff-sharing', default='shared',
                        choices=['shared', 'independent'])
    parser.add_argument('--supervision-mode', default='legacy',
                        choices=['legacy', 'native'])
    parser.add_argument('--amp-phase-mode', default='off',
                        choices=['off', 'lf_shared'])
    parser.add_argument('--use-lfds', action='store_true')
    parser.add_argument('--use-tct', action='store_true')
    parser.add_argument('--encoder-fusion-mode', default='hfea',
                        choices=['hfea', 'rephfea_pyr'])
    parser.add_argument('--boundary-mode', default=None,
                        choices=['off', 'edgegate', 'bdsr'])
    parser.add_argument('--consistency-mode', default='off',
                        choices=['off', 'baic'])
    parser.add_argument('--device', default=None,
                        help='cpu/cuda; defaults to CUDA when available')
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model = BaseNet(
        use_eaom=args.use_eaom,
        use_edgegate=args.use_edgegate,
        use_sfif=args.use_sfif,
        grmsa_mode=args.grmsa_mode,
        decoder_mode=args.decoder_mode,
        head_mode=args.head_mode,
        scale_fusion_mode=args.scale_fusion,
        diff_mode=args.diff_mode,
        diff_sharing=args.diff_sharing,
        supervision_mode=args.supervision_mode,
        amp_phase_mode=args.amp_phase_mode,
        use_lfds=args.use_lfds,
        use_tct=args.use_tct,
        encoder_fusion_mode=args.encoder_fusion_mode,
        boundary_mode=args.boundary_mode,
        consistency_mode=args.consistency_mode,
    ).to(device).eval()
    inputs = (
        torch.randn(1, 3, 256, 256, device=device),
        torch.randn(1, 3, 256, 256, device=device),
    )

    print(f'Decoder mode: {args.decoder_mode}')
    print(f'Head mode: {args.head_mode}')
    print_complexity('Training graph', model, inputs)

    if args.decoder_mode in ('rep_dw', 'rep_dw_shared'):
        converted = model.switch_to_deploy()
        print(f'RepDW blocks converted: {converted}')
        print_complexity('Deploy graph', model, inputs)


if __name__ == '__main__':
    main()
