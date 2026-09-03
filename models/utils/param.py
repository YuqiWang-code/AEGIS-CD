"""AEGIS-CD model-complexity reporting.

Reports separately:
  - trainable/model parameters
  - THOP-registered operators
  - analytic SDTR relation MACs

FFT/DWT and other functional operators are explicitly noted rather than
silently folded into a misleading total-FLOPs number.
"""

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
    """Analytic SDTR relation MACs omitted by THOP.

    Counts only:
      - shallow bidirectional cosine matching + value accumulation
      - deep bidirectional QK + AV matrix products

    Conv/BN/projection operators remain in THOP registered ops.
    Batch size is fixed to 1 for this analytic report.
    """
    plan = getattr(
        model,
        'temporal_relation_plan',
        ('off', 'off', 'off', 'off'),
    )

    if len(plan) != 4:
        raise ValueError(
            'temporal_relation_plan must contain exactly four scales; '
            f'got {plan!r}'
        )

    scale_specs = (
        # scale_idx, spatial_size, shallow_window
        (0, 64, 5),
        (1, 32, 3),
        (2, 16, None),
        (3, 8, None),
    )

    total = 0

    for scale_idx, size, window in scale_specs:
        relation_mode = plan[scale_idx]
        positions = size * size

        if relation_mode == 'off':
            continue

        if relation_mode == 'shallow_replace':
            if window is None:
                raise ValueError(
                    f'shallow_replace is invalid at scale {scale_idx}'
                )

            candidates = window * window

            # Two temporal directions.
            # Per direction:
            #   cosine matching     = C * N * K
            #   value accumulation  = C * N * K
            total += (
                4
                * channels
                * positions
                * candidates
            )
            continue

        if relation_mode in (
            'deep_replace',
            'deep_residual',
        ):
            # Two temporal directions.
            # Per direction:
            #   QK = C * N^2
            #   AV = C * N^2
            total += (
                4
                * channels
                * positions
                * positions
            )
            continue

        raise ValueError(
            f'Unknown temporal relation mode '
            f'{relation_mode!r} at scale {scale_idx}'
        )

    return total


def print_complexity(label, model, inputs):
    params, registered_ops = complexity(
        model,
        inputs,
    )

    relation_macs = estimate_sdtr_relation_macs(
        model
    )

    print(f'{label}:')
    print(
        f'  Params                 : '
        f'{params / 1e6:.6f} M'
    )
    print(
        f'  THOP registered ops    : '
        f'{registered_ops / 1e9:.6f} G'
    )
    print(
        f'  SDTR relation MACs     : '
        f'{relation_macs / 1e9:.6f} G'
    )
    print(
        '  NOTE: THOP registered ops and SDTR relation MACs '
        'are reported separately and must not be summed and '
        'reported as exact total FLOPs.'
    )
    print(
        '  NOTE: FFT/DWT and other functional operators may '
        'remain outside THOP accounting.'
    )


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
    parser.add_argument(
        '--use-eaom',
        action='store_true',
    )
    parser.add_argument(
        '--use-edgegate',
        action='store_true',
    )
    parser.add_argument(
        '--use-sfif',
        action='store_true',
    )
    parser.add_argument(
        '--use-prior',
        action='store_true',
    )
    parser.add_argument(
        '--use-msca',
        action='store_true',
    )
    parser.add_argument('--diff-mode', default=None,
                        choices=['cfdm', 'eaom', 'sdtr'])
    parser.add_argument(
        '--diff-sharing',
        default='shared',
        choices=['shared', 'independent'],
    )
    parser.add_argument(
        '--sdtr-scope',
        default='all',
        choices=['all', 'shallow', 'deep'],
    )
    parser.add_argument(
        '--temporal-relation-mode',
        default=None,
        choices=[
            'off',
            'shallow_replace',
            'deep_replace',
            'deep_residual',
        ],
    )
    parser.add_argument(
        '--supervision-mode',
        default='legacy',
        choices=['legacy', 'native'],
    )
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
        use_prior=args.use_prior,
        use_msca=args.use_msca,
        grmsa_mode=args.grmsa_mode,
        decoder_mode=args.decoder_mode,
        head_mode=args.head_mode,
        scale_fusion_mode=args.scale_fusion,
        diff_mode=args.diff_mode,
        diff_sharing=args.diff_sharing,
        sdtr_scope=args.sdtr_scope,
        temporal_relation_mode=args.temporal_relation_mode,
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
    print(f'Diff mode: {args.diff_mode}')
    print(f'Diff sharing: {args.diff_sharing}')
    print(f'SDTR scope: {args.sdtr_scope}')
    print(
        f'Temporal relation mode: '
        f'{args.temporal_relation_mode}'
    )
    print(
        f'Effective base diff: '
        f'{model.base_diff_mode}'
    )
    print(
        f'Effective temporal relation plan: '
        f'{list(model.temporal_relation_plan)}'
    )
    print(f'MSCA: {args.use_msca}')
    print(f'Prior: {args.use_prior}')
    print(f'TCT: {args.use_tct}')

    print_complexity(
        'Training graph',
        model,
        inputs,
    )

    if args.decoder_mode in ('rep_dw', 'rep_dw_shared'):
        converted = model.switch_to_deploy()
        print(f'RepDW blocks converted: {converted}')
        print_complexity('Deploy graph', model, inputs)


if __name__ == '__main__':
    main()
