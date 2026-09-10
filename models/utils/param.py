"""AEGIS-CD model-complexity reporting.

Reports trainable/model parameters and THOP-registered operators.

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


def print_complexity(label, model, inputs):
    params, registered_ops = complexity(model, inputs)
    print(f'{label}:')
    print(f'  Params                 : {params / 1e6:.6f} M')
    print(f'  THOP registered ops    : {registered_ops / 1e9:.6f} G')
    print(
        '  NOTE: FFT/DWT and other functional operators may '
        'remain outside THOP accounting.'
    )


def main():
    parser = argparse.ArgumentParser(description='AEGIS-CD complexity report')
    parser.add_argument(
        '--decoder-mode', default='rep_dw', choices=['msa', 'rep_dw'],
    )
    parser.add_argument(
        '--head-mode', default='independent', choices=['shared', 'independent'],
    )
    parser.add_argument('--diff-mode', default='eaom', choices=['eaom', 'none'])
    parser.add_argument(
        '--diff-sharing', default='independent', choices=['shared', 'independent'],
    )
    parser.add_argument(
        '--supervision-mode', default='native', choices=['legacy', 'native'],
    )
    parser.add_argument(
        '--boundary-mode', default='edgegate', choices=['off', 'edgegate'],
    )
    parser.add_argument(
        '--device', default=None,
        help='cpu/cuda; defaults to CUDA when available',
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    model = BaseNet(
        diff_mode=args.diff_mode,
        diff_sharing=args.diff_sharing,
        supervision_mode=args.supervision_mode,
        decoder_mode=args.decoder_mode,
        head_mode=args.head_mode,
        boundary_mode=args.boundary_mode,
    ).to(device).eval()
    inputs = (
        torch.randn(1, 3, 256, 256, device=device),
        torch.randn(1, 3, 256, 256, device=device),
    )

    print(f'Decoder mode: {args.decoder_mode}')
    print(f'Head mode: {args.head_mode}')
    print(f'Diff mode: {args.diff_mode}')
    print(f'Diff sharing: {args.diff_sharing}')

    print_complexity('Training graph', model, inputs)

    if args.decoder_mode == 'rep_dw':
        converted = model.switch_to_deploy()
        print(f'RepDW blocks converted: {converted}')
        print_complexity('Deploy graph', model, inputs)


if __name__ == '__main__':
    main()
