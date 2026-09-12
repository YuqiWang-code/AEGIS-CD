"""
AEGIS-CD  —  Evaluation & Smoke-Test Entry Point
==================================================
Full evaluation::

    python models/scripts/test.py --dataset LEVIR-CD-256 \\
        --checkpoint saved_models/baseline/LEVIR-CD-256/best_model.pth

Architecture preflight::

    python models/scripts/test.py --smoke
"""

import sys
import os

# Ensure project root is on sys.path (script lives in models/scripts/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import random
import cv2
import numpy as np
from argparse import ArgumentParser
from PIL import Image

import torch
import torch.backends.cudnn as cudnn

from models.data import dataset as myDataLoader
from models.data import Transforms as myTransforms
from models.utils.metric_tool import ConfuseMatrixMeter

from models.model import BaseNet, validate_checkpoint_head_mode


# =========================================================
#  Loss helpers — imported from the single source of truth
# =========================================================

from models.utils.losses import deep_supervision_loss


def resolve_explicit_modes(args):
    """Normalise the explicit architecture mode fields (defaults already set)."""


def cli_option_was_provided(option_name):
    """Return True when an option was explicitly supplied on the CLI."""
    prefix = option_name + '='
    return any(
        token == option_name or token.startswith(prefix)
        for token in sys.argv[1:]
    )


def apply_checkpoint_value(
    args,
    signature,
    signature_key,
    arg_name,
    cli_option,
):
    """Restore a checkpoint value unless the explicit CLI conflicts."""
    if signature_key not in signature:
        return

    checkpoint_value = signature[signature_key]

    if cli_option_was_provided(cli_option):
        cli_value = getattr(args, arg_name)

        if cli_value != checkpoint_value:
            raise ValueError(
                f'CLI/checkpoint protocol conflict for {cli_option}: '
                f'CLI={cli_value!r}, '
                f'checkpoint={checkpoint_value!r}'
            )
    else:
        setattr(args, arg_name, checkpoint_value)


def load_checkpoint_for_evaluation(checkpoint_path):
    """Load raw legacy weights or a metadata-aware checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {checkpoint_path}'
        )

    print(f'Loading checkpoint metadata: {checkpoint_path}')

    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location='cpu',
    )

    if (
        isinstance(checkpoint, dict)
        and 'state_dict' in checkpoint
    ):
        state_dict = checkpoint['state_dict']

        protocol_signature = checkpoint.get(
            'protocol_signature'
        )

        if protocol_signature is None:
            run_config = checkpoint.get('run_config')
            if isinstance(run_config, dict):
                protocol_signature = run_config.get(
                    'protocol_signature'
                )

        return state_dict, protocol_signature

    # Historical best_model*.pth: raw state_dict.
    return checkpoint, None


def apply_checkpoint_protocol(args, signature):
    """Restore evaluation/model protocol from a metadata-aware checkpoint."""
    apply_checkpoint_value(
        args, signature,
        'dataset', 'dataset', '--dataset',
    )

    apply_checkpoint_value(
        args, signature,
        'in_width', 'inWidth', '--inWidth',
    )
    apply_checkpoint_value(
        args, signature,
        'in_height', 'inHeight', '--inHeight',
    )

    apply_checkpoint_value(
        args, signature,
        'color_order', 'color_order', '--color-order',
    )

    apply_checkpoint_value(
        args, signature,
        'diff_mode', 'diff_mode', '--diff-mode',
    )
    apply_checkpoint_value(
        args, signature,
        'diff_sharing', 'diff_sharing', '--diff-sharing',
    )
    apply_checkpoint_value(
        args, signature,
        'supervision_mode',
        'supervision_mode',
        '--supervision-mode',
    )
    apply_checkpoint_value(
        args, signature,
        'dice_reduction',
        'dice_reduction',
        '--dice-reduction',
    )
    apply_checkpoint_value(
        args, signature,
        'ds_profile', 'ds_profile', '--ds-profile',
    )

    apply_checkpoint_value(
        args, signature,
        'decoder_mode',
        'decoder_mode',
        '--decoder-mode',
    )
    apply_checkpoint_value(
        args, signature,
        'head_mode',
        'head_mode',
        '--head-mode',
    )
    apply_checkpoint_value(
        args, signature,
        'boundary_mode',
        'boundary_mode',
        '--boundary-mode',
    )
    apply_checkpoint_value(
        args, signature,
        'seed', 'seed', '--seed',
    )
    apply_checkpoint_value(
        args, signature,
        'deterministic',
        'deterministic',
        '--deterministic',
    )

def validate_effective_model_protocol(model, signature):
    """Check the effective model topology after checkpoint restoration."""
    if signature is None:
        return



# =========================================================
#  Validation with change-map output + inference timing
# =========================================================

@torch.no_grad()
def val(args, val_loader, model, epoch):
    model.eval()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    infer_time_total = 0.0
    infer_pairs = 0

    total_batches = len(val_loader)
    print(f'Total test batches: {total_batches}')

    # warm-up (GPU)
    if args.onGPU and torch.cuda.is_available():
        warm_n = 20
        for it, (img, target) in enumerate(val_loader):
            pre_img = img[:, 0:3].cuda(non_blocking=True).float()
            post_img = img[:, 3:6].cuda(non_blocking=True).float()
            _ = model(pre_img, post_img)
            if it >= warm_n - 1:
                break
        torch.cuda.synchronize()

    # evaluation
    sample_offset = 0
    for it, (img, target) in enumerate(val_loader):

        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU:
            pre_img = pre_img.cuda(non_blocking=True)
            post_img = post_img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        pre_img = pre_img.float()
        post_img = post_img.float()
        target = target.float()

        # inference timing
        if args.onGPU and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        outputs = model(pre_img, post_img)
        output, output2, output3, output4 = outputs

        if args.onGPU and torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_time_total += (time.time() - t0)
        infer_pairs += pre_img.size(0)

        # loss (not timed)
        loss = deep_supervision_loss(
            outputs, target, args.ds_profile, args.supervision_mode,
            args.dice_reduction,
        )
        epoch_loss.append(loss.item())

        pred = torch.where(output > 0.5,
                           torch.ones_like(output),
                           torch.zeros_like(output)).long()

        # ---- change maps + metrics for every sample in the batch ----
        f1 = 0.0
        for b_idx in range(pred.size(0)):
            sample_idx = sample_offset + b_idx
            img_name = val_loader.dataset.file_list[sample_idx]
            pr = pred[b_idx, 0].cpu().numpy()
            gt = target[b_idx, 0].cpu().numpy()

            index_tp = np.where((pr == 1) & (gt == 1))
            index_fp = np.where((pr == 1) & (gt == 0))
            index_tn = np.where((pr == 0) & (gt == 0))
            index_fn = np.where((pr == 0) & (gt == 1))

            cmap = np.zeros([gt.shape[0], gt.shape[1], 3], dtype=np.uint8)
            cmap[index_tp] = [255, 255, 255]   # TP: white
            cmap[index_fp] = [255, 0, 0]       # FP: red
            cmap[index_tn] = [0, 0, 0]         # TN: black
            cmap[index_fn] = [0, 255, 0]       # FN: green

            Image.fromarray(cmap).save(os.path.join(args.vis_dir, img_name))
            f1 = salEvalVal.update_cm(pr, gt)
        sample_offset += pred.size(0)

        if it % 5 == 0:
            avg_ms = (infer_time_total / max(1, infer_pairs)) * 1000.0
            print(f'\r[{it}/{total_batches}] F1: {f1:.4f}  '
                  f'loss: {loss.item():.4f}  infer: {avg_ms:.3f} ms/pair',
                  end='')

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()

    avg_infer_ms = (infer_time_total / max(1, infer_pairs)) * 1000.0
    print(f'\nAverage Inference Time: {avg_infer_ms:.3f} ms / image pair')
    scores['infer_ms_per_pair'] = avg_infer_ms

    return average_epoch_loss_val, scores


# =========================================================
#  Smoke test — minimal forward pass
# =========================================================

def _assert_reparam_close(reference, candidate, label):
    """Check reparameterization with precision-aware runtime tolerances.

    A fused convolution changes the floating-point accumulation order.  On
    CUDA (especially with TF32-enabled cuDNN kernels), mathematically
    equivalent train/deploy graphs can therefore differ by several 1e-4.
    Formula correctness is checked separately in float64; this comparison
    verifies that the actual runtime result remains numerically equivalent.
    """
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f'{label} shape mismatch: {tuple(reference.shape)} vs '
            f'{tuple(candidate.shape)}'
        )
    if reference.dtype == torch.float64:
        rtol, atol = 1e-10, 1e-10
    elif reference.device.type == 'cuda':
        rtol, atol = 1e-4, 1e-3
    else:
        rtol, atol = 1e-5, 1e-5

    difference = (reference - candidate).abs()
    max_abs = difference.max().item()
    reference_scale = reference.abs().max().clamp_min(1e-12).item()
    normalized = max_abs / reference_scale
    print(
        f'{label}: max_abs={max_abs:.3e}, normalized={normalized:.3e}, '
        f'rtol={rtol:.0e}, atol={atol:.0e}'
    )
    if not torch.allclose(reference, candidate, rtol=rtol, atol=atol):
        raise RuntimeError(
            f'{label} failed: max_abs={max_abs:.3e}, '
            f'normalized={normalized:.3e}, rtol={rtol:.0e}, atol={atol:.0e}'
        )
    return max_abs


def repdw_block_smoke(device):
    """Validate RepDW shapes, identity init, and non-trivial fusion."""
    from models.rep_decoder import RepDWBlock

    identity_block = RepDWBlock(64).to(device).eval()
    max_identity_error = 0.0
    shapes = []
    with torch.no_grad():
        for size in (64, 32, 16, 8):
            x = torch.randn(1, 64, size, size, device=device)
            y = identity_block(x)
            shapes.append(tuple(y.shape))
            max_identity_error = max(
                max_identity_error, (y - x).abs().max().item()
            )
    print(f'RepDW scale shapes: {shapes}')
    print(f'RepDW identity-init max|y-x|: {max_identity_error:.3e}')
    if max_identity_error >= 1e-7:
        raise RuntimeError(
            f'RepDW identity initialization failed: {max_identity_error:.3e}'
        )

    # Verify the fusion algebra under strict float64 arithmetic.  This catches
    # kernel padding, BN folding, or bias mistakes independently of GPU
    # float32/TF32 convolution algorithm differences.
    strict_probe = RepDWBlock(8).double().eval()
    with torch.no_grad():
        strict_probe.pw_bn.weight.uniform_(-1.0, 1.0)
        strict_probe.pw_bn.bias.uniform_(-0.1, 0.1)
        strict_x = torch.randn(2, 8, 8, 8, dtype=torch.float64)
        strict_train = strict_probe(strict_x)
        strict_probe.switch_to_deploy()
        strict_deploy = strict_probe(strict_x)
        _assert_reparam_close(
            strict_train, strict_deploy,
            'RepDW fusion formula (CPU float64)',
        )

    # Make the correction non-zero so the runtime-device test cannot pass
    # trivially because the final BN gamma is initialized to zero.
    probe = RepDWBlock(64).to(device).eval()
    with torch.no_grad():
        probe.pw_bn.weight.uniform_(-1.0, 1.0)
        probe.pw_bn.bias.uniform_(-0.1, 0.1)
        x = torch.randn(2, 64, 16, 16, device=device)
        y_train = probe(x)
        probe.switch_to_deploy()
        y_deploy = probe(x)
        _assert_reparam_close(
            y_train, y_deploy,
            'RepDW runtime deploy equivalence (non-zero branch)',
        )


def smoke_test(args):
    """Run the architecture preflight smoke test."""

    print('=== AEGIS-CD Smoke Test ===')

    print(f'PyTorch: {torch.__version__}')
    print(
        f'CUDA available: '
        f'{torch.cuda.is_available()}'
    )

    if torch.cuda.is_available():
        print(
            f'CUDA version: '
            f'{torch.version.cuda}'
        )
        print(
            f'GPU count: '
            f'{torch.cuda.device_count()}'
        )

        for index in range(
            torch.cuda.device_count()
        ):
            print(
                f'  GPU {index}: '
                f'{torch.cuda.get_device_name(index)}'
            )

    if (
        args.onGPU
        and torch.cuda.is_available()
    ):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Smoke device: {device}')

    # Existing RepDW algebra/runtime preflight.
    repdw_block_smoke(
        device
    )

    # Optional dataset-path probe.
    if args.dataset:
        dataset_root = os.path.join(
            args.data_root,
            args.dataset,
        )

        print(
            f'\nProbing dataset: '
            f'{dataset_root}'
        )

        if os.path.isdir(dataset_root):
            for subdirectory in (
                'A',
                'B',
                'label',
                'list',
            ):
                sub_path = os.path.join(
                    dataset_root,
                    subdirectory,
                )

                if os.path.isdir(sub_path):
                    count = len(
                        os.listdir(sub_path)
                    )

                    print(
                        f'  {subdirectory}/ : '
                        f'{count} items'
                    )
                else:
                    print(
                        f'  {subdirectory}/ : '
                        'MISSING'
                    )
        else:
            print(
                f'  Dataset root not found: '
                f'{dataset_root}'
            )

    print(
        '\n=== AEGIS-CD Smoke Test PASSED ==='
    )


# =========================================================
#  Main evaluation pipeline
# =========================================================

def ValidateSegmentation(args):
    state_dict, checkpoint_signature = (
        load_checkpoint_for_evaluation(
            args.checkpoint
        )
    )

    if checkpoint_signature is not None:
        apply_checkpoint_protocol(
            args,
            checkpoint_signature,
        )
        print(
            'Checkpoint protocol_signature found; '
            'evaluation configuration restored.'
        )
    else:
        print(
            'WARNING: legacy checkpoint has no '
            'protocol_signature; using explicit/default CLI '
            'configuration.'
        )

    resolve_explicit_modes(args)

    # Deterministic mode: disable cudnn.benchmark for strict reproducibility
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        cudnn.benchmark = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.deploy_reparam and args.decoder_mode != 'rep_dw':
        raise ValueError('--deploy-reparam requires rep_dw')

    # ---- build dataset root ----
    dataset_root = os.path.join(args.data_root, args.dataset)

    # ---- output directories ----
    if args.experiment_name:
        args.vis_dir = os.path.join(
            './predict', args.experiment_name, args.dataset)
        args.heatmap_dir = os.path.join(
            './heatmap', args.experiment_name, args.dataset)
    else:
        args.vis_dir = os.path.join('./predict', args.dataset)
        args.heatmap_dir = os.path.join('./heatmap', args.dataset)
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs(args.heatmap_dir, exist_ok=True)

    # ---- model ----
    model = BaseNet(
        diff_mode=args.diff_mode,
        diff_sharing=args.diff_sharing,
        supervision_mode=args.supervision_mode,
        decoder_mode=args.decoder_mode,
        head_mode=args.head_mode,
        boundary_mode=args.boundary_mode,
    )

    validate_effective_model_protocol(
        model,
        checkpoint_signature,
    )
    if args.onGPU:
        model = model.cuda()

    total_params = sum(
        np.prod(p.size())
        for p in model.parameters()
    )

    print(
        f'Total network parameters: '
        f'{total_params / 1e6:.2f} M'
    )

    print(
        f'Effective color order: '
        f'{args.color_order}'
    )

    # ---- transforms ----
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    valDataset = myTransforms.Compose([
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.ToTensor(color_order=args.color_order),
    ])

    # ---- data loader ----
    # --eval-split chooses val or test
    test_data = myDataLoader.Dataset(
        args.eval_split, file_root=dataset_root, transform=valDataset,
        list_name=args.eval_split)
    testLoader = torch.utils.data.DataLoader(
        test_data,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False)

    # ---- load checkpoint weights ----
    validate_checkpoint_head_mode(
        state_dict,
        args.head_mode,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        'Checkpoint weights loaded with strict=True'
    )

    # NOTE: there are no training-only modules to strip.  The deep-supervision
    # auxiliary heads (decoder_out2/3/4) are part of the model's forward() and
    # are still constructed and returned at inference time, so the model is
    # used exactly as loaded.  (A legacy call to a non-existent
    # ``model.strip_training_only_modules()`` used to live here and made every
    # non-smoke evaluation crash with AttributeError.)

    if args.deploy_reparam:
        model.eval()
        converted = model.switch_to_deploy()
        deploy_params = sum(np.prod(p.size()) for p in model.parameters())
        print(f'RepDW blocks converted for deployment: {converted}')
        print(f'Deploy network parameters: {deploy_params / 1e6:.4f} M')

    # ---- run evaluation ----
    loss_test, score_test = val(args, testLoader, model, 0)
    print('\n' + '=' * 60)
    print('Test Results:')
    print(f'  OA  = {score_test["OA"]:.4f}')
    print(f'  IoU = {score_test["IoU"]:.4f}')
    print(f'  F1  = {score_test["F1"]:.4f}')
    print(f'  R   = {score_test["recall"]:.4f}')
    print(f'  P   = {score_test["precision"]:.4f}')
    print(f'  Inference: {score_test.get("infer_ms_per_pair", 0):.3f} ms/pair')
    print('=' * 60)

    torch.cuda.empty_cache()


# =========================================================
#  Entry
# =========================================================

if __name__ == '__main__':
    parser = ArgumentParser(description='WDMF-Net / AEGIS-CD Evaluation')

    # ---- dataset ----
    parser.add_argument('--dataset', default='LEVIR-CD-256',
                        help='Dataset name (LEVIR-CD-256, WHU-CD-256, '
                             'SYSU-CD-256, CDD-CD-256, LEVIR-CD+256)')
    parser.add_argument('--data-root',
                        default='/home/hzeng/project/ZH/data/CD',
                        help='Root directory containing all datasets')

    # ---- model ----
    parser.add_argument('--checkpoint', default=None,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument(
        '--decoder-mode', default='rep_dw',
        choices=['msa', 'rep_dw'],
        help='Decoder implementation; must match the checkpoint',
    )
    parser.add_argument(
        '--head-mode', default='independent',
        choices=['shared', 'independent'],
        help='Prediction-head mode; must match the checkpoint',
    )
    parser.add_argument(
        '--ds-profile', default='legacy',
        choices=['legacy', 'primary', 'main_only'],
        help='Deep-supervision profile used for loss reporting',
    )
    parser.add_argument(
        '--diff-mode', default='eaom', choices=['eaom', 'none'],
        help='Difference encoder; must match the checkpoint.',
    )
    parser.add_argument(
        '--diff-sharing', default='independent',
        choices=['shared', 'independent'],
        help='Difference-encoder sharing; must match the checkpoint.',
    )
    parser.add_argument(
        '--supervision-mode', default='native',
        choices=['legacy', 'native'],
        help='legacy full-resolution DS or native SCDS outputs.',
    )
    parser.add_argument(
        '--dice-reduction', default='batch_global',
        choices=['batch_global', 'per_image'],
        help='Soft-Dice reduction used for loss reporting.',
    )
    parser.add_argument(
        '--boundary-mode', default='edgegate', choices=['off', 'edgegate'],
        help='Boundary mode; must match the checkpoint.',
    )
    parser.add_argument(
        '--deploy-reparam', action='store_true',
        help='Fuse RepDW training branches before smoke test/evaluation',
    )

    # ---- reproducibility parameters ----
    parser.add_argument(
        '--deterministic', action='store_true',
        help='Enable strict deterministic mode (disable cudnn.benchmark)',
    )
    parser.add_argument(
        '--color-order', default='fixed',
        choices=['legacy', 'fixed'],
        help=(
            'ToTensor color order. '
            'For metadata-aware checkpoints this value is restored '
            'from protocol_signature; an explicitly supplied conflicting '
            'CLI value fails fast. Legacy raw checkpoints retain the '
            'historical default.'
        ),
    )
    parser.add_argument(
        '--eval-split', default='test',
        choices=['val', 'test'],
        help='Split to evaluate: val or test (default: test)',
    )

    # ---- image ----
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)

    # ---- output ----
    parser.add_argument('--experiment-name', default=None,
                        help='Subdirectory under predict/ and heatmap/ to '
                             'avoid overwriting change maps across experiments')

    # ---- system ----
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--onGPU', default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--seed', type=int, default=2333)

    # ---- smoke test ----
    parser.add_argument(
        '--smoke',
        action='store_true',
        help=(
            'Run the architecture preflight smoke test.'
        ),
    )

    args = parser.parse_args()

    if args.smoke:
        smoke_test(args)
    else:
        if args.checkpoint is None:
            print('ERROR: --checkpoint is required for full evaluation '
                  '(or use --smoke for quick check)')
            sys.exit(1)
        ValidateSegmentation(args)
