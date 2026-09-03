"""
AEGIS-CD  —  Training Entry Point
===================================
Usage::

    # Baseline (CFDM)
    python models/scripts/train.py --dataset LEVIR-CD-256 --epochs 200 \\
        --batch-size 48 --lr 5e-4 --val-interval 10 --val-split val

    # EAOM ablation
    python models/scripts/train.py --dataset LEVIR-CD-256 --epochs 200 \\
        --batch-size 48 --lr 5e-4 --val-interval 10 --val-split val --use-eaom
"""

import sys
import os
import glob

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import datetime
import random
import numpy as np
import json
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn
from models.data import dataset as myDataLoader
from models.data import Transforms as myTransforms
from models.utils.metric_tool import ConfuseMatrixMeter

from models.model import BaseNet, validate_checkpoint_head_mode


def seed_worker(worker_id):
    """Seed Python/NumPy in each DataLoader worker from PyTorch's worker seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


PROTOCOL_SIGNATURE_VERSION = 1


def build_protocol_signature(args, model):
    """Build the strict experiment protocol used for full-checkpoint resume."""
    return {
        'version': PROTOCOL_SIGNATURE_VERSION,

        # Dataset / input protocol
        'dataset': args.dataset,
        'in_width': args.inWidth,
        'in_height': args.inHeight,
        'batch_size': args.batch_size,
        'color_order': args.color_order,
        'val_split': args.val_split,

        # Optimisation / reproducibility protocol
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'lr_mode': args.lr_mode,
        'step_loss': args.step_loss,
        'val_interval': args.val_interval,
        'seed': args.seed,
        'deterministic': args.deterministic,
        'num_workers': args.num_workers,

        # Data augmentation
        'amp_mix': args.amp_mix,

        # Difference / temporal relation
        'diff_mode': args.diff_mode,
        'base_diff_mode': model.base_diff_mode,
        'diff_sharing': args.diff_sharing,
        'sdtr_scope': args.sdtr_scope,
        'temporal_relation_mode': args.temporal_relation_mode,
        'temporal_relation_plan': list(model.temporal_relation_plan),

        # Supervision
        'supervision_mode': args.supervision_mode,
        'dice_reduction': args.dice_reduction,
        'ds_profile': args.ds_profile,

        # Architecture
        'use_prior': args.use_prior,
        'use_msca': args.use_msca,
        'use_tct': args.use_tct,
        'use_lfds': args.use_lfds,
        'amp_phase_mode': args.amp_phase_mode,
        'encoder_fusion_mode': args.encoder_fusion_mode,
        'decoder_mode': args.decoder_mode,
        'head_mode': args.head_mode,
        'scale_fusion': args.scale_fusion,
        'boundary_mode': args.boundary_mode,
        'consistency_mode': args.consistency_mode,

        # Historical architecture switches
        'use_sfif': args.use_sfif,
        'use_stargate': args.use_stargate,
        'grmsa_mode': args.grmsa_mode,

        # Fixed auxiliary-loss coefficients
        'lambda_freq': 0.2,
        'lambda_boundary': 0.2,
        'lambda_consistency': 0.1,
    }


def validate_protocol_signature(checkpoint_signature, current_signature):
    """Fail-fast when a full checkpoint uses a different experiment protocol."""
    if not isinstance(checkpoint_signature, dict):
        raise ValueError(
            'Checkpoint protocol_signature is not a dictionary.'
        )

    mismatches = []

    all_keys = sorted(
        set(checkpoint_signature.keys())
        | set(current_signature.keys())
    )

    for key in all_keys:
        checkpoint_value = checkpoint_signature.get(
            key, '<MISSING>'
        )
        current_value = current_signature.get(
            key, '<MISSING>'
        )

        if checkpoint_value != current_value:
            mismatches.append(
                f'  {key}: '
                f'checkpoint={checkpoint_value!r}, '
                f'current={current_value!r}'
            )

    if mismatches:
        details = '\n'.join(mismatches)
        raise ValueError(
            'Checkpoint protocol mismatch. '
            'Refusing full resume.\n'
            f'{details}'
        )


# =========================================================
#  Loss helpers — imported from the single source of truth
# =========================================================

from models.utils.losses import DS_PROFILES, run13_supervised_loss
from models.utils.consistency import (
    amplitude_mix,
    consistency_loss,
    freeze_batchnorm_running_stats,
)


# Normalize runs on OpenCV BGR channels, then ToTensor converts each temporal
# image to RGB.  Model inputs therefore carry RGB-ordered statistics.
_RGB_MEAN_6 = (0.485, 0.456, 0.406, 0.485, 0.456, 0.406)
_RGB_STD_6 = (0.229, 0.224, 0.225, 0.229, 0.224, 0.225)


def supervised_forward(args, model, pre_img, post_img, targets):
    """Forward main predictions plus training-only Run13 heads when needed."""
    need_aux = args.use_lfds or args.boundary_mode == 'bdsr'
    if need_aux:
        outputs, aux = model(pre_img, post_img, return_aux=True)
    else:
        outputs = model(pre_img, post_img)
        aux = {}
    loss, components = run13_supervised_loss(
        outputs,
        targets,
        profile=args.ds_profile,
        supervision_mode=args.supervision_mode,
        aux=aux,
        lambda_freq=0.2,
        lambda_boundary=0.2,
        dice_reduction=args.dice_reduction,
    )
    return outputs, loss, components


# =========================================================
#  Validation
# =========================================================

@torch.no_grad()
def val(args, val_loader, model, _epoch):
    model.eval()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(val_loader)

    for iter, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU:
            pre_img = pre_img.cuda()
            post_img = post_img.cuda()
            target = target.cuda()

        pre_img_var = pre_img.float()
        post_img_var = post_img.float()
        target_var = target.float()

        outputs, loss, _components = supervised_forward(
            args, model, pre_img_var, post_img_var, target_var
        )
        output, output2, output3, output4 = outputs

        pred = torch.where(output > 0.5,
                           torch.ones_like(output),
                           torch.zeros_like(output)).long()
        epoch_loss.append(loss.item())

        f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(),
                                  gt=target_var.cpu().numpy())

        if iter % 5 == 0:
            print(f'\r  val [{iter}/{total_batches}] F1={f1:.4f}  loss={loss.item():.4f}',
                  end='')

    avg_loss = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()
    return avg_loss, scores


# =========================================================
#  Training loop (one epoch)
# =========================================================

def train_one_epoch(args, train_loader, model, optimizer, epoch,
                    max_batches, cur_iter=0, lr_factor=1.):
    model.train()
    salEvalVal = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total = len(train_loader)

    for iter, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        lr = adjust_learning_rate(args, optimizer, epoch,
                                  iter + cur_iter, max_batches,
                                  lr_factor=lr_factor)

        if args.onGPU:
            pre_img = pre_img.cuda()
            post_img = post_img.cuda()
            target = target.cuda()

        pre_img_var = pre_img.float()
        post_img_var = post_img.float()
        target_var = target.float()

        outputs, loss, components = supervised_forward(
            args, model, pre_img_var, post_img_var, target_var
        )
        output, output2, output3, output4 = outputs

        if args.consistency_mode == 'baic':
            pair = torch.cat([pre_img_var, post_img_var], dim=1)
            mixed = amplitude_mix(
                pair, r=0.125, mean=_RGB_MEAN_6, std=_RGB_STD_6
            )
            with freeze_batchnorm_running_stats(model):
                student_outputs = model(mixed[:, 0:3], mixed[:, 3:6])
            components['consistency'] = consistency_loss(
                output, student_outputs[0], conf=0.9
            )
            loss = loss + 0.1 * components['consistency']

        pred = torch.where(output > 0.5,
                           torch.ones_like(output),
                           torch.zeros_like(output)).long()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        with torch.no_grad():
            f1 = salEvalVal.update_cm(pr=pred.cpu().numpy(),
                                      gt=target_var.cpu().numpy())

        if iter % 5 == 0:
            global_step = iter + cur_iter
            total_steps = max_batches * args.epochs
            print(f'\r  train [{global_step}/{total_steps}] '
                  f'F1={f1:.3f}  lr={lr:.7f}  loss={loss.item():.4f}',
                  end='')

    avg_loss = sum(epoch_loss) / len(epoch_loss)
    scores = salEvalVal.get_scores()
    return avg_loss, scores, lr


# =========================================================
#  LR scheduling
# =========================================================

def adjust_learning_rate(args, optimizer, epoch, iter, max_batches, lr_factor=1):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        max_iter = max_batches * args.epochs
        lr = args.lr * (1 - (iter) * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError(f'Unknown lr_mode: {args.lr_mode}')

    if epoch == 0 and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr

    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# =========================================================
#  Helper: count FLOPs
# =========================================================

def count_flops(model, input_shape=(1, 3, 256, 256), device='cuda'):
    """Return THOP-registered ops in G; functional FFT/relation ops are omitted."""
    was_training = model.training

    try:
        from thop import profile

        model.eval()

        a = torch.randn(*input_shape).to(device)
        b = torch.randn(*input_shape).to(device)

        with torch.no_grad():
            flops, _params = profile(
                model,
                inputs=(a, b),
                verbose=False,
            )

        return flops / 1e9

    except Exception:
        return None

    finally:
        model.train(was_training)


# =========================================================
#  Banner: print full configuration before training
# =========================================================

def print_banner(args, model, dataset_root, train_loader, val_loader, test_loader):
    """Print a comprehensive configuration banner to stdout and return a
    multi-line string for the log file."""
    sep = '=' * 68
    lines = [sep]

    # ---- Dataset ----
    lines.append(f'  Dataset       : {args.dataset}')
    lines.append(f'  Data root     : {dataset_root}')
    lines.append(f'  Train / Val / Test : '
                 f'{len(train_loader.dataset)} / '
                 f'{len(val_loader.dataset)} / '
                 f'{len(test_loader.dataset)}')
    lines.append(f'  Image size    : {args.inWidth} × {args.inHeight}')

    # ---- Model ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f'  Params (total / trainable) : '
                 f'{total_params / 1e6:.2f} M / {trainable_params / 1e6:.2f} M')

    flops = count_flops(model)
    if flops is not None:
        lines.append(f'  THOP ops      : {flops:.2f} G (excludes FFT/relation functional ops)')
    else:
        lines.append(f'  THOP ops      : (THOP unavailable or profiling failed)')

    # ---- Module switches ----
    switches = []
    for sw in ['use_eaom', 'use_sfif', 'use_prior', 'use_msca',
               'use_stargate', 'use_edgegate', 'use_lfds', 'use_tct']:
        val = getattr(args, sw, False)
        marker = '✅ ON' if val else '❌ off'
        switches.append(f'{sw}={marker}')
    switches.append(f'grmsa_mode={args.grmsa_mode}')
    switches.append(f'decoder_mode={args.decoder_mode}')
    switches.append(f'head_mode={args.head_mode}')
    switches.append(f'ds_profile={args.ds_profile}')
    switches.append(f'diff_mode={args.diff_mode}')
    switches.append(f'diff_sharing={args.diff_sharing}')
    switches.append(f'sdtr_scope={args.sdtr_scope}')
    switches.append(
        f'temporal_relation_mode={args.temporal_relation_mode}'
    )
    switches.append(f'supervision_mode={args.supervision_mode}')
    switches.append(f'amp_phase_mode={args.amp_phase_mode}')
    switches.append(f'encoder_fusion_mode={args.encoder_fusion_mode}')
    switches.append(f'boundary_mode={args.boundary_mode}')
    switches.append(f'consistency_mode={args.consistency_mode}')
    lines.append(f'  Modules       : {", ".join(switches)}')

    # Run14: also print the effective normalised topology.
    lines.append(
        f'  Base diff     : {model.base_diff_mode}'
    )
    lines.append(
        f'  Relation plan : {list(model.temporal_relation_plan)}'
    )

    if args.use_sfif:
        deployable = 'no (legacy SFIF)'
        rep_branches = 'n/a'
        scale_sharing = 'independent (4× SFIF)'
    elif args.decoder_mode == 'rep_dw':
        deployable = 'yes (RepDW branches → DW5×5)'
        rep_branches = 'DW5×5 + DW3×3 + DW1×1 + identity'
        scale_sharing = 'independent (4× RepDW)'
    elif args.decoder_mode == 'rep_dw_shared':
        deployable = 'yes (RepDW branches → DW5×5)'
        rep_branches = 'DW5×5 + DW3×3 + DW1×1 + identity'
        scale_sharing = 'shared (1× RepDW across 4 scales)'
    elif args.decoder_mode == 'plain_dw':
        deployable = 'no reparameterization (single DW5×5 branch)'
        rep_branches = 'DW5×5 only'
        scale_sharing = 'independent (4× PlainDW)'
    else:
        deployable = 'no (legacy MSA/GR-MSA)'
        rep_branches = 'n/a'
        scale_sharing = 'shared (1× MSA across 4 scales)'
    lines.append(f'  Decoder mode  : {args.decoder_mode}')
    lines.append(f'  Decoder deployable : {deployable}')
    lines.append(f'  Rep branches  : {rep_branches}')
    lines.append(f'  Scale sharing : {scale_sharing}')

    # ---- Hardware ----
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        lines.append(f'  GPU           : {gpu_count}× {gpu_name} '
                     f'({mem_total:.0f} GB), CUDA {torch.version.cuda}')
    else:
        lines.append(f'  GPU           : (none — running on CPU)')

    # ---- Hyperparameters ----
    lines.append(f'  Optimizer     : Adam (β=0.9, 0.99), weight_decay={args.weight_decay}')
    lines.append(f'  LR            : {args.lr}  ({args.lr_mode} schedule)')
    lines.append(f'  Batch size    : {args.batch_size}')
    lines.append(f'  Epochs        : {args.epochs}')
    lines.append(f'  Val interval  : every {args.val_interval} epoch(s)')
    lines.append(f'  Checkpoint selection : {args.val_split}.txt')
    lines.append(f'  Head mode     : {args.head_mode}')
    lines.append(
        f'  DS profile    : {args.ds_profile} '
        f'weights={DS_PROFILES[args.ds_profile]}'
    )
    lines.append(
        f'  Loss          : SCDS({args.supervision_mode}) '
        f'weights={DS_PROFILES[args.ds_profile]}, '
        f'dice={args.dice_reduction}'
    )
    if args.use_lfds:
        lines.append('  LFDS loss     : 0.2 × BCEDice (16×16 local frequency)')
    if args.boundary_mode == 'bdsr':
        lines.append('  Boundary loss : 0.2 × BCEDice (dilate-erode target)')
    if args.consistency_mode == 'baic':
        lines.append('  BAIC loss     : 0.1 × confidence-masked L1')

    # ---- Run12 configuration ----
    lines.append(f'  Color order   : {args.color_order}')
    lines.append(f'  Scale fusion  : {args.scale_fusion}')
    lines.append(f'  Amp augment   : {"ON" if args.amp_mix else "OFF"}')
    lines.append(f'  Deterministic : {"ON" if args.deterministic else "OFF"}')

    lines.append(f'  Seed          : {args.seed}')
    lines.append(f'  Num workers   : {args.num_workers}')

    # ---- Output ----
    lines.append(f'  Save dir      : {args.savedir}')
    lines.append(f'  Log file      : {args.logFile}')
    lines.append(f'  Start time    : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(sep)

    banner = '\n'.join(lines)
    print('\n' + banner + '\n')
    return banner


# =========================================================
#  Main training orchestrator
# =========================================================

def trainValidateSegmentation(args):
    t_start = time.time()

    # Resolve historical boolean flags into the explicit Run13 modes before
    # logging/config serialization.  Old command lines remain reproducible,
    # while new runs have one unambiguous source of truth.
    if args.diff_mode is None:
        args.diff_mode = 'eaom' if args.use_eaom else 'cfdm'
    if args.boundary_mode is None:
        args.boundary_mode = 'edgegate' if args.use_edgegate else 'off'

    # Run12: deterministic mode disables cudnn.benchmark for strict reproducibility
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        cudnn.benchmark = True

    SEED = getattr(args, 'seed', 2333)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- mutual exclusion checks ----
    if args.grmsa_mode != 'off' and args.use_sfif:
        raise ValueError('--grmsa-mode is mutually exclusive with --use-sfif')
    if args.decoder_mode != 'msa' and args.use_sfif:
        raise ValueError('--decoder-mode other than msa is mutually exclusive with --use-sfif')
    if args.decoder_mode != 'msa' and args.grmsa_mode != 'off':
        raise ValueError('--grmsa-mode applies only with --decoder-mode msa')

    # ---- build dataset path ----
    dataset_root = os.path.join(args.data_root, args.dataset)

    # ---- output directory ----
    args.savedir = os.path.join(args.savedir, args.dataset)
    os.makedirs(args.savedir, exist_ok=True)
    stale_completion = os.path.join(args.savedir, '.run_complete')
    if os.path.isfile(stale_completion):
        os.remove(stale_completion)

    # ---- model ----
    model = BaseNet(
        use_eaom=args.use_eaom,
        use_sfif=args.use_sfif,
        use_prior=args.use_prior,
        use_msca=args.use_msca,
        use_stargate=args.use_stargate,
        use_edgegate=args.use_edgegate,
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
        sdtr_scope=args.sdtr_scope,
        temporal_relation_mode=args.temporal_relation_mode,
    )

    current_protocol_signature = build_protocol_signature(
        args, model
    )

    print(
        'Run14 protocol signature prepared '
        f'(version={current_protocol_signature["version"]})'
    )

    if args.onGPU:
        model = model.cuda()

    # ---- data transforms ----
    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # Run12: build transform list dynamically
    # CRITICAL: AmpMix must be BEFORE Normalize (operates on raw [0,255] images)
    train_transforms = [
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
    ]
    if args.amp_mix:
        train_transforms.append(myTransforms.AmpMix(prob=0.5))
    # Normalize must be AFTER AmpMix
    train_transforms.append(myTransforms.Normalize(mean=mean, std=std))
    train_transforms.append(myTransforms.ToTensor(color_order=args.color_order))

    trainDataset_main = myTransforms.Compose(train_transforms)

    valDataset = myTransforms.Compose([
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.ToTensor(color_order=args.color_order),
    ])

    # ---- data loaders ----
    train_data = myDataLoader.Dataset(
        'train', file_root=dataset_root, transform=trainDataset_main,
        list_name='train')
    train_generator = make_generator(SEED)
    val_generator = make_generator(SEED + 1)
    test_generator = make_generator(SEED + 2)

    trainLoader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True,
        worker_init_fn=seed_worker, generator=train_generator)

    # Validation selects the best checkpoint; Run9 uses val.txt explicitly.
    val_data = myDataLoader.Dataset(
        'val', file_root=dataset_root, transform=valDataset,
        list_name=args.val_split)
    valLoader = torch.utils.data.DataLoader(
        val_data, shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False,
        worker_init_fn=seed_worker, generator=val_generator)

    # Held-out test.txt is evaluated once after loading the best val checkpoint.
    test_data = myDataLoader.Dataset(
        'test', file_root=dataset_root, transform=valDataset,
        list_name='test')
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=False,
        worker_init_fn=seed_worker, generator=test_generator)

    max_batches = len(trainLoader)

    # ---- print comprehensive banner ----
    banner = print_banner(args, model, dataset_root,
                          trainLoader, valLoader, testLoader)

    # ---- optimizer ----
    optimizer = torch.optim.Adam(
        model.parameters(), args.lr, (0.9, 0.99),
        eps=1e-08, weight_decay=args.weight_decay)

    # ---- resume ----
    start_epoch = 0
    cur_iter = 0
    max_F1_val = 0.0
    best_epoch = 0
    full_resume = False
    strict_protocol_resume = False

    if args.resume and os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, weights_only=False)
        if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
            # Full checkpoint resume: model + optimizer + epoch + best_f1

            checkpoint_signature = checkpoint.get(
                'protocol_signature'
            )

            if checkpoint_signature is not None:
                validate_protocol_signature(
                    checkpoint_signature,
                    current_protocol_signature,
                )
                strict_protocol_resume = True
                print(
                    '=> protocol_signature matched; '
                    'strict full resume allowed'
                )
            else:
                # Run14's explicit temporal-relation API requires a strict
                # protocol_signature.  Never resume it from an unverifiable
                # historical full checkpoint.
                if args.temporal_relation_mode is not None:
                    raise ValueError(
                        'Checkpoint has no protocol_signature, but the '
                        'current run uses the Run14 '
                        '--temporal-relation-mode API. '
                        'Refusing unverifiable full resume.'
                    )

                print(
                    'WARNING: legacy full checkpoint has no '
                    'protocol_signature; falling back to legacy '
                    'dataset/batch-size/head-mode validation.'
                )

                if (
                    checkpoint.get('dataset', args.dataset)
                    != args.dataset
                ):
                    raise ValueError(
                        f"Checkpoint dataset="
                        f"{checkpoint.get('dataset')!r} does "
                        f"not match CLI dataset={args.dataset!r}"
                    )

                if (
                    checkpoint.get(
                        'batch_size', args.batch_size
                    )
                    != args.batch_size
                ):
                    raise ValueError(
                        f"Checkpoint batch_size="
                        f"{checkpoint.get('batch_size')} "
                        f"does not match CLI batch_size="
                        f"{args.batch_size}"
                    )

            validate_checkpoint_head_mode(
                checkpoint['state_dict'],
                args.head_mode,
            )
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            cur_iter = checkpoint.get('cur_iter', start_epoch * max_batches)
            max_F1_val = checkpoint.get('best_f1', 0.0)
            best_epoch = checkpoint.get('best_epoch', start_epoch)

            # Restore the exact next-epoch data order and stochastic state.
            if 'train_generator_state' in checkpoint:
                train_generator.set_state(checkpoint['train_generator_state'])
            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'])
            if args.onGPU and checkpoint.get('cuda_rng_state_all') is not None:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
            if 'numpy_rng_state' in checkpoint:
                np.random.set_state(checkpoint['numpy_rng_state'])
            if 'python_rng_state' in checkpoint:
                random.setstate(checkpoint['python_rng_state'])

            full_resume = True
            print(
                f"=> resumed after epoch {start_epoch}, "
                f"best_f1={max_F1_val:.4f} @ epoch {best_epoch}"
            )
        else:
            # Weights-only: best_model.pth or legacy format.
            # This is not an exact training resume.
            state_dict = checkpoint.get('state_dict', checkpoint)
            validate_checkpoint_head_mode(
                state_dict,
                args.head_mode,
            )
            model.load_state_dict(state_dict)
            print(
                'WARNING: loaded weights only; no full-resume '
                'protocol validation is available. '
                'Using a fresh optimizer and starting epoch 0.'
            )

    # ---- log file header ----
    logFileLoc = os.path.join(args.savedir, args.logFile)
    if full_resume and os.path.isfile(logFileLoc):
        # Append to existing log
        logger = open(logFileLoc, 'a')
        logger.write(
            f'\n# Resumed at epoch {start_epoch}\n'
        )
        logger.write(
            '# Protocol resume: '
            f'{"STRICT" if strict_protocol_resume else "LEGACY_FALLBACK"}\n'
        )
    else:
        logger = open(logFileLoc, 'w')
        logger.write(banner + '\n')
        logger.write('%-6s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s\t%-10s\n' %
                     ('Epoch', 'TrLoss', 'VaLoss', 'OA(val)', 'IoU(val)',
                      'F1(val)', 'R(val)', 'P(val)', 'BestF1'))
    logger.flush()

    # ---- save run configuration as JSON for reproducibility ----
    config_dict = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'lr_mode': args.lr_mode,
        'seed': args.seed,
        'use_eaom': args.use_eaom,
        'use_sfif': args.use_sfif,
        'use_prior': args.use_prior,
        'use_msca': args.use_msca,
        'use_stargate': args.use_stargate,
        'use_edgegate': args.use_edgegate,
        'grmsa_mode': args.grmsa_mode,
        'decoder_mode': args.decoder_mode,
        'head_mode': args.head_mode,
        'ds_profile': args.ds_profile,
        'scale_fusion': args.scale_fusion,
        'color_order': args.color_order,
        'deterministic': args.deterministic,
        'amp_mix': args.amp_mix,
        'val_split': args.val_split,
        'diff_mode': args.diff_mode,
        'diff_sharing': args.diff_sharing,
        'sdtr_scope': args.sdtr_scope,
        'temporal_relation_mode': args.temporal_relation_mode,

        # Effective normalised Run14 topology.
        'base_diff_mode': model.base_diff_mode,
        'temporal_relation_plan': list(model.temporal_relation_plan),

        'supervision_mode': args.supervision_mode,
        'amp_phase_mode': args.amp_phase_mode,
        'use_lfds': args.use_lfds,
        'use_tct': args.use_tct,
        'encoder_fusion_mode': args.encoder_fusion_mode,
        'boundary_mode': args.boundary_mode,
        'consistency_mode': args.consistency_mode,
        'lambda_freq': 0.2,
        'lambda_boundary': 0.2,
        'lambda_consistency': 0.1,
        'dice_reduction': args.dice_reduction,

        # Strict full-resume experiment protocol.
        'protocol_signature': current_protocol_signature,
    }
    config_path = os.path.join(args.savedir, 'run_config.json')

    if full_resume and not strict_protocol_resume:
        legacy_resume_config_path = os.path.join(
            args.savedir,
            'run_config_legacy_resume_attempt.json',
        )
        with open(
            legacy_resume_config_path,
            'w',
            encoding='utf-8',
        ) as f:
            json.dump(config_dict, f, indent=2)

        print(
            'Legacy resume configuration saved to: '
            f'{legacy_resume_config_path}'
        )
    else:
        with open(
            config_path,
            'w',
            encoding='utf-8',
        ) as f:
            json.dump(config_dict, f, indent=2)

        print(f'Configuration saved to: {config_path}')

    # ---- training loop ----
    total_epochs = args.epochs
    last_checkpoint_path = os.path.join(
        args.savedir, 'last_checkpoint.pth.tar'
    )
    last_checkpoint_tmp = last_checkpoint_path + '.tmp'

    for epoch in range(start_epoch, total_epochs):
        t_epoch = time.time()

        lossTr, score_tr, lr = train_one_epoch(
            args, trainLoader, model, optimizer, epoch,
            max_batches, cur_iter)
        cur_iter += len(trainLoader)

        torch.cuda.empty_cache()

        elapsed_total = time.time() - t_start
        eta_total = (elapsed_total / (epoch + 1 - start_epoch)) * (total_epochs - epoch - 1) \
            if epoch + 1 > start_epoch else 0

        # ---- validation ----
        if epoch == 0 or (epoch + 1) % args.val_interval == 0:
            lossVal, score_val = val(args, valLoader, model, epoch)
            torch.cuda.empty_cache()

            is_best = score_val['F1'] > max_F1_val
            if is_best:
                max_F1_val = score_val['F1']
                best_epoch = epoch + 1
                # remove old best-model files (weights + checkpoint)
                for old in glob.glob(os.path.join(args.savedir, 'best_model*.pth')):
                    os.remove(old)
                for old in glob.glob(os.path.join(args.savedir, 'best_model*.pth.tar')):
                    os.remove(old)
                best_name = f'best_model_F1={max_F1_val:.4f}.pth'

                best_checkpoint_payload = {
                    'state_dict': model.state_dict(),
                    'protocol_signature': current_protocol_signature,
                    'run_config': config_dict,
                    'best_f1': max_F1_val,
                    'best_epoch': best_epoch,
                    'dataset': args.dataset,
                }

                torch.save(
                    best_checkpoint_payload,
                    os.path.join(args.savedir, best_name),
                )

            logger.write('%-6d\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-10.4f\n' %
                         (epoch + 1, lossTr, lossVal,
                          score_val['OA'], score_val['IoU'],
                          score_val['F1'], score_val['recall'],
                          score_val['precision'], max_F1_val))
            logger.flush()

            ts = datetime.datetime.now().strftime('%H:%M:%S')
            best_mark = ' ★ BEST' if is_best else ''
            summary = (f'[{ts}] Epoch {epoch + 1}/{total_epochs} | '
                       f'TrLoss={lossTr:.4f} | VaLoss={lossVal:.4f} | '
                       f'F1(train)={score_tr["F1"]:.4f} | F1(val)={score_val["F1"]:.4f} | '
                       f'best={max_F1_val:.4f} | '
                       f'elapsed={elapsed_total/60:.0f}m ETA={eta_total/60:.0f}m'
                       f'{best_mark}')
            print('\n' + summary)
            logger.write(summary + '\n')
            logger.flush()
        else:
            ts = datetime.datetime.now().strftime('%H:%M:%S')
            summary = (f'[{ts}] Epoch {epoch + 1}/{total_epochs} | '
                       f'TrLoss={lossTr:.4f} | F1(train)={score_tr["F1"]:.4f} | '
                       f'elapsed={elapsed_total/60:.0f}m ETA={eta_total/60:.0f}m '
                       f'(val skipped)')
            print('\n' + summary)
            logger.write(summary + '\n')
            logger.flush()

        # Save the state after every fully completed epoch.  os.replace keeps
        # the previous checkpoint intact if the process is killed mid-write.
        checkpoint_payload = {
            'epoch': epoch + 1,
            'cur_iter': cur_iter,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_f1': max_F1_val,
            'best_epoch': best_epoch,
            'dataset': args.dataset,
            'batch_size': args.batch_size,

            # Strict Run14 full-resume protocol.
            'protocol_signature': current_protocol_signature,

            # Full human-readable experiment configuration.
            'run_config': config_dict,

            'train_generator_state': train_generator.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state_all': (
                torch.cuda.get_rng_state_all() if args.onGPU else None
            ),
            'numpy_rng_state': np.random.get_state(),
            'python_rng_state': random.getstate(),
        }
        torch.save(checkpoint_payload, last_checkpoint_tmp)
        os.replace(last_checkpoint_tmp, last_checkpoint_path)

        torch.cuda.empty_cache()

    # ---- final test with best model ----
    best_files = sorted(glob.glob(os.path.join(args.savedir, 'best_model*.pth')))
    if best_files:
        best_path = best_files[-1]  # latest (should be only one)
        print(f"\nLoading best model: {os.path.basename(best_path)}")

        best_checkpoint = torch.load(
            best_path,
            weights_only=False,
        )

        best_state_dict = best_checkpoint.get(
            'state_dict',
            best_checkpoint,
        )

        model.load_state_dict(best_state_dict)
    else:
        raise RuntimeError(
            f"No best_model*.pth found in {args.savedir}. "
            f"Training likely failed or no validation was performed. "
            f"Cannot evaluate on test set without a validated checkpoint."
        )

    _loss_test, score_test = val(args, testLoader, model, 0)
    total_time = (time.time() - t_start) / 60

    test_summary = (
        f'  TEST RESULTS  |  '
        f'OA={score_test["OA"]:.4f}  IoU={score_test["IoU"]:.4f}  '
        f'F1={score_test["F1"]:.4f}  R={score_test["recall"]:.4f}  '
        f'P={score_test["precision"]:.4f}  |  '
        f'BestF1={max_F1_val:.4f} @ epoch {best_epoch}  |  '
        f'Time={total_time:.1f}min'
    )
    print('\n' + '=' * 68)
    print(test_summary)
    print('=' * 68)

    # ---- log SCRF alpha values if enabled ----
    if args.scale_fusion == 'scrf':
        scrf_module = getattr(model.decoder_fusion, 'scrf', None)
        if scrf_module is not None and hasattr(scrf_module, 'gammas'):
            alpha_values = []
            for i, gamma in enumerate(scrf_module.gammas):
                alpha = 1.0 + torch.tanh(gamma).item()
                alpha_values.append(f'scale{i}={alpha:.4f}')
            scrf_summary = '  SCRF Alpha: ' + ', '.join(alpha_values)
            print(scrf_summary)
            logger.write('\n' + scrf_summary + '\n')

    logger.write('\n' + test_summary + '\n')
    logger.write('\n%-6s\t-\t-\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-8.4f\t%-10.4f\n' %
                 ('Test', score_test['OA'], score_test['IoU'],
                  score_test['F1'], score_test['recall'],
                  score_test['precision'], max_F1_val))
    logger.flush()
    logger.close()

    # Run13 master scripts use this marker strictly for crash-safe resume, not
    # for result-based experiment selection.  It is created only after the
    # best-val checkpoint has completed its one held-out test evaluation.
    completion_path = os.path.join(args.savedir, '.run_complete')
    with open(completion_path, 'w', encoding='utf-8') as completion_file:
        completion_file.write(
            f'completed_at={datetime.datetime.now().isoformat()}\n'
            f'best_val_f1={max_F1_val:.6f}\n'
            f'best_epoch={best_epoch}\n'
            f'test_f1={score_test["F1"]:.6f}\n'
        )

    # A completed job is resumed via .run_complete, so its rolling training
    # checkpoint is no longer needed.  Keep only the selected best weights.
    for checkpoint_file in (last_checkpoint_path, last_checkpoint_tmp):
        if os.path.isfile(checkpoint_file):
            os.remove(checkpoint_file)

    print(f"\nBest val F1 = {max_F1_val:.4f} @ epoch {best_epoch}")
    print(f"Output: {args.savedir}")
    print(f"Completion marker: {completion_path}")


# =========================================================
#  CLI
# =========================================================

if __name__ == '__main__':
    parser = ArgumentParser(description='AEGIS-CD Training')

    # ---- dataset ----
    parser.add_argument('--dataset', default='LEVIR-CD-256',
                        help='Dataset name')
    parser.add_argument('--data-root',
                        default='/home/hzeng/project/ZH/data/CD',
                        help='Root directory containing all datasets')

    # ---- training ----
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-mode', default='poly', choices=['poly', 'step'])
    parser.add_argument('--step-loss', type=int, default=100)

    # ---- validation ----
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--val-split', default='val', choices=['val'],
                        help='Split used for checkpoint selection (must be val)')

    # ---- image ----
    parser.add_argument('--inWidth', type=int, default=256)
    parser.add_argument('--inHeight', type=int, default=256)

    # ---- system ----
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--onGPU', default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--seed', type=int, default=2333)

    # ---- I/O ----
    parser.add_argument('--savedir', default='./saved_models/baseline')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--logFile', default='trainValLog.txt')

    # ---- AEGIS module toggles ----
    parser.add_argument('--use-eaom', action='store_true',
                        help='Enable EAOM (Edge-Aware Oracle Module)')
    parser.add_argument('--use-sfif', action='store_true',
                        help='Enable SFIF (Spatial-Frequency Interactive Fusion)')
    parser.add_argument('--use-prior', action='store_true',
                        help='Enable HFC Prior Injector')
    parser.add_argument('--use-msca', action='store_true',
                        help='Enable MSCA module')
    parser.add_argument('--use-stargate', action='store_true',
                        help='Enable StarGate module (Star-Operation cross-temporal gating)')
    parser.add_argument('--use-edgegate', action='store_true',
                        help='Enable EdgeGate module (Edge-Guided Boundary Refinement)')
    parser.add_argument(
        '--grmsa-mode', default='off',
        choices=['off', 'mask', 'residual', 'full'],
        help='Decoder MSA ablation: off, mask fix, residual fix, or full GR-MSA',
    )
    parser.add_argument(
        '--decoder-mode', default='msa',
        choices=['msa', 'plain_dw', 'rep_dw', 'rep_dw_shared'],
        help='Decoder implementation (Run10 uses plain_dw/rep_dw variants)',
    )
    parser.add_argument(
        '--head-mode', default='shared',
        choices=['shared', 'independent'],
        help='Prediction-head sharing across decoder scales (Run11)',
    )
    parser.add_argument(
        '--ds-profile', default='legacy',
        choices=['legacy', 'primary', 'main_only'],
        help='Deep-supervision weight profile (Run11)',
    )

    # ---- Run13 explicit architecture/supervision modes ----
    parser.add_argument(
        '--diff-mode', default=None, choices=['cfdm', 'eaom', 'sdtr'],
        help='Run13 difference encoder. Omit to preserve legacy --use-eaom/CFDM behavior.',
    )
    parser.add_argument(
        '--diff-sharing', default='shared',
        choices=['shared', 'independent'],
        help='Share one difference encoder or use four scale-specific instances.',
    )
    parser.add_argument(
        '--sdtr-scope', default='all',
        choices=['all', 'shallow', 'deep'],
        help=(
            'Legacy Run13 SDTR scope used only with --diff-mode sdtr: '
            'all, shallow, or deep. '
            'Run14 should prefer --temporal-relation-mode.'
        ),
    )
    parser.add_argument(
        '--temporal-relation-mode', default=None,
        choices=['off', 'shallow_replace', 'deep_replace', 'deep_residual'],
        help=(
            'Run14 temporal-relation mode. '
            'Use with --diff-mode eaom and --diff-sharing independent. '
            'Omit for historical Run13 compatibility.'
        ),
    )
    parser.add_argument(
        '--supervision-mode', default='legacy',
        choices=['legacy', 'native'],
        help='legacy full-resolution DS or native-resolution SCDS.',
    )
    parser.add_argument(
        '--dice-reduction', default='batch_global',
        choices=['batch_global', 'per_image'],
        help='Soft-Dice reduction. Run13 fixes batch_global for a clean Run12 control.',
    )
    parser.add_argument(
        '--amp-phase-mode', default='off',
        choices=['off', 'lf_shared'],
        help='APID-LF feature amplitude/phase mode.',
    )
    parser.add_argument(
        '--use-lfds', action='store_true',
        help='Enable the training-only local frequency prediction head.',
    )
    parser.add_argument(
        '--use-tct', action='store_true',
        help='Enable Temporal Change Tokens at 16x16 and 8x8.',
    )
    parser.add_argument(
        '--encoder-fusion-mode', default='hfea',
        choices=['hfea', 'rephfea_pyr'],
        help='Legacy HFEA or Run13 RepHFEA-Pyramid.',
    )
    parser.add_argument(
        '--boundary-mode', default=None, choices=['off', 'edgegate', 'bdsr'],
        help='Run13 boundary refinement. Omit to preserve --use-edgegate/off.',
    )
    parser.add_argument(
        '--consistency-mode', default='off', choices=['off', 'baic'],
        help='Training-time batch amplitude-invariance consistency.',
    )

    # ---- Run12 parameters ----
    parser.add_argument(
        '--scale-fusion', default='plain',
        choices=['plain', 'scrf'],
        help='Scale fusion mode: plain (fixed +) or scrf (learnable calibration)',
    )
    parser.add_argument(
        '--deterministic', action='store_true',
        help='Enable strict deterministic mode (disable cudnn.benchmark)',
    )
    parser.add_argument(
        '--color-order', default='legacy',
        choices=['legacy', 'fixed'],
        help='ToTensor color order: legacy (Run10/11 bug) or fixed (correct T1/T2). '
             'Default legacy for historical compatibility; Run12 scripts should explicitly use fixed.',
    )
    parser.add_argument(
        '--amp-mix', action='store_true',
        help='Enable amplitude-invariant augmentation (AmpMix)',
    )

    args = parser.parse_args()
    trainValidateSegmentation(args)
