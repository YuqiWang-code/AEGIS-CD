"""
Shared loss utilities — single source of truth for BCEDiceLoss and
deep-supervision weight profiles (Run11).

Both ``train.py`` and ``test.py`` import from here so the DS profiles can
never silently drift apart.
"""

import torch.nn.functional as F


DS_PROFILES = {
    # Historical Run10 behaviour.
    'legacy':   (1.0, 0.8, 0.4, 0.2),
    # Run11: primary-output-oriented deep supervision.
    # Keep main loss coefficient fixed at 1.0 and reduce auxiliary pressure.
    'primary':  (1.0, 0.5, 0.25, 0.125),
    # Diagnostic anchor: remove direct supervision from f2/f3/f4.
    'main_only': (1.0, 0.0, 0.0, 0.0),
}


VALID_DICE_REDUCTIONS = ('batch_global', 'per_image')


def BCEDiceLoss(inputs, targets, dice_reduction='batch_global'):
    """Binary cross-entropy plus selectable soft-Dice reduction.

    ``batch_global`` exactly preserves the historical Run12 loss and is the
    fixed Run13 protocol, keeping E1 a clean control. ``per_image`` remains
    available as an explicit future ablation rather than a silent code drift.
    """
    if inputs.shape != targets.shape:
        raise ValueError(
            f'BCEDiceLoss shape mismatch: inputs={tuple(inputs.shape)} '
            f'targets={tuple(targets.shape)}'
        )
    targets = targets.to(dtype=inputs.dtype)
    if dice_reduction not in VALID_DICE_REDUCTIONS:
        raise ValueError(
            f'Unknown dice_reduction={dice_reduction!r}; '
            f'expected one of {VALID_DICE_REDUCTIONS}'
        )
    bce = F.binary_cross_entropy(inputs, targets)

    eps = 1e-5
    if dice_reduction == 'batch_global':
        inter = (inputs * targets).sum()
        dice_loss = 1 - (
            (2 * inter + eps) /
            (inputs.sum() + targets.sum() + eps)
        )
    else:
        inputs_flat = inputs.flatten(1)
        targets_flat = targets.flatten(1)
        inter = (inputs_flat * targets_flat).sum(dim=1)
        dice_per_image = (
            (2 * inter + eps) /
            (inputs_flat.sum(dim=1) + targets_flat.sum(dim=1) + eps)
        )
        dice_loss = (1 - dice_per_image).mean()

    return bce + dice_loss


def deep_supervision_loss(outputs, targets, profile,
                          supervision_mode='legacy',
                          dice_reduction='batch_global'):
    """Weighted sum of BCEDiceLoss over the four decoder outputs.

    Args:
        outputs: (output1, output2, output3, output4), all [B,1,H,W]
            probabilities after sigmoid.
        targets: [B,1,H,W] float full-resolution targets.
        profile: one of DS_PROFILES keys.
        supervision_mode: ``legacy`` supervises four full-resolution
            predictions. ``native`` area-averages the target independently to
            each prediction resolution (SCDS, Run13).
    """
    if profile not in DS_PROFILES:
        raise ValueError(
            f'Unknown ds_profile={profile!r}; '
            f'expected one of {tuple(DS_PROFILES.keys())}'
        )
    if supervision_mode not in ('legacy', 'native'):
        raise ValueError(
            f'Unknown supervision_mode={supervision_mode!r}; '
            f'expected legacy or native'
        )

    weights = DS_PROFILES[profile]

    if len(outputs) != len(weights):
        raise ValueError(
            f'Expected {len(weights)} decoder outputs, '
            f'got {len(outputs)}'
        )

    loss = outputs[0].new_zeros(())

    for weight, output in zip(weights, outputs):
        # Important for main_only: do not even build the auxiliary loss
        # graph when its weight is zero.
        if weight > 0.0:
            if supervision_mode == 'native':
                target_i = F.adaptive_avg_pool2d(
                    targets.to(dtype=output.dtype), output.shape[-2:]
                )
            else:
                if output.shape[-2:] != targets.shape[-2:]:
                    raise ValueError(
                        'legacy supervision requires every output at target '
                        f'resolution; got output={tuple(output.shape)} and '
                        f'target={tuple(targets.shape)}'
                    )
                target_i = targets
            loss = loss + weight * BCEDiceLoss(
                output, target_i, dice_reduction=dice_reduction
            )

    return loss


def boundary_target(targets, output_size, kernel_size=3):
    """Create ``dilate(Y) - erode(Y)`` and area-pool it to output_size."""
    if kernel_size % 2 != 1 or kernel_size < 3:
        raise ValueError('kernel_size must be an odd integer >= 3')
    targets = targets.float()
    padding = kernel_size // 2
    dilated = F.max_pool2d(
        targets, kernel_size, stride=1, padding=padding
    )
    eroded = 1.0 - F.max_pool2d(
        1.0 - targets, kernel_size, stride=1, padding=padding
    )
    boundary = (dilated - eroded).clamp_(0.0, 1.0)
    return F.adaptive_avg_pool2d(boundary, output_size)


def run13_supervised_loss(outputs, targets, profile='legacy',
                          supervision_mode='legacy', aux=None,
                          lambda_freq=0.2, lambda_boundary=0.2,
                          dice_reduction='batch_global'):
    """SCDS plus the optional LFDS and supervised BDSR losses.

    Returns ``(total, components)`` so training logs can audit every fixed
    Run13 coefficient without duplicating the loss definitions in scripts.
    BAIC consistency is intentionally added by the training loop because it
    requires a second augmented forward pass.
    """
    components = {
        'scds': deep_supervision_loss(
            outputs, targets, profile, supervision_mode, dice_reduction
        )
    }
    total = components['scds']
    aux = aux or {}

    if 'frequency' in aux:
        prediction = aux['frequency']
        freq_target = F.adaptive_avg_pool2d(
            targets.to(dtype=prediction.dtype), prediction.shape[-2:]
        )
        components['frequency'] = BCEDiceLoss(
            prediction, freq_target, dice_reduction=dice_reduction
        )
        total = total + lambda_freq * components['frequency']

    if 'boundary' in aux:
        prediction = aux['boundary']
        target_b = boundary_target(targets, prediction.shape[-2:]).to(
            dtype=prediction.dtype
        )
        components['boundary'] = BCEDiceLoss(
            prediction, target_b, dice_reduction=dice_reduction
        )
        total = total + lambda_boundary * components['boundary']

    return total, components
