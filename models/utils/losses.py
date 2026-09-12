"""
Shared loss utilities — single source of truth for BCEDiceLoss and
deep-supervision weight profiles.

Both ``train.py`` and ``test.py`` import from here so the DS profiles can
never silently drift apart.
"""

import torch.nn.functional as F


DS_PROFILES = {
    # Default deep supervision: progressively weaker auxiliary heads.
    'legacy':   (1.0, 0.8, 0.4, 0.2),
    # Primary-output-oriented deep supervision: keep the main loss
    # coefficient fixed at 1.0 and reduce auxiliary pressure.
    'primary':  (1.0, 0.5, 0.25, 0.125),
    # Diagnostic anchor: remove direct supervision from f2/f3/f4.
    'main_only': (1.0, 0.0, 0.0, 0.0),
}


VALID_DICE_REDUCTIONS = ('batch_global', 'per_image')


def BCEDiceLoss(inputs, targets, dice_reduction='batch_global'):
    """Binary cross-entropy plus selectable soft-Dice reduction.

    ``batch_global`` accumulates the Dice numerator and denominator over the
    whole batch and is the default training protocol. ``per_image`` computes
    Dice per sample and averages, available as an explicit alternative rather
    than a silent code drift.
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
            each prediction resolution (SCDS).
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


def scds_supervised_loss(outputs, targets, profile='legacy',
                          supervision_mode='legacy',
                          dice_reduction='batch_global'):
    """SCDS deep-supervision loss.

    Returns ``(total, components)`` so training logs can audit the loss.
    """
    components = {
        'scds': deep_supervision_loss(
            outputs, targets, profile, supervision_mode, dice_reduction
        )
    }
    total = components['scds']
    return total, components
