"""
Run14 protocol / reproducibility preflight tests.

Coverage
--------
1. Protocol signature contains every Run14-critical experiment field.
2. Exact protocol matches are accepted.
3. Protocol mismatches fail fast, including:
   - dataset
   - color order
   - diff mode / sharing / SDTR scope
   - temporal-relation mode / plan
   - supervision mode
   - Dice reduction
   - MSCA / Prior / TCT
   - decoder / head / DS profile
   - seed
4. Fixed color-order preserves T1/T2 identity while converting BGR -> RGB.
5. Native supervision area-pools GT independently to every output scale.
6. Per-image Dice implements per-sample Dice averaging rather than
   batch-global aggregation.
7. RNG checkpoint state restores the exact next Python / NumPy / Torch /
   DataLoader-generator random sequence.
8. Run14 best checkpoints retain protocol and best-validation metadata.
9. Evaluation restores protocol metadata from checkpoints.
10. Explicit evaluation CLI values conflicting with checkpoint metadata
    fail fast.

All tests are CPU-only and do not require datasets, pretrained weights,
or a GPU.
"""

from argparse import Namespace
import copy
import importlib.util
import inspect
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F


# ===========================================================================
# Repository imports
# ===========================================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from models.data.Transforms import ToTensor
import models.utils.losses as losses_module
from models.utils.losses import (
    BCEDiceLoss,
    deep_supervision_loss,
)


def load_script_module(module_name, relative_path):
    """Load a script file without executing its __main__ CLI block."""
    path = ROOT / relative_path

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f'Unable to import script module from {path}'
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


train_script = load_script_module(
    'aegis_train_script_for_protocol_tests',
    'models/scripts/train.py',
)

test_script = load_script_module(
    'aegis_test_script_for_protocol_tests',
    'models/scripts/test.py',
)


# ===========================================================================
# Helpers
# ===========================================================================

def make_run14_args(**overrides):
    """Construct the Run14 R0/E4-style protocol namespace."""
    values = {
        # Dataset / input
        'dataset': 'SYSU-CD-256',
        'inWidth': 256,
        'inHeight': 256,
        'batch_size': 64,
        'color_order': 'fixed',
        'val_split': 'val',

        # Optimisation / reproducibility
        'epochs': 200,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'lr_mode': 'poly',
        'step_loss': 100,
        'val_interval': 10,
        'seed': 2333,
        'deterministic': True,
        'num_workers': 4,

        # Augmentation
        'amp_mix': False,

        # Difference / temporal relation
        'diff_mode': 'eaom',
        'diff_sharing': 'independent',
        'sdtr_scope': 'all',
        'temporal_relation_mode': 'off',

        # Supervision
        'supervision_mode': 'native',
        'dice_reduction': 'batch_global',
        'ds_profile': 'legacy',

        # Architecture
        'use_prior': False,
        'use_msca': False,
        'use_tct': False,
        'use_lfds': False,
        'amp_phase_mode': 'off',
        'encoder_fusion_mode': 'hfea',
        'decoder_mode': 'rep_dw',
        'head_mode': 'independent',
        'scale_fusion': 'plain',
        'boundary_mode': 'edgegate',
        'consistency_mode': 'off',

        # Historical compatibility switches
        'use_sfif': False,
        'use_stargate': False,
        'grmsa_mode': 'off',

        # Compatibility booleans used by evaluation restore
        'use_eaom': True,
        'use_edgegate': True,
    }

    values.update(
        overrides
    )

    return Namespace(
        **values
    )


def make_dummy_model(
    base_diff_mode='eaom',
    temporal_relation_plan=None,
):
    """Minimal model-like object required by build_protocol_signature()."""
    if temporal_relation_plan is None:
        temporal_relation_plan = (
            'off',
            'off',
            'off',
            'off',
        )

    return Namespace(
        base_diff_mode=base_diff_mode,
        temporal_relation_plan=tuple(
            temporal_relation_plan
        ),
    )


def build_signature(
    args=None,
    model=None,
):
    if args is None:
        args = make_run14_args()

    if model is None:
        model = make_dummy_model()

    return train_script.build_protocol_signature(
        args,
        model,
    )


def clone_signature(signature):
    return copy.deepcopy(
        signature
    )


def make_eval_args(**overrides):
    """Minimal Namespace needed by apply_checkpoint_protocol()."""
    values = {
        'dataset': 'LEVIR-CD-256',

        'inWidth': 256,
        'inHeight': 256,

        'color_order': 'legacy',

        'diff_mode': 'eaom',
        'diff_sharing': 'independent',
        'sdtr_scope': 'all',
        'temporal_relation_mode': 'off',

        'supervision_mode': 'legacy',
        'dice_reduction': 'batch_global',
        'ds_profile': 'legacy',

        'use_prior': False,
        'use_msca': False,
        'use_tct': False,
        'use_lfds': False,

        'amp_phase_mode': 'off',
        'encoder_fusion_mode': 'hfea',
        'decoder_mode': 'rep_dw',
        'head_mode': 'independent',
        'scale_fusion': 'plain',
        'boundary_mode': 'off',
        'consistency_mode': 'off',

        'use_sfif': False,
        'use_stargate': False,
        'grmsa_mode': 'off',

        'seed': 2333,
        'deterministic': False,

        'use_eaom': True,
        'use_edgegate': False,
    }

    values.update(
        overrides
    )

    return Namespace(
        **values
    )


# ===========================================================================
# 1. Protocol signature schema
# ===========================================================================

def test_protocol_signature_version_is_current():
    assert (
        train_script.PROTOCOL_SIGNATURE_VERSION
        == 1
    )

    signature = build_signature()

    assert signature['version'] == 1


def test_protocol_signature_contains_run14_critical_fields():
    signature = build_signature()

    required = {
        # Dataset / input
        'version',
        'dataset',
        'in_width',
        'in_height',
        'batch_size',
        'color_order',
        'val_split',

        # Training / reproducibility
        'epochs',
        'lr',
        'weight_decay',
        'lr_mode',
        'step_loss',
        'val_interval',
        'seed',
        'deterministic',
        'num_workers',
        'amp_mix',

        # Difference / relation
        'diff_mode',
        'base_diff_mode',
        'diff_sharing',
        'sdtr_scope',
        'temporal_relation_mode',
        'temporal_relation_plan',

        # Supervision
        'supervision_mode',
        'dice_reduction',
        'ds_profile',

        # Architecture
        'use_prior',
        'use_msca',
        'use_tct',
        'use_lfds',
        'amp_phase_mode',
        'encoder_fusion_mode',
        'decoder_mode',
        'head_mode',
        'scale_fusion',
        'boundary_mode',
        'consistency_mode',

        # Historical compatibility
        'use_sfif',
        'use_stargate',
        'grmsa_mode',

        # Fixed loss coefficients
        'lambda_freq',
        'lambda_boundary',
        'lambda_consistency',
    }

    missing = required - set(
        signature.keys()
    )

    assert not missing, (
        f'protocol_signature is missing fields: '
        f'{sorted(missing)}'
    )


def test_protocol_signature_records_run14_r1_per_image():
    args = make_run14_args(
        dice_reduction='per_image',
    )

    signature = build_signature(
        args=args,
    )

    assert (
        signature['dice_reduction']
        == 'per_image'
    )


def test_protocol_signature_records_deep_residual_plan():
    args = make_run14_args(
        temporal_relation_mode='deep_residual',
    )

    model = make_dummy_model(
        temporal_relation_plan=(
            'off',
            'off',
            'deep_residual',
            'deep_residual',
        ),
    )

    signature = build_signature(
        args=args,
        model=model,
    )

    assert (
        signature['temporal_relation_mode']
        == 'deep_residual'
    )

    assert signature['temporal_relation_plan'] == [
        'off',
        'off',
        'deep_residual',
        'deep_residual',
    ]


# ===========================================================================
# 2. Exact protocol matching
# ===========================================================================

def test_identical_protocol_signature_is_accepted():
    current = build_signature()

    checkpoint = clone_signature(
        current
    )

    result = train_script.validate_protocol_signature(
        checkpoint,
        current,
    )

    assert result is None


@pytest.mark.parametrize(
    'field,new_value',
    (
        ('dataset', 'LEVIR-CD-256'),
        ('color_order', 'legacy'),
        ('diff_mode', 'cfdm'),
        ('diff_sharing', 'shared'),
        ('sdtr_scope', 'deep'),
        (
            'temporal_relation_mode',
            'deep_residual',
        ),
        ('supervision_mode', 'legacy'),
        ('dice_reduction', 'per_image'),
        ('use_msca', True),
        ('use_prior', True),
        ('use_tct', True),
        ('decoder_mode', 'msa'),
        ('head_mode', 'shared'),
        ('ds_profile', 'primary'),
        ('seed', 3407),
    ),
)
def test_protocol_mismatch_fails_fast(
    field,
    new_value,
):
    current = build_signature()

    checkpoint = clone_signature(
        current
    )

    checkpoint[field] = new_value

    with pytest.raises(
        ValueError,
        match='Checkpoint protocol mismatch',
    ):
        train_script.validate_protocol_signature(
            checkpoint,
            current,
        )


def test_temporal_relation_plan_mismatch_fails_fast():
    current = build_signature()

    checkpoint = clone_signature(
        current
    )

    checkpoint['temporal_relation_plan'] = [
        'off',
        'off',
        'deep_replace',
        'deep_replace',
    ]

    with pytest.raises(
        ValueError,
        match='Checkpoint protocol mismatch',
    ):
        train_script.validate_protocol_signature(
            checkpoint,
            current,
        )


def test_missing_protocol_field_fails_fast():
    current = build_signature()

    checkpoint = clone_signature(
        current
    )

    checkpoint.pop(
        'color_order'
    )

    with pytest.raises(
        ValueError,
        match='Checkpoint protocol mismatch',
    ):
        train_script.validate_protocol_signature(
            checkpoint,
            current,
        )


def test_protocol_signature_must_be_dictionary():
    current = build_signature()

    with pytest.raises(
        ValueError,
        match='not a dictionary',
    ):
        train_script.validate_protocol_signature(
            None,
            current,
        )


# ===========================================================================
# 3. Fixed ToTensor color order
# ===========================================================================

def test_fixed_color_order_preserves_temporal_identity():
    # Input convention before ToTensor:
    #
    #   T1 = [B1, G1, R1] = [1, 2, 3]
    #   T2 = [B2, G2, R2] = [4, 5, 6]
    #
    # Correct fixed result:
    #
    #   T1 = [R1, G1, B1] = [3, 2, 1]
    #   T2 = [R2, G2, B2] = [6, 5, 4]
    image = np.array(
        [
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ]
            ]
        ],
        dtype=np.float32,
    )

    label = np.zeros(
        (1, 1),
        dtype=np.uint8,
    )

    transform = ToTensor(
        color_order='fixed',
    )

    image_tensor, _label_tensor = transform(
        image.copy(),
        label.copy(),
    )

    actual = image_tensor[
        :,
        0,
        0,
    ]

    expected = torch.tensor(
        [
            3.0,
            2.0,
            1.0,
            6.0,
            5.0,
            4.0,
        ],
        dtype=image_tensor.dtype,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_legacy_color_order_reproduces_historical_temporal_swap():
    image = np.array(
        [
            [
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                ]
            ]
        ],
        dtype=np.float32,
    )

    label = np.zeros(
        (1, 1),
        dtype=np.uint8,
    )

    transform = ToTensor(
        color_order='legacy',
    )

    image_tensor, _label_tensor = transform(
        image.copy(),
        label.copy(),
    )

    actual = image_tensor[
        :,
        0,
        0,
    ]

    expected = torch.tensor(
        [
            6.0,
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
        ],
        dtype=image_tensor.dtype,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_fixed_and_legacy_color_orders_are_not_equivalent():
    image = np.array(
        [
            [
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                ]
            ]
        ],
        dtype=np.float32,
    )

    label = np.zeros(
        (1, 1),
        dtype=np.uint8,
    )

    fixed, _ = ToTensor(
        color_order='fixed',
    )(
        image.copy(),
        label.copy(),
    )

    legacy, _ = ToTensor(
        color_order='legacy',
    )(
        image.copy(),
        label.copy(),
    )

    assert not torch.equal(
        fixed,
        legacy,
    )


# ===========================================================================
# 4. Native SCDS supervision
# ===========================================================================

def test_native_supervision_area_pools_target_to_each_output_scale(
    monkeypatch,
):
    captured_targets = []

    def fake_bce_dice(
        inputs,
        targets,
        dice_reduction='batch_global',
    ):
        del dice_reduction

        captured_targets.append(
            targets.detach().clone()
        )

        return inputs.sum() * 0.0

    monkeypatch.setattr(
        losses_module,
        'BCEDiceLoss',
        fake_bce_dice,
    )

    target = torch.arange(
        64,
        dtype=torch.float32,
    ).reshape(
        1,
        1,
        8,
        8,
    )

    target = target / 63.0

    outputs = (
        torch.full(
            (1, 1, 8, 8),
            0.5,
        ),
        torch.full(
            (1, 1, 4, 4),
            0.5,
        ),
        torch.full(
            (1, 1, 2, 2),
            0.5,
        ),
        torch.full(
            (1, 1, 1, 1),
            0.5,
        ),
    )

    loss = deep_supervision_loss(
        outputs,
        target,
        profile='legacy',
        supervision_mode='native',
        dice_reduction='per_image',
    )

    assert torch.isfinite(
        loss
    )

    assert len(
        captured_targets
    ) == 4

    for captured, output in zip(
        captured_targets,
        outputs,
    ):
        expected = F.adaptive_avg_pool2d(
            target,
            output.shape[-2:],
        )

        torch.testing.assert_close(
            captured,
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_native_supervision_accepts_soft_area_targets():
    target = torch.zeros(
        1,
        1,
        8,
        8,
    )

    # Four changed pixels in one 4x4 pooling region produce a fractional
    # occupancy target after native area pooling.
    target[
        :,
        :,
        0:2,
        0:2,
    ] = 1.0

    outputs = (
        torch.full(
            (1, 1, 8, 8),
            0.5,
        ),
        torch.full(
            (1, 1, 4, 4),
            0.5,
        ),
        torch.full(
            (1, 1, 2, 2),
            0.5,
        ),
        torch.full(
            (1, 1, 1, 1),
            0.5,
        ),
    )

    loss = deep_supervision_loss(
        outputs,
        target,
        profile='legacy',
        supervision_mode='native',
        dice_reduction='batch_global',
    )

    assert loss.ndim == 0
    assert torch.isfinite(
        loss
    )


# ===========================================================================
# 5. Per-image Dice
# ===========================================================================

def test_per_image_dice_matches_manual_formula():
    inputs = torch.tensor(
        [
            [
                [
                    [0.90, 0.80],
                    [0.70, 0.60],
                ]
            ],
            [
                [
                    [0.80, 0.70],
                    [0.60, 0.50],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    actual = BCEDiceLoss(
        inputs,
        targets,
        dice_reduction='per_image',
    )

    bce = F.binary_cross_entropy(
        inputs,
        targets,
    )

    eps = 1e-5

    inputs_flat = inputs.flatten(
        1
    )

    targets_flat = targets.flatten(
        1
    )

    intersection = (
        inputs_flat
        * targets_flat
    ).sum(
        dim=1
    )

    dice_per_image = (
        2.0 * intersection
        + eps
    ) / (
        inputs_flat.sum(dim=1)
        + targets_flat.sum(dim=1)
        + eps
    )

    expected = (
        bce
        + (
            1.0
            - dice_per_image
        ).mean()
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_per_image_and_batch_global_dice_are_distinct():
    inputs = torch.tensor(
        [
            [
                [
                    [0.95, 0.90],
                    [0.85, 0.80],
                ]
            ],
            [
                [
                    [0.80, 0.75],
                    [0.70, 0.65],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [1.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    per_image = BCEDiceLoss(
        inputs,
        targets,
        dice_reduction='per_image',
    )

    batch_global = BCEDiceLoss(
        inputs,
        targets,
        dice_reduction='batch_global',
    )

    assert not torch.isclose(
        per_image,
        batch_global,
        rtol=1e-6,
        atol=1e-7,
    )


def test_invalid_dice_reduction_fails_fast():
    inputs = torch.full(
        (1, 1, 2, 2),
        0.5,
    )

    targets = torch.zeros_like(
        inputs
    )

    with pytest.raises(
        ValueError,
        match='Unknown dice_reduction',
    ):
        BCEDiceLoss(
            inputs,
            targets,
            dice_reduction='invalid',
        )


# ===========================================================================
# 6. Resume RNG state
# ===========================================================================

def test_resume_rng_state_round_trip(tmp_path):
    seed = 2333

    random.seed(
        seed
    )
    np.random.seed(
        seed
    )
    torch.manual_seed(
        seed
    )

    train_generator = train_script.make_generator(
        seed
    )

    # Consume an initial sequence to represent completed training work.
    random.random()
    np.random.random()
    torch.rand(1)
    torch.rand(
        1,
        generator=train_generator,
    )

    checkpoint_payload = {
        'train_generator_state':
            train_generator.get_state(),

        'torch_rng_state':
            torch.get_rng_state(),

        'cuda_rng_state_all':
            None,

        'numpy_rng_state':
            np.random.get_state(),

        'python_rng_state':
            random.getstate(),
    }

    checkpoint_path = (
        tmp_path
        / 'last_checkpoint.pth.tar'
    )

    torch.save(
        checkpoint_payload,
        checkpoint_path,
    )

    # This is the exact sequence that an uninterrupted run would see next.
    expected_python = [
        random.random()
        for _ in range(4)
    ]

    expected_numpy = np.random.random(
        4
    )

    expected_torch = torch.rand(
        4
    )

    expected_loader = torch.rand(
        4,
        generator=train_generator,
    )

    # Disturb every state.
    for _ in range(10):
        random.random()

    np.random.random(
        10
    )

    torch.rand(
        10
    )

    torch.rand(
        10,
        generator=train_generator,
    )

    # Simulate a fresh process and restore exactly as train.py does.
    restored = torch.load(
        checkpoint_path,
        weights_only=False,
    )

    resumed_generator = train_script.make_generator(
        0
    )

    resumed_generator.set_state(
        restored['train_generator_state']
    )

    torch.set_rng_state(
        restored['torch_rng_state']
    )

    np.random.set_state(
        restored['numpy_rng_state']
    )

    random.setstate(
        restored['python_rng_state']
    )

    actual_python = [
        random.random()
        for _ in range(4)
    ]

    actual_numpy = np.random.random(
        4
    )

    actual_torch = torch.rand(
        4
    )

    actual_loader = torch.rand(
        4,
        generator=resumed_generator,
    )

    assert actual_python == pytest.approx(
        expected_python,
        rel=0.0,
        abs=0.0,
    )

    np.testing.assert_array_equal(
        actual_numpy,
        expected_numpy,
    )

    torch.testing.assert_close(
        actual_torch,
        expected_torch,
        rtol=0.0,
        atol=0.0,
    )

    torch.testing.assert_close(
        actual_loader,
        expected_loader,
        rtol=0.0,
        atol=0.0,
    )


def test_training_resume_source_restores_all_cpu_rng_states():
    source = inspect.getsource(
        train_script.trainValidateSegmentation
    )

    required_restore_calls = (
        'train_generator.set_state',
        'torch.set_rng_state',
        'np.random.set_state',
        'random.setstate',
    )

    for expression in required_restore_calls:
        assert expression in source, (
            f'Training resume no longer restores '
            f'{expression}'
        )


def test_training_checkpoint_source_saves_all_rng_states():
    source = inspect.getsource(
        train_script.trainValidateSegmentation
    )

    required_checkpoint_fields = (
        "'train_generator_state'",
        "'torch_rng_state'",
        "'cuda_rng_state_all'",
        "'numpy_rng_state'",
        "'python_rng_state'",
    )

    for field in required_checkpoint_fields:
        assert field in source, (
            f'Training checkpoint no longer saves '
            f'{field}'
        )


# ===========================================================================
# 7. Best-validation checkpoint metadata
# ===========================================================================

def test_training_source_best_checkpoint_contains_required_metadata():
    source = inspect.getsource(
        train_script.trainValidateSegmentation
    )

    assert 'best_checkpoint_payload = {' in source

    best_block = source.split(
        'best_checkpoint_payload = {',
        1,
    )[1].split(
        '}',
        1,
    )[0]

    required_fields = (
        "'state_dict'",
        "'protocol_signature'",
        "'run_config'",
        "'best_f1'",
        "'best_epoch'",
        "'dataset'",
    )

    for field in required_fields:
        assert field in best_block, (
            f'Best checkpoint metadata missing '
            f'{field}'
        )


def test_metadata_aware_best_checkpoint_round_trip(
    tmp_path,
):
    signature = build_signature()

    state_dict = {
        'dummy.weight': torch.tensor(
            [1.0, 2.0, 3.0]
        ),
    }

    payload = {
        'state_dict': state_dict,
        'protocol_signature': signature,
        'run_config': {
            'protocol_signature':
                signature,
        },
        'best_f1': 0.8388,
        'best_epoch': 100,
        'dataset': 'SYSU-CD-256',
    }

    checkpoint_path = (
        tmp_path
        / 'best_model_F1=0.8388.pth'
    )

    torch.save(
        payload,
        checkpoint_path,
    )

    loaded_state_dict, loaded_signature = (
        test_script.load_checkpoint_for_evaluation(
            str(checkpoint_path)
        )
    )

    assert loaded_signature == signature

    torch.testing.assert_close(
        loaded_state_dict['dummy.weight'],
        state_dict['dummy.weight'],
        rtol=0.0,
        atol=0.0,
    )

    raw = torch.load(
        checkpoint_path,
        weights_only=False,
    )

    assert raw['best_f1'] == pytest.approx(
        0.8388
    )

    assert raw['best_epoch'] == 100

    assert (
        raw['dataset']
        == 'SYSU-CD-256'
    )


def test_checkpoint_loader_recovers_signature_from_run_config(
    tmp_path,
):
    signature = build_signature()

    payload = {
        'state_dict': {
            'dummy.weight':
                torch.tensor([1.0]),
        },
        'run_config': {
            'protocol_signature':
                signature,
        },
    }

    checkpoint_path = (
        tmp_path
        / 'metadata_nested.pth'
    )

    torch.save(
        payload,
        checkpoint_path,
    )

    _state_dict, loaded_signature = (
        test_script.load_checkpoint_for_evaluation(
            str(checkpoint_path)
        )
    )

    assert loaded_signature == signature


def test_legacy_raw_state_dict_checkpoint_has_no_signature(
    tmp_path,
):
    raw_state_dict = {
        'dummy.weight':
            torch.tensor([1.0]),
    }

    checkpoint_path = (
        tmp_path
        / 'legacy_best_model.pth'
    )

    torch.save(
        raw_state_dict,
        checkpoint_path,
    )

    loaded_state_dict, loaded_signature = (
        test_script.load_checkpoint_for_evaluation(
            str(checkpoint_path)
        )
    )

    assert loaded_signature is None

    torch.testing.assert_close(
        loaded_state_dict['dummy.weight'],
        raw_state_dict['dummy.weight'],
        rtol=0.0,
        atol=0.0,
    )


# ===========================================================================
# 8. Evaluation protocol restoration
# ===========================================================================

def test_evaluation_restores_run14_protocol_from_checkpoint(
    monkeypatch,
):
    args = make_eval_args()

    signature = {
        'dataset':
            'SYSU-CD-256',

        'in_width':
            256,

        'in_height':
            256,

        'color_order':
            'fixed',

        'diff_mode':
            'eaom',

        'diff_sharing':
            'independent',

        'sdtr_scope':
            'all',

        'temporal_relation_mode':
            'deep_residual',

        'supervision_mode':
            'native',

        'dice_reduction':
            'per_image',

        'ds_profile':
            'legacy',

        'use_prior':
            True,

        'use_msca':
            True,

        'use_tct':
            False,

        'use_lfds':
            False,

        'amp_phase_mode':
            'off',

        'encoder_fusion_mode':
            'hfea',

        'decoder_mode':
            'rep_dw',

        'head_mode':
            'independent',

        'scale_fusion':
            'plain',

        'boundary_mode':
            'edgegate',

        'consistency_mode':
            'off',

        'use_sfif':
            False,

        'use_stargate':
            False,

        'grmsa_mode':
            'off',

        'seed':
            3407,

        'deterministic':
            True,
    }

    monkeypatch.setattr(
        sys,
        'argv',
        ['test.py'],
    )

    test_script.apply_checkpoint_protocol(
        args,
        signature,
    )

    assert args.dataset == 'SYSU-CD-256'
    assert args.color_order == 'fixed'

    assert args.diff_mode == 'eaom'
    assert args.diff_sharing == 'independent'
    assert args.sdtr_scope == 'all'

    assert (
        args.temporal_relation_mode
        == 'deep_residual'
    )

    assert (
        args.supervision_mode
        == 'native'
    )

    assert (
        args.dice_reduction
        == 'per_image'
    )

    assert args.use_prior is True
    assert args.use_msca is True

    assert (
        args.decoder_mode
        == 'rep_dw'
    )

    assert (
        args.head_mode
        == 'independent'
    )

    assert args.seed == 3407
    assert args.deterministic is True

    # Historical compatibility flags are synchronised by test.py.
    assert args.use_eaom is True
    assert args.use_edgegate is True


@pytest.mark.parametrize(
    'argv,argument_name,checkpoint_value',
    (
        (
            [
                'test.py',
                '--color-order',
                'legacy',
            ],
            'color_order',
            'fixed',
        ),
        (
            [
                'test.py',
                '--temporal-relation-mode',
                'deep_replace',
            ],
            'temporal_relation_mode',
            'deep_residual',
        ),
        (
            [
                'test.py',
                '--dice-reduction',
                'batch_global',
            ],
            'dice_reduction',
            'per_image',
        ),
        (
            [
                'test.py',
                '--seed',
                '2333',
            ],
            'seed',
            3407,
        ),
    ),
)
def test_explicit_eval_cli_conflict_fails_fast(
    monkeypatch,
    argv,
    argument_name,
    checkpoint_value,
):
    args = make_eval_args()

    # Match the explicitly supplied CLI value represented by argv.
    if argument_name == 'color_order':
        args.color_order = 'legacy'

    elif argument_name == 'temporal_relation_mode':
        args.temporal_relation_mode = (
            'deep_replace'
        )

    elif argument_name == 'dice_reduction':
        args.dice_reduction = (
            'batch_global'
        )

    elif argument_name == 'seed':
        args.seed = 2333

    monkeypatch.setattr(
        sys,
        'argv',
        argv,
    )

    signature = {
        argument_name:
            checkpoint_value,
    }

    with pytest.raises(
        ValueError,
        match='CLI/checkpoint protocol conflict',
    ):
        test_script.apply_checkpoint_protocol(
            args,
            signature,
        )


# ===========================================================================
# 9. Full checkpoint protocol + RNG schema
# ===========================================================================

def test_full_checkpoint_contract_contains_protocol_and_rng_fields():
    source = inspect.getsource(
        train_script.trainValidateSegmentation
    )

    checkpoint_marker = (
        'checkpoint_payload = {'
    )

    assert checkpoint_marker in source

    checkpoint_block = source.split(
        checkpoint_marker,
        1,
    )[1].split(
        '}',
        1,
    )[0]

    required_fields = (
        "'state_dict'",
        "'optimizer'",
        "'best_f1'",
        "'best_epoch'",
        "'dataset'",
        "'batch_size'",
        "'protocol_signature'",
        "'run_config'",
        "'train_generator_state'",
        "'torch_rng_state'",
        "'cuda_rng_state_all'",
        "'numpy_rng_state'",
        "'python_rng_state'",
    )

    for field in required_fields:
        assert field in checkpoint_block, (
            f'Full checkpoint contract missing '
            f'{field}'
        )


# ===========================================================================
# 10. Run14 critical configuration examples
# ===========================================================================

@pytest.mark.parametrize(
    'dice_reduction,relation_mode,relation_plan',
    (
        (
            'batch_global',
            'off',
            (
                'off',
                'off',
                'off',
                'off',
            ),
        ),
        (
            'per_image',
            'off',
            (
                'off',
                'off',
                'off',
                'off',
            ),
        ),
        (
            'batch_global',
            'shallow_replace',
            (
                'shallow_replace',
                'shallow_replace',
                'off',
                'off',
            ),
        ),
        (
            'batch_global',
            'deep_replace',
            (
                'off',
                'off',
                'deep_replace',
                'deep_replace',
            ),
        ),
        (
            'batch_global',
            'deep_residual',
            (
                'off',
                'off',
                'deep_residual',
                'deep_residual',
            ),
        ),
    ),
)
def test_run14_protocol_examples_are_distinguishable(
    dice_reduction,
    relation_mode,
    relation_plan,
):
    args = make_run14_args(
        dice_reduction=dice_reduction,
        temporal_relation_mode=relation_mode,
    )

    model = make_dummy_model(
        temporal_relation_plan=relation_plan,
    )

    signature = build_signature(
        args=args,
        model=model,
    )

    assert (
        signature['dice_reduction']
        == dice_reduction
    )

    assert (
        signature['temporal_relation_mode']
        == relation_mode
    )

    assert (
        signature['temporal_relation_plan']
        == list(relation_plan)
    )

    assert (
        signature['color_order']
        == 'fixed'
    )

    assert (
        signature['supervision_mode']
        == 'native'
    )

    assert (
        signature['diff_sharing']
        == 'independent'
    )

    assert (
        signature['head_mode']
        == 'independent'
    )

    assert (
        signature['ds_profile']
        == 'legacy'
    )