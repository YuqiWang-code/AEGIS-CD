"""
Run14 preflight tests for SDTR scale dispatch and deep residual topology.

Coverage
--------
1. Legacy Run13 E5:
   sdtr_scope='all' keeps the historical
   shallow5 / shallow3 / deep-replace / deep-replace topology.

2. Run14 scale dispatch:
   - off
   - shallow_replace
   - deep_replace
   - deep_residual

3. Module identity:
   unselected scales use EAOM and do not instantiate hidden SDTR modules.

4. Four-scale output shapes.

5. Gradient flow:
   every effective per-scale difference module receives gradients.

6. SDTR temporal-swap symmetry:
   shallow and deep relation representations are symmetric under T1/T2 swap.

7. Deep residual warm start:
   alpha_deep == 0 makes deep-residual output numerically identical to its
   EAOM anchor.

8. Deep residual gradient behaviour:
   at alpha=0 the EAOM anchor and alpha receive gradients while the relation
   branch is intentionally gated off; after alpha becomes non-zero, the
   relation branch also receives gradients.

These tests intentionally avoid ImageNet checkpoint I/O by monkeypatching
MobileNetV2 construction to pretrained=False.
"""

from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Make repository root importable when this file is executed directly by
# pytest from models/tests/.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import models.model as model_module
import models.backbone.mobilenet_v2 as backbone_mobilenet_module

from models.eaom import EAOM
from models.model import BaseNet
from models.temporal_relation import SDTR


CHANNELS = 64


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def disable_imagenet_pretrained(monkeypatch):
    """Prevent BaseNet construction from reading ImageNet weights."""
    real_factory = backbone_mobilenet_module.mobilenet_v2

    def build_without_pretrained(
        pretrained=True,
        progress=True,
        **kwargs,
    ):
        return real_factory(
            pretrained=False,
            progress=progress,
            **kwargs,
        )

    monkeypatch.setattr(
        model_module.mobilenet_v2,
        'mobilenet_v2',
        build_without_pretrained,
    )


def build_run14_model(
    temporal_relation_mode='off',
):
    """Construct the Run14 E4-style independent-diff topology."""
    return BaseNet(
        decoder_mode='rep_dw',
        head_mode='independent',
        scale_fusion_mode='plain',

        diff_mode='eaom',
        diff_sharing='independent',
        sdtr_scope='all',
        temporal_relation_mode=temporal_relation_mode,

        supervision_mode='native',
        amp_phase_mode='off',
        encoder_fusion_mode='hfea',
        boundary_mode='off',
        consistency_mode='off',

        use_prior=False,
        use_msca=False,
        use_tct=False,
        use_lfds=False,
        use_sfif=False,
        use_stargate=False,
    )


def build_legacy_e5_model():
    """Construct the historical Run13 E5 SDTR configuration."""
    return BaseNet(
        decoder_mode='rep_dw',
        head_mode='independent',
        scale_fusion_mode='plain',

        diff_mode='sdtr',
        diff_sharing='independent',
        sdtr_scope='all',
        temporal_relation_mode=None,

        supervision_mode='native',
        amp_phase_mode='off',
        encoder_fusion_mode='hfea',
        boundary_mode='off',
        consistency_mode='off',

        use_prior=False,
        use_msca=False,
        use_tct=False,
        use_lfds=False,
        use_sfif=False,
        use_stargate=False,
    )


def diff_modules(model):
    return (
        model.diff1,
        model.diff2,
        model.diff3,
        model.diff4,
    )


def has_any_nonzero_grad(module):
    """Return True when at least one learnable parameter has non-zero grad."""
    for parameter in module.parameters():
        if parameter.grad is None:
            continue

        if torch.count_nonzero(parameter.grad).item() > 0:
            return True

    return False


def relation_branch_parameters(module):
    """Return only the deep temporal-relation branch parameters."""
    prefixes = (
        'wq.',
        'wk.',
        'wv.',
        'out_proj.',
    )

    return [
        parameter
        for name, parameter in module.named_parameters()
        if name.startswith(prefixes)
    ]


def any_nonzero_grad(parameters):
    for parameter in parameters:
        if parameter.grad is None:
            continue

        if torch.count_nonzero(parameter.grad).item() > 0:
            return True

    return False


def all_zero_or_none_grad(parameters):
    for parameter in parameters:
        if parameter.grad is None:
            continue

        if torch.count_nonzero(parameter.grad).item() != 0:
            return False

    return True


def make_feature_pair(size, requires_grad=False):
    f1 = torch.randn(
        1,
        CHANNELS,
        size,
        size,
        requires_grad=requires_grad,
    )

    f2 = torch.randn(
        1,
        CHANNELS,
        size,
        size,
        requires_grad=requires_grad,
    )

    return f1, f2


class SyntheticEncoder(nn.Module):
    """Cheap encoder replacement used only for full-forward shape tests."""

    def forward(self, x):
        return x, x, x, x, x


class SyntheticEncoderFusion(nn.Module):
    """Generate the four real AEGIS-CD feature resolutions cheaply."""

    SIZES = (
        64,
        32,
        16,
        8,
    )

    def forward(
        self,
        x1,
        x2,
        x3,
        x4,
        x5,
    ):
        del x2, x3, x4, x5

        base = x1.mean(
            dim=1,
            keepdim=True,
        )

        outputs = []

        for size in self.SIZES:
            feature = F.interpolate(
                base,
                size=(size, size),
                mode='bilinear',
                align_corners=False,
            )

            feature = feature.repeat(
                1,
                CHANNELS,
                1,
                1,
            )

            outputs.append(feature)

        return tuple(outputs)


def install_synthetic_frontend(model):
    """Remove backbone/HFEA cost while preserving the real diff/decoder path."""
    model.encoder = SyntheticEncoder()
    model.encoder_fusion = SyntheticEncoderFusion()
    model.diff_adapters = None

    return model


def deep_relation_delta(module, f1, f2):
    """Compute only the symmetric deep relation correction."""
    r12 = module._cross_attend(
        f1,
        f2,
    )

    r21 = module._cross_attend(
        f2,
        f1,
    )

    d_relation = 0.5 * (
        torch.abs(f1 - r12)
        + torch.abs(f2 - r21)
    )

    return module.out_proj(
        d_relation
    )


# ===========================================================================
# 1. Legacy Run13 E5 compatibility
# ===========================================================================

def test_legacy_all_scope_matches_run13_e5_topology():
    model = build_legacy_e5_model()

    assert model.diff_mode == 'sdtr'
    assert model.diff_sharing == 'independent'
    assert model.sdtr_scope == 'all'
    assert model.temporal_relation_mode is None

    assert model.base_diff_mode == 'eaom'

    assert tuple(model.temporal_relation_plan) == (
        'shallow_replace',
        'shallow_replace',
        'deep_replace',
        'deep_replace',
    )

    assert isinstance(
        model.diff1,
        SDTR,
    )
    assert isinstance(
        model.diff2,
        SDTR,
    )
    assert isinstance(
        model.diff3,
        SDTR,
    )
    assert isinstance(
        model.diff4,
        SDTR,
    )

    assert model.diff1.mode == 'shallow'
    assert model.diff1.window == 5

    assert model.diff2.mode == 'shallow'
    assert model.diff2.window == 3

    assert model.diff3.mode == 'deep'
    assert model.diff3.deep_relation_mode == 'replace'

    assert model.diff4.mode == 'deep'
    assert model.diff4.deep_relation_mode == 'replace'

    # Historical deep replacement must not contain the Run14 EAOM anchor.
    assert not hasattr(
        model.diff3,
        'deep_anchor',
    )
    assert not hasattr(
        model.diff3,
        'alpha_deep',
    )

    assert not hasattr(
        model.diff4,
        'deep_anchor',
    )
    assert not hasattr(
        model.diff4,
        'alpha_deep',
    )


# ===========================================================================
# 2. New Run14 dispatch identity
# ===========================================================================

def test_off_dispatch_is_four_independent_eaom_modules():
    model = build_run14_model(
        temporal_relation_mode='off',
    )

    assert tuple(model.temporal_relation_plan) == (
        'off',
        'off',
        'off',
        'off',
    )

    modules = diff_modules(model)

    assert all(
        isinstance(module, EAOM)
        for module in modules
    )

    # Independent means distinct parameter/module objects.
    assert model.diff1 is not model.diff2
    assert model.diff2 is not model.diff3
    assert model.diff3 is not model.diff4


def test_shallow_replace_dispatch_identity():
    model = build_run14_model(
        temporal_relation_mode='shallow_replace',
    )

    assert tuple(model.temporal_relation_plan) == (
        'shallow_replace',
        'shallow_replace',
        'off',
        'off',
    )

    assert isinstance(
        model.diff1,
        SDTR,
    )
    assert isinstance(
        model.diff2,
        SDTR,
    )
    assert isinstance(
        model.diff3,
        EAOM,
    )
    assert isinstance(
        model.diff4,
        EAOM,
    )

    assert model.diff1.mode == 'shallow'
    assert model.diff1.window == 5

    assert model.diff2.mode == 'shallow'
    assert model.diff2.window == 3

    # Shallow-only instances must not contain unused deep relation blocks.
    for module in (
        model.diff1,
        model.diff2,
    ):
        assert not hasattr(
            module,
            'wq',
        )
        assert not hasattr(
            module,
            'wk',
        )
        assert not hasattr(
            module,
            'wv',
        )
        assert not hasattr(
            module,
            'out_proj',
        )
        assert not hasattr(
            module,
            'deep_anchor',
        )
        assert not hasattr(
            module,
            'alpha_deep',
        )


def test_deep_replace_dispatch_identity():
    model = build_run14_model(
        temporal_relation_mode='deep_replace',
    )

    assert tuple(model.temporal_relation_plan) == (
        'off',
        'off',
        'deep_replace',
        'deep_replace',
    )

    assert isinstance(
        model.diff1,
        EAOM,
    )
    assert isinstance(
        model.diff2,
        EAOM,
    )
    assert isinstance(
        model.diff3,
        SDTR,
    )
    assert isinstance(
        model.diff4,
        SDTR,
    )

    for module in (
        model.diff3,
        model.diff4,
    ):
        assert module.mode == 'deep'
        assert module.deep_relation_mode == 'replace'

        # Deep-only instances must not contain shallow-only blocks.
        assert not hasattr(
            module,
            'metric_proj',
        )
        assert not hasattr(
            module,
            'phi',
        )
        assert not hasattr(
            module,
            'beta_shallow',
        )

        # Historical replacement must not instantiate residual anchor params.
        assert not hasattr(
            module,
            'deep_anchor',
        )
        assert not hasattr(
            module,
            'alpha_deep',
        )


def test_deep_residual_dispatch_identity():
    model = build_run14_model(
        temporal_relation_mode='deep_residual',
    )

    assert tuple(model.temporal_relation_plan) == (
        'off',
        'off',
        'deep_residual',
        'deep_residual',
    )

    assert isinstance(
        model.diff1,
        EAOM,
    )
    assert isinstance(
        model.diff2,
        EAOM,
    )
    assert isinstance(
        model.diff3,
        SDTR,
    )
    assert isinstance(
        model.diff4,
        SDTR,
    )

    for module in (
        model.diff3,
        model.diff4,
    ):
        assert module.mode == 'deep'
        assert module.deep_relation_mode == 'residual'

        assert isinstance(
            module.deep_anchor,
            EAOM,
        )

        assert hasattr(
            module,
            'alpha_deep',
        )

        torch.testing.assert_close(
            module.alpha_deep.detach(),
            torch.tensor(0.0),
            rtol=0.0,
            atol=0.0,
        )

        assert not hasattr(
            module,
            'metric_proj',
        )
        assert not hasattr(
            module,
            'phi',
        )
        assert not hasattr(
            module,
            'beta_shallow',
        )


# ===========================================================================
# 3. Invalid dispatch combinations must fail fast
# ===========================================================================

@pytest.mark.parametrize(
    'relation_mode',
    (
        'shallow_replace',
        'deep_replace',
        'deep_residual',
    ),
)
def test_run14_relation_requires_independent_diff_sharing(
    relation_mode,
):
    with pytest.raises(ValueError):
        BaseNet(
            diff_mode='eaom',
            diff_sharing='shared',
            temporal_relation_mode=relation_mode,
        )


def test_legacy_partial_sdtr_scope_requires_independent_sharing():
    with pytest.raises(ValueError):
        BaseNet(
            diff_mode='sdtr',
            diff_sharing='shared',
            sdtr_scope='deep',
            temporal_relation_mode=None,
        )


# ===========================================================================
# 4. Four effective difference outputs preserve per-scale shapes
# ===========================================================================

@pytest.mark.parametrize(
    'relation_mode',
    (
        'off',
        'shallow_replace',
        'deep_replace',
        'deep_residual',
    ),
)
def test_four_difference_outputs_preserve_input_shapes(
    relation_mode,
):
    torch.manual_seed(2333)

    model = build_run14_model(
        temporal_relation_mode=relation_mode,
    ).eval()

    sizes = (
        16,
        8,
        8,
        4,
    )

    f1 = tuple(
        torch.randn(
            1,
            CHANNELS,
            size,
            size,
        )
        for size in sizes
    )

    f2 = tuple(
        torch.randn(
            1,
            CHANNELS,
            size,
            size,
        )
        for size in sizes
    )

    with torch.no_grad():
        outputs = model._compute_differences(
            f1,
            f2,
        )

    assert len(outputs) == 4

    for output, expected_size in zip(
        outputs,
        sizes,
    ):
        assert output.shape == (
            1,
            CHANNELS,
            expected_size,
            expected_size,
        )


# ===========================================================================
# 5. Full native-SCDS output shapes
# ===========================================================================

@pytest.mark.parametrize(
    'relation_mode',
    (
        'shallow_replace',
        'deep_replace',
        'deep_residual',
    ),
)
def test_full_forward_native_output_shapes(
    relation_mode,
):
    torch.manual_seed(2333)

    model = build_run14_model(
        temporal_relation_mode=relation_mode,
    )

    model = install_synthetic_frontend(
        model
    )

    model.eval()

    t1 = torch.randn(
        1,
        3,
        256,
        256,
    )

    t2 = torch.randn(
        1,
        3,
        256,
        256,
    )

    with torch.no_grad():
        outputs = model(
            t1,
            t2,
        )

    assert len(outputs) == 4

    assert outputs[0].shape == (
        1,
        1,
        256,
        256,
    )

    assert outputs[1].shape == (
        1,
        1,
        32,
        32,
    )

    assert outputs[2].shape == (
        1,
        1,
        16,
        16,
    )

    assert outputs[3].shape == (
        1,
        1,
        8,
        8,
    )


# ===========================================================================
# 6. Every effective dispatched module participates in backprop
# ===========================================================================

@pytest.mark.parametrize(
    'relation_mode',
    (
        'off',
        'shallow_replace',
        'deep_replace',
        'deep_residual',
    ),
)
def test_every_effective_diff_module_receives_gradient(
    relation_mode,
):
    torch.manual_seed(2333)

    model = build_run14_model(
        temporal_relation_mode=relation_mode,
    )

    model.eval()
    model.zero_grad(
        set_to_none=True
    )

    # Reduced even resolutions keep the gradient preflight fast while
    # exercising all four actual dispatched modules.
    sizes = (
        16,
        8,
        8,
        4,
    )

    f1 = tuple(
        torch.randn(
            1,
            CHANNELS,
            size,
            size,
            requires_grad=True,
        )
        for size in sizes
    )

    f2 = tuple(
        torch.randn(
            1,
            CHANNELS,
            size,
            size,
            requires_grad=True,
        )
        for size in sizes
    )

    outputs = model._compute_differences(
        f1,
        f2,
    )

    loss = sum(
        output.square().mean()
        for output in outputs
    )

    loss.backward()

    for index, module in enumerate(
        diff_modules(model),
        start=1,
    ):
        assert has_any_nonzero_grad(module), (
            f'diff{index} received no non-zero parameter gradient '
            f'for relation_mode={relation_mode!r}'
        )

    for index, feature in enumerate(
        f1,
        start=1,
    ):
        assert feature.grad is not None, (
            f'T1 scale {index} received no input gradient'
        )

        assert torch.isfinite(
            feature.grad
        ).all()

    for index, feature in enumerate(
        f2,
        start=1,
    ):
        assert feature.grad is not None, (
            f'T2 scale {index} received no input gradient'
        )

        assert torch.isfinite(
            feature.grad
        ).all()


# ===========================================================================
# 7. Temporal swap symmetry
# ===========================================================================

def test_shallow_sdtr_is_t1_t2_swap_symmetric():
    torch.manual_seed(2333)

    module = SDTR(
        channels=8,
        mode='shallow',
        window=5,
        temperature=0.2,
    ).eval()

    f1 = torch.randn(
        1,
        8,
        8,
        8,
    )

    f2 = torch.randn(
        1,
        8,
        8,
        8,
    )

    with torch.no_grad():
        forward_order = module(
            f1,
            f2,
        )

        reverse_order = module(
            f2,
            f1,
        )

    torch.testing.assert_close(
        forward_order,
        reverse_order,
        rtol=1e-5,
        atol=1e-6,
    )


def test_deep_replace_sdtr_is_t1_t2_swap_symmetric():
    torch.manual_seed(2333)

    module = SDTR(
        channels=8,
        num_heads=4,
        mode='deep',
        deep_relation_mode='replace',
    ).eval()

    f1 = torch.randn(
        1,
        8,
        4,
        4,
    )

    f2 = torch.randn(
        1,
        8,
        4,
        4,
    )

    with torch.no_grad():
        forward_order = module(
            f1,
            f2,
        )

        reverse_order = module(
            f2,
            f1,
        )

    torch.testing.assert_close(
        forward_order,
        reverse_order,
        rtol=1e-5,
        atol=1e-6,
    )


def test_deep_residual_relation_delta_is_t1_t2_swap_symmetric():
    torch.manual_seed(2333)

    module = SDTR(
        channels=8,
        num_heads=4,
        mode='deep',
        deep_relation_mode='residual',
    ).eval()

    f1 = torch.randn(
        1,
        8,
        4,
        4,
    )

    f2 = torch.randn(
        1,
        8,
        4,
        4,
    )

    with torch.no_grad():
        forward_delta = deep_relation_delta(
            module,
            f1,
            f2,
        )

        reverse_delta = deep_relation_delta(
            module,
            f2,
            f1,
        )

    torch.testing.assert_close(
        forward_delta,
        reverse_delta,
        rtol=1e-5,
        atol=1e-6,
    )


# ===========================================================================
# 8. Deep residual alpha=0 identity warm start
# ===========================================================================

def test_deep_residual_alpha_zero_equals_eaom_anchor():
    torch.manual_seed(2333)

    module = SDTR(
        channels=8,
        num_heads=4,
        mode='deep',
        deep_relation_mode='residual',
    ).eval()

    assert module.alpha_deep.item() == pytest.approx(
        0.0,
        abs=0.0,
    )

    f1 = torch.randn(
        1,
        8,
        8,
        8,
    )

    f2 = torch.randn(
        1,
        8,
        8,
        8,
    )

    with torch.no_grad():
        residual_output = module(
            f1,
            f2,
        )

        eaom_output = module.deep_anchor(
            f1,
            f2,
        )

    torch.testing.assert_close(
        residual_output,
        eaom_output,
        rtol=0.0,
        atol=1e-7,
    )


# ===========================================================================
# 9. Deep residual gradient gating
# ===========================================================================

def test_deep_residual_alpha_zero_gates_relation_parameter_gradients():
    torch.manual_seed(2333)

    module = SDTR(
        channels=8,
        num_heads=4,
        mode='deep',
        deep_relation_mode='residual',
    ).eval()

    f1 = torch.randn(
        1,
        8,
        8,
        8,
        requires_grad=True,
    )

    f2 = torch.randn(
        1,
        8,
        8,
        8,
        requires_grad=True,
    )

    output = module(
        f1,
        f2,
    )

    # relation_delta is ReLU-ended, so mean gives alpha a stable
    # non-zero first-step signal under the fixed seed.
    loss = output.mean()
    loss.backward()

    assert module.alpha_deep.grad is not None
    assert torch.isfinite(
        module.alpha_deep.grad
    ).all()

    assert torch.count_nonzero(
        module.alpha_deep.grad
    ).item() > 0

    assert has_any_nonzero_grad(
        module.deep_anchor
    )

    relation_parameters = relation_branch_parameters(
        module
    )

    assert relation_parameters

    # Exact zero residual scale intentionally gates the relation branch
    # at the warm-start step.
    assert all_zero_or_none_grad(
        relation_parameters
    )


def test_deep_residual_relation_branch_receives_gradient_after_alpha_opens():
    torch.manual_seed(2333)

    module = SDTR(
        channels=8,
        num_heads=4,
        mode='deep',
        deep_relation_mode='residual',
    ).eval()

    with torch.no_grad():
        module.alpha_deep.fill_(
            0.25
        )

    module.zero_grad(
        set_to_none=True
    )

    f1 = torch.randn(
        1,
        8,
        8,
        8,
        requires_grad=True,
    )

    f2 = torch.randn(
        1,
        8,
        8,
        8,
        requires_grad=True,
    )

    output = module(
        f1,
        f2,
    )

    loss = output.square().mean()
    loss.backward()

    relation_parameters = relation_branch_parameters(
        module
    )

    assert any_nonzero_grad(
        relation_parameters
    )

    assert has_any_nonzero_grad(
        module.deep_anchor
    )

    assert module.alpha_deep.grad is not None
    assert torch.isfinite(
        module.alpha_deep.grad
    ).all()


# ===========================================================================
# 10. SDTR constructor invariants
# ===========================================================================

def test_deep_residual_requires_explicit_deep_mode():
    with pytest.raises(
        ValueError,
        match='requires mode="deep"',
    ):
        SDTR(
            channels=8,
            mode='auto',
            deep_relation_mode='residual',
        )


def test_shallow_module_has_no_permanently_unused_deep_parameters():
    module = SDTR(
        channels=8,
        mode='shallow',
        window=5,
    )

    parameter_names = {
        name
        for name, _parameter in module.named_parameters()
    }

    assert not any(
        name.startswith('wq.')
        for name in parameter_names
    )

    assert not any(
        name.startswith('wk.')
        for name in parameter_names
    )

    assert not any(
        name.startswith('wv.')
        for name in parameter_names
    )

    assert not any(
        name.startswith('out_proj.')
        for name in parameter_names
    )


def test_deep_replace_module_has_no_permanently_unused_shallow_parameters():
    module = SDTR(
        channels=8,
        num_heads=4,
        mode='deep',
        deep_relation_mode='replace',
    )

    parameter_names = {
        name
        for name, _parameter in module.named_parameters()
    }

    assert not any(
        name.startswith('metric_proj.')
        for name in parameter_names
    )

    assert not any(
        name.startswith('phi.')
        for name in parameter_names
    )

    assert 'beta_shallow' not in parameter_names
    assert 'alpha_deep' not in parameter_names

    assert not any(
        name.startswith('deep_anchor.')
        for name in parameter_names
    )