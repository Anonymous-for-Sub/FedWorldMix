import numpy as np
import pytest

from fedsplat.safemix import (
    SAFEMIX_EVENT_BALANCED_SAMPLING,
    SafeMixContract,
    combine_safemix_gradients,
    safemix_contract_errors,
    validate_safemix_contract,
)


def _spec(sampling_mode: str = "uniform") -> dict:
    return {
        "gradient_mode": "augmentation_portfolio",
        "synthetic_objective": "action_bc",
        "synthetic_gradient_scope": "perception",
        "use_synthetic_quality_weight": True,
        "max_auxiliary_to_real_norm": 1.0,
        "projection_cosine_threshold": -0.02,
        "update_contract": "combined",
        "local_trainable_module_patterns": [
            "model.vision_model.*",
            "model.connector.*",
        ],
        "augmentation_portfolio": {
            "schema": "fedsplat.augmentation_portfolio.v1",
            "gs_ratio": 0.25,
            "equal_auxiliary_budget": True,
            "two_d_source": {
                "type": "visual_proxy",
                "provenance": "proxy_visual_augmentation",
            },
        },
        "synthetic_source": {
            "type": "lerobot_dataset",
            "quality_manifest": "/tmp/quality.json",
            "selection": {"sampling_mode": sampling_mode},
        },
    }


def test_uniform_contract_matches_frozen_development_candidate():
    contract = validate_safemix_contract(_spec())

    assert contract == SafeMixContract()
    assert contract.two_d_ratio == 0.75
    assert contract.gs_ratio == 0.25


def test_event_balanced_contract_is_explicit_unvalidated_variant():
    spec = _spec(SAFEMIX_EVENT_BALANCED_SAMPLING)
    contract = validate_safemix_contract(
        spec,
        sampling_mode=SAFEMIX_EVENT_BALANCED_SAMPLING,
    )

    assert contract.sampling_mode == SAFEMIX_EVENT_BALANCED_SAMPLING
    assert safemix_contract_errors(spec) == [
        "sampling_mode must be 'uniform', got 'gripper_event_balanced'"
    ]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("gs_ratio", 0.5, "SafeMix-v1 requires gs_ratio=0.25"),
        ("equal_auxiliary_budget", False, "equal_auxiliary_budget must be true"),
    ],
)
def test_contract_rejects_changed_mixture(key, value, message):
    spec = _spec()
    spec["augmentation_portfolio"][key] = value

    assert message in safemix_contract_errors(spec)


def test_contract_rejects_policy_wide_updates():
    spec = _spec()
    spec["local_trainable_module_patterns"].append("model.action_expert.*")

    with pytest.raises(ValueError, match="perception-only"):
        validate_safemix_contract(spec)


def test_safemix_wrapper_freezes_ratio_and_preserves_projection():
    result = combine_safemix_gradients(
        [1.0, 0.0],
        [0.0, 1.0],
        [-2.0, -2.0],
        gs_quality_score=1.0,
        max_gs_to_base_norm=1.0,
        projection_cosine_threshold=0.0,
    )

    assert result.gs_ratio == 0.25
    assert result.two_d_ratio == 0.75
    assert result.gs_combination.projection_applied
    np.testing.assert_allclose(
        result.base_combination.combined_gradient,
        [1.0, 0.75],
    )
