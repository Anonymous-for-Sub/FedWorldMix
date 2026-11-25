"""VistaMix contract (legacy SafeMix API retained for compatibility)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .adaptive_augmentation import (
    PortfolioGradientResult,
    combine_augmentation_portfolio_gradients,
)


FEDWORLDMIX_TITLE = (
    "FedVista: Robust Federated VLA Learning from Diverse Visual Worlds"
)
SAFEMIX_GS_RATIO = 0.25
SAFEMIX_TWO_D_RATIO = 0.75
SAFEMIX_V1_SAMPLING = "uniform"
SAFEMIX_EVENT_BALANCED_SAMPLING = "gripper_event_balanced"
SAFEMIX_ALLOWED_SAMPLING = (
    SAFEMIX_V1_SAMPLING,
    SAFEMIX_EVENT_BALANCED_SAMPLING,
)


@dataclass(frozen=True)
class SafeMixContract:
    """Machine-readable description of the frozen local update."""

    schema: str = "fedworldmix.safemix_contract.v1"
    gs_ratio: float = SAFEMIX_GS_RATIO
    two_d_ratio: float = SAFEMIX_TWO_D_RATIO
    equal_auxiliary_budget: bool = True
    synthetic_objective: str = "action_bc"
    synthetic_gradient_scope: str = "perception"
    use_synthetic_quality_weight: bool = True
    update_contract: str = "combined"
    server_aggregation: str = "sample_weighted_fedavg"
    sampling_mode: str = SAFEMIX_V1_SAMPLING

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safemix_contract_errors(
    spec: Mapping[str, Any],
    *,
    sampling_mode: str = SAFEMIX_V1_SAMPLING,
) -> list[str]:
    """Return deviations from the evidence-compatible SafeMix contract."""
    errors: list[str] = []
    if sampling_mode not in SAFEMIX_ALLOWED_SAMPLING:
        return [f"unsupported SafeMix sampling mode: {sampling_mode}"]
    if spec.get("gradient_mode") != "augmentation_portfolio":
        errors.append("gradient_mode must be augmentation_portfolio")

    portfolio = spec.get("augmentation_portfolio")
    if not isinstance(portfolio, Mapping):
        errors.append("augmentation_portfolio must be an object")
    else:
        if portfolio.get("schema") != "fedsplat.augmentation_portfolio.v1":
            errors.append("augmentation_portfolio schema changed")
        if not _close(portfolio.get("gs_ratio"), SAFEMIX_GS_RATIO):
            errors.append("SafeMix-v1 requires gs_ratio=0.25")
        if portfolio.get("equal_auxiliary_budget") is not True:
            errors.append("equal_auxiliary_budget must be true")
        two_d_source = portfolio.get("two_d_source")
        if (
            not isinstance(two_d_source, Mapping)
            or two_d_source.get("type") != "visual_proxy"
            or two_d_source.get("provenance")
            != "proxy_visual_augmentation"
        ):
            errors.append("2D source provenance changed")

    expected = {
        "synthetic_objective": "action_bc",
        "synthetic_gradient_scope": "perception",
        "use_synthetic_quality_weight": True,
        "update_contract": "combined",
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            errors.append(f"{key} must be {value!r}")

    max_norm = spec.get("max_auxiliary_to_real_norm")
    if (
        not isinstance(max_norm, (int, float))
        or not math.isfinite(float(max_norm))
        or float(max_norm) <= 0.0
    ):
        errors.append("max_auxiliary_to_real_norm must be positive")
    threshold = spec.get("projection_cosine_threshold")
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not -1.0 <= float(threshold) <= 0.0
    ):
        errors.append("projection_cosine_threshold must be in [-1, 0]")

    source = spec.get("synthetic_source")
    if not isinstance(source, Mapping):
        errors.append("synthetic_source must be an object")
    else:
        if source.get("type") != "lerobot_dataset":
            errors.append("synthetic_source.type must be lerobot_dataset")
        if not source.get("quality_manifest"):
            errors.append("GS quality_manifest is required")
        selection = source.get("selection")
        if not isinstance(selection, Mapping):
            errors.append("synthetic_source.selection must be an object")
        elif selection.get("sampling_mode") != sampling_mode:
            errors.append(
                "sampling_mode must be "
                f"{sampling_mode!r}, got {selection.get('sampling_mode')!r}"
            )

    patterns = spec.get("local_trainable_module_patterns")
    if not isinstance(patterns, list) or not patterns:
        errors.append("perception-only trainable module patterns are required")
    elif any(
        "vision_model" not in str(pattern) and "connector" not in str(pattern)
        for pattern in patterns
    ):
        errors.append("trainable module patterns must remain perception-only")
    return errors


def validate_safemix_contract(
    spec: Mapping[str, Any],
    *,
    sampling_mode: str = SAFEMIX_V1_SAMPLING,
) -> SafeMixContract:
    """Validate a local-update config and return its frozen contract."""
    errors = safemix_contract_errors(spec, sampling_mode=sampling_mode)
    if errors:
        raise ValueError("Invalid SafeMix contract: " + "; ".join(errors))
    return SafeMixContract(sampling_mode=sampling_mode)


def combine_safemix_gradients(
    real_gradient: Sequence[float],
    two_d_gradient: Sequence[float],
    gs_gradient: Sequence[float],
    *,
    gs_quality_score: float,
    max_gs_to_base_norm: float,
    projection_cosine_threshold: float,
) -> PortfolioGradientResult:
    """Apply the frozen 75% 2D / 25% quality-gated GS update."""
    return combine_augmentation_portfolio_gradients(
        real_gradient,
        two_d_gradient,
        gs_gradient,
        gs_ratio=SAFEMIX_GS_RATIO,
        gs_quality_score=gs_quality_score,
        max_gs_to_base_norm=max_gs_to_base_norm,
        projection_cosine_threshold=projection_cosine_threshold,
    )


def _close(value: Any, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, abs_tol=1e-12)
    )
