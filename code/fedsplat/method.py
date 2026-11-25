"""Core FedSplat reliability and communication primitives.

The functions in this module implement the method equations documented in
``idea/method_math.md`` without depending on a particular VLA framework.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


REQUIRED_FEASIBILITY_CHECKS = (
    "schema",
    "joint_limits",
    "ik",
    "collision",
    "visibility",
    "instruction_action",
)


@dataclass(frozen=True)
class FeasibilityResult:
    accepted: bool
    checks: dict[str, bool]
    failed_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectionResult:
    calibrated_gradient: tuple[float, ...]
    alpha: float
    real_norm: float
    synthetic_norm: float
    calibrated_norm: float
    real_norm_sq: float
    dot_before: float
    dot_after: float
    cosine_before: float
    cosine_after: float
    branch: str
    non_conflict_guaranteed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GradientCombinationResult:
    mode: str
    combined_gradient: tuple[float, ...]
    synthetic_gradient_used: tuple[float, ...] | None
    synthetic_weight: float
    quality_score: float
    effective_synthetic_weight: float
    auxiliary_trust_scale: float
    weighted_synthetic_norm: float
    projection_applied: bool
    projection: ProjectionResult | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def feasibility_gate(
    checks: Mapping[str, bool],
    required: Sequence[str] = REQUIRED_FEASIBILITY_CHECKS,
) -> FeasibilityResult:
    """Reject a trajectory when a required physical/schema check is missing or false."""
    normalized = {str(key): bool(value) for key, value in checks.items()}
    failed = tuple(name for name in required if not normalized.get(name, False))
    return FeasibilityResult(accepted=not failed, checks=normalized, failed_checks=failed)


def geometric_quality_score(
    scores: Mapping[str, float],
    weights: Mapping[str, float] | None = None,
    feasibility: FeasibilityResult | bool = True,
    epsilon: float = 1e-8,
) -> float:
    """Compute a feasibility-gated weighted geometric mean in ``[0, 1]``."""
    if not scores:
        raise ValueError("At least one soft quality score is required")
    accepted = feasibility.accepted if isinstance(feasibility, FeasibilityResult) else bool(feasibility)
    if not accepted:
        return 0.0
    if epsilon <= 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must be in (0, 1]")

    normalized_scores: dict[str, float] = {}
    for name, value in scores.items():
        value = float(value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"Quality score {name!r} must be finite and in [0, 1]")
        normalized_scores[str(name)] = value

    if weights is None:
        normalized_weights = {name: 1.0 / len(normalized_scores) for name in normalized_scores}
    else:
        unknown = set(weights) - set(normalized_scores)
        missing = set(normalized_scores) - set(weights)
        if unknown or missing:
            raise ValueError(f"Quality weights must match scores; missing={sorted(missing)}, unknown={sorted(unknown)}")
        total = sum(float(value) for value in weights.values())
        if total <= 0.0 or any(float(value) < 0.0 for value in weights.values()):
            raise ValueError("Quality weights must be nonnegative with positive sum")
        normalized_weights = {name: float(weights[name]) / total for name in normalized_scores}

    log_score = sum(
        normalized_weights[name] * math.log(max(value, epsilon))
        for name, value in normalized_scores.items()
    )
    return float(math.exp(log_score))


def cosine_similarity(left: Sequence[float], right: Sequence[float], epsilon: float = 1e-12) -> float:
    a = _vector(left, "left")
    b = _vector(right, "right")
    if a.shape != b.shape:
        raise ValueError(f"Vector shape mismatch: {a.shape} != {b.shape}")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= epsilon:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def calibrate_synthetic_gradient(
    real_gradient: Sequence[float],
    synthetic_gradient: Sequence[float],
    projection_cosine_threshold: float = 0.0,
    epsilon: float = 1e-12,
) -> ProjectionResult:
    """Project severe conflicts; retain mild conflicts for norm-based trust control."""
    real = _vector(real_gradient, "real_gradient")
    synthetic = _vector(synthetic_gradient, "synthetic_gradient")
    if real.shape != synthetic.shape:
        raise ValueError(f"Gradient shape mismatch: {real.shape} != {synthetic.shape}")

    real_norm_sq = float(np.dot(real, real))
    dot_before = float(np.dot(synthetic, real))
    cosine_before = cosine_similarity(synthetic, real, epsilon=epsilon)
    if not -1.0 <= projection_cosine_threshold <= 0.0:
        raise ValueError("projection_cosine_threshold must be in [-1, 0]")
    if real_norm_sq <= epsilon:
        alpha = 0.0
        calibrated = synthetic.copy()
        branch = "near_zero_real_gradient"
        guaranteed = False
    elif dot_before < 0.0 and cosine_before < projection_cosine_threshold:
        alpha = -dot_before / real_norm_sq
        calibrated = synthetic + alpha * real
        branch = "projected"
        guaranteed = True
    elif dot_before < 0.0:
        alpha = 0.0
        calibrated = synthetic.copy()
        branch = "mild_conflict_retained"
        guaranteed = False
    else:
        alpha = 0.0
        calibrated = synthetic.copy()
        branch = "already_aligned"
        guaranteed = True

    dot_after = float(np.dot(calibrated, real))
    tolerance = 1e-10 * max(1.0, abs(dot_before), real_norm_sq)
    if guaranteed and dot_after < -tolerance:
        raise ArithmeticError(f"Projection violated non-conflict tolerance: {dot_after}")
    if guaranteed and abs(dot_after) <= tolerance:
        dot_after = 0.0

    return ProjectionResult(
        calibrated_gradient=tuple(float(value) for value in calibrated),
        alpha=float(alpha),
        real_norm=float(np.linalg.norm(real)),
        synthetic_norm=float(np.linalg.norm(synthetic)),
        calibrated_norm=float(np.linalg.norm(calibrated)),
        real_norm_sq=real_norm_sq,
        dot_before=dot_before,
        dot_after=dot_after,
        cosine_before=cosine_before,
        cosine_after=cosine_similarity(calibrated, real, epsilon=epsilon),
        branch=branch,
        non_conflict_guaranteed=guaranteed,
    )


def combine_real_synthetic_gradients(
    real_gradient: Sequence[float],
    synthetic_gradient: Sequence[float] | None = None,
    *,
    mode: str = "real_only",
    synthetic_weight: float = 1.0,
    quality_score: float = 1.0,
    max_auxiliary_to_real_norm: float | None = None,
    projection_cosine_threshold: float = 0.0,
    epsilon: float = 1e-12,
) -> GradientCombinationResult:
    """Combine real and auxiliary gradients using an explicit local-update mode.

    ``projected`` is the FedSplat client-side calibration rule. The caller is
    responsible for preserving provenance: an auxiliary visual transform is
    not a Gaussian demonstration merely because it enters this interface.
    """
    if mode not in {"real_only", "naive", "projected"}:
        raise ValueError(f"Unknown gradient combination mode: {mode}")
    real = _vector(real_gradient, "real_gradient")
    synthetic_weight = float(synthetic_weight)
    quality_score = float(quality_score)
    if not math.isfinite(synthetic_weight) or synthetic_weight < 0.0:
        raise ValueError("synthetic_weight must be finite and nonnegative")
    if not math.isfinite(quality_score) or not 0.0 <= quality_score <= 1.0:
        raise ValueError("quality_score must be finite and in [0, 1]")
    if max_auxiliary_to_real_norm is not None:
        max_auxiliary_to_real_norm = float(max_auxiliary_to_real_norm)
        if not math.isfinite(max_auxiliary_to_real_norm) or max_auxiliary_to_real_norm <= 0.0:
            raise ValueError("max_auxiliary_to_real_norm must be finite and positive")

    if mode == "real_only":
        if synthetic_gradient is not None:
            synthetic = _vector(synthetic_gradient, "synthetic_gradient")
            if synthetic.shape != real.shape:
                raise ValueError(f"Gradient shape mismatch: {real.shape} != {synthetic.shape}")
        return GradientCombinationResult(
            mode=mode,
            combined_gradient=tuple(float(value) for value in real),
            synthetic_gradient_used=None,
            synthetic_weight=synthetic_weight,
            quality_score=quality_score,
            effective_synthetic_weight=0.0,
            auxiliary_trust_scale=1.0,
            weighted_synthetic_norm=0.0,
            projection_applied=False,
            projection=None,
        )

    if synthetic_gradient is None:
        raise ValueError(f"synthetic_gradient is required for mode={mode}")
    synthetic = _vector(synthetic_gradient, "synthetic_gradient")
    if synthetic.shape != real.shape:
        raise ValueError(f"Gradient shape mismatch: {real.shape} != {synthetic.shape}")
    projection = calibrate_synthetic_gradient(
        real,
        synthetic,
        projection_cosine_threshold=projection_cosine_threshold,
        epsilon=epsilon,
    )
    used = np.asarray(
        projection.calibrated_gradient if mode == "projected" else synthetic,
        dtype=np.float64,
    )
    effective_weight = synthetic_weight * quality_score
    auxiliary_trust_scale = 1.0
    weighted_synthetic_norm = effective_weight * float(np.linalg.norm(used))
    if max_auxiliary_to_real_norm is not None:
        real_norm = float(np.linalg.norm(real))
        allowed_norm = max_auxiliary_to_real_norm * real_norm
        if real_norm <= epsilon:
            auxiliary_trust_scale = 0.0
        elif weighted_synthetic_norm > allowed_norm:
            auxiliary_trust_scale = allowed_norm / max(weighted_synthetic_norm, epsilon)
        effective_weight *= auxiliary_trust_scale
        weighted_synthetic_norm *= auxiliary_trust_scale
    combined = real + effective_weight * used
    return GradientCombinationResult(
        mode=mode,
        combined_gradient=tuple(float(value) for value in combined),
        synthetic_gradient_used=tuple(float(value) for value in used),
        synthetic_weight=synthetic_weight,
        quality_score=quality_score,
        effective_synthetic_weight=effective_weight,
        auxiliary_trust_scale=auxiliary_trust_scale,
        weighted_synthetic_norm=weighted_synthetic_norm,
        projection_applied=mode == "projected" and projection.branch == "projected",
        projection=projection,
    )


def heldout_real_reliability(loss_before: float, loss_after: float, epsilon: float = 1e-8) -> float:
    """Return clipped relative held-out-real loss improvement."""
    before = float(loss_before)
    after = float(loss_after)
    if not math.isfinite(before) or not math.isfinite(after):
        raise ValueError("Validation losses must be finite")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return float(np.clip((before - after) / (abs(before) + epsilon), 0.0, 1.0))


def reliability_raw_weight(
    n_real: float,
    reliability: float,
    staleness: float,
    reliability_floor: float = 0.05,
    staleness_gamma: float = 0.1,
) -> float:
    """Compute the unnormalized reliability/staleness aggregation weight."""
    if not 0.0 < reliability_floor <= 1.0:
        raise ValueError("reliability_floor must be in (0, 1]")
    if staleness_gamma < 0.0:
        raise ValueError("staleness_gamma must be nonnegative")
    n_real = max(float(n_real), 0.0)
    reliability = float(np.clip(reliability, 0.0, 1.0))
    staleness = max(float(staleness), 0.0)
    return n_real * (reliability_floor + (1.0 - reliability_floor) * reliability) * math.exp(
        -staleness_gamma * staleness
    )


def update_novelty(candidate: Sequence[float], selected: Sequence[Sequence[float]]) -> float:
    """Compute update novelty in ``[0, 1]`` relative to selected updates."""
    if not selected:
        return 1.0
    max_cosine = max(cosine_similarity(candidate, other) for other in selected)
    return float(np.clip(1.0 - (1.0 + max_cosine) / 2.0, 0.0, 1.0))


def communication_utility(
    reliability: float,
    novelty: float,
    serialized_bytes: int,
    compute_time_s: float,
    novelty_weight: float = 0.25,
    time_cost_bytes_per_s: float = 0.0,
    epsilon: float = 1.0,
) -> float:
    """Score a client update by real reliability and novelty per measured cost."""
    if serialized_bytes < 0 or compute_time_s < 0.0:
        raise ValueError("Communication and compute costs must be nonnegative")
    if novelty_weight < 0.0 or time_cost_bytes_per_s < 0.0 or epsilon <= 0.0:
        raise ValueError("Utility hyperparameters must be nonnegative and epsilon positive")
    numerator = float(np.clip(reliability, 0.0, 1.0)) * (
        1.0 + novelty_weight * float(np.clip(novelty, 0.0, 1.0))
    )
    denominator = float(serialized_bytes) + time_cost_bytes_per_s * float(compute_time_s) + epsilon
    return numerator / denominator


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector
