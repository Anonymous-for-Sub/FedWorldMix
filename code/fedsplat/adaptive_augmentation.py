"""Client-adaptive 2D/GS augmentation portfolio primitives."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .method import GradientCombinationResult, combine_real_synthetic_gradients


PORTFOLIO_RATIOS = (0.0, 0.25, 0.5)


@dataclass(frozen=True)
class PortfolioGradientResult:
    gs_ratio: float
    two_d_ratio: float
    base_combination: GradientCombinationResult
    gs_combination: GradientCombinationResult

    @property
    def combined_gradient(self) -> tuple[float, ...]:
        return self.gs_combination.combined_gradient

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateCalibration:
    client_id: str
    gs_ratio: float
    ood_outcomes: tuple[bool, ...]
    canonical_outcomes: tuple[bool, ...]
    gs_eligible: bool = True

    @property
    def ood_success_rate(self) -> float:
        return _success_rate(self.ood_outcomes, "ood_outcomes")

    @property
    def canonical_success_rate(self) -> float:
        return _success_rate(
            self.canonical_outcomes,
            "canonical_outcomes",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "ood_success_rate": self.ood_success_rate,
            "canonical_success_rate": self.canonical_success_rate,
        }


@dataclass(frozen=True)
class CandidateDecision:
    gs_ratio: float
    utility: float
    ood_gain: float
    canonical_gain: float
    superiority_confidence: float
    selected: bool
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioSelection:
    scope: str
    selected_ratio: float
    fallback_used: bool
    decisions: tuple[CandidateDecision, ...]
    client_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_portfolio_ratio(gs_ratio: float) -> float:
    ratio = float(gs_ratio)
    if not math.isfinite(ratio) or ratio not in PORTFOLIO_RATIOS:
        raise ValueError(
            f"gs_ratio must be one of {PORTFOLIO_RATIOS}, got {gs_ratio!r}"
        )
    return ratio


def combine_augmentation_portfolio_gradients(
    real_gradient: Sequence[float],
    two_d_gradient: Sequence[float],
    gs_gradient: Sequence[float],
    *,
    gs_ratio: float,
    gs_quality_score: float,
    max_gs_to_base_norm: float,
    projection_cosine_threshold: float,
) -> PortfolioGradientResult:
    """Form real+2D base, then project/cap GS against that complete base."""
    ratio = validate_portfolio_ratio(gs_ratio)
    two_d_ratio = 1.0 - ratio
    base = combine_real_synthetic_gradients(
        real_gradient,
        two_d_gradient,
        mode="naive",
        synthetic_weight=two_d_ratio,
        quality_score=1.0,
        max_auxiliary_to_real_norm=None,
        projection_cosine_threshold=projection_cosine_threshold,
    )
    gs = combine_real_synthetic_gradients(
        base.combined_gradient,
        gs_gradient,
        mode="projected",
        synthetic_weight=ratio,
        quality_score=gs_quality_score,
        max_auxiliary_to_real_norm=max_gs_to_base_norm,
        projection_cosine_threshold=projection_cosine_threshold,
    )
    return PortfolioGradientResult(
        gs_ratio=ratio,
        two_d_ratio=two_d_ratio,
        base_combination=base,
        gs_combination=gs,
    )


def paired_superiority_confidence(
    candidate: Sequence[bool],
    baseline: Sequence[bool],
) -> float:
    """One-sided exact sign-test confidence from paired binary outcomes."""
    candidate_values = _outcomes(candidate, "candidate")
    baseline_values = _outcomes(baseline, "baseline")
    if len(candidate_values) != len(baseline_values):
        raise ValueError("Paired outcome sequences must have equal length")
    wins = sum(left and not right for left, right in zip(
        candidate_values,
        baseline_values,
        strict=True,
    ))
    losses = sum(right and not left for left, right in zip(
        candidate_values,
        baseline_values,
        strict=True,
    ))
    discordant = wins + losses
    if discordant == 0:
        return 0.5
    p_value = sum(
        math.comb(discordant, value)
        for value in range(wins, discordant + 1)
    ) / float(2**discordant)
    return float(np.clip(1.0 - p_value, 0.0, 1.0))


def select_client_portfolio(
    candidates: Iterable[CandidateCalibration],
    *,
    confidence_threshold: float,
    canonical_tolerance: float = 0.05,
    canonical_penalty: float = 1.0,
) -> PortfolioSelection:
    records = tuple(candidates)
    client_ids = {record.client_id for record in records}
    if len(client_ids) != 1:
        raise ValueError("Client selection requires exactly one client_id")
    return _select(
        records,
        scope="client",
        client_ids=tuple(sorted(client_ids)),
        confidence_threshold=confidence_threshold,
        canonical_tolerance=canonical_tolerance,
        canonical_penalty=canonical_penalty,
    )


def select_global_portfolio(
    candidates: Iterable[CandidateCalibration],
    *,
    confidence_threshold: float,
    canonical_tolerance: float = 0.05,
    canonical_penalty: float = 1.0,
) -> PortfolioSelection:
    records = tuple(candidates)
    by_client: dict[str, dict[float, CandidateCalibration]] = {}
    for record in records:
        ratio = validate_portfolio_ratio(record.gs_ratio)
        if ratio in by_client.setdefault(record.client_id, {}):
            raise ValueError(
                f"Duplicate ratio {ratio} for client {record.client_id}"
            )
        by_client[record.client_id][ratio] = record
    if not by_client:
        raise ValueError("Global selection requires at least one client")
    expected = set(PORTFOLIO_RATIOS)
    for client_id, client_records in by_client.items():
        if set(client_records) != expected:
            raise ValueError(
                f"Client {client_id} must provide ratios {sorted(expected)}"
            )

    pooled = []
    for ratio in PORTFOLIO_RATIOS:
        ratio_records = [by_client[key][ratio] for key in sorted(by_client)]
        pooled.append(
            CandidateCalibration(
                client_id="__pooled_global__",
                gs_ratio=ratio,
                ood_outcomes=tuple(
                    value
                    for record in ratio_records
                    for value in record.ood_outcomes
                ),
                canonical_outcomes=tuple(
                    value
                    for record in ratio_records
                    for value in record.canonical_outcomes
                ),
                gs_eligible=all(record.gs_eligible for record in ratio_records),
            )
        )
    return _select(
        tuple(pooled),
        scope="global",
        client_ids=tuple(sorted(by_client)),
        confidence_threshold=confidence_threshold,
        canonical_tolerance=canonical_tolerance,
        canonical_penalty=canonical_penalty,
    )


def _select(
    records: tuple[CandidateCalibration, ...],
    *,
    scope: str,
    client_ids: tuple[str, ...],
    confidence_threshold: float,
    canonical_tolerance: float,
    canonical_penalty: float,
) -> PortfolioSelection:
    if not 0.5 <= float(confidence_threshold) < 1.0:
        raise ValueError("confidence_threshold must be in [0.5, 1)")
    if not 0.0 <= float(canonical_tolerance) <= 1.0:
        raise ValueError("canonical_tolerance must be in [0, 1]")
    if not math.isfinite(float(canonical_penalty)) or canonical_penalty < 0.0:
        raise ValueError("canonical_penalty must be finite and nonnegative")

    by_ratio = {}
    for record in records:
        ratio = validate_portfolio_ratio(record.gs_ratio)
        if ratio in by_ratio:
            raise ValueError(f"Duplicate calibration ratio: {ratio}")
        _outcomes(record.ood_outcomes, "ood_outcomes")
        _outcomes(record.canonical_outcomes, "canonical_outcomes")
        by_ratio[ratio] = record
    if set(by_ratio) != set(PORTFOLIO_RATIOS):
        raise ValueError(
            f"Calibration must provide ratios {list(PORTFOLIO_RATIOS)}"
        )

    baseline = by_ratio[0.0]
    decisions = []
    for ratio in PORTFOLIO_RATIOS:
        record = by_ratio[ratio]
        ood_gain = record.ood_success_rate - baseline.ood_success_rate
        canonical_gain = (
            record.canonical_success_rate - baseline.canonical_success_rate
        )
        confidence = (
            1.0
            if ratio == 0.0
            else paired_superiority_confidence(
                record.ood_outcomes,
                baseline.ood_outcomes,
            )
        )
        reasons = []
        if ratio != 0.0 and not record.gs_eligible:
            reasons.append("gs_ineligible")
        if ratio != 0.0 and confidence < confidence_threshold:
            reasons.append("insufficient_ood_confidence")
        if canonical_gain < -canonical_tolerance:
            reasons.append("canonical_tolerance_exceeded")
        eligible = ratio == 0.0 or not reasons
        utility = record.ood_success_rate - canonical_penalty * max(
            0.0,
            -canonical_gain - canonical_tolerance,
        )
        decisions.append(
            CandidateDecision(
                gs_ratio=ratio,
                utility=utility,
                ood_gain=ood_gain,
                canonical_gain=canonical_gain,
                superiority_confidence=confidence,
                selected=False,
                eligible=eligible,
                rejection_reasons=tuple(reasons),
            )
        )
    selected_ratio = max(
        (
            decision.utility,
            -decision.gs_ratio,
            decision.gs_ratio,
        )
        for decision in decisions
        if decision.eligible
    )[2]
    finalized = tuple(
        CandidateDecision(
            **{
                **decision.to_dict(),
                "selected": decision.gs_ratio == selected_ratio,
            }
        )
        for decision in decisions
    )
    return PortfolioSelection(
        scope=scope,
        selected_ratio=selected_ratio,
        fallback_used=selected_ratio == 0.0,
        decisions=finalized,
        client_ids=client_ids,
    )


def _success_rate(values: Sequence[bool], name: str) -> float:
    outcomes = _outcomes(values, name)
    return sum(outcomes) / float(len(outcomes))


def _outcomes(values: Sequence[bool], name: str) -> tuple[bool, ...]:
    outcomes = tuple(values)
    if not outcomes:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, (bool, np.bool_)) for value in outcomes):
        raise ValueError(f"{name} must contain only booleans")
    return tuple(bool(value) for value in outcomes)
