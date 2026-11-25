"""Client-local cross-fit selection for Gaussian demonstration trajectories."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TrajectoryGradientEvidence:
    episode_id: int
    variation: str
    quality: float
    feasible: bool
    gradient: np.ndarray
    source_demo: str | None = None


@dataclass(frozen=True)
class TrajectoryScore:
    episode_id: int
    variation: str
    source_demo: str | None
    quality: float
    feasible: bool
    alignments: tuple[float, ...]
    mean_alignment: float
    alignment_std: float
    lower_confidence_bound: float
    positive_fraction: float
    score: float
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrajectorySelection:
    selected_episode_ids: tuple[int, ...]
    selected_variations: tuple[str, ...]
    scores: tuple[TrajectoryScore, ...]
    candidate_count: int
    eligible_count: int

    def to_dict(self) -> dict:
        return {
            "selected_episode_ids": list(self.selected_episode_ids),
            "selected_variations": list(self.selected_variations),
            "candidate_count": self.candidate_count,
            "eligible_count": self.eligible_count,
            "scores": [item.to_dict() for item in self.scores],
        }


def cosine_similarity(left: np.ndarray, right: np.ndarray, epsilon: float = 1e-12) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Gradient signature shape mismatch: {left.shape} versus {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Gradient signatures must be finite")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= epsilon:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def score_trajectory(
    candidate: TrajectoryGradientEvidence,
    anchor_gradients: Iterable[np.ndarray],
    *,
    lcb_z: float,
    minimum_lcb: float,
    minimum_positive_fraction: float,
) -> TrajectoryScore:
    """Score one candidate against disjoint calibration-gradient signatures."""
    anchors = tuple(np.asarray(value) for value in anchor_gradients)
    if not anchors:
        raise ValueError("At least one calibration-gradient signature is required")
    if not math.isfinite(lcb_z) or lcb_z < 0.0:
        raise ValueError("lcb_z must be finite and nonnegative")
    if not math.isfinite(minimum_lcb) or not -1.0 <= minimum_lcb <= 1.0:
        raise ValueError("minimum_lcb must be finite and in [-1, 1]")
    if not math.isfinite(minimum_positive_fraction) or not 0.0 <= minimum_positive_fraction <= 1.0:
        raise ValueError("minimum_positive_fraction must be in [0, 1]")
    if not math.isfinite(candidate.quality) or not 0.0 <= candidate.quality <= 1.0:
        raise ValueError("candidate quality must be finite and in [0, 1]")

    alignments = tuple(cosine_similarity(candidate.gradient, anchor) for anchor in anchors)
    mean = float(np.mean(alignments))
    std = float(np.std(alignments, ddof=1)) if len(alignments) > 1 else 0.0
    standard_error = std / math.sqrt(len(alignments))
    lower_bound = mean - lcb_z * standard_error
    positive_fraction = float(np.mean(np.asarray(alignments) > 0.0))
    # Rank by margin above the declared acceptance threshold. This preserves
    # deterministic ordering when a small negative numerical tolerance is
    # configured instead of collapsing all near-orthogonal candidates to zero.
    score = candidate.quality * max(0.0, lower_bound - minimum_lcb)
    reasons = []
    if not candidate.feasible:
        reasons.append("feasibility_gate")
    if candidate.quality <= 0.0:
        reasons.append("zero_quality")
    if lower_bound <= minimum_lcb:
        reasons.append("nonpositive_crossfit_lcb")
    if positive_fraction < minimum_positive_fraction:
        reasons.append("insufficient_positive_folds")
    return TrajectoryScore(
        episode_id=candidate.episode_id,
        variation=candidate.variation,
        source_demo=candidate.source_demo,
        quality=candidate.quality,
        feasible=candidate.feasible,
        alignments=alignments,
        mean_alignment=mean,
        alignment_std=std,
        lower_confidence_bound=lower_bound,
        positive_fraction=positive_fraction,
        score=score,
        eligible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def select_trajectories(
    candidates: Iterable[TrajectoryGradientEvidence],
    anchor_gradients: Iterable[np.ndarray],
    *,
    max_selected: int,
    max_per_variation: int,
    minimum_variations: int = 1,
    lcb_z: float = 1.0,
    minimum_lcb: float = 0.0,
    minimum_positive_fraction: float = 0.5,
) -> TrajectorySelection:
    """Select high-confidence candidates with deterministic variation coverage."""
    candidates = tuple(candidates)
    anchors = tuple(anchor_gradients)
    if not candidates:
        raise ValueError("At least one candidate trajectory is required")
    if len({item.episode_id for item in candidates}) != len(candidates):
        raise ValueError("Candidate episode IDs must be unique")
    for name, value in (
        ("max_selected", max_selected),
        ("max_per_variation", max_per_variation),
        ("minimum_variations", minimum_variations),
    ):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if minimum_variations > max_selected:
        raise ValueError("minimum_variations cannot exceed max_selected")

    scores = tuple(
        score_trajectory(
            candidate,
            anchors,
            lcb_z=lcb_z,
            minimum_lcb=minimum_lcb,
            minimum_positive_fraction=minimum_positive_fraction,
        )
        for candidate in candidates
    )
    eligible = sorted(
        (item for item in scores if item.eligible),
        key=lambda item: (-item.score, -item.lower_confidence_bound, item.episode_id),
    )
    by_variation: dict[str, list[TrajectoryScore]] = {}
    for item in eligible:
        by_variation.setdefault(item.variation, []).append(item)

    selected: list[TrajectoryScore] = []
    counts: dict[str, int] = {}
    variation_order = sorted(
        by_variation,
        key=lambda name: (-by_variation[name][0].score, name),
    )
    for variation in variation_order[:minimum_variations]:
        item = by_variation[variation][0]
        selected.append(item)
        counts[variation] = 1
        if len(selected) >= max_selected:
            break

    selected_ids = {item.episode_id for item in selected}
    for item in eligible:
        if len(selected) >= max_selected:
            break
        if item.episode_id in selected_ids:
            continue
        if counts.get(item.variation, 0) >= max_per_variation:
            continue
        selected.append(item)
        selected_ids.add(item.episode_id)
        counts[item.variation] = counts.get(item.variation, 0) + 1

    selected.sort(key=lambda item: (-item.score, item.episode_id))
    return TrajectorySelection(
        selected_episode_ids=tuple(item.episode_id for item in selected),
        selected_variations=tuple(item.variation for item in selected),
        scores=tuple(sorted(scores, key=lambda item: item.episode_id)),
        candidate_count=len(candidates),
        eligible_count=len(eligible),
    )
