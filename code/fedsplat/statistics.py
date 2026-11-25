"""Paired binary statistics for matched robot rollout experiments."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def holm_bonferroni(
    pvalues: Mapping[str, float], *, alpha: float = 0.05
) -> dict:
    """Apply Holm's step-down family-wise error correction."""
    if not pvalues:
        raise ValueError("Holm correction requires at least one p-value")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    normalized = {str(name): float(value) for name, value in pvalues.items()}
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in normalized.values()
    ):
        raise ValueError("p-values must be finite and in [0, 1]")

    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    rejected: dict[str, bool] = {name: False for name in normalized}
    adjusted: dict[str, float] = {}
    running_adjusted = 0.0
    continue_rejecting = True
    for rank, (name, value) in enumerate(ordered):
        multiplier = count - rank
        running_adjusted = max(running_adjusted, multiplier * value)
        adjusted[name] = min(1.0, running_adjusted)
        threshold = alpha / multiplier
        rejected[name] = bool(continue_rejecting and value <= threshold)
        if value > threshold:
            continue_rejecting = False
    return {
        "method": "holm_bonferroni",
        "alpha": alpha,
        "raw_pvalues": normalized,
        "adjusted_pvalues": adjusted,
        "rejected": rejected,
        "all_rejected": all(rejected.values()),
    }


def exact_mcnemar_pvalue(treatment: Sequence[int], control: Sequence[int]) -> dict:
    """Return a two-sided exact McNemar test for matched binary outcomes."""
    treatment_array = np.asarray(treatment, dtype=np.int8).reshape(-1)
    control_array = np.asarray(control, dtype=np.int8).reshape(-1)
    if treatment_array.size == 0 or treatment_array.shape != control_array.shape:
        raise ValueError("Matched outcomes must be nonempty and have the same shape")
    if not np.isin(treatment_array, (0, 1)).all() or not np.isin(control_array, (0, 1)).all():
        raise ValueError("Matched outcomes must be binary")
    treatment_only = int(np.sum((treatment_array == 1) & (control_array == 0)))
    control_only = int(np.sum((treatment_array == 0) & (control_array == 1)))
    discordant = treatment_only + control_only
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(min(treatment_only, control_only) + 1))
        pvalue = min(1.0, 2.0 * tail / (2.0**discordant))
    return {
        "treatment_only_successes": treatment_only,
        "control_only_successes": control_only,
        "discordant_pairs": discordant,
        "pvalue_two_sided_exact": pvalue,
    }


def paired_bootstrap_difference(
    treatment_groups: Sequence[Sequence[int]],
    control_groups: Sequence[Sequence[int]],
    *,
    samples: int = 20_000,
    confidence: float = 0.95,
    seed: int = 20260722,
) -> dict:
    """Hierarchically bootstrap matched seeds and episodes within each seed."""
    treatment = np.asarray(treatment_groups, dtype=np.float64)
    control = np.asarray(control_groups, dtype=np.float64)
    if treatment.ndim != 2 or treatment.shape != control.shape or treatment.size == 0:
        raise ValueError("Bootstrap groups must be nonempty matched rectangular matrices")
    if samples <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("samples must be positive and confidence must be in (0, 1)")
    if not np.isin(treatment, (0.0, 1.0)).all() or not np.isin(control, (0.0, 1.0)).all():
        raise ValueError("Bootstrap outcomes must be binary")
    rng = np.random.default_rng(seed)
    num_groups, group_size = treatment.shape
    differences = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected_groups = rng.integers(0, num_groups, size=num_groups)
        values = []
        for group_index in selected_groups:
            selected_items = rng.integers(0, group_size, size=group_size)
            values.append(
                treatment[group_index, selected_items] - control[group_index, selected_items]
            )
        differences[sample_index] = float(np.mean(np.concatenate(values)))
    alpha = (1.0 - confidence) / 2.0
    observed = float(np.mean(treatment - control))
    return {
        "difference": observed,
        "confidence": confidence,
        "bootstrap_samples": samples,
        "ci_lower": float(np.quantile(differences, alpha)),
        "ci_upper": float(np.quantile(differences, 1.0 - alpha)),
        "probability_positive": float(np.mean(differences > 0.0)),
        "seed": seed,
    }


def paired_binary_comparison(
    treatment_groups: Sequence[Sequence[int]],
    control_groups: Sequence[Sequence[int]],
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 20260722,
) -> dict:
    treatment = np.asarray(treatment_groups, dtype=np.int8)
    control = np.asarray(control_groups, dtype=np.int8)
    if treatment.ndim != 2 or treatment.shape != control.shape:
        raise ValueError("Comparison groups must have the same two-dimensional shape")
    return {
        "num_pairs": int(treatment.size),
        "treatment_success_rate": float(treatment.mean()),
        "control_success_rate": float(control.mean()),
        **exact_mcnemar_pvalue(treatment.reshape(-1), control.reshape(-1)),
        "paired_bootstrap": paired_bootstrap_difference(
            treatment,
            control,
            samples=bootstrap_samples,
            seed=seed,
        ),
    }
