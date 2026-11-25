import numpy as np
import pytest

from fedsplat.adaptive_augmentation import (
    CandidateCalibration,
    combine_augmentation_portfolio_gradients,
    paired_superiority_confidence,
    select_client_portfolio,
    select_global_portfolio,
)


def _record(
    client: str,
    ratio: float,
    ood: list[bool],
    canonical: list[bool],
    *,
    eligible: bool = True,
) -> CandidateCalibration:
    return CandidateCalibration(
        client_id=client,
        gs_ratio=ratio,
        ood_outcomes=tuple(ood),
        canonical_outcomes=tuple(canonical),
        gs_eligible=eligible,
    )


def test_portfolio_gradient_uses_real_plus_2d_as_gs_projection_base():
    result = combine_augmentation_portfolio_gradients(
        [1.0, 0.0],
        [0.0, 1.0],
        [-2.0, -2.0],
        gs_ratio=0.25,
        gs_quality_score=1.0,
        max_gs_to_base_norm=1.0,
        projection_cosine_threshold=0.0,
    )

    np.testing.assert_allclose(
        result.base_combination.combined_gradient,
        [1.0, 0.75],
    )
    assert result.gs_combination.projection_applied
    assert result.gs_combination.projection.dot_after == pytest.approx(0.0)
    assert result.gs_combination.effective_synthetic_weight == pytest.approx(
        0.25
    )


def test_zero_gs_ratio_is_exact_real_plus_2d_candidate():
    result = combine_augmentation_portfolio_gradients(
        [1.0, 0.0],
        [0.0, 2.0],
        [100.0, 100.0],
        gs_ratio=0.0,
        gs_quality_score=1.0,
        max_gs_to_base_norm=1.0,
        projection_cosine_threshold=-0.02,
    )

    np.testing.assert_allclose(result.combined_gradient, [1.0, 2.0])
    assert result.gs_combination.effective_synthetic_weight == 0.0


def test_paired_confidence_rewards_consistent_wins():
    baseline = [False] * 6
    candidate = [True] * 5 + [False]
    assert paired_superiority_confidence(candidate, baseline) == pytest.approx(
        0.96875
    )


def test_discovery_threshold_accepts_two_clean_wins_but_not_one():
    assert paired_superiority_confidence(
        [True, True],
        [False, False],
    ) == pytest.approx(0.75)
    assert paired_superiority_confidence(
        [True],
        [False],
    ) == pytest.approx(0.5)


def test_client_selector_uses_gs_when_gain_is_confident_and_safe():
    records = [
        _record("owner0", 0.0, [False] * 6, [True] * 10),
        _record(
            "owner0",
            0.25,
            [True] * 5 + [False],
            [True] * 10,
        ),
        _record(
            "owner0",
            0.5,
            [True] * 4 + [False] * 2,
            [True] * 8 + [False] * 2,
        ),
    ]

    selected = select_client_portfolio(
        records,
        confidence_threshold=0.9,
    )

    assert selected.selected_ratio == 0.25
    assert not selected.fallback_used


def test_client_selector_falls_back_for_quality_or_canonical_failure():
    records = [
        _record("owner0", 0.0, [False] * 6, [True] * 10),
        _record(
            "owner0",
            0.25,
            [True] * 5 + [False],
            [True] * 10,
            eligible=False,
        ),
        _record(
            "owner0",
            0.5,
            [True] * 6,
            [True] * 8 + [False] * 2,
        ),
    ]

    selected = select_client_portfolio(
        records,
        confidence_threshold=0.9,
        canonical_tolerance=0.05,
    )

    assert selected.selected_ratio == 0.0
    assert selected.fallback_used


def test_global_selector_uses_one_ratio_for_all_clients():
    records = []
    for client in ("owner0", "owner1"):
        records.extend(
            [
                _record(client, 0.0, [False] * 6, [True] * 10),
                _record(
                    client,
                    0.25,
                    [True] * 5 + [False],
                    [True] * 10,
                ),
                _record(
                    client,
                    0.5,
                    [True] * 4 + [False] * 2,
                    [True] * 9 + [False],
                ),
            ]
        )

    selected = select_global_portfolio(
        records,
        confidence_threshold=0.9,
    )

    assert selected.scope == "global"
    assert selected.client_ids == ("owner0", "owner1")
    assert selected.selected_ratio == 0.25
