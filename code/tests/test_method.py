import unittest

import numpy as np

from fedsplat.method import (
    calibrate_synthetic_gradient,
    combine_real_synthetic_gradients,
    feasibility_gate,
    geometric_quality_score,
    heldout_real_reliability,
    reliability_raw_weight,
    update_novelty,
)


class MethodTest(unittest.TestCase):
    def test_hard_gate_overrides_soft_quality(self) -> None:
        checks = {
            "schema": True,
            "joint_limits": True,
            "ik": True,
            "collision": False,
            "visibility": True,
            "instruction_action": True,
        }
        gate = feasibility_gate(checks)
        self.assertFalse(gate.accepted)
        self.assertEqual(gate.failed_checks, ("collision",))
        self.assertEqual(geometric_quality_score({"render": 0.99}, feasibility=gate), 0.0)

    def test_geometric_quality_penalizes_weak_component(self) -> None:
        score = geometric_quality_score({"render": 1.0, "action": 0.01})
        self.assertAlmostEqual(score, 0.1, places=7)

    def test_conflicting_gradient_is_projected(self) -> None:
        result = calibrate_synthetic_gradient([1.0, 0.0], [-2.0, 1.0])
        self.assertEqual(result.branch, "projected")
        self.assertAlmostEqual(result.dot_after, 0.0, places=10)
        np.testing.assert_allclose(result.calibrated_gradient, [0.0, 1.0], atol=1e-10)

    def test_aligned_gradient_is_unchanged(self) -> None:
        result = calibrate_synthetic_gradient([1.0, 0.0], [0.5, 1.0])
        self.assertEqual(result.branch, "already_aligned")
        np.testing.assert_allclose(result.calibrated_gradient, [0.5, 1.0])

    def test_mild_conflict_can_be_left_to_trust_cap(self) -> None:
        result = calibrate_synthetic_gradient(
            [1.0, 0.0],
            [-0.01, 1.0],
            projection_cosine_threshold=-0.05,
        )
        self.assertEqual(result.branch, "mild_conflict_retained")
        self.assertFalse(result.non_conflict_guaranteed)
        np.testing.assert_allclose(result.calibrated_gradient, [-0.01, 1.0])

    def test_real_only_and_naive_gradient_combination(self) -> None:
        real_only = combine_real_synthetic_gradients([1.0, 2.0], mode="real_only")
        np.testing.assert_allclose(real_only.combined_gradient, [1.0, 2.0])
        naive = combine_real_synthetic_gradients(
            [1.0, 0.0],
            [-2.0, 1.0],
            mode="naive",
            synthetic_weight=0.5,
            quality_score=0.8,
        )
        np.testing.assert_allclose(naive.combined_gradient, [0.2, 0.4])
        self.assertAlmostEqual(naive.effective_synthetic_weight, 0.4)

    def test_projected_combination_removes_real_conflict(self) -> None:
        result = combine_real_synthetic_gradients(
            [1.0, 0.0],
            [-2.0, 1.0],
            mode="projected",
        )
        np.testing.assert_allclose(result.combined_gradient, [1.0, 1.0], atol=1e-10)
        self.assertTrue(result.projection_applied)
        self.assertEqual(result.projection.branch, "projected")

    def test_auxiliary_trust_ratio_caps_weighted_norm(self) -> None:
        result = combine_real_synthetic_gradients(
            [1.0, 0.0],
            [0.0, 2.0],
            mode="projected",
            synthetic_weight=10.0,
            max_auxiliary_to_real_norm=0.5,
        )
        np.testing.assert_allclose(result.combined_gradient, [1.0, 0.5], atol=1e-10)
        self.assertAlmostEqual(result.auxiliary_trust_scale, 0.025)
        self.assertAlmostEqual(result.weighted_synthetic_norm, 0.5)

    def test_near_zero_real_gradient_has_explicit_branch(self) -> None:
        result = calibrate_synthetic_gradient([0.0, 0.0], [1.0, 2.0])
        self.assertFalse(result.non_conflict_guaranteed)
        self.assertEqual(result.branch, "near_zero_real_gradient")

    def test_reliability_and_staleness(self) -> None:
        self.assertAlmostEqual(heldout_real_reliability(2.0, 1.0), 0.5, places=7)
        fresh = reliability_raw_weight(100, 0.4, 0)
        stale = reliability_raw_weight(100, 0.4, 5)
        self.assertGreater(fresh, stale)

    def test_update_novelty(self) -> None:
        self.assertEqual(update_novelty([1.0, 0.0], []), 1.0)
        self.assertAlmostEqual(update_novelty([1.0, 0.0], [[1.0, 0.0]]), 0.0)
        self.assertAlmostEqual(update_novelty([1.0, 0.0], [[-1.0, 0.0]]), 1.0)


if __name__ == "__main__":
    unittest.main()
