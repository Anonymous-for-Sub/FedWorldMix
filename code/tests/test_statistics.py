import unittest

from fedsplat.statistics import (
    exact_mcnemar_pvalue,
    holm_bonferroni,
    paired_binary_comparison,
)


class PairedStatisticsTest(unittest.TestCase):
    def test_exact_mcnemar_detects_one_sided_discordance(self):
        result = exact_mcnemar_pvalue([1] * 8, [0] * 8)
        self.assertEqual(result["treatment_only_successes"], 8)
        self.assertEqual(result["control_only_successes"], 0)
        self.assertAlmostEqual(result["pvalue_two_sided_exact"], 2 / 256)

    def test_paired_bootstrap_is_reproducible(self):
        treatment = [[1, 1, 0, 1], [1, 0, 1, 1]]
        control = [[0, 1, 0, 0], [0, 0, 1, 0]]
        first = paired_binary_comparison(treatment, control, bootstrap_samples=500, seed=7)
        second = paired_binary_comparison(treatment, control, bootstrap_samples=500, seed=7)
        self.assertEqual(first, second)
        self.assertGreater(first["paired_bootstrap"]["difference"], 0.0)

    def test_holm_bonferroni_controls_two_primary_comparisons(self):
        passed = holm_bonferroni({"gaussian": 0.01, "anchor_real": 0.03})
        self.assertTrue(passed["all_rejected"])
        self.assertEqual(
            passed["adjusted_pvalues"],
            {"gaussian": 0.02, "anchor_real": 0.03},
        )

        failed = holm_bonferroni({"gaussian": 0.03, "anchor_real": 0.04})
        self.assertFalse(failed["all_rejected"])
        self.assertFalse(failed["rejected"]["gaussian"])
        self.assertFalse(failed["rejected"]["anchor_real"])


if __name__ == "__main__":
    unittest.main()
