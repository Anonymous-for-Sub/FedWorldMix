import tempfile
import unittest
from pathlib import Path

import numpy as np

from fedsplat.lora_updates import (
    ClientAdapterUpdate,
    DualChannelAdapterUpdate,
    LoRALayerUpdate,
    aggregate_dual_channel_updates,
    aggregate_lora_updates,
    client_update_artifact_fingerprint,
    clip_layers_by_dense_delta_norm,
    load_dual_channel_update,
    load_client_update,
    low_rank_dense_difference_norm,
    low_rank_dense_frobenius_norm,
    save_dual_channel_update,
    save_client_update,
)


class LoRAUpdateTest(unittest.TestCase):
    def _update(self, client_id: str, scale: float, rank: int) -> ClientAdapterUpdate:
        a = np.arange(rank * 3, dtype=np.float32).reshape(rank, 3) / 10.0 + 0.1
        b = np.arange(4 * rank, dtype=np.float32).reshape(4, rank) / 10.0 + 0.2
        return ClientAdapterUpdate(
            client_id=client_id,
            layers=(LoRALayerUpdate("policy.head", a * scale, b),),
            n_real=10,
            reliability=0.5,
        )

    def test_round_trip_and_measured_bytes(self) -> None:
        update = self._update("client_a", 1.0, 2)
        with tempfile.TemporaryDirectory() as tmp:
            metadata = save_client_update(update, Path(tmp))
            restored = load_client_update(Path(tmp))
            fingerprint = client_update_artifact_fingerprint(Path(tmp))
            np.testing.assert_allclose(restored.layers[0].dense(), update.layers[0].dense())
            self.assertGreater(metadata["total_upload_bytes"], metadata["serialized_bytes"])
            self.assertEqual(metadata["metadata_bytes"], (Path(tmp) / "adapter_update.json").stat().st_size)
            self.assertEqual(metadata["payload_sha256"], fingerprint["payload_sha256"])

    def test_low_rank_norms_match_dense_materialization(self) -> None:
        current = self._update("client_a", 1.0, 2).layers[0]
        previous = self._update("client_a", 0.4, 1).layers[0]
        self.assertAlmostEqual(
            low_rank_dense_frobenius_norm(current.a, current.b),
            float(np.linalg.norm(current.dense())),
            places=5,
        )
        self.assertAlmostEqual(
            low_rank_dense_difference_norm(current, previous),
            float(np.linalg.norm(current.dense() - previous.dense())),
            places=5,
        )

    def test_server_trust_region_is_relative_to_previous_global(self) -> None:
        previous = self._update("previous", 0.8, 2).layers
        current = self._update("current", 1.8, 2).layers
        before = float(np.linalg.norm(current[0].dense() - previous[0].dense()))
        cap = before * 0.3
        clipped, audit = clip_layers_by_dense_delta_norm(
            current, previous, cap, server_rank=2
        )
        after = float(np.linalg.norm(clipped[0].dense() - previous[0].dense()))
        self.assertLessEqual(after, cap * (1.0 + 1e-6))
        self.assertTrue(audit["clipped"])
        self.assertEqual(audit["reference"], "previous_global_adapter")
        self.assertAlmostEqual(audit["dense_update_norm_before"], before, places=5)
        self.assertAlmostEqual(audit["dense_update_norm_after"], after, places=5)

    def test_heterogeneous_rank_aggregation_matches_dense_average(self) -> None:
        first = self._update("client_a", 1.0, 1)
        second = self._update("client_b", 0.5, 2)
        result = aggregate_lora_updates([first, second], {"client_a": 0.25, "client_b": 0.75}, server_rank=3)
        expected = 0.25 * first.layers[0].dense() + 0.75 * second.layers[0].dense()
        np.testing.assert_allclose(result[0].dense(), expected, atol=1e-6)

    def test_low_rank_aggregation_matches_dense_truncated_svd(self) -> None:
        rng = np.random.default_rng(7)

        def update(client_id: str, rank: int) -> ClientAdapterUpdate:
            return ClientAdapterUpdate(
                client_id=client_id,
                layers=(LoRALayerUpdate(
                    "wide.layer",
                    rng.normal(size=(rank, 17)).astype(np.float32),
                    rng.normal(size=(13, rank)).astype(np.float32),
                ),),
                n_real=1,
                reliability=1.0,
            )

        updates = [update("a", 2), update("b", 3), update("c", 4)]
        weights = {"a": 0.2, "b": 0.3, "c": 0.5}
        result = aggregate_lora_updates(updates, weights, server_rank=3)[0].dense()
        dense = sum(weights[item.client_id] * item.layers[0].dense() for item in updates)
        u, singular_values, vt = np.linalg.svd(dense.astype(np.float64), full_matrices=False)
        expected = (u[:, :3] * singular_values[:3]) @ vt[:3]
        np.testing.assert_allclose(result, expected, atol=2e-5, rtol=2e-5)

    def test_dual_channel_round_trip_and_optional_residual_loading(self) -> None:
        real = self._update("client_a", 1.0, 2)
        residual = self._update("client_a", 0.25, 1)
        package = DualChannelAdapterUpdate(
            "client_a", real, residual, marginal_reliability=0.4, compute_time_s=3.5
        )
        with tempfile.TemporaryDirectory() as tmp:
            metadata = save_dual_channel_update(package, Path(tmp))
            restored = load_dual_channel_update(Path(tmp))
            real_only = load_dual_channel_update(Path(tmp), load_residual=False)
            np.testing.assert_allclose(restored.real.layers[0].dense(), real.layers[0].dense())
            np.testing.assert_allclose(
                restored.synthetic_residual.layers[0].dense(), residual.layers[0].dense()
            )
            self.assertIsNone(real_only.synthetic_residual)
            self.assertEqual(real_only.marginal_reliability, 0.0)
            self.assertAlmostEqual(restored.compute_time_s, 3.5)
            self.assertGreater(metadata["beacon_bytes"], 0)
            self.assertGreater(metadata["real_upload_bytes"], 0)
            self.assertGreater(metadata["synthetic_residual_upload_bytes"], 0)

    def test_dual_channel_aggregation_keeps_real_and_selects_residual(self) -> None:
        real_a = self._update("client_a", 1.0, 2)
        real_b = self._update("client_b", 0.5, 2)
        residual_a = self._update("client_a", 0.2, 1)
        residual_b = self._update("client_b", 0.8, 1)
        updates = [
            DualChannelAdapterUpdate("client_a", real_a, residual_a, 0.8),
            DualChannelAdapterUpdate("client_b", real_b, residual_b, 0.1),
        ]
        result = aggregate_dual_channel_updates(
            updates,
            real_weights={"client_a": 1.0, "client_b": 1.0},
            residual_weights={"client_a": 1.0, "client_b": 0.0},
            server_rank=3,
        )
        expected = 0.5 * real_a.layers[0].dense() + 0.5 * real_b.layers[0].dense()
        expected += 0.5 * residual_a.layers[0].dense()
        np.testing.assert_allclose(result[0].dense(), expected, atol=1e-6)

        all_selected = aggregate_dual_channel_updates(
            updates,
            real_weights={"client_a": 1.0, "client_b": 1.0},
            residual_weights={"client_a": 1.0, "client_b": 1.0},
            server_rank=3,
        )
        expected += 0.5 * residual_b.layers[0].dense()
        np.testing.assert_allclose(all_selected[0].dense(), expected, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
