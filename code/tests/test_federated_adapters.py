import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fedsplat.federated_adapters import aggregate_adapter_manifest, validate_federated_adapter_spec
from fedsplat.lora_updates import ClientAdapterUpdate, LoRALayerUpdate, load_client_update, save_client_update


class FederatedAdapterTest(unittest.TestCase):
    def _update(self, client_id: str, value: float, reliability: float) -> ClientAdapterUpdate:
        return ClientAdapterUpdate(
            client_id=client_id,
            layers=(LoRALayerUpdate("policy.head", np.array([[1.0, 0.0]]), np.array([[value], [0.0]])),),
            n_real=10,
            reliability=reliability,
        )

    def test_validation_requires_multiple_updates(self) -> None:
        errors = validate_federated_adapter_spec({
            "aggregation_mode": "fedavg",
            "server_rank": 1,
            "global_update_id": "global",
            "updates": [{"path": "one"}],
        })
        self.assertIn("updates must contain at least two client update entries", errors)

    def test_reliability_aggregation_emits_replayable_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            first = root / "updates" / "first"
            second = root / "updates" / "second"
            save_client_update(self._update("a", 1.0, 0.0), first)
            save_client_update(self._update("b", 3.0, 1.0), second)
            config = root / "configs" / "aggregate.json"
            config.write_text(json.dumps({
                "aggregation_mode": "reliability",
                "server_rank": 1,
                "global_update_id": "global",
                "updates": [
                    {"path": "updates/first"},
                    {"path": "updates/second"},
                ],
            }), encoding="utf-8")
            report = aggregate_adapter_manifest(config, root / "out")
            self.assertTrue(report["aggregation_ready"])
            self.assertGreater(report["weights"]["b"], report["weights"]["a"])
            restored = load_client_update(root / "out" / "adapter_upload")
            expected = report["weights"]["a"] * 1.0 + report["weights"]["b"] * 3.0
            self.assertAlmostEqual(float(restored.layers[0].dense()[0, 0]), expected, places=6)

    def test_runpack_config_resolves_project_relative_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fedsplat").mkdir()
            run_dir = root / "runs" / "run-001"
            run_dir.mkdir(parents=True)
            first = root / "updates" / "first"
            second = root / "updates" / "second"
            save_client_update(self._update("a", 1.0, 0.5), first)
            save_client_update(self._update("b", 3.0, 0.5), second)
            config = run_dir / "config.json"
            config.write_text(json.dumps({
                "aggregation_mode": "fedavg",
                "server_rank": 1,
                "global_update_id": "global",
                "updates": [
                    {"path": "updates/first"},
                    {"path": "updates/second"},
                ],
            }), encoding="utf-8")

            report = aggregate_adapter_manifest(config, run_dir / "artifacts")

            self.assertTrue(report["aggregation_ready"])
            restored = load_client_update(run_dir / "artifacts" / "adapter_upload")
            self.assertAlmostEqual(float(restored.layers[0].dense()[0, 0]), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
