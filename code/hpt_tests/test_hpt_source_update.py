from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from hpt_source_update import (  # noqa: E402
    compose_protected_gradients,
    photometric_twin,
    sample_validity,
    structured_multiview_twin,
)


def test_source_twins_do_not_mutate_real_batch() -> None:
    batch = {
        "observation.images.corner": torch.full((2, 3, 4, 4), 0.5),
        "observation.images.top": torch.full((2, 3, 4, 4), 0.6),
        "observation.images.shoulder": torch.full((2, 3, 4, 4), 0.7),
        "action": torch.ones((2, 4)),
    }
    original = {key: value.clone() for key, value in batch.items()}
    two_d = photometric_twin(batch)
    structured = structured_multiview_twin(batch, batch_index=1)
    assert all(torch.equal(batch[key], original[key]) for key in batch)
    assert not torch.equal(two_d["observation.images.corner"], batch["observation.images.corner"])
    assert int(sum(torch.count_nonzero(structured[key]).item() == 0 for key in structured if "images" in key)) == 1


def test_sample_validity_is_sample_level() -> None:
    batch = {
        "observation.images.corner": torch.ones((2, 3, 2, 2)),
        "action": torch.ones((2, 4)),
    }
    batch["action"][1, 0] = float("nan")
    quality, accepted, total = sample_validity(batch)
    assert (quality, accepted, total) == (0.5, 1, 2)


def test_projection_removes_first_order_conflict_and_caps_norm() -> None:
    names = ["model.trunk.block.weight", "model.stems.metaworld_state.weight"]
    base = [torch.tensor([1.0, 0.0]), torch.tensor([2.0])]
    correction = [torch.tensor([-2.0, 2.0]), torch.tensor([-9.0])]
    combined, diagnostics = compose_protected_gradients(
        names,
        base,
        correction,
        correction_weight=0.25,
        quality=1.0,
        norm_cap=0.5,
    )
    applied_correction = (combined[0] - base[0]) / 0.25
    assert diagnostics["projection_applied"] is True
    assert float(torch.dot(base[0], applied_correction)) >= -1.0e-6
    assert float(torch.linalg.vector_norm(applied_correction)) <= 0.5 * float(torch.linalg.vector_norm(base[0])) + 1.0e-6
    assert torch.equal(combined[1], base[1])

