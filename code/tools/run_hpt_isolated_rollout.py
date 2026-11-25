#!/usr/bin/env python3
"""Evaluate an HPT checkpoint under a frozen visual condition."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEDVLA_ROOT = ROOT
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(FEDVLA_ROOT))

from hpt_source_update import photometric_twin, structured_multiview_twin  # noqa: E402
from phaseb.hpt.modeling_hpt import HPTPolicy  # noqa: E402
from phaseb.metaworld_rollout_hpt import evaluate_metaworld_rollouts_hpt  # noqa: E402


def transform_batch(batch: dict[str, object], domain: str) -> dict[str, object]:
    if domain == "canonical":
        return batch
    if domain == "photometric":
        return photometric_twin(batch)
    if domain == "camera_dropout":
        return structured_multiview_twin(batch, batch_index=1)
    if domain == "mixed":
        return structured_multiview_twin(photometric_twin(batch), batch_index=1)
    raise ValueError(f"Unknown domain: {domain}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--budget", type=int, choices=[20, 200], required=True)
    parser.add_argument("--domain", choices=["canonical", "photometric", "camera_dropout", "mixed"], required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--tasks", default="door-lock,close-drawer,sweep-into,open-window")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-root", default="")
    parser.add_argument("--save-media", action="store_true")
    args = parser.parse_args()

    original = HPTPolicy.select_action

    def select_action_with_domain(self: HPTPolicy, batch: dict[str, object]) -> object:
        return original(self, transform_batch(batch, str(args.domain)))

    HPTPolicy.select_action = select_action_with_domain
    try:
        report = evaluate_metaworld_rollouts_hpt(
            checkpoint_path=str(args.checkpoint),
            tasks=[item.strip() for item in str(args.tasks).split(",") if item.strip()],
            seed=int(args.evaluation_seed),
            trials=int(args.trials),
            repeats=int(args.repeats),
            max_steps_per_episode=int(args.max_steps),
            render_width=128,
            render_height=128,
            camera_views=["corner", "top", "shoulder"],
            torch_device=str(args.device),
            media_root=str(args.media_root),
            media_max_episodes=1 if args.save_media else 0,
            media_max_frames=120,
            media_fps=10,
            media_save_frames=False,
            media_save_gif=bool(args.save_media),
        )
    finally:
        HPTPolicy.select_action = original
    report.update(
        {
            "schema": "fedworldmix.isolated_hpt_rollout.v1",
            "method": str(args.method),
            "training_seed": int(args.training_seed),
            "budget": int(args.budget),
            "domain": str(args.domain),
            "evaluation_seed": int(args.evaluation_seed),
            "claim_boundary": "recorded multiview, not 3DGS",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exit_code))
