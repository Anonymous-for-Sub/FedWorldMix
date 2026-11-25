#!/usr/bin/env python3
"""Run an isolated HPT baseline or heterogeneous-source federated method."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEDVLA_ROOT = ROOT
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(FEDVLA_ROOT))

from hpt_source_update import HPTSourceEpoch, SourceUpdateConfig  # noqa: E402
from phaseb.config import ModelConfig, RunConfig, TaskSplitConfig, TrainConfig  # noqa: E402
from phaseb import trainer_hpt  # noqa: E402


def _dataclass_from_dict(cls: type, payload: dict) -> object:
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def load_config(path: Path) -> RunConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return RunConfig(
        task_split=_dataclass_from_dict(TaskSplitConfig, payload["task_split"]),
        train=_dataclass_from_dict(TrainConfig, payload["train"]),
        model=_dataclass_from_dict(ModelConfig, payload["model"]),
        output_dir=str(payload.get("output_dir", "runs/phase_b")),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["centralized_hpt", "fedavg_hpt", "fedvla_hpt", "naive_joint_hpt", "fedworldmix_hpt"],
        required=True,
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--local-epochs", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--norm-cap", type=float, default=0.50)
    args = parser.parse_args()

    config = load_config(args.base_config)
    config.train.seed = int(args.seed)
    config.train.rounds = int(args.rounds)
    if int(args.local_epochs) > 0:
        config.train.local_epochs = int(args.local_epochs)
    config.train.max_samples_per_client = int(args.max_samples)
    config.train.torch_device = str(args.device)
    config.train.progress_every_rounds = min(10, max(int(args.rounds), 1))
    config.train.checkpoint_every_rounds = min(10, max(int(args.rounds), 1))
    config.train.grad_clip_norm = 1.0
    config.train.skip_nonfinite_updates = True
    config.train.experiment_mode = {
        "centralized_hpt": "centralized",
        "fedavg_hpt": "fedavg",
        "fedvla_hpt": "fedvla",
        "naive_joint_hpt": "fedavg",
        "fedworldmix_hpt": "fedavg",
    }[args.method]
    config.train.aggregation = "eda" if args.method == "fedvla_hpt" else "fedavg"
    config.train.eda_debug = args.method == "fedvla_hpt"
    config.output_dir = str(args.output.parent)
    config.output_file = str(args.output)
    config.save_checkpoint = True
    config.checkpoint_output = str(args.checkpoint)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    if args.diagnostics.exists():
        args.diagnostics.unlink()

    original_train_epoch = trainer_hpt._train_epoch
    if args.method in {"naive_joint_hpt", "fedworldmix_hpt"}:
        trainer_hpt._train_epoch = HPTSourceEpoch(
            SourceUpdateConfig(method=args.method, norm_cap=float(args.norm_cap)),
            diagnostics_path=args.diagnostics,
        )
    try:
        report = trainer_hpt.run_phaseb_hpt(config, dataset_root=str(args.dataset_root))
    finally:
        trainer_hpt._train_epoch = original_train_epoch

    report["isolated_method"] = str(args.method)
    report["source_diagnostics"] = str(args.diagnostics)
    report["source_semantics"] = (
        "recorded_multiview_camera_availability_not_3dgs"
        if args.method in {"naive_joint_hpt", "fedworldmix_hpt"}
        else "real_only_baseline"
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Transformers/PyTorch helper threads can otherwise keep a completed
    # detached job alive after its atomic reports and checkpoint are closed.
    os._exit(int(exit_code))
