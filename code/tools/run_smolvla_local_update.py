#!/usr/bin/env python3
"""Run one audited SmolVLA local update from a relocatable base config."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fedsplat.smolvla_local_update import (
    _worker,
    _write_worker_report,
    validate_local_update_spec,
)


def run(config_path: Path, out_dir: Path) -> dict:
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    spec["out_dir"] = str(out_dir)
    errors = validate_local_update_spec(spec)
    if errors:
        raise ValueError("Invalid worker config: " + "; ".join(errors))
    worker_config = out_dir / "worker_config.json"
    worker_config.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    report = _worker(spec)
    report["runtime_wall_s"] = time.perf_counter() - started
    (out_dir / "local_update_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_worker_report(report, out_dir / "SMOLVLA_LOCAL_UPDATE_REPORT.md")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.config, args.out_dir)
    print(
        json.dumps(
            {
                "report": str((args.out_dir / "local_update_report.json").resolve()),
                "local_update_ready": report["local_update_ready"],
                "selected_branch": (report.get("crossfit") or {}).get(
                    "selected_branch"
                ),
                "steps_completed": report["steps_completed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
