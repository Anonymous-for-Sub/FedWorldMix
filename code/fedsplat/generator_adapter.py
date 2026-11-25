"""FedSplat client-side Gaussian generation adapter.

The adapter runs a project-local generator runtime prepared under
`materials/`.  The first supported runtime is `robosplat_compatible`, which
uses the same command contract as the RoboSplat seed code after it has been
imported into this project.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import ClientSpec, GenerationJob, GenerationResult


class GaussianClientGenerator:
    def __init__(
        self,
        material_root: Path,
        python: str = "python3",
        output_root: Path | None = None,
        backend: str = "robosplat_compatible",
        default_ref_demo_path: str = "data/source_demo/real_000000.h5",
        default_xy_step: tuple[int, int] = (2, 2),
        default_image_size: int = 256,
    ) -> None:
        self.material_root = Path(material_root)
        self.python = python
        self.output_root = Path(output_root or "outputs/client_generation")
        self.backend = backend
        self.default_ref_demo_path = default_ref_demo_path
        self.default_xy_step = default_xy_step
        self.default_image_size = default_image_size

    def build_command(self, job: GenerationJob, output_dir: Path) -> list[str]:
        if self.backend != "robosplat_compatible":
            raise ValueError(f"Unsupported generator backend: {self.backend}")
        ref_demo = job.ref_demo_path or self.default_ref_demo_path
        xy_step = job.xy_step or self.default_xy_step
        image_size = job.image_size or self.default_image_size
        return [
            self.python,
            "data_aug/generate_demo.py",
            "--image_size",
            str(image_size),
            "--save",
            "True",
            "--save_video",
            "True",
            "--ref_demo_path",
            ref_demo,
            "--xy_step_str",
            f"[{xy_step[0]}, {xy_step[1]}]",
            "--augment_lighting",
            str(job.augment_lighting),
            "--augment_appearance",
            str(job.augment_appearance),
            "--augment_camera_pose",
            str(job.augment_camera_pose),
            "--output_path",
            str(output_dir),
        ]

    def run_job(self, client: ClientSpec, job: GenerationJob, dry_run: bool = True) -> GenerationResult:
        output_dir = self.output_root / client.client_id / job.label
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(job, output_dir)

        status = "planned"
        error = None
        if not dry_run:
            self._validate_runtime()
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.material_root,
                    check=True,
                    text=True,
                    capture_output=True,
                )
                (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
                (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
                status = "ok"
            except subprocess.CalledProcessError as exc:
                status = "failed"
                error = f"returncode={exc.returncode}"
                (output_dir / "stdout.txt").write_text(exc.stdout or "", encoding="utf-8")
                (output_dir / "stderr.txt").write_text(exc.stderr or "", encoding="utf-8")

        result = self.collect_result(client.client_id, job.label, output_dir, command, status, error)
        self.write_manifest(client, job, result, dry_run=dry_run)
        return result

    def collect_result(
        self,
        client_id: str,
        job_label: str,
        output_dir: Path,
        command: list[str],
        status: str,
        error: str | None,
    ) -> GenerationResult:
        h5_files = sorted(str(path) for path in output_dir.rglob("*.h5"))
        mp4_files = sorted(str(path) for path in output_dir.rglob("*.mp4"))
        image_files = sorted(
            str(path)
            for path in output_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        report_files = sorted(
            str(path)
            for path in output_dir.rglob("*")
            if path.suffix.lower() in {".md", ".json", ".txt"}
        )
        manifest_path = output_dir / "fedsplat_manifest.json"
        return GenerationResult(
            client_id=client_id,
            job_label=job_label,
            backend=self.backend,
            status=status,
            output_dir=str(output_dir),
            command=command,
            h5_files=h5_files,
            mp4_files=mp4_files,
            image_files=image_files,
            report_files=report_files,
            manifest_path=str(manifest_path),
            error=error,
        )

    def write_manifest(
        self,
        client: ClientSpec,
        job: GenerationJob,
        result: GenerationResult,
        dry_run: bool,
    ) -> None:
        payload: dict[str, Any] = {
            "schema": "fedsplat.client_generation_manifest.v2",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "client": {
                "client_id": client.client_id,
                "robot": client.robot,
                "task": client.task,
                "object_categories": list(client.object_categories),
                "domain": asdict(client.domain),
            },
            "job": asdict(job),
            "material_contract": {
                "backend": self.backend,
                "material_root": str(self.material_root),
                "runtime_prepared_inside_project": True,
            },
            "privacy_boundary": {
                "local_only": [
                    "raw demonstrations",
                    "raw multi-view images",
                    "3D Gaussian scenes",
                    "generated demonstrations",
                    "videos",
                    "trajectories",
                ],
                "upload_allowed": [
                    "model updates",
                    "low-dimensional quality statistics",
                    "low-dimensional domain statistics",
                    "communication/resource status",
                ],
            },
            "result": result.to_dict(),
            "preserved_generator_outputs": [".h5", ".mp4", ".png/.jpg screenshots", ".md/.json/.txt reports"],
        }
        manifest_path = Path(result.manifest_path or Path(result.output_dir) / "fedsplat_manifest.json")
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _validate_runtime(self) -> None:
        script = self.material_root / "data_aug" / "generate_demo.py"
        if not script.exists():
            raise FileNotFoundError(
                f"Missing generator runtime script: {script}. "
                "Run `python -m fedsplat.cli prepare-materials --source <seed_source>` first."
            )


def load_client_config(path: Path) -> tuple[dict[str, Any], list[ClientSpec]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    clients = [ClientSpec.from_dict(item) for item in payload.get("clients", [])]
    return payload, clients
