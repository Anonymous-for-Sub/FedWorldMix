"""Shared schemas for FedSplat orchestration.

These dataclasses intentionally stay dependency-light so they can be used by
generation scripts, FL training code, and report builders without importing
RoboSplat or VLA training stacks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DomainProfile:
    camera: str = "unknown"
    lighting: str = "unknown"
    background: str = "unknown"
    appearance: str = "unknown"
    embodiment: str = "unknown"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DomainProfile":
        payload = payload or {}
        return cls(
            camera=str(payload.get("camera", "unknown")),
            lighting=str(payload.get("lighting", "unknown")),
            background=str(payload.get("background", "unknown")),
            appearance=str(payload.get("appearance", "unknown")),
            embodiment=str(payload.get("embodiment", "unknown")),
        )


@dataclass(frozen=True)
class GenerationJob:
    label: str
    augment_lighting: bool = False
    augment_appearance: bool = False
    augment_camera_pose: bool = False
    ref_demo_path: str | None = None
    xy_step: tuple[int, int] | None = None
    image_size: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationJob":
        xy_step = payload.get("xy_step")
        return cls(
            label=str(payload["label"]),
            augment_lighting=bool(payload.get("augment_lighting", False)),
            augment_appearance=bool(payload.get("augment_appearance", False)),
            augment_camera_pose=bool(payload.get("augment_camera_pose", False)),
            ref_demo_path=payload.get("ref_demo_path"),
            xy_step=tuple(xy_step) if xy_step is not None else None,
            image_size=payload.get("image_size"),
        )


@dataclass(frozen=True)
class ClientSpec:
    client_id: str
    robot: str
    task: str
    object_categories: tuple[str, ...]
    domain: DomainProfile
    jobs: tuple[GenerationJob, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClientSpec":
        return cls(
            client_id=str(payload["client_id"]),
            robot=str(payload.get("robot", "unknown")),
            task=str(payload.get("task", "unknown")),
            object_categories=tuple(payload.get("object_categories", [])),
            domain=DomainProfile.from_dict(payload.get("domain")),
            jobs=tuple(GenerationJob.from_dict(job) for job in payload.get("jobs", [])),
        )


@dataclass
class GenerationResult:
    client_id: str
    job_label: str
    backend: str
    status: str
    output_dir: str
    command: list[str]
    h5_files: list[str] = field(default_factory=list)
    mp4_files: list[str] = field(default_factory=list)
    image_files: list[str] = field(default_factory=list)
    report_files: list[str] = field(default_factory=list)
    manifest_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClientStats:
    client_id: str
    n_real: float = 0.0
    n_aug: float = 0.0
    q_aug: float = 1.0
    real_synthetic_divergence: float = 0.0
    domain_novelty: float = 1.0
    validation_score: float = 1.0
    upload_cost_mb: float = 1.0
    reliability: float = 1.0
    staleness: float = 0.0
    serialized_bytes: int = 0
    compute_time_s: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClientStats":
        upload_cost_mb = max(float(payload.get("upload_cost_mb", 1.0)), 1e-6)
        serialized_bytes = int(payload.get("serialized_bytes", round(upload_cost_mb * 1_000_000)))
        reliability = payload.get(
            "reliability",
            payload.get("heldout_real_reliability", payload.get("validation_score", 1.0)),
        )
        return cls(
            client_id=str(payload["client_id"]),
            n_real=float(payload.get("n_real", 0.0)),
            n_aug=float(payload.get("n_aug", 0.0)),
            q_aug=float(payload.get("q_aug", 1.0)),
            real_synthetic_divergence=float(payload.get("real_synthetic_divergence", 0.0)),
            domain_novelty=float(payload.get("domain_novelty", 1.0)),
            validation_score=float(payload.get("validation_score", 1.0)),
            upload_cost_mb=upload_cost_mb,
            reliability=min(max(float(reliability), 0.0), 1.0),
            staleness=max(float(payload.get("staleness", payload.get("staleness_rounds", 0.0))), 0.0),
            serialized_bytes=max(serialized_bytes, 0),
            compute_time_s=max(float(payload.get("compute_time_s", 0.0)), 0.0),
        )


def project_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
