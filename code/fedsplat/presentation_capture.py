"""Native high-resolution capture helpers for qualitative paper evidence."""

from __future__ import annotations

import contextlib
import hashlib
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np


MINIMUM_SHORT_DIMENSION = 720
MINIMUM_CRF = 16
MAXIMUM_CRF = 18
PRESENTATION_SOURCE = "policy_domain_observer"
NATIVE_RESOLUTION_STRUCTURAL_GRID = 32
MINIMUM_NATIVE_RESOLUTION_STRUCTURAL_CORRELATION = 0.80
ORIENTATION_TRANSFORMS = {
    "identity": lambda value: value,
    "flip_vertical": lambda value: value[::-1],
    "flip_horizontal": lambda value: value[:, ::-1],
    "flip_both": lambda value: value[::-1, ::-1],
}
MUJOCO_RENDER_STATE_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "mocap_pos",
    "mocap_quat",
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "userdata",
)
MUJOCO_RENDER_MODEL_FIELDS = (
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "jnt_pos",
    "jnt_axis",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_rgba",
    "geom_matid",
    "geom_group",
    "site_pos",
    "site_quat",
    "site_size",
    "site_rgba",
    "site_matid",
    "site_group",
    "cam_pos",
    "cam_quat",
    "cam_fovy",
    "cam_ipd",
    "light_pos",
    "light_dir",
    "light_ambient",
    "light_diffuse",
    "light_specular",
    "light_active",
    "light_castshadow",
    "light_attenuation",
    "light_cutoff",
    "light_exponent",
    "mat_rgba",
    "mat_texid",
    "mat_texrepeat",
    "mat_texuniform",
    "mat_emission",
    "mat_specular",
    "mat_shininess",
    "mat_reflectance",
    "tex_rgb",
    "mesh_vert",
    "mesh_normal",
    "mesh_face",
    "hfield_data",
)


def validate_presentation_capture(
    config: dict[str, Any],
    *,
    rollout_fps: int,
    environment_count: int,
) -> list[str]:
    """Validate the strict separation between presentation and statistics."""
    errors: list[str] = []
    if config.get("enabled") is not True:
        errors.append("presentation_capture.enabled must be true")
    if config.get("used_for_statistical_evidence") is not False:
        errors.append(
            "presentation_capture.used_for_statistical_evidence must be false"
        )
    if config.get("source") != PRESENTATION_SOURCE:
        errors.append(
            f"presentation_capture.source must equal {PRESENTATION_SOURCE}"
        )
    width = config.get("width")
    height = config.get("height")
    for key, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or value <= 0 or value % 2:
            errors.append(f"presentation_capture.{key} must be a positive even integer")
    if (
        isinstance(width, int)
        and isinstance(height, int)
        and min(width, height) < MINIMUM_SHORT_DIMENSION
    ):
        errors.append(
            "presentation_capture shorter dimension must be at least "
            f"{MINIMUM_SHORT_DIMENSION}"
        )
    fps = config.get("fps")
    if fps != rollout_fps:
        errors.append("presentation_capture.fps must equal rollout fps")
    crf = config.get("crf")
    if (
        not isinstance(crf, int)
        or crf < MINIMUM_CRF
        or crf > MAXIMUM_CRF
    ):
        errors.append(
            f"presentation_capture.crf must be in [{MINIMUM_CRF}, {MAXIMUM_CRF}]"
        )
    camera_name = config.get("camera_name")
    if not isinstance(camera_name, str) or not camera_name.strip():
        errors.append("presentation_capture.camera_name must be a nonempty string")
    environment_index = config.get("environment_index")
    if (
        not isinstance(environment_index, int)
        or environment_index < 0
        or environment_index >= environment_count
    ):
        errors.append(
            "presentation_capture.environment_index must select an existing environment"
        )
    if config.get("native_render_only") is not True:
        errors.append("presentation_capture.native_render_only must be true")
    if config.get("dedicated_native_resolution_rendering_twin") is not True:
        errors.append(
            "presentation_capture.dedicated_native_resolution_rendering_twin "
            "must be true"
        )
    if config.get("super_resolution_used") is not False:
        errors.append("presentation_capture.super_resolution_used must be false")
    if config.get("frame_interpolation_used") is not False:
        errors.append("presentation_capture.frame_interpolation_used must be false")
    component_budget = config.get("component_gs_max_gaussians")
    if component_budget is not None and (
        not isinstance(component_budget, int) or component_budget <= 0
    ):
        errors.append(
            "presentation_capture.component_gs_max_gaussians must be a positive integer"
        )
    return errors


def scaled_camera_intrinsic(
    intrinsic: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Scale a pinhole intrinsic without changing camera pose or field of view."""
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Camera dimensions must be positive")
    matrix = np.asarray(intrinsic, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Camera intrinsic must have shape [3,3]")
    scaled = matrix.copy()
    scaled[0, :] *= float(target_width) / float(source_width)
    scaled[1, :] *= float(target_height) / float(source_height)
    return scaled


def render_simulator_observer(
    sim: Any,
    *,
    camera_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    """Render one MuJoCo camera in the policy image orientation."""
    frame = sim.render(
        camera_name=camera_name,
        width=int(width),
        height=int(height),
        segmentation=False,
    )
    # LeRobot's LIBERO wrapper uses this exact transform for visualization.
    frame = np.ascontiguousarray(
        np.asarray(frame, dtype=np.uint8)[::-1, ::-1]
    )
    expected = (int(height), int(width), 3)
    if frame.shape != expected:
        raise ValueError(
            f"Native simulator observer returned {frame.shape}, expected {expected}"
        )
    return frame


def enforce_offscreen_context_ownership(sim: Any) -> dict[str, Any]:
    """Make every offscreen render activate the simulator's own GL context."""
    context = getattr(sim, "_render_context_offscreen", None)
    if context is None:
        raise ValueError("Simulator has no offscreen render context")
    gl_context = getattr(context, "gl_ctx", None)
    if gl_context is None or not callable(getattr(gl_context, "make_current", None)):
        raise ValueError("Simulator offscreen context cannot be made current")
    marker = "_fedsplat_context_ownership_enforced"
    if getattr(context, marker, False):
        return {
            "schema": "fedsplat.offscreen_context_ownership.v1",
            "enforced": True,
            "already_enforced": True,
            "device_id": int(getattr(context, "device_id", -1)),
        }

    original_render = context.render

    def render_with_owned_context(*args, **kwargs):
        gl_context.make_current()
        return original_render(*args, **kwargs)

    context.render = render_with_owned_context
    setattr(context, marker, True)
    return {
        "schema": "fedsplat.offscreen_context_ownership.v1",
        "enforced": True,
        "already_enforced": False,
        "device_id": int(getattr(context, "device_id", -1)),
    }


def mirror_mujoco_render_state(
    source_sim: Any,
    target_sim: Any,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Mirror one MuJoCo state into a rendering-only simulator."""
    copied_fields = []
    for field in MUJOCO_RENDER_STATE_FIELDS:
        if not hasattr(source_sim.data, field) or not hasattr(target_sim.data, field):
            continue
        source_value = getattr(source_sim.data, field)
        target_value = getattr(target_sim.data, field)
        if source_value is None or target_value is None:
            continue
        source = np.asarray(source_value)
        target = np.asarray(target_value)
        if source.shape != target.shape:
            raise ValueError(
                f"Presentation state-mirror field {field} shape mismatch: "
                f"{source.shape} != {target.shape}"
            )
        if source.size:
            target_value[...] = source_value
        copied_fields.append(field)
    if hasattr(source_sim.data, "time") and hasattr(target_sim.data, "time"):
        target_sim.data.time = float(source_sim.data.time)
        copied_fields.append("time")
    target_sim.forward()
    qpos_delta = float(
        np.max(
            np.abs(
                np.asarray(source_sim.data.qpos, dtype=np.float64)
                - np.asarray(target_sim.data.qpos, dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    qvel_delta = float(
        np.max(
            np.abs(
                np.asarray(source_sim.data.qvel, dtype=np.float64)
                - np.asarray(target_sim.data.qvel, dtype=np.float64)
            ),
            initial=0.0,
        )
    )
    required = {"qpos", "qvel", "time"}
    return {
        "schema": "fedsplat.presentation_render_state_mirror.v1",
        "source": "policy_environment_physics_state",
        "destination": "dedicated_native_resolution_rendering_twin",
        "tolerance": float(tolerance),
        "copied_fields": copied_fields,
        "maximum_qpos_delta": qpos_delta,
        "maximum_qvel_delta": qvel_delta,
        "canonical_physics_stepped": False,
        "reward_or_success_queried": False,
        "verified": bool(
            required.issubset(copied_fields)
            and qpos_delta <= tolerance
            and qvel_delta <= tolerance
        ),
    }


def mirror_mujoco_render_model_state(
    source_sim: Any,
    target_sim: Any,
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Mirror static geometry and appearance into a rendering-only simulator."""
    copied_fields = []
    pre_copy_maximum_delta = 0.0
    post_copy_maximum_delta = 0.0
    for field in MUJOCO_RENDER_MODEL_FIELDS:
        if not hasattr(source_sim.model, field) or not hasattr(target_sim.model, field):
            continue
        source_value = getattr(source_sim.model, field)
        target_value = getattr(target_sim.model, field)
        if source_value is None or target_value is None:
            continue
        source = np.asarray(source_value)
        target = np.asarray(target_value)
        if source.shape != target.shape:
            raise ValueError(
                f"Presentation model-mirror field {field} shape mismatch: "
                f"{source.shape} != {target.shape}"
            )
        if source.size:
            pre_copy_maximum_delta = max(
                pre_copy_maximum_delta,
                float(
                    np.max(
                        np.abs(
                            source.astype(np.float64)
                            - target.astype(np.float64)
                        ),
                        initial=0.0,
                    )
                ),
            )
            target_value[...] = source_value
            post_copy_maximum_delta = max(
                post_copy_maximum_delta,
                float(
                    np.max(
                        np.abs(
                            source.astype(np.float64)
                            - np.asarray(target_value).astype(np.float64)
                        ),
                        initial=0.0,
                    )
                ),
            )
        copied_fields.append(field)
    target_sim.forward()
    required = {"body_pos", "geom_rgba", "cam_pos"}
    return {
        "schema": "fedsplat.presentation_render_model_mirror.v1",
        "source": "policy_environment_static_render_model",
        "destination": "dedicated_native_resolution_rendering_twin",
        "tolerance": float(tolerance),
        "copied_fields": copied_fields,
        "pre_copy_maximum_absolute_delta": pre_copy_maximum_delta,
        "post_copy_maximum_absolute_delta": post_copy_maximum_delta,
        "reward_or_success_queried": False,
        "verified": bool(
            required.issubset(copied_fields)
            and post_copy_maximum_delta <= tolerance
        ),
    }


def compare_native_resolution_frames(
    source: np.ndarray,
    target: np.ndarray,
    *,
    maximum_mean_absolute_uint8_error: float = 20.0,
    minimum_rgb_correlation: float = (
        MINIMUM_NATIVE_RESOLUTION_STRUCTURAL_CORRELATION
    ),
) -> dict[str, Any]:
    """Compare native renders while tolerating resolution-dependent edge coverage."""
    from PIL import Image

    source = np.asarray(source, dtype=np.uint8)
    target = np.asarray(target, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"Source frame must be HxWx3, got {source.shape}")
    if target.ndim != 3 or target.shape[2] != 3:
        raise ValueError(f"Target frame must be HxWx3, got {target.shape}")
    source_height, source_width = source.shape[:2]
    target_height, target_width = target.shape[:2]
    downsampled = np.asarray(
        Image.fromarray(target).resize(
            (source_width, source_height),
            Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )
    source_float = source.astype(np.float64)
    target_float = downsampled.astype(np.float64)
    mean_absolute_error = float(np.abs(source_float - target_float).mean())

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        left_float = np.asarray(left, dtype=np.float64)
        right_float = np.asarray(right, dtype=np.float64)
        left_centered = left_float.reshape(-1) - float(left_float.mean())
        right_centered = right_float.reshape(-1) - float(right_float.mean())
        denominator = float(
            np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        )
        if denominator <= 1e-12:
            return 1.0 if np.array_equal(left, right) else 0.0
        return float(np.dot(left_centered, right_centered) / denominator)

    raw_correlation = correlation(source, downsampled)
    structural_size = min(
        NATIVE_RESOLUTION_STRUCTURAL_GRID,
        source_width,
        source_height,
    )
    structural_shape = (structural_size, structural_size)
    structural_source = np.asarray(
        Image.fromarray(source).resize(
            structural_shape,
            Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )
    structural_target = np.asarray(
        Image.fromarray(downsampled).resize(
            structural_shape,
            Image.Resampling.LANCZOS,
        ),
        dtype=np.uint8,
    )
    structural_correlation = correlation(
        structural_source,
        structural_target,
    )
    passed = bool(
        mean_absolute_error <= maximum_mean_absolute_uint8_error
        and structural_correlation >= minimum_rgb_correlation
    )
    return {
        "schema": "fedsplat.native_resolution_frame_comparison.v1",
        "source_resolution": [int(source_width), int(source_height)],
        "target_resolution": [int(target_width), int(target_height)],
        "comparison": "target_lanczos_downsampled_to_source_resolution",
        "mean_absolute_uint8_error": mean_absolute_error,
        "maximum_mean_absolute_uint8_error": float(
            maximum_mean_absolute_uint8_error
        ),
        "rgb_correlation": structural_correlation,
        "raw_rgb_correlation": raw_correlation,
        "structural_rgb_correlation": structural_correlation,
        "structural_comparison_resolution": [
            structural_size,
            structural_size,
        ],
        "minimum_rgb_correlation": float(minimum_rgb_correlation),
        "framebuffer_tiling_or_scene_mismatch_detected": not passed,
        "outcome_or_success_used": False,
        "passed": passed,
    }


def audit_native_resolution_rendering_twin(
    source_sim: Any,
    target_sim: Any,
    *,
    camera_name: str,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    maximum_mean_absolute_uint8_error: float = 20.0,
    minimum_rgb_correlation: float = (
        MINIMUM_NATIVE_RESOLUTION_STRUCTURAL_CORRELATION
    ),
) -> dict[str, Any]:
    """Reject framebuffer tiling or scene mismatch before native capture."""
    source = render_simulator_observer(
        source_sim,
        camera_name=camera_name,
        width=source_width,
        height=source_height,
    )
    target = render_simulator_observer(
        target_sim,
        camera_name=camera_name,
        width=target_width,
        height=target_height,
    )
    comparison = compare_native_resolution_frames(
        source,
        target,
        maximum_mean_absolute_uint8_error=maximum_mean_absolute_uint8_error,
        minimum_rgb_correlation=minimum_rgb_correlation,
    )
    return {
        **comparison,
        "schema": "fedsplat.native_resolution_rendering_twin_audit.v1",
        "camera_name": camera_name,
    }


@contextlib.contextmanager
def _component_render_resolution(
    scene: Any,
    *,
    width: int,
    height: int,
) -> Iterator[None]:
    """Temporarily scale a component-GS camera for a presentation-only render."""
    import torch

    source_width = int(scene.width)
    source_height = int(scene.height)
    original_intrinsic = scene.intrinsic
    original_width = scene.width
    original_height = scene.height
    matrix = original_intrinsic.detach().cpu().numpy()[0]
    scaled = scaled_camera_intrinsic(
        matrix,
        source_width=source_width,
        source_height=source_height,
        target_width=int(width),
        target_height=int(height),
    )
    scene.width = int(width)
    scene.height = int(height)
    scene.intrinsic = torch.as_tensor(
        scaled,
        dtype=original_intrinsic.dtype,
        device=original_intrinsic.device,
    )[None]
    try:
        yield
    finally:
        scene.width = original_width
        scene.height = original_height
        scene.intrinsic = original_intrinsic


def render_component_gs_observer(
    scene: Any,
    sim: Any,
    *,
    width: int,
    height: int,
    orientation_reference: np.ndarray | None = None,
) -> np.ndarray:
    """Render the policy's component-GS domain at native presentation size."""
    with _component_render_resolution(scene, width=width, height=height):
        rgb, _alpha = scene.render(sim)
    orientation = getattr(scene, "_presentation_orientation", None)
    if orientation is None:
        reference = orientation_reference
        selection_source = "matched_native_simulator_observer"
        if reference is None:
            reference = getattr(scene, "reference_rgb", None)
            selection_source = "native_component_scene_reference_rgb"
        if reference is None:
            raise ValueError(
                "Component-GS presentation scene lacks its native policy reference"
            )
        reference = np.asarray(reference, dtype=np.uint8)
        if reference.shape != rgb.shape:
            raise ValueError(
                "Component-GS presentation reference and render shapes differ: "
                f"{reference.shape} versus {rgb.shape}"
            )
        scores = {}
        for name, transform in ORIENTATION_TRANSFORMS.items():
            candidate = np.asarray(transform(rgb), dtype=np.int16)
            scores[name] = float(
                np.abs(candidate - reference.astype(np.int16)).mean()
            )
        orientation = min(scores, key=scores.get)
        scene._presentation_orientation = orientation
        scene.initialization_report["presentation_orientation_audit"] = {
            "schema": "fedsplat.presentation_orientation_audit.v1",
            "selection_source": selection_source,
            "selected_transform": orientation,
            "mean_absolute_uint8_by_transform": scores,
            "outcome_or_success_used": False,
            "frozen_after_first_frame": True,
        }
    rgb = ORIENTATION_TRANSFORMS[orientation](rgb)
    expected = (int(height), int(width), 3)
    if rgb.shape != expected:
        raise ValueError(
            f"Native component-GS observer returned {rgb.shape}, expected {expected}"
        )
    return np.ascontiguousarray(rgb, dtype=np.uint8)


class DirectPresentationRecorder:
    """Write direct native frames without mosaicking or post-hoc upscaling."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        crf: int,
        immutable_native_sensor_resolution: bool = False,
    ) -> None:
        import imageio.v2 as imageio

        if (
            min(width, height) < MINIMUM_SHORT_DIMENSION
            and not immutable_native_sensor_resolution
        ):
            raise ValueError("Presentation recorder source is below the quality floor")
        if width % 2 or height % 2:
            raise ValueError("H.264 presentation dimensions must be even")
        if not MINIMUM_CRF <= crf <= MAXIMUM_CRF:
            raise ValueError("Presentation recorder CRF is outside the frozen range")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.crf = int(crf)
        self.immutable_native_sensor_resolution = bool(
            immutable_native_sensor_resolution
        )
        self.frame_count = 0
        self._writer = imageio.get_writer(
            self.path,
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=None,
            ffmpeg_params=[
                "-profile:v",
                "high",
                "-crf",
                str(self.crf),
                "-movflags",
                "+faststart",
            ],
        )

    def append(self, frame: np.ndarray) -> None:
        value = np.asarray(frame, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if value.shape != expected:
            raise ValueError(
                f"Presentation frame has shape {value.shape}, expected {expected}"
            )
        self._writer.append_data(np.ascontiguousarray(value))
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "DirectPresentationRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def manifest_entry(self, *, source: str, camera_name: str) -> dict[str, Any]:
        if self._writer is not None:
            raise RuntimeError("Close the presentation recorder before manifesting")
        if not self.path.is_file() or self.path.stat().st_size <= 0:
            raise RuntimeError("Presentation video is missing or empty")
        duration = self.frame_count / float(self.fps)
        if not math.isfinite(duration) or duration <= 0:
            raise RuntimeError("Presentation video has no frames")
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {
            "schema": "fedsplat.native_presentation_capture.v1",
            "path": self.path.name,
            "source": source,
            "camera_name": camera_name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "crf": self.crf,
            "codec": "h264",
            "pixel_format": "yuv420p",
            "frame_count": self.frame_count,
            "duration_s": duration,
            "native_render_only": True,
            "super_resolution_used": False,
            "frame_interpolation_used": False,
            "used_for_statistical_evidence": False,
            "immutable_native_sensor_resolution": (
                self.immutable_native_sensor_resolution
            ),
            "bytes": self.path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
