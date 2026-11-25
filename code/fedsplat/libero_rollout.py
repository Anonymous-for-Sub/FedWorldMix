"""Project-owned single-task LIBERO rollout adapter for external LeRobot runtimes."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any


def validate_rollout_spec(spec: dict[str, Any]) -> list[str]:
    errors = []
    for key in ("checkpoint_path", "libero_datasets_path", "suite", "device"):
        if not isinstance(spec.get(key), str) or not str(spec[key]).strip():
            errors.append(f"{key} must be a nonempty string")
    for key in ("task_id", "seed"):
        if not isinstance(spec.get(key), int) or int(spec[key]) < 0:
            errors.append(f"{key} must be a nonnegative integer")
    if isinstance(spec.get("task_id"), int) and int(spec["task_id"]) > 9:
        errors.append("task_id must be in [0, 9] for a ten-task LIBERO suite")
    for key in ("num_episodes", "episode_length", "fps", "observation_width", "observation_height"):
        if not isinstance(spec.get(key), int) or int(spec[key]) <= 0:
            errors.append(f"{key} must be a positive integer")
    if "deterministic" in spec and not isinstance(spec["deterministic"], bool):
        errors.append("deterministic must be a boolean")
    if "libero_resource_root" in spec and (
        not isinstance(spec["libero_resource_root"], str)
        or not spec["libero_resource_root"].strip()
    ):
        errors.append("libero_resource_root must be a nonempty string when provided")
    for key in ("expected_bddl_tokens", "forbidden_bddl_tokens"):
        if key in spec and (
            not isinstance(spec[key], list)
            or any(not isinstance(value, str) or not value for value in spec[key])
        ):
            errors.append(f"{key} must be a string list")
    target_pattern = spec.get("diagnostic_target_body_pattern")
    if target_pattern is not None and (
        not isinstance(target_pattern, str) or not target_pattern.strip()
    ):
        errors.append("diagnostic_target_body_pattern must be a nonempty string when provided")
    adapter_bank = spec.get("adapter_bank_manifest")
    visual_router = spec.get("visual_router_manifest")
    for key, value in (
        ("adapter_bank_manifest", adapter_bank),
        ("visual_router_manifest", visual_router),
    ):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{key} must be a nonempty string when provided")
    if (adapter_bank is None) != (visual_router is None):
        errors.append(
            "adapter_bank_manifest and visual_router_manifest must be provided together"
        )
    visual_shift = spec.get("visual_domain_shift")
    if visual_shift is not None:
        if not isinstance(visual_shift, dict):
            errors.append("visual_domain_shift must be an object")
        else:
            if not isinstance(visual_shift.get("name"), str) or not visual_shift.get("name", "").strip():
                errors.append("visual_domain_shift.name must be a nonempty string")
            camera_name = visual_shift.get("camera_name", "agentview")
            if not isinstance(camera_name, str) or not camera_name.strip():
                errors.append("visual_domain_shift.camera_name must be a nonempty string")
            camera_delta = visual_shift.get("camera_position_delta_m", [0.0, 0.0, 0.0])
            if (
                not isinstance(camera_delta, list)
                or len(camera_delta) != 3
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or abs(float(value)) > 0.25
                    for value in camera_delta
                )
            ):
                errors.append(
                    "visual_domain_shift.camera_position_delta_m must contain three finite values in [-0.25, 0.25]"
                )
            camera_rotation = visual_shift.get("camera_rotation_delta_deg", [0.0, 0.0, 0.0])
            if (
                not isinstance(camera_rotation, list)
                or len(camera_rotation) != 3
                or any(
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or abs(float(value)) > 30.0
                    for value in camera_rotation
                )
            ):
                errors.append(
                    "visual_domain_shift.camera_rotation_delta_deg must contain three finite values in [-30, 30]"
                )
            for key in ("light_ambient_scale", "light_diffuse_scale", "light_specular_scale"):
                value = visual_shift.get(key, 1.0)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    or float(value) > 4.0
                ):
                    errors.append(f"visual_domain_shift.{key} must be finite and in (0, 4]")
            body_pattern = visual_shift.get("target_body_pattern")
            if body_pattern is not None and (
                not isinstance(body_pattern, str) or not body_pattern.strip()
            ):
                errors.append("visual_domain_shift.target_body_pattern must be a nonempty string")
            target_rgb = visual_shift.get("target_rgb")
            if target_rgb is not None and (
                not isinstance(target_rgb, list)
                or len(target_rgb) != 3
                or any(
                    not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
                    for value in target_rgb
                )
            ):
                errors.append("visual_domain_shift.target_rgb must contain three values in [0, 1]")
            background_patterns = visual_shift.get("background_body_patterns", [])
            if (
                not isinstance(background_patterns, list)
                or any(not isinstance(value, str) or not value.strip() for value in background_patterns)
            ):
                errors.append("visual_domain_shift.background_body_patterns must be a string list")
            background_rgb = visual_shift.get("background_rgb")
            if background_rgb is not None and (
                not isinstance(background_rgb, list)
                or len(background_rgb) != 3
                or any(
                    not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0
                    for value in background_rgb
                )
            ):
                errors.append("visual_domain_shift.background_rgb must contain three values in [0, 1]")
            if background_rgb is not None and not background_patterns:
                errors.append(
                    "visual_domain_shift.background_body_patterns is required with background_rgb"
                )
            has_mutation = (
                any(abs(float(value)) > 1e-12 for value in camera_delta)
                or any(abs(float(value)) > 1e-12 for value in camera_rotation)
                or any(
                    abs(float(visual_shift.get(key, 1.0)) - 1.0) > 1e-12
                    for key in (
                        "light_ambient_scale",
                        "light_diffuse_scale",
                        "light_specular_scale",
                    )
                )
                or target_rgb is not None
                or background_rgb is not None
            )
            if not has_mutation:
                errors.append("visual_domain_shift must request at least one non-identity mutation")
    provider = spec.get("policy_observation_provider")
    if provider is not None:
        if not isinstance(provider, dict):
            errors.append("policy_observation_provider must be an object")
        else:
            provider_modes = {
                "simulator_canonical_shadow",
                "simulator_canonical_state_mirror",
                "component_gs_state_mirror_pose_oracle",
            }
            if provider.get("mode") not in provider_modes:
                errors.append(
                    "policy_observation_provider.mode must be one of "
                    f"{sorted(provider_modes)}"
                )
            component_gs_mode = (
                provider.get("mode") == "component_gs_state_mirror_pose_oracle"
            )
            expected_role = (
                "development_component_gs_pose_oracle_only"
                if component_gs_mode
                else "development_oracle_upper_bound_only"
            )
            if provider.get("evidence_role") != expected_role:
                errors.append(
                    f"policy_observation_provider.evidence_role must be {expected_role}"
                )
            tolerance = provider.get("state_sync_tolerance", 1e-9)
            if (
                not isinstance(tolerance, (int, float))
                or not math.isfinite(float(tolerance))
                or not 0.0 < float(tolerance) <= 1e-6
            ):
                errors.append(
                    "policy_observation_provider.state_sync_tolerance must be finite and in (0, 1e-6]"
                )
            if component_gs_mode:
                if provider.get("camera_name", "agentview") != "agentview":
                    errors.append(
                        "component-GS provider currently requires camera_name=agentview"
                    )
                if provider.get("canonicalized_pixel_keys") != ["image"]:
                    errors.append(
                        "component-GS provider canonicalized_pixel_keys must equal ['image']"
                    )
                for key in (
                    "base_scale_factor",
                    "base_opacity",
                    "surfel_normal_scale_ratio",
                    "auxiliary_scale_factor",
                    "auxiliary_opacity",
                    "dynamic_auxiliary_scale_factor",
                    "dynamic_auxiliary_opacity",
                    "auxiliary_novel_voxel_size_m",
                    "dynamic_auxiliary_novel_voxel_size_m",
                    "auxiliary_occlusion_margin_m",
                    "maximum_dynamic_body_distance_median_m",
                    "maximum_dynamic_body_distance_p95_m",
                    "max_depth_m",
                ):
                    value = provider.get(key)
                    if (
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) <= 0.0
                    ):
                        errors.append(f"policy_observation_provider.{key} must be positive")
                support_margin = provider.get("dynamic_geom_support_margin_m")
                if support_margin is not None and (
                    not isinstance(support_margin, (int, float))
                    or not math.isfinite(float(support_margin))
                    or not 0.0 <= float(support_margin) <= 0.05
                ):
                    errors.append(
                        "policy_observation_provider.dynamic_geom_support_margin_m "
                        "must be finite and in [0, 0.05]"
                    )
                for key in (
                    "base_opacity",
                    "auxiliary_opacity",
                    "dynamic_auxiliary_opacity",
                ):
                    value = provider.get(key)
                    if isinstance(value, (int, float)) and not 0.0 < float(value) < 1.0:
                        errors.append(f"policy_observation_provider.{key} must lie in (0, 1)")
                ratio = provider.get("surfel_normal_scale_ratio")
                if isinstance(ratio, (int, float)) and not 0.0 < float(ratio) < 1.0:
                    errors.append(
                        "policy_observation_provider.surfel_normal_scale_ratio must lie in (0, 1)"
                    )
                if (
                    not isinstance(provider.get("max_gaussians"), int)
                    or int(provider["max_gaussians"]) <= 0
                ):
                    errors.append(
                        "policy_observation_provider.max_gaussians must be a positive integer"
                    )
                if provider.get("auxiliary_dynamic_visibility_mode") not in {
                    "all_visible",
                    "occluded_only",
                }:
                    errors.append(
                        "policy_observation_provider.auxiliary_dynamic_visibility_mode "
                        "must be all_visible or occluded_only"
                    )
                calibration_body_patterns = provider.get(
                    "articulation_calibration_body_patterns"
                )
                calibration_joint_patterns = provider.get(
                    "articulation_calibration_joint_patterns"
                )
                for key, patterns in (
                    ("articulation_calibration_body_patterns", calibration_body_patterns),
                    ("articulation_calibration_joint_patterns", calibration_joint_patterns),
                ):
                    if (
                        not isinstance(patterns, list)
                        or not patterns
                        or any(
                            not isinstance(pattern, str) or not pattern.strip()
                            for pattern in patterns
                        )
                    ):
                        errors.append(
                            f"policy_observation_provider.{key} must be a non-empty string list"
                        )
                calibration_offsets = provider.get(
                    "articulation_calibration_qpos_offsets"
                )
                if (
                    not isinstance(calibration_offsets, list)
                    or not 1 <= len(calibration_offsets) <= 16
                    or any(not isinstance(offsets, list) for offsets in calibration_offsets)
                    or any(not offsets for offsets in calibration_offsets)
                    or len({len(offsets) for offsets in calibration_offsets}) != 1
                    or any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for offsets in calibration_offsets
                        for value in offsets
                    )
                ):
                    errors.append(
                        "policy_observation_provider.articulation_calibration_qpos_offsets "
                        "must contain 1--16 equal-length finite numeric lists"
                    )
                if not isinstance(provider.get("static_disocclusion_completion"), bool):
                    errors.append(
                        "policy_observation_provider.static_disocclusion_completion "
                        "must be a boolean"
                    )
                if not isinstance(
                    provider.get("dynamic_actor_local_voxel_fusion", False), bool
                ):
                    errors.append(
                        "policy_observation_provider.dynamic_actor_local_voxel_fusion "
                        "must be a boolean"
                    )
                albedo_scale = provider.get("articulation_material_albedo_scale")
                if (
                    not isinstance(albedo_scale, (int, float))
                    or not math.isfinite(float(albedo_scale))
                    or not 0.0 < float(albedo_scale) <= 1.5
                ):
                    errors.append(
                        "policy_observation_provider.articulation_material_albedo_scale "
                        "must be finite and in (0, 1.5]"
                    )
                albedo_blend = provider.get("articulation_material_albedo_blend")
                if (
                    not isinstance(albedo_blend, (int, float))
                    or not math.isfinite(float(albedo_blend))
                    or not 0.0 <= float(albedo_blend) <= 1.0
                ):
                    errors.append(
                        "policy_observation_provider.articulation_material_albedo_blend "
                        "must be finite and in [0, 1]"
                    )
                if not isinstance(provider.get("articulation_backface_culling"), bool):
                    errors.append(
                        "policy_observation_provider.articulation_backface_culling "
                        "must be a boolean"
                    )
                relighting_intensity = provider.get(
                    "articulation_relighting_intensity_scale"
                )
                if (
                    not isinstance(relighting_intensity, (int, float))
                    or not math.isfinite(float(relighting_intensity))
                    or not 0.0 < float(relighting_intensity) <= 2.0
                ):
                    errors.append(
                        "policy_observation_provider."
                        "articulation_relighting_intensity_scale "
                        "must be finite and in (0, 2]"
                    )
                relighting_gamma = provider.get("articulation_relighting_gamma")
                if (
                    not isinstance(relighting_gamma, (int, float))
                    or not math.isfinite(float(relighting_gamma))
                    or not 0.0 < float(relighting_gamma) <= 5.0
                ):
                    errors.append(
                        "policy_observation_provider.articulation_relighting_gamma "
                        "must be finite and in (0, 5]"
                    )
                auxiliary_views = provider.get("auxiliary_views")
                if not isinstance(auxiliary_views, list) or len(auxiliary_views) > 8:
                    errors.append(
                        "policy_observation_provider.auxiliary_views must be a list of at most 8 views"
                    )
                else:
                    for index, view in enumerate(auxiliary_views):
                        if not isinstance(view, dict):
                            errors.append(f"auxiliary_views[{index}] must be an object")
                            continue
                        for key in ("position_delta_m", "rotation_delta_deg"):
                            values = view.get(key)
                            if (
                                not isinstance(values, list)
                                or len(values) != 3
                                or any(
                                    not isinstance(value, (int, float))
                                    or not math.isfinite(float(value))
                                    for value in values
                                )
                            ):
                                errors.append(
                                    f"auxiliary_views[{index}].{key} must contain three finite numbers"
                                )
                background = provider.get("background_rgb", [0.0, 0.0, 0.0])
                if (
                    not isinstance(background, list)
                    or len(background) != 3
                    or any(
                        not isinstance(value, (int, float))
                        or not 0.0 <= float(value) <= 1.0
                        for value in background
                    )
                ):
                    errors.append(
                        "policy_observation_provider.background_rgb must contain three values in [0, 1]"
                    )
                quality_gate = provider.get("startup_quality_gate")
                if not isinstance(quality_gate, dict):
                    errors.append(
                        "policy_observation_provider.startup_quality_gate must be an object"
                    )
                elif any(
                    key not in quality_gate
                    for key in (
                        "minimum_psnr_db",
                        "minimum_ssim",
                        "minimum_mean_alpha",
                        "maximum_mean_absolute_uint8",
                    )
                ):
                    errors.append(
                        "policy_observation_provider.startup_quality_gate is incomplete"
                    )
                diagnostic_steps = provider.get("diagnostic_reference_steps", [])
                if (
                    not isinstance(diagnostic_steps, list)
                    or any(not isinstance(value, int) or value < 0 for value in diagnostic_steps)
                    or diagnostic_steps != sorted(set(diagnostic_steps))
                    or any(value > int(spec.get("episode_length", 0)) for value in diagnostic_steps)
                ):
                    errors.append(
                        "policy_observation_provider.diagnostic_reference_steps must be sorted unique steps within the episode"
                    )
    presentation = spec.get("presentation_capture")
    if presentation is not None:
        if not isinstance(presentation, dict):
            errors.append("presentation_capture must be an object")
        else:
            from .presentation_capture import validate_presentation_capture

            errors.extend(
                validate_presentation_capture(
                    presentation,
                    rollout_fps=int(spec.get("fps", 0)),
                    environment_count=int(spec.get("num_episodes", 0)),
                )
            )
    return errors


def run_libero_rollout(
    config_path: Path,
    out_dir: Path,
    runtime_python: Path,
    libero_site: Path,
    pythonpath: list[Path] | None = None,
    adapter_update: Path | None = None,
    adapter_bank_manifest: Path | None = None,
    visual_router_manifest: Path | None = None,
    cuda_visible_device: str | None = None,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = _find_project_root(config_path)
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    errors = validate_rollout_spec(spec)
    if errors:
        raise ValueError("Invalid LIBERO rollout config: " + "; ".join(errors))
    if adapter_update is not None and adapter_bank_manifest is not None:
        raise ValueError("Static adapter_update and routed adapter bank are mutually exclusive")
    if (adapter_bank_manifest is None) != (visual_router_manifest is None):
        raise ValueError(
            "adapter_bank_manifest and visual_router_manifest must be provided together"
        )

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_spec = dict(spec)
    worker_spec["checkpoint_path"] = str(_project_path(project_root, spec["checkpoint_path"]))
    worker_spec["libero_datasets_path"] = str(_project_path(project_root, spec["libero_datasets_path"]))
    worker_spec["libero_site"] = str(libero_site.resolve())
    if spec.get("libero_resource_root"):
        worker_spec["libero_resource_root"] = str(
            _project_path(project_root, spec["libero_resource_root"])
        )
    worker_spec["adapter_update"] = str(adapter_update.resolve()) if adapter_update else None
    worker_spec["adapter_bank_manifest"] = (
        str(adapter_bank_manifest.resolve()) if adapter_bank_manifest else None
    )
    worker_spec["visual_router_manifest"] = (
        str(visual_router_manifest.resolve()) if visual_router_manifest else None
    )
    worker_spec["out_dir"] = str(out_dir)
    worker_config = out_dir / "worker_config.json"
    worker_config.write_text(json.dumps(worker_spec, indent=2) + "\n", encoding="utf-8")

    runtime_python = runtime_python.absolute()
    if not runtime_python.exists():
        raise FileNotFoundError(f"Runtime Python does not exist: {runtime_python}")
    if not libero_site.exists():
        raise FileNotFoundError(f"LIBERO site-packages directory does not exist: {libero_site}")
    env = os.environ.copy()
    paths = [project_root, *(pythonpath or [])]
    prefix = os.pathsep.join(str(path.resolve()) for path in paths)
    env["PYTHONPATH"] = prefix + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    # LIBERO init-state files are trusted local benchmark artifacts saved with
    # pickle. PyTorch >=2.6 otherwise rejects them when callers omit the flag.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if bool(spec.get("deterministic", True)):
        env["PYTHONHASHSEED"] = str(spec["seed"])
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if cuda_visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)
        env["MUJOCO_EGL_DEVICE_ID"] = str(cuda_visible_device)
    completed = subprocess.run(
        [str(runtime_python), "-m", "fedsplat.libero_rollout", "--worker-config", str(worker_config)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
    )
    (out_dir / "worker_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "worker_stderr.log").write_text(completed.stderr, encoding="utf-8")
    report_path = out_dir / "rollout_report.json"
    if completed.returncode != 0 or not report_path.exists():
        detail = (completed.stderr or completed.stdout or "worker produced no report")[-5000:]
        _write_json_atomic(out_dir / "rollout_failure.json", {
            "schema": "fedsplat.libero_rollout_failure.v1",
            "returncode": completed.returncode,
            "worker_config": str(worker_config),
            "detail": detail,
        })
        raise RuntimeError(f"LIBERO rollout worker failed ({completed.returncode}): {detail}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _find_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "fedsplat").is_dir() and (candidate / "materials").is_dir():
            return candidate
    return config_path.parent.parent


def _bootstrap_libero(
    libero_site: Path,
    config_root: Path,
    datasets_root: Path,
    resource_root: Path | None = None,
) -> dict[str, str]:
    import sys
    import yaml

    libero_site = libero_site.resolve()
    candidates = [
        libero_site / "libero" / "libero",
        libero_site / "libero",
    ]
    package_root = next(
        (candidate for candidate in candidates if (candidate / "bddl_files").exists() and (candidate / "init_files").exists()),
        None,
    )
    if package_root is None:
        raise FileNotFoundError(f"LIBERO bddl/init resources not found from {libero_site}")
    resource_root = resource_root.resolve() if resource_root else package_root
    for required in ("bddl_files", "init_files"):
        if not (resource_root / required).is_dir():
            raise FileNotFoundError(f"LIBERO resource root is missing {required}: {resource_root}")

    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(resource_root / "bddl_files"),
        "init_states": str(resource_root / "init_files"),
        "datasets": str(datasets_root),
        "assets": str(package_root / "assets"),
    }
    (config_root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    # LIBERO captures LIBERO_CONFIG_PATH at import time. Set both the resource
    # config and import precedence before any package import in this worker.
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)
    outer_package = package_root.parent
    python_root = outer_package.parent
    import_roots = [python_root]
    if not (outer_package / "__init__.py").is_file():
        # Some LIBERO-PRO checkouts rely on an editable install to synthesize
        # the outer package. A tiny namespace shim prevents a preinstalled
        # legacy LIBERO from silently taking precedence.
        overlay_root = config_root / "python_overlay"
        overlay_package = overlay_root / "libero"
        overlay_package.mkdir(parents=True, exist_ok=True)
        (overlay_package / "__init__.py").write_text(
            "from pkgutil import extend_path\n"
            "__path__ = extend_path(__path__, __name__)\n",
            encoding="utf-8",
        )
        import_roots.insert(0, overlay_root)
        config["compatibility_overlay"] = str(overlay_root)
    for root in reversed(import_roots):
        root_string = str(root)
        if root_string in sys.path:
            sys.path.remove(root_string)
        sys.path.insert(0, root_string)
    config["python_root"] = str(python_root)
    return config


def _apply_static_adapter(policy, update_path: Path) -> list[str]:
    import torch
    from torch import nn
    from torch.nn import functional as functional

    from .lora_updates import load_client_update

    class StaticAdapterLinear(nn.Module):
        def __init__(self, base: nn.Linear, a, b) -> None:
            super().__init__()
            self.base = base
            self.register_buffer("adapter_a", torch.as_tensor(a, dtype=torch.float32, device=base.weight.device))
            self.register_buffer("adapter_b", torch.as_tensor(b, dtype=torch.float32, device=base.weight.device))

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        @property
        def in_features(self) -> int:
            return self.base.in_features

        @property
        def out_features(self) -> int:
            return self.base.out_features

        def forward(self, inputs):
            base_output = self.base(inputs)
            update = functional.linear(functional.linear(inputs.float(), self.adapter_a), self.adapter_b)
            return base_output + update.to(dtype=base_output.dtype)

    update = load_client_update(update_path)
    applied = []
    for layer in update.layers:
        parent_name, separator, attribute = layer.layer_name.rpartition(".")
        parent = policy if not separator else policy.get_submodule(parent_name)
        base = getattr(parent, attribute)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Adapter target is not torch.nn.Linear: {layer.layer_name}")
        setattr(parent, attribute, StaticAdapterLinear(base, layer.a, layer.b))
        applied.append(layer.layer_name)
    return applied


def _success_flags(info: dict[str, Any], num_envs: int):
    import numpy as np

    success = np.zeros(num_envs, dtype=bool)
    for key in ("is_success", "success"):
        if key in info:
            values = np.asarray(info[key], dtype=bool).reshape(-1)
            success[: min(len(values), num_envs)] = values[:num_envs]
    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        for key in ("is_success", "success"):
            if key in final_info:
                values = np.asarray(final_info[key], dtype=bool).reshape(-1)
                success[: min(len(values), num_envs)] |= values[:num_envs]
    elif final_info is not None:
        for index, item in enumerate(final_info):
            if isinstance(item, dict):
                success[index] |= bool(item.get("is_success", item.get("success", False)))
    return success


def _frames_from_observation(raw_observation):
    import numpy as np

    frames = np.asarray(raw_observation["pixels"]["image"])
    if frames.ndim == 3:
        frames = frames[None, ...]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected vector RGB observations [N,H,W,3], got {frames.shape}")
    return np.ascontiguousarray(frames[:, ::-1, ::-1], dtype=np.uint8)


def _stack_vector_observations(values):
    import numpy as np

    first = values[0]
    if isinstance(first, dict):
        return {
            key: _stack_vector_observations([value[key] for value in values])
            for key in first
        }
    return np.stack(values)


def _compose_canonical_policy_observation(execution_observation, canonical_observation):
    """Use canonical pixels with execution-side proprioception and task state."""
    import numpy as np

    execution_pixels = execution_observation.get("pixels")
    canonical_pixels = canonical_observation.get("pixels")
    if not isinstance(execution_pixels, dict) or not isinstance(canonical_pixels, dict):
        raise ValueError("Canonical observation composition requires pixel dictionaries")
    if set(execution_pixels) != set(canonical_pixels):
        raise ValueError(
            "Execution/canonical pixel keys differ: "
            f"{sorted(execution_pixels)} != {sorted(canonical_pixels)}"
        )
    policy_pixels = {}
    for key in execution_pixels:
        execution_value = np.asarray(execution_pixels[key])
        canonical_value = np.asarray(canonical_pixels[key])
        if execution_value.shape != canonical_value.shape:
            raise ValueError(
                f"Execution/canonical pixel shape differs for {key}: "
                f"{execution_value.shape} != {canonical_value.shape}"
            )
        policy_pixels[key] = canonical_pixels[key]
    composed = dict(execution_observation)
    composed["pixels"] = policy_pixels
    return composed


def _compare_environment_physics(
    execution_environment,
    canonical_environment,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Compare vectorized MuJoCo state without consulting reward or policy output."""
    import numpy as np

    execution_sims = _environment_sims(execution_environment)
    canonical_sims = _environment_sims(canonical_environment)
    if len(execution_sims) != len(canonical_sims):
        raise ValueError("Execution/canonical environment counts differ")
    rows = []
    for index, (execution_sim, canonical_sim) in enumerate(
        zip(execution_sims, canonical_sims, strict=True)
    ):
        execution_qpos = np.asarray(execution_sim.data.qpos, dtype=np.float64)
        canonical_qpos = np.asarray(canonical_sim.data.qpos, dtype=np.float64)
        execution_qvel = np.asarray(execution_sim.data.qvel, dtype=np.float64)
        canonical_qvel = np.asarray(canonical_sim.data.qvel, dtype=np.float64)
        if execution_qpos.shape != canonical_qpos.shape:
            raise ValueError(f"qpos shape mismatch in environment {index}")
        if execution_qvel.shape != canonical_qvel.shape:
            raise ValueError(f"qvel shape mismatch in environment {index}")
        qpos_delta = float(np.max(np.abs(execution_qpos - canonical_qpos), initial=0.0))
        qvel_delta = float(np.max(np.abs(execution_qvel - canonical_qvel), initial=0.0))
        rows.append(
            {
                "environment_index": index,
                "maximum_qpos_delta": qpos_delta,
                "maximum_qvel_delta": qvel_delta,
                "verified": qpos_delta <= tolerance and qvel_delta <= tolerance,
            }
        )
    return {
        "schema": "fedsplat.canonical_provider_physics_sync.v1",
        "tolerance": float(tolerance),
        "environment_count": len(rows),
        "maximum_qpos_delta": max(
            (row["maximum_qpos_delta"] for row in rows), default=0.0
        ),
        "maximum_qvel_delta": max(
            (row["maximum_qvel_delta"] for row in rows), default=0.0
        ),
        "rows": rows,
        "verified": bool(rows) and all(row["verified"] for row in rows),
    }


def _mirror_environment_render_state(
    execution_environment,
    canonical_environment,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Mirror execution state into a rendering-only canonical MuJoCo twin."""
    from .presentation_capture import mirror_mujoco_render_state

    execution_sims = _environment_sims(execution_environment)
    canonical_sims = _environment_sims(canonical_environment)
    if len(execution_sims) != len(canonical_sims):
        raise ValueError("Execution/canonical environment counts differ")
    rows = []
    for index, (execution_sim, canonical_sim) in enumerate(
        zip(execution_sims, canonical_sims, strict=True)
    ):
        audit = mirror_mujoco_render_state(
            execution_sim,
            canonical_sim,
            tolerance=tolerance,
        )
        rows.append(
            {
                "environment_index": index,
                "copied_fields": audit["copied_fields"],
                "maximum_qpos_delta": audit["maximum_qpos_delta"],
                "maximum_qvel_delta": audit["maximum_qvel_delta"],
                "verified": audit["verified"],
            }
        )
    required = {"qpos", "qvel", "time"}
    return {
        "schema": "fedsplat.canonical_render_state_mirror.v1",
        "tolerance": float(tolerance),
        "environment_count": len(rows),
        "source": "execution_environment_physics_state",
        "destination": "rendering_only_canonical_twin",
        "canonical_physics_stepped": False,
        "canonical_reward_or_success_queried": False,
        "required_fields": sorted(required),
        "maximum_qpos_delta": max(
            (row["maximum_qpos_delta"] for row in rows), default=0.0
        ),
        "maximum_qvel_delta": max(
            (row["maximum_qvel_delta"] for row in rows), default=0.0
        ),
        "rows": rows,
        "verified": bool(rows)
        and all(row["verified"] for row in rows)
        and all(required.issubset(row["copied_fields"]) for row in rows),
    }


def _refresh_vector_observation(environment):
    """Render current states after model-only visual edits without stepping physics."""
    observations = []
    for subenvironment in getattr(environment, "envs", []):
        libero_environment = getattr(subenvironment, "_env", None)
        if libero_environment is None or not hasattr(libero_environment, "env"):
            raise RuntimeError("LIBERO sub-environment cannot refresh its current observation")
        raw = libero_environment.env._get_observations(force_update=True)
        observations.append(subenvironment._format_raw_obs(raw))
    if not observations:
        raise RuntimeError("LIBERO vector environment exposes no observations to refresh")
    return _stack_vector_observations(observations)


def _audit_policy_observation_render_sync(
    raw_observation,
    environment,
    camera_name: str,
) -> dict[str, Any]:
    """Prove that the RGB tensor consumed by the policy reflects the edited renderer."""
    import numpy as np

    observed = np.asarray(raw_observation["pixels"]["image"], dtype=np.uint8)
    if observed.ndim == 3:
        observed = observed[None, ...]
    transforms = {
        "identity": lambda value: value,
        "flip_vertical": lambda value: value[::-1],
        "flip_horizontal": lambda value: value[:, ::-1],
        "flip_both": lambda value: value[::-1, ::-1],
    }
    rows = []
    for environment_index, sim in enumerate(_environment_sims(environment)):
        height, width = observed[environment_index].shape[:2]
        rendered = np.asarray(
            sim.render(
                camera_name=camera_name,
                height=height,
                width=width,
                segmentation=False,
            ),
            dtype=np.uint8,
        )
        differences = {}
        for name, transform in transforms.items():
            delta = np.abs(
                np.asarray(observed[environment_index], dtype=np.int16)
                - np.asarray(transform(rendered), dtype=np.int16)
            )
            differences[name] = {
                "mean_absolute_uint8": float(delta.mean()),
                "maximum_absolute_uint8": int(delta.max()),
            }
        best_name = min(
            differences,
            key=lambda name: differences[name]["mean_absolute_uint8"],
        )
        best = differences[best_name]
        rows.append({
            "environment_index": environment_index,
            "best_orientation": best_name,
            **best,
            "verified": bool(
                best["mean_absolute_uint8"] <= 1.0
                and best["maximum_absolute_uint8"] <= 5
            ),
        })
    return {
        "schema": "fedsplat.policy_observation_render_sync.v1",
        "camera_name": camera_name,
        "environments": rows,
        "verified": bool(rows) and all(row["verified"] for row in rows),
    }


def _mosaic_frame(raw_observation, done, success, tile_size: int = 128):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    frames = _frames_from_observation(raw_observation)
    count = len(frames)
    done = np.asarray(done, dtype=bool).reshape(-1)
    success = np.asarray(success, dtype=bool).reshape(-1)
    if len(done) != count or len(success) != count:
        raise ValueError("Mosaic status vectors must match the number of environments")
    columns = min(5, count)
    rows = int(math.ceil(count / columns))
    label_height = 22
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "black")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()
    for index, frame in enumerate(frames):
        column = index % columns
        row = index // columns
        x = column * tile_size
        y = row * (tile_size + label_height)
        panel = Image.fromarray(frame).resize((tile_size, tile_size), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x, y))
        status = "OK" if success[index] else "DONE" if done[index] else "RUN"
        color = (72, 220, 122) if success[index] else (238, 190, 70) if done[index] else (225, 230, 238)
        draw.text((x + 5, y + tile_size + 3), f"E{index:02d} {status}", fill=color, font=font)
    array = np.asarray(canvas, dtype=np.uint8)
    if array.shape[0] % 2 or array.shape[1] % 2:
        padded = np.zeros((array.shape[0] + array.shape[0] % 2, array.shape[1] + array.shape[1] % 2, 3), dtype=np.uint8)
        padded[: array.shape[0], : array.shape[1]] = array
        array = padded
    return np.ascontiguousarray(array)


def _save_startup_observation_audit(
    raw_observation,
    out_dir: Path,
    *,
    prefix: str = "startup_agentview",
) -> dict[str, Any]:
    """Persist the exact startup RGB tensor consumed by routing and the policy."""
    import numpy as np
    from PIL import Image

    images = np.asarray(raw_observation["pixels"]["image"])
    if images.ndim == 3:
        images = images[None, ...]
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            "Startup agent-view observations must be [N,H,W,3], "
            f"observed {images.shape}"
        )
    images = np.ascontiguousarray(images, dtype=np.uint8)
    if not prefix or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in prefix):
        raise ValueError("Startup observation artifact prefix must be lowercase ASCII")
    payload_path = out_dir / f"{prefix}_rgb.npz"
    np.savez_compressed(payload_path, images=images)
    preview_path = out_dir / f"{prefix}_preview.png"
    status = np.zeros(len(images), dtype=bool)
    Image.fromarray(_mosaic_frame(raw_observation, status, status)).save(preview_path)
    return {
        "schema": "fedsplat.startup_agentview_rgb_audit.v1",
        "routing_and_policy_input": True,
        "local_audit_artifact_only": True,
        "federated_raw_image_upload": False,
        "environment_count": int(len(images)),
        "shape": list(images.shape),
        "dtype": str(images.dtype),
        "payload": {
            "path": payload_path.name,
            "sha256": _sha256(payload_path),
            "bytes": payload_path.stat().st_size,
            "array_key": "images",
        },
        "preview": {
            "path": preview_path.name,
            "sha256": _sha256(preview_path),
            "bytes": preview_path.stat().st_size,
        },
    }


def _decode_video_frames(video_path: Path, indices: list[int]) -> dict[int, Any]:
    import imageio.v2 as imageio

    requested = set(indices)
    decoded = {}
    reader = imageio.get_reader(video_path)
    try:
        for index, frame in enumerate(reader):
            if index in requested:
                decoded[index] = frame
            if len(decoded) == len(requested):
                break
    finally:
        reader.close()
    missing = requested - set(decoded)
    if missing:
        raise RuntimeError(f"Video decode missed requested frame indices: {sorted(missing)}")
    return decoded


def _classify_episode_diagnostic(
    success: bool,
    min_eef_target_distance_m: float | None,
    target_displacement_m: float | None,
    first_close_step: int | None,
) -> str:
    if success:
        return "success"
    if target_displacement_m is not None and target_displacement_m >= 0.03:
        return "object_displaced_not_completed"
    if min_eef_target_distance_m is not None and min_eef_target_distance_m <= 0.10:
        return "reached_target_no_transport" if first_close_step is not None else "reached_target_no_gripper_close"
    return "no_target_reach" if min_eef_target_distance_m is not None else "diagnostic_unavailable"


def _environment_sims(environment):
    sims = []
    for subenvironment in getattr(environment, "envs", []):
        inner = getattr(subenvironment, "_env", None)
        sim = getattr(inner, "sim", None)
        if sim is None:
            raise RuntimeError("LIBERO vector sub-environment does not expose _env.sim")
        sims.append(sim)
    if not sims:
        raise RuntimeError("LIBERO vector environment exposes no sub-environments")
    return sims


def _matching_body_ids(model, pattern: str) -> list[int]:
    return [
        index
        for index, name in enumerate(model.body_names)
        if name is not None and fnmatch.fnmatchcase(str(name), pattern)
    ]


def _target_visual_geom_ids(model, pattern: str) -> tuple[list[int], list[int]]:
    matched_bodies = _matching_body_ids(model, pattern)
    if not matched_bodies:
        raise ValueError(f"Visual target body pattern matched no body: {pattern}")
    root_id = min(matched_bodies, key=lambda index: len(str(model.body_names[index])))
    descendant_ids = set()
    for body_id in range(int(model.nbody)):
        ancestor = body_id
        while ancestor > 0:
            if ancestor == root_id:
                descendant_ids.add(body_id)
                break
            ancestor = int(model.body_parentid[ancestor])
    geom_ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in descendant_ids
        and int(model.geom_group[geom_id]) == 1
    ]
    if not geom_ids:
        raise ValueError(f"Visual target body has no group-1 render geom: {pattern}")
    return matched_bodies, geom_ids


def _visual_geom_ids_for_body_patterns(model, patterns: list[str]) -> tuple[list[int], list[int]]:
    """Resolve render geoms under matched bodies without changing collision state."""
    matched_bodies: set[int] = set()
    for pattern in patterns:
        pattern_matches = _matching_body_ids(model, pattern)
        if not pattern_matches and not any(token in pattern for token in "*?["):
            pattern_matches = _matching_body_ids(model, f"*{pattern}*")
        matched_bodies.update(pattern_matches)
    if not matched_bodies:
        raise ValueError(f"Background body patterns matched no body: {patterns}")
    descendants: set[int] = set()
    for body_id in range(int(model.nbody)):
        ancestor = body_id
        while ancestor > 0:
            if ancestor in matched_bodies:
                descendants.add(body_id)
                break
            ancestor = int(model.body_parentid[ancestor])
    geom_ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in descendants
        and int(model.geom_group[geom_id]) == 1
    ]
    if not geom_ids:
        raise ValueError(f"Background body patterns have no group-1 render geom: {patterns}")
    return sorted(matched_bodies), geom_ids


def _quaternion_multiply_wxyz(left, right):
    import numpy as np

    w1, x1, y1, z1 = np.asarray(left, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(right, dtype=np.float64)
    value = np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Camera quaternion has zero norm")
    return value / norm


def _quaternion_from_euler_xyz_deg(rotation_deg):
    """Return a local XYZ Euler increment as a MuJoCo wxyz quaternion."""
    import numpy as np

    roll, pitch, yaw = np.deg2rad(np.asarray(rotation_deg, dtype=np.float64)) / 2.0
    qx = np.asarray([np.cos(roll), np.sin(roll), 0.0, 0.0])
    qy = np.asarray([np.cos(pitch), 0.0, np.sin(pitch), 0.0])
    qz = np.asarray([np.cos(yaw), 0.0, 0.0, np.sin(yaw)])
    return _quaternion_multiply_wxyz(
        _quaternion_multiply_wxyz(qz, qy), qx
    )


def _target_body_positions(environment, pattern: str | None):
    import numpy as np

    if not pattern:
        return None, []
    positions = []
    names = []
    for sim in _environment_sims(environment):
        matches = _matching_body_ids(sim.model, pattern)
        if not matches:
            raise ValueError(f"Target body pattern matched no simulator body: {pattern}")
        body_id = min(matches, key=lambda index: len(str(sim.model.body_names[index])))
        names.append(str(sim.model.body_names[body_id]))
        positions.append(np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy())
    return np.stack(positions), names


def _capture_target_appearance(
    environment,
    camera_name: str,
    target_pattern: str,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    """Measure target pixels with MuJoCo element segmentation, independent of RGB heuristics."""
    import mujoco
    import numpy as np

    rows = []
    geom_object_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    for environment_index, sim in enumerate(_environment_sims(environment)):
        _matched_bodies, geom_ids = _target_visual_geom_ids(sim.model, target_pattern)
        rgb = np.asarray(
            sim.render(
                camera_name=camera_name,
                height=height,
                width=width,
                segmentation=False,
            ),
            dtype=np.uint8,
        )
        segmentation = np.asarray(
            sim.render(
                camera_name=camera_name,
                height=height,
                width=width,
                segmentation=True,
            )
        )
        mask = (segmentation[..., 0] == geom_object_type) & np.isin(
            segmentation[..., 1], np.asarray(geom_ids, dtype=np.int32)
        )
        pixel_count = int(mask.sum())
        mean_rgb = (
            np.asarray(rgb[mask], dtype=np.float64).mean(axis=0).tolist()
            if pixel_count
            else None
        )
        rows.append({
            "environment_index": environment_index,
            "target_geom_ids": geom_ids,
            "target_pixel_count": pixel_count,
            "target_pixel_fraction": float(mask.mean()),
            "target_mean_rgb_uint8": mean_rgb,
        })
    return rows


def _audit_target_appearance(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    target_rgb: list[float],
    minimum_pixels: int = 16,
) -> dict[str, Any]:
    import numpy as np

    requested = np.asarray(target_rgb, dtype=np.float64)
    requested_chroma = requested / max(float(requested.sum()), 1e-8)
    after_by_index = {int(row["environment_index"]): row for row in after}
    rows = []
    for before_row in before:
        index = int(before_row["environment_index"])
        after_row = after_by_index[index]
        before_mean_value = before_row.get("target_mean_rgb_uint8")
        after_mean_value = after_row.get("target_mean_rgb_uint8")
        visible = (
            int(before_row["target_pixel_count"]) >= minimum_pixels
            and int(after_row["target_pixel_count"]) >= minimum_pixels
            and before_mean_value is not None
            and after_mean_value is not None
        )
        if visible:
            before_mean = np.asarray(before_mean_value, dtype=np.float64) / 255.0
            after_mean = np.asarray(after_mean_value, dtype=np.float64) / 255.0
            before_chroma = before_mean / max(float(before_mean.sum()), 1e-8)
            after_chroma = after_mean / max(float(after_mean.sum()), 1e-8)
            before_distance = float(np.abs(before_chroma - requested_chroma).sum())
            after_distance = float(np.abs(after_chroma - requested_chroma).sum())
            mean_rgb_l1_delta = float(np.abs(after_mean - before_mean).mean())
            chroma_improvement = before_distance - after_distance
        else:
            before_distance = None
            after_distance = None
            mean_rgb_l1_delta = None
            chroma_improvement = None
        verified = bool(
            visible
            and mean_rgb_l1_delta is not None
            and mean_rgb_l1_delta >= 0.10
            and after_distance is not None
            and after_distance <= 0.35
            and chroma_improvement is not None
            and chroma_improvement >= 0.05
        )
        rows.append({
            "environment_index": index,
            "target_pixel_count_before": int(before_row["target_pixel_count"]),
            "target_pixel_count_after": int(after_row["target_pixel_count"]),
            "target_mean_rgb_before_uint8": before_mean_value,
            "target_mean_rgb_after_uint8": after_mean_value,
            "mean_rgb_l1_delta_0_1": mean_rgb_l1_delta,
            "target_chromaticity_l1_before": before_distance,
            "target_chromaticity_l1_after": after_distance,
            "target_chromaticity_improvement": chroma_improvement,
            "appearance_shift_verified": verified,
        })
    return {
        "schema": "fedsplat.target_appearance_audit.v1",
        "camera_name": None,
        "requested_rgb": requested.tolist(),
        "minimum_target_pixels": minimum_pixels,
        "minimum_mean_rgb_l1_delta_0_1": 0.10,
        "maximum_target_chromaticity_l1": 0.35,
        "minimum_chromaticity_improvement": 0.05,
        "environments": rows,
        "verified": bool(rows) and all(row["appearance_shift_verified"] for row in rows),
    }


def _audit_target_visibility(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    minimum_pixels: int = 16,
    minimum_retained_fraction: float = 0.25,
) -> dict[str, Any]:
    """Ensure a nuisance shift does not remove the language-referenced target."""
    after_by_index = {int(row["environment_index"]): row for row in after}
    rows = []
    for before_row in before:
        index = int(before_row["environment_index"])
        after_row = after_by_index[index]
        before_pixels = int(before_row["target_pixel_count"])
        after_pixels = int(after_row["target_pixel_count"])
        retained = after_pixels / max(before_pixels, 1)
        verified = bool(
            before_pixels >= minimum_pixels
            and after_pixels >= minimum_pixels
            and retained >= minimum_retained_fraction
        )
        rows.append({
            "environment_index": index,
            "target_pixel_count_before": before_pixels,
            "target_pixel_count_after": after_pixels,
            "retained_pixel_fraction": retained,
            "visibility_preserved": verified,
        })
    return {
        "schema": "fedsplat.target_visibility_audit.v1",
        "minimum_target_pixels": minimum_pixels,
        "minimum_retained_pixel_fraction": minimum_retained_fraction,
        "environments": rows,
        "verified": bool(rows) and all(row["visibility_preserved"] for row in rows),
    }


def _apply_visual_domain_shift(environment, visual_shift: dict[str, Any] | None) -> dict[str, Any]:
    import numpy as np

    if not visual_shift:
        return {"applied": False, "name": "nominal", "environments": []}
    camera_name = str(visual_shift.get("camera_name", "agentview"))
    camera_delta = np.asarray(visual_shift.get("camera_position_delta_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    camera_rotation = np.asarray(
        visual_shift.get("camera_rotation_delta_deg", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    camera_rotation_quaternion = _quaternion_from_euler_xyz_deg(camera_rotation)
    target_pattern = visual_shift.get("target_body_pattern")
    target_rgb = visual_shift.get("target_rgb")
    background_patterns = list(visual_shift.get("background_body_patterns", []))
    background_rgb = visual_shift.get("background_rgb")
    rows = []
    for environment_index, sim in enumerate(_environment_sims(environment)):
        model = sim.model
        camera_id = model.camera_name2id(camera_name)
        camera_before = np.asarray(model.cam_pos[camera_id], dtype=np.float64).copy()
        camera_quaternion_before = np.asarray(
            model.cam_quat[camera_id], dtype=np.float64
        ).copy()
        model.cam_pos[camera_id] = camera_before + camera_delta
        expected_camera_quaternion = _quaternion_multiply_wxyz(
            camera_quaternion_before, camera_rotation_quaternion
        )
        model.cam_quat[camera_id] = expected_camera_quaternion
        qpos_before = np.asarray(sim.data.qpos, dtype=np.float64).copy()
        qvel_before = np.asarray(sim.data.qvel, dtype=np.float64).copy()
        light_ambient_before = np.asarray(model.light_ambient, dtype=np.float64).copy()
        light_diffuse_before = np.asarray(model.light_diffuse, dtype=np.float64).copy()
        light_specular_before = np.asarray(model.light_specular, dtype=np.float64).copy()
        light_before = {
            "ambient_mean": float(light_ambient_before.mean()),
            "diffuse_mean": float(light_diffuse_before.mean()),
            "specular_mean": float(light_specular_before.mean()),
        }
        expected_ambient = np.clip(
            light_ambient_before * float(visual_shift.get("light_ambient_scale", 1.0)),
            0.0,
            1.0,
        )
        expected_diffuse = np.clip(
            light_diffuse_before * float(visual_shift.get("light_diffuse_scale", 1.0)),
            0.0,
            1.0,
        )
        expected_specular = np.clip(
            light_specular_before * float(visual_shift.get("light_specular_scale", 1.0)),
            0.0,
            1.0,
        )
        model.light_ambient[:] = expected_ambient
        model.light_diffuse[:] = expected_diffuse
        model.light_specular[:] = expected_specular
        matched_bodies = _matching_body_ids(model, str(target_pattern)) if target_pattern else []
        matched_geom_ids: list[int] = []
        matched_visual_geom_names: list[str] = []
        matched_material_ids: set[int] = set()
        source_material_ids_by_geom: dict[str, int] = {}
        detached_geom_names: list[str] = []
        if target_rgb is not None:
            matched_bodies, matched_geom_ids = _target_visual_geom_ids(
                model, str(target_pattern)
            )
            rgb = np.asarray(target_rgb, dtype=np.float64)
            for geom_id in matched_geom_ids:
                geom_name = model.geom_names[geom_id]
                if geom_name is not None:
                    matched_visual_geom_names.append(str(geom_name))
                material_id = int(model.geom_matid[geom_id])
                if material_id >= 0:
                    matched_material_ids.add(material_id)
                stable_name = str(geom_name) if geom_name is not None else f"geom_{geom_id}"
                source_material_ids_by_geom[stable_name] = material_id
                # Detaching the visual geom avoids renderer-specific texture caches
                # and prevents a shared source material from recoloring other objects.
                model.geom_matid[geom_id] = -1
                model.geom_rgba[geom_id, :3] = rgb
                model.geom_rgba[geom_id, 3] = 1.0
                detached_geom_names.append(stable_name)
            material_shift_verified = bool(matched_geom_ids) and all(
                int(model.geom_matid[geom_id]) == -1
                and np.allclose(model.geom_rgba[geom_id, :3], rgb, atol=1e-6)
                and float(model.geom_rgba[geom_id, 3]) >= 0.999
                for geom_id in matched_geom_ids
            )
        else:
            material_shift_verified = True

        background_body_ids: list[int] = []
        background_geom_ids: list[int] = []
        background_geom_names: list[str] = []
        if background_rgb is not None:
            background_body_ids, background_geom_ids = _visual_geom_ids_for_body_patterns(
                model, background_patterns
            )
            if target_pattern:
                _target_body_ids, target_geom_ids = _target_visual_geom_ids(
                    model, str(target_pattern)
                )
                target_geom_id_set = set(target_geom_ids)
                background_geom_ids = [
                    geom_id for geom_id in background_geom_ids
                    if geom_id not in target_geom_id_set
                ]
            if not background_geom_ids:
                raise ValueError("Background mutation has no non-target render geoms")
            background_color = np.asarray(background_rgb, dtype=np.float64)
            for geom_id in background_geom_ids:
                geom_name = model.geom_names[geom_id]
                background_geom_names.append(
                    str(geom_name) if geom_name is not None else f"geom_{geom_id}"
                )
                model.geom_matid[geom_id] = -1
                model.geom_rgba[geom_id, :3] = background_color
                model.geom_rgba[geom_id, 3] = 1.0
            background_shift_verified = all(
                int(model.geom_matid[geom_id]) == -1
                and np.allclose(
                    model.geom_rgba[geom_id, :3], background_color, atol=1e-6
                )
                and float(model.geom_rgba[geom_id, 3]) >= 0.999
                for geom_id in background_geom_ids
            )
        else:
            background_shift_verified = True
        sim.forward()
        camera_position_verified = bool(
            np.allclose(
                model.cam_pos[camera_id], camera_before + camera_delta, atol=1e-10
            )
        )
        camera_quaternion_after = np.asarray(
            model.cam_quat[camera_id], dtype=np.float64
        )
        camera_rotation_verified = bool(
            abs(float(np.dot(camera_quaternion_after, expected_camera_quaternion)))
            >= 1.0 - 1e-8
        )
        light_shift_verified = bool(
            np.allclose(model.light_ambient, expected_ambient, atol=1e-8)
            and np.allclose(model.light_diffuse, expected_diffuse, atol=1e-8)
            and np.allclose(model.light_specular, expected_specular, atol=1e-8)
        )
        qpos_delta = float(
            np.max(
                np.abs(np.asarray(sim.data.qpos, dtype=np.float64) - qpos_before),
                initial=0.0,
            )
        )
        qvel_delta = float(
            np.max(
                np.abs(np.asarray(sim.data.qvel, dtype=np.float64) - qvel_before),
                initial=0.0,
            )
        )
        physics_state_unchanged = qpos_delta <= 1e-12 and qvel_delta <= 1e-12
        row_verified = bool(
            camera_position_verified
            and camera_rotation_verified
            and light_shift_verified
            and material_shift_verified
            and background_shift_verified
            and physics_state_unchanged
        )
        rows.append({
            "environment_index": environment_index,
            "camera_name": camera_name,
            "camera_position_before_m": camera_before.tolist(),
            "camera_position_after_m": np.asarray(model.cam_pos[camera_id], dtype=float).tolist(),
            "camera_quaternion_before_wxyz": camera_quaternion_before.tolist(),
            "camera_quaternion_after_wxyz": camera_quaternion_after.tolist(),
            "camera_rotation_delta_deg": camera_rotation.tolist(),
            "camera_position_verified": camera_position_verified,
            "camera_rotation_verified": camera_rotation_verified,
            "light_before": light_before,
            "light_after": {
                "ambient_mean": float(np.asarray(model.light_ambient).mean()),
                "diffuse_mean": float(np.asarray(model.light_diffuse).mean()),
                "specular_mean": float(np.asarray(model.light_specular).mean()),
            },
            "light_shift_verified": light_shift_verified,
            "matched_target_bodies": [str(model.body_names[index]) for index in matched_bodies],
            "matched_visual_geom_count": len(matched_geom_ids),
            "matched_visual_geom_names": matched_visual_geom_names,
            "matched_material_count": len(matched_material_ids),
            "appearance_override": "detached_solid_rgba",
            "source_material_ids_by_geom": source_material_ids_by_geom,
            "detached_geom_names": detached_geom_names,
            "material_shift_verified": material_shift_verified,
            "background_body_patterns": background_patterns,
            "matched_background_bodies": [
                str(model.body_names[index]) for index in background_body_ids
            ],
            "matched_background_geom_count": len(background_geom_ids),
            "matched_background_geom_names": background_geom_names,
            "background_shift_verified": background_shift_verified,
            "maximum_qpos_delta": qpos_delta,
            "maximum_qvel_delta": qvel_delta,
            "physics_state_unchanged": physics_state_unchanged,
            "shift_contract_verified": row_verified,
        })
    return {
        "applied": True,
        "name": visual_shift["name"],
        "requested": visual_shift,
        "environments": rows,
        "target_recolored": target_rgb is not None,
        "semantics_preserved_by_configuration": target_rgb is None,
        "mutation_verified": all(row["shift_contract_verified"] for row in rows),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def adapter_dense_fingerprint(path: Path) -> str:
    """Hash canonical dense adapter values, independent of NPZ container bytes."""
    import numpy as np

    from .lora_updates import load_client_update

    update = load_client_update(path)
    digest = hashlib.sha256()
    for layer in update.layers:
        dense = np.ascontiguousarray(layer.dense(), dtype=np.float32)
        digest.update(layer.layer_name.encode("utf-8"))
        digest.update(str(tuple(dense.shape)).encode("ascii"))
        digest.update(dense.tobytes())
    return digest.hexdigest()


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np
    import torch

    out_dir = Path(spec["out_dir"])
    libero_runtime = _bootstrap_libero(
        Path(spec["libero_site"]),
        out_dir / "runtime_libero_config",
        Path(spec["libero_datasets_path"]),
        Path(spec["libero_resource_root"]) if spec.get("libero_resource_root") else None,
    )
    from .runtime_compat import bootstrap_smolvla_processors

    bootstrap_smolvla_processors()

    # Some robosuite releases hard-code a shared /tmp log path. Redirect that
    # constructor during import so independent users and runs cannot collide.
    import logging

    original_file_handler = logging.FileHandler

    def redirected_file_handler(filename, *args, **kwargs):
        if str(filename) == "/tmp/robosuite.log":
            filename = out_dir / "robosuite.log"
        return original_file_handler(filename, *args, **kwargs)

    logging.FileHandler = redirected_file_handler
    try:
        import sys
        import gymnasium as gym

        # LIBERO-PRO still imports the legacy `gym` module in its vector-env
        # helpers. The API surface used there is supplied by Gymnasium.
        sys.modules.setdefault("gym", gym)
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.factory import make_env_config, make_env_pre_post_processors
        from lerobot.envs.libero import create_libero_envs
        from lerobot.envs.utils import add_envs_task, preprocess_observation
        from lerobot.policies.factory import make_policy, make_pre_post_processors
    finally:
        logging.FileHandler = original_file_handler

    seed = int(spec["seed"])
    deterministic = bool(spec.get("deterministic", True))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    suite = spec["suite"]
    task_id = int(spec["task_id"])
    num_episodes = int(spec["num_episodes"])
    from libero.libero import benchmark as libero_benchmark, get_libero_path

    benchmark_mapping = libero_benchmark.get_benchmark_dict()
    if suite not in benchmark_mapping:
        raise KeyError(f"Requested LIBERO suite is not registered: {suite}")
    benchmark_suite = benchmark_mapping[suite]()
    benchmark_task = benchmark_suite.get_task(task_id)
    bddl_path = (
        Path(get_libero_path("bddl_files"))
        / benchmark_task.problem_folder
        / benchmark_task.bddl_file
    )
    bddl_text = bddl_path.read_text(encoding="utf-8")
    expected_bddl_tokens = list(spec.get("expected_bddl_tokens", []))
    forbidden_bddl_tokens = list(spec.get("forbidden_bddl_tokens", []))
    missing_tokens = [token for token in expected_bddl_tokens if token not in bddl_text]
    present_forbidden_tokens = [token for token in forbidden_bddl_tokens if token in bddl_text]
    if missing_tokens or present_forbidden_tokens:
        raise ValueError(
            "LIBERO environment contract failed: "
            f"missing={missing_tokens}, forbidden_present={present_forbidden_tokens}, bddl={bddl_path}"
        )
    environment_contract = {
        "suite": suite,
        "task_id": task_id,
        "language": benchmark_task.language,
        "bddl_path": str(bddl_path),
        "bddl_sha256": _sha256(bddl_path),
        "expected_bddl_tokens": expected_bddl_tokens,
        "forbidden_bddl_tokens": forbidden_bddl_tokens,
        "expected_tokens_present": not missing_tokens,
        "forbidden_tokens_absent": not present_forbidden_tokens,
    }
    env_config = make_env_config(
        "libero",
        task=suite,
        obs_type="pixels_agent_pos",
        observation_height=int(spec["observation_height"]),
        observation_width=int(spec["observation_width"]),
        camera_name="agentview_image,robot0_eye_in_hand_image",
        episode_length=int(spec["episode_length"]),
        render_mode="rgb_array",
        control_mode="relative",
    )
    environments = create_libero_envs(
        task=suite,
        n_envs=num_episodes,
        gym_kwargs={
            "task_ids": [task_id],
            "obs_type": "pixels_agent_pos",
            "render_mode": "rgb_array",
            "observation_height": int(spec["observation_height"]),
            "observation_width": int(spec["observation_width"]),
        },
        camera_name="agentview_image,robot0_eye_in_hand_image",
        init_states=True,
        env_cls=gym.vector.SyncVectorEnv,
        control_mode="relative",
        episode_length=int(spec["episode_length"]),
    )
    environment = environments[suite][task_id]
    provider_config = spec.get("policy_observation_provider")
    provider_mode = provider_config.get("mode") if provider_config is not None else None
    component_gs_provider_mode = (
        provider_mode == "component_gs_state_mirror_pose_oracle"
    )
    state_mirror_provider = provider_mode in {
        "simulator_canonical_state_mirror",
        "component_gs_state_mirror_pose_oracle",
    }
    canonical_environment = None
    presentation_environment = None
    component_gs_provider = None
    if provider_config is not None:
        canonical_environments = create_libero_envs(
            task=suite,
            n_envs=num_episodes,
            gym_kwargs={
                "task_ids": [task_id],
                "obs_type": "pixels_agent_pos",
                "render_mode": "rgb_array",
                "observation_height": int(spec["observation_height"]),
                "observation_width": int(spec["observation_width"]),
            },
            camera_name="agentview_image,robot0_eye_in_hand_image",
            init_states=True,
            env_cls=gym.vector.SyncVectorEnv,
            control_mode="relative",
            episode_length=int(spec["episode_length"]),
        )
        canonical_environment = canonical_environments[suite][task_id]
    visual_shift_report = {"applied": False, "name": "nominal", "environments": []}
    started = time.perf_counter()
    try:
        checkpoint = Path(spec["checkpoint_path"])
        policy_config = PreTrainedConfig.from_pretrained(checkpoint)
        policy_config.pretrained_path = checkpoint
        policy_config.device = spec["device"]
        policy_config.use_amp = False
        policy = make_policy(cfg=policy_config, env_cfg=env_config)
        applied_layers = []
        adapter_fingerprint = None
        routed_controller = None
        routed_bank_audit = None
        adapter_routing_report = None
        if spec.get("adapter_update"):
            adapter_fingerprint = adapter_dense_fingerprint(Path(spec["adapter_update"]))
            applied_layers = _apply_static_adapter(policy, Path(spec["adapter_update"]))
        elif spec.get("adapter_bank_manifest"):
            from .routed_adapters import apply_routed_adapter_bank

            routed_controller, applied_layers, routed_bank_audit = apply_routed_adapter_bank(
                policy,
                Path(spec["adapter_bank_manifest"]),
                environment_count=num_episodes,
            )
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_config,
            policy_cfg=policy_config,
        )

        policy.reset()
        raw_observation, _ = environment.reset(seed=seed)
        canonical_observation = None
        provider_report = None
        provider_sync_checks = 0
        provider_state_transfer_checks = 0
        provider_max_qpos_delta = 0.0
        provider_max_qvel_delta = 0.0
        if canonical_environment is not None:
            canonical_observation, _ = canonical_environment.reset(seed=seed)
            if not state_mirror_provider:
                initial_sync = _compare_environment_physics(
                    environment,
                    canonical_environment,
                    tolerance=float(
                        provider_config.get("state_sync_tolerance", 1e-9)
                    ),
                )
                if initial_sync["verified"] is not True:
                    raise RuntimeError(
                        "Canonical provider initial physics synchronization failed"
                    )
                provider_sync_checks = 1
                provider_max_qpos_delta = float(
                    initial_sync["maximum_qpos_delta"]
                )
                provider_max_qvel_delta = float(
                    initial_sync["maximum_qvel_delta"]
                )
        nominal_observation = raw_observation
        if spec.get("visual_domain_shift") is not None:
            visual_shift = spec["visual_domain_shift"]
            target_pattern = visual_shift.get("target_body_pattern")
            target_rgb = visual_shift.get("target_rgb")
            appearance_before = None
            if target_pattern:
                appearance_before = _capture_target_appearance(
                    environment,
                    str(visual_shift.get("camera_name", "agentview")),
                    str(target_pattern),
                    int(spec["observation_height"]),
                    int(spec["observation_width"]),
                )
            visual_shift_report = _apply_visual_domain_shift(
                environment,
                visual_shift,
            )
            raw_observation = _refresh_vector_observation(environment)
            if appearance_before is not None:
                appearance_after = _capture_target_appearance(
                    environment,
                    str(visual_shift.get("camera_name", "agentview")),
                    str(target_pattern),
                    int(spec["observation_height"]),
                    int(spec["observation_width"]),
                )
                visibility_audit = _audit_target_visibility(
                    appearance_before,
                    appearance_after,
                )
                visibility_audit["camera_name"] = str(
                    visual_shift.get("camera_name", "agentview")
                )
                visual_shift_report["target_visibility_audit"] = visibility_audit
                if target_rgb is not None:
                    appearance_audit = _audit_target_appearance(
                        appearance_before,
                        appearance_after,
                        target_rgb,
                    )
                    appearance_audit["camera_name"] = str(
                        visual_shift.get("camera_name", "agentview")
                    )
                    visual_shift_report["appearance_audit"] = appearance_audit
            observation_sync = _audit_policy_observation_render_sync(
                raw_observation,
                environment,
                str(visual_shift.get("camera_name", "agentview")),
            )
            visual_shift_report["policy_observation_render_sync"] = observation_sync
            target_audit_verified = bool(
                visual_shift_report.get("appearance_audit", {}).get("verified")
                if target_rgb is not None
                else visual_shift_report.get("target_visibility_audit", {}).get(
                    "verified", True
                )
            )
            visual_shift_report["contract_verified"] = bool(
                visual_shift_report.get("mutation_verified")
                and observation_sync["verified"]
                and target_audit_verified
            )
        startup_observation_audit = _save_startup_observation_audit(
            raw_observation,
            out_dir,
        )
        policy_raw_observation = raw_observation
        if canonical_environment is not None:
            state_transfer = None
            if state_mirror_provider:
                state_transfer = _mirror_environment_render_state(
                    environment,
                    canonical_environment,
                    tolerance=float(
                        provider_config.get("state_sync_tolerance", 1e-9)
                    ),
                )
                if state_transfer["verified"] is not True:
                    raise RuntimeError(
                        "Canonical state-mirror provider failed startup transfer"
                    )
                provider_state_transfer_checks = 1
                provider_sync_checks = 1
                provider_max_qpos_delta = float(
                    state_transfer["maximum_qpos_delta"]
                )
                provider_max_qvel_delta = float(
                    state_transfer["maximum_qvel_delta"]
                )
                if component_gs_provider_mode:
                    from .libero_component_gs import LiberoComponentGSProvider

                    component_gs_provider = LiberoComponentGSProvider.build(
                        _environment_sims(canonical_environment),
                        provider_config,
                        width=int(spec["observation_width"]),
                        height=int(spec["observation_height"]),
                        seed=seed,
                        out_dir=out_dir,
                    )
                    policy_raw_observation, component_startup_quality = (
                        component_gs_provider.render_policy_observation(
                            raw_observation,
                            _environment_sims(canonical_environment),
                        )
                    )
                    component_gs_provider.record_diagnostic_reference(
                        policy_raw_observation,
                        _environment_sims(canonical_environment),
                        step=0,
                        out_dir=out_dir,
                    )
                else:
                    canonical_observation = _refresh_vector_observation(
                        canonical_environment
                    )
            if not component_gs_provider_mode:
                policy_raw_observation = _compose_canonical_policy_observation(
                    raw_observation,
                    canonical_observation,
                )
            canonical_startup_audit = _save_startup_observation_audit(
                policy_raw_observation,
                out_dir,
                prefix="canonical_policy_startup",
            )
            if component_gs_provider_mode:
                canonical_render_sync = component_startup_quality
                _write_json_atomic(
                    out_dir / "component_gs_startup_quality.json",
                    canonical_render_sync,
                )
                _write_json_atomic(
                    out_dir / "component_gs_provider_audit.json",
                    component_gs_provider.report(),
                )
                if canonical_render_sync["passed"] is not True:
                    raise RuntimeError("Component-GS provider startup quality gate failed")
            else:
                canonical_render_sync = _audit_policy_observation_render_sync(
                    policy_raw_observation,
                    canonical_environment,
                    "agentview",
                )
                if canonical_render_sync["verified"] is not True:
                    raise RuntimeError("Canonical provider RGB does not match its renderer")
            execution_rgb = np.asarray(raw_observation["pixels"]["image"], dtype=np.int16)
            policy_rgb = np.asarray(
                policy_raw_observation["pixels"]["image"], dtype=np.int16
            )
            provider_report = {
                "schema": "fedsplat.canonical_policy_observation_provider.v1",
                "mode": provider_mode,
                "evidence_role": provider_config["evidence_role"],
                "oracle_upper_bound_only": not component_gs_provider_mode,
                "pose_oracle_development_only": component_gs_provider_mode,
                "deployment_eligible": False,
                "three_d_gaussian_renderer_used": component_gs_provider_mode,
                "simulator_hidden_state_dependency": True,
                "policy_rgb_source": (
                    "persistent_actor_local_component_gaussians"
                    if component_gs_provider_mode
                    else (
                        "state_mirrored_nominal_shadow_renderer"
                        if state_mirror_provider
                        else "synchronized_nominal_shadow_renderer"
                    )
                ),
                "execution_rgb_source": "visually_shifted_execution_environment",
                "execution_robot_state_retained": True,
                "same_action_applied_to_both_environments": (
                    not state_mirror_provider
                ),
                "canonical_physics_stepped": not state_mirror_provider,
                "canonical_reward_or_success_queried": False,
                "state_transfer": (
                    "execution_to_rendering_only_canonical_twin_before_every_policy_render"
                    if state_mirror_provider
                    else None
                ),
                "task_success_or_reward_used_for_policy_input": False,
                "simulator_rgb_per_policy_step": False if component_gs_provider_mode else True,
                "raw_rgb_federated_upload": False,
                "federated_upload_bytes": 0,
                "startup_execution_policy_mean_absolute_rgb_delta": float(
                    np.abs(execution_rgb - policy_rgb).mean()
                ),
                "startup_policy_observation": canonical_startup_audit,
                "startup_policy_render_sync": canonical_render_sync,
                "physics_sync": {
                    "tolerance": float(
                        provider_config.get("state_sync_tolerance", 1e-9)
                    ),
                    "checks": provider_sync_checks,
                    "maximum_qpos_delta": provider_max_qpos_delta,
                    "maximum_qvel_delta": provider_max_qvel_delta,
                    "termination_mismatch_count": 0,
                    "success_mismatch_count": 0,
                    "verified": True,
                },
            }
            if state_mirror_provider:
                provider_report["state_mirror"] = {
                    **state_transfer,
                    "checks": provider_state_transfer_checks,
                    "performed_before_every_policy_render": True,
                }
            if component_gs_provider_mode:
                provider_report["component_gs"] = component_gs_provider.report()
                _write_json_atomic(
                    out_dir / "component_gs_provider_audit.json",
                    provider_report["component_gs"],
                )
        if routed_controller is not None:
            router_path = Path(spec["visual_router_manifest"])
            router_manifest = json.loads(router_path.read_text(encoding="utf-8"))
            if router_manifest.get("audit_passed") is not True:
                raise ValueError("Image-only router manifest did not pass its unseen-client audit")
            from .visual_router import route_visual_images

            decisions = route_visual_images(
                np.asarray(raw_observation["pixels"]["image"]),
                router_manifest,
                available_routes=set(routed_controller.route_names),
            )
            selected_routes = [
                row["selected_route"] for row in decisions["rows"]
            ]
            routed_controller.set_routes(selected_routes)
            policy.reset()
            adapter_routing_report = {
                "schema": "fedsplat.image_only_adapter_routing.v1",
                "routing_input": "startup_agentview_rgb_only",
                "domain_metadata_used": False,
                "task_success_or_reward_used": False,
                "router_manifest": str(router_path.resolve()),
                "router_manifest_sha256": _sha256(router_path),
                "router_unseen_client_audit_passed": True,
                "adapter_bank": routed_bank_audit,
                "decisions": decisions,
            }
            _write_json_atomic(
                out_dir / "adapter_routing_startup_audit.json",
                adapter_routing_report,
            )
        done = np.zeros(num_episodes, dtype=bool)
        success = np.zeros(num_episodes, dtype=bool)
        if visual_shift_report.get("applied") is True:
            from PIL import Image

            before_preview_path = out_dir / "visual_shift_before.png"
            before_preview_frame = _mosaic_frame(nominal_observation, done, success)
            Image.fromarray(before_preview_frame).save(before_preview_path)
            preview_path = out_dir / "visual_shift_preview.png"
            preview_frame = _mosaic_frame(raw_observation, done, success)
            Image.fromarray(preview_frame).save(preview_path)
            comparison_path = out_dir / "visual_shift_before_after.png"
            comparison_frame = np.concatenate((before_preview_frame, preview_frame), axis=1)
            Image.fromarray(comparison_frame).save(comparison_path)
            visual_shift_report["before_preview"] = {
                "path": before_preview_path.name,
                "sha256": _sha256(before_preview_path),
                "bytes": before_preview_path.stat().st_size,
                "width": int(before_preview_frame.shape[1]),
                "height": int(before_preview_frame.shape[0]),
                "environment_count": num_episodes,
            }
            visual_shift_report["preview"] = {
                "path": preview_path.name,
                "sha256": _sha256(preview_path),
                "bytes": preview_path.stat().st_size,
                "width": int(preview_frame.shape[1]),
                "height": int(preview_frame.shape[0]),
                "environment_count": num_episodes,
            }
            visual_shift_report["before_after_comparison"] = {
                "path": comparison_path.name,
                "sha256": _sha256(comparison_path),
                "bytes": comparison_path.stat().st_size,
                "width": int(comparison_frame.shape[1]),
                "height": int(comparison_frame.shape[0]),
                "environment_count": num_episodes,
            }
            _write_json_atomic(
                out_dir / "visual_shift_startup_audit.json",
                visual_shift_report,
            )
            if visual_shift_report.get("contract_verified") is not True:
                raise RuntimeError(
                    "Visual shift did not pass mutation, target, and policy-observation audits"
                )
        success_step = np.full(num_episodes, -1, dtype=int)
        inference_seconds = []
        dummy_actions = np.zeros((num_episodes, 7), dtype=np.float32)
        dummy_actions[:, -1] = -1.0
        max_steps = int(spec["episode_length"])
        progress_path = out_dir / "rollout_progress.json"
        video_path = out_dir / "libero_rollout.mp4"
        policy_video_path = (
            out_dir / "canonical_policy_rollout.mp4"
            if canonical_environment is not None
            else None
        )
        presentation_config = spec.get("presentation_capture")
        presentation_path = None
        presentation_recorder = None
        presentation_renderer = None
        presentation_source = None
        presentation_component_scene = None
        presentation_resolution_audit = None
        presentation_resolution_checks = 0
        presentation_resolution_failed_checks = 0
        presentation_resolution_maximum_mae = 0.0
        presentation_resolution_minimum_correlation = 1.0
        presentation_context_ownership = []
        presentation_state_mirror_checks = 0
        presentation_state_mirror_max_qpos_delta = 0.0
        presentation_state_mirror_max_qvel_delta = 0.0
        if presentation_config is not None:
            from .presentation_capture import (
                DirectPresentationRecorder,
                compare_native_resolution_frames,
                enforce_offscreen_context_ownership,
                mirror_mujoco_render_model_state,
                mirror_mujoco_render_state,
                render_component_gs_observer,
                render_simulator_observer,
            )

            presentation_index = int(
                presentation_config["environment_index"]
            )
            presentation_width = int(presentation_config["width"])
            presentation_height = int(presentation_config["height"])
            presentation_camera = str(presentation_config["camera_name"])
            presentation_environments = create_libero_envs(
                task=suite,
                n_envs=1,
                gym_kwargs={
                    "task_ids": [task_id],
                    "obs_type": "pixels_agent_pos",
                    "render_mode": "rgb_array",
                    "observation_height": presentation_height,
                    "observation_width": presentation_width,
                },
                camera_name="agentview_image,robot0_eye_in_hand_image",
                init_states=True,
                env_cls=gym.vector.SyncVectorEnv,
                control_mode="relative",
                episode_length=int(spec["episode_length"]),
            )
            presentation_environment = presentation_environments[suite][task_id]
            presentation_environment.reset(seed=seed)
            presentation_source_environment = (
                canonical_environment
                if canonical_environment is not None
                else environment
            )
            presentation_source_sim = _environment_sims(
                presentation_source_environment
            )[presentation_index]
            presentation_sim = _environment_sims(presentation_environment)[0]
            initial_presentation_model_mirror = (
                mirror_mujoco_render_model_state(
                    presentation_source_sim,
                    presentation_sim,
                    tolerance=float(
                        presentation_config.get("model_sync_tolerance", 1e-9)
                    ),
                )
            )
            if initial_presentation_model_mirror["verified"] is not True:
                raise RuntimeError(
                    "Native presentation rendering twin lost model-level visual sync"
                )
            owned_simulators = []
            for owned_environment in (
                environment,
                canonical_environment,
                presentation_environment,
            ):
                if owned_environment is None:
                    continue
                for owned_sim in _environment_sims(owned_environment):
                    if all(owned_sim is not existing for existing in owned_simulators):
                        owned_simulators.append(owned_sim)
                        presentation_context_ownership.append(
                            enforce_offscreen_context_ownership(owned_sim)
                        )

            def sync_presentation_render_state():
                nonlocal presentation_state_mirror_checks
                nonlocal presentation_state_mirror_max_qpos_delta
                nonlocal presentation_state_mirror_max_qvel_delta
                audit = mirror_mujoco_render_state(
                    presentation_source_sim,
                    presentation_sim,
                    tolerance=float(
                        presentation_config.get("state_sync_tolerance", 1e-9)
                    ),
                )
                presentation_state_mirror_checks += 1
                presentation_state_mirror_max_qpos_delta = max(
                    presentation_state_mirror_max_qpos_delta,
                    float(audit["maximum_qpos_delta"]),
                )
                presentation_state_mirror_max_qvel_delta = max(
                    presentation_state_mirror_max_qvel_delta,
                    float(audit["maximum_qvel_delta"]),
                )
                if audit["verified"] is not True:
                    raise RuntimeError(
                        "Native presentation rendering twin lost state sync"
                    )
                return audit

            def render_and_audit_presentation_simulator():
                nonlocal presentation_resolution_checks
                nonlocal presentation_resolution_failed_checks
                nonlocal presentation_resolution_maximum_mae
                nonlocal presentation_resolution_minimum_correlation
                source_frame = render_simulator_observer(
                    presentation_sim,
                    camera_name=presentation_camera,
                    width=int(spec["observation_width"]),
                    height=int(spec["observation_height"]),
                )
                target_frame = render_simulator_observer(
                    presentation_sim,
                    camera_name=presentation_camera,
                    width=presentation_width,
                    height=presentation_height,
                )
                audit = compare_native_resolution_frames(
                    source_frame,
                    target_frame,
                )
                audit["render_source"] = (
                    "dedicated_rendering_twin_at_policy_and_presentation_resolutions"
                )
                presentation_resolution_checks += 1
                presentation_resolution_failed_checks += int(
                    audit["passed"] is not True
                )
                presentation_resolution_maximum_mae = max(
                    presentation_resolution_maximum_mae,
                    float(audit["mean_absolute_uint8_error"]),
                )
                presentation_resolution_minimum_correlation = min(
                    presentation_resolution_minimum_correlation,
                    float(audit["rgb_correlation"]),
                )
                if audit["passed"] is not True:
                    from PIL import Image

                    diagnostic_prefix = (
                        "presentation_resolution_failure__"
                        f"check-{presentation_resolution_checks:04d}"
                    )
                    source_diagnostic = (
                        out_dir / f"{diagnostic_prefix}__source.png"
                    )
                    target_diagnostic = (
                        out_dir / f"{diagnostic_prefix}__target.png"
                    )
                    audit_diagnostic = (
                        out_dir / f"{diagnostic_prefix}__audit.json"
                    )
                    Image.fromarray(source_frame).save(source_diagnostic)
                    Image.fromarray(target_frame).save(target_diagnostic)
                    _write_json_atomic(
                        audit_diagnostic,
                        {
                            **audit,
                            "source_path": source_diagnostic.name,
                            "target_path": target_diagnostic.name,
                            "initial_model_mirror": (
                                initial_presentation_model_mirror
                            ),
                            "initial_state_mirror": (
                                initial_presentation_state_mirror
                            ),
                        },
                    )
                    raise RuntimeError(
                        "Native presentation rendering twin failed full-trajectory "
                        f"source-resolution consistency at check "
                        f"{presentation_resolution_checks}"
                    )
                return target_frame, audit

            initial_presentation_state_mirror = sync_presentation_render_state()
            _, presentation_resolution_audit = (
                render_and_audit_presentation_simulator()
            )
            if component_gs_provider_mode:
                from .libero_component_gs import ComponentGaussianScene

                presentation_build_config = dict(provider_config)
                presentation_build_config.update(
                    {
                        "width": presentation_width,
                        "height": presentation_height,
                        "seed": seed + 900_001,
                        "max_gaussians": int(
                            presentation_config.get(
                                "component_gs_max_gaussians", 1_200_000
                            )
                        ),
                    }
                )
                presentation_component_scene = ComponentGaussianScene.build(
                    presentation_sim,
                    presentation_build_config,
                    0,
                )

                def presentation_renderer():
                    sync_presentation_render_state()
                    target_frame, _ = render_and_audit_presentation_simulator()
                    return render_component_gs_observer(
                        presentation_component_scene,
                        presentation_sim,
                        width=presentation_width,
                        height=presentation_height,
                        orientation_reference=target_frame,
                    )

                presentation_source = (
                    "native_high_resolution_actor_local_component_gaussians"
                )
            else:
                def presentation_renderer():
                    sync_presentation_render_state()
                    target_frame, _ = render_and_audit_presentation_simulator()
                    return target_frame

                presentation_source = (
                    "state_mirrored_nominal_shadow_renderer"
                    if canonical_environment is not None
                    else "execution_simulator_renderer"
                )
            presentation_path = (
                out_dir / "presentation__policy-domain__agentview.mp4"
            )
            presentation_recorder = DirectPresentationRecorder(
                presentation_path,
                width=presentation_width,
                height=presentation_height,
                fps=int(presentation_config["fps"]),
                crf=int(presentation_config["crf"]),
            )
        target_pattern = spec.get("diagnostic_target_body_pattern")
        initial_target_positions, diagnostic_target_names = _target_body_positions(
            environment,
            target_pattern,
        )
        min_eef_target_distance = np.full(num_episodes, np.inf, dtype=np.float64)
        max_target_displacement = np.zeros(num_episodes, dtype=np.float64)
        first_close_step = np.full(num_episodes, -1, dtype=int)
        frame_count = 0
        writer = imageio.get_writer(
            video_path,
            fps=int(spec["fps"]),
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
        )
        policy_writer = (
            imageio.get_writer(
                policy_video_path,
                fps=int(spec["fps"]),
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
            )
            if policy_video_path is not None
            else None
        )
        try:
            for step in range(max_steps):
                writer.append_data(_mosaic_frame(raw_observation, done, success))
                if policy_writer is not None:
                    policy_writer.append_data(
                        _mosaic_frame(policy_raw_observation, done, success)
                    )
                if presentation_recorder is not None:
                    presentation_recorder.append(presentation_renderer())
                frame_count += 1
                if initial_target_positions is not None:
                    target_positions, _ = _target_body_positions(environment, target_pattern)
                    eef_positions = np.asarray(raw_observation["robot_state"]["eef"]["pos"], dtype=np.float64)
                    active = ~done
                    distances = np.linalg.norm(eef_positions - target_positions, axis=1)
                    displacements = np.linalg.norm(target_positions - initial_target_positions, axis=1)
                    min_eef_target_distance[active] = np.minimum(
                        min_eef_target_distance[active], distances[active]
                    )
                    max_target_displacement[active] = np.maximum(
                        max_target_displacement[active], displacements[active]
                    )
                observation = preprocess_observation(policy_raw_observation)
                observation = add_envs_task(environment, observation)
                observation = env_preprocessor(observation)
                observation = preprocessor(observation)
                query_started = time.perf_counter()
                with torch.no_grad():
                    action = policy.select_action(observation)
                inference_seconds.append(time.perf_counter() - query_started)
                action = postprocessor(action)
                action_transition = env_postprocessor({"action": action})
                action_array = action_transition["action"].detach().cpu().numpy()
                newly_closing = (action_array[:, -1] > 0.0) & (first_close_step < 0) & (~done)
                first_close_step[newly_closing] = step + 1
                action_array[done] = dummy_actions[done]
                raw_observation, _reward, terminated, truncated, info = environment.step(action_array)
                new_success = _success_flags(info, num_episodes)
                if canonical_environment is not None:
                    if state_mirror_provider:
                        state_transfer = _mirror_environment_render_state(
                            environment,
                            canonical_environment,
                            tolerance=float(
                                provider_config.get("state_sync_tolerance", 1e-9)
                            ),
                        )
                        provider_state_transfer_checks += 1
                        provider_sync_checks += 1
                        provider_max_qpos_delta = max(
                            provider_max_qpos_delta,
                            float(state_transfer["maximum_qpos_delta"]),
                        )
                        provider_max_qvel_delta = max(
                            provider_max_qvel_delta,
                            float(state_transfer["maximum_qvel_delta"]),
                        )
                        if state_transfer["verified"] is not True:
                            _write_json_atomic(
                                out_dir / "canonical_sync_failure.json",
                                {
                                    "schema": "fedsplat.canonical_sync_failure.v1",
                                    "mode": provider_mode,
                                    "step": step + 1,
                                    "state_transfer": state_transfer,
                                },
                            )
                            raise RuntimeError(
                                "Canonical state-mirror transfer failed exact-state audit"
                            )
                        if component_gs_provider_mode:
                            policy_raw_observation, _ = (
                                component_gs_provider.render_policy_observation(
                                    raw_observation,
                                    _environment_sims(canonical_environment),
                                )
                            )
                            reference_block = (
                                component_gs_provider.record_diagnostic_reference(
                                    policy_raw_observation,
                                    _environment_sims(canonical_environment),
                                    step=step + 1,
                                    out_dir=out_dir,
                                )
                            )
                            if reference_block is not None:
                                _write_json_atomic(
                                    out_dir / "component_gs_sparse_reference_audit.json",
                                    component_gs_provider.report()["sparse_reference_audit"],
                                )
                        else:
                            canonical_observation = _refresh_vector_observation(
                                canonical_environment
                            )
                        provider_report["state_mirror"].update(
                            {
                                "checks": provider_state_transfer_checks,
                                "maximum_qpos_delta": provider_max_qpos_delta,
                                "maximum_qvel_delta": provider_max_qvel_delta,
                                "performed_before_every_policy_render": True,
                                "verified": True,
                            }
                        )
                    else:
                        (
                            canonical_observation,
                            _canonical_reward,
                            canonical_terminated,
                            canonical_truncated,
                            canonical_info,
                        ) = canonical_environment.step(action_array.copy())
                        sync = _compare_environment_physics(
                            environment,
                            canonical_environment,
                            tolerance=float(
                                provider_config.get("state_sync_tolerance", 1e-9)
                            ),
                        )
                        provider_sync_checks += 1
                        provider_max_qpos_delta = max(
                            provider_max_qpos_delta,
                            float(sync["maximum_qpos_delta"]),
                        )
                        provider_max_qvel_delta = max(
                            provider_max_qvel_delta,
                            float(sync["maximum_qvel_delta"]),
                        )
                        execution_done_step = np.asarray(
                            terminated | truncated, dtype=bool
                        )
                        canonical_done_step = np.asarray(
                            canonical_terminated | canonical_truncated, dtype=bool
                        )
                        termination_match = bool(
                            np.array_equal(execution_done_step, canonical_done_step)
                        )
                        canonical_success = _success_flags(
                            canonical_info, num_episodes
                        )
                        success_match = bool(
                            np.array_equal(new_success, canonical_success)
                        )
                        if (
                            not sync["verified"]
                            or not termination_match
                            or not success_match
                        ):
                            _write_json_atomic(
                                out_dir / "canonical_sync_failure.json",
                                {
                                    "schema": "fedsplat.canonical_sync_failure.v1",
                                    "mode": provider_mode,
                                    "step": step + 1,
                                    "physics_sync": sync,
                                    "termination_match": termination_match,
                                    "success_match": success_match,
                                    "execution_success": new_success.tolist(),
                                    "canonical_success": canonical_success.tolist(),
                                },
                            )
                            raise RuntimeError(
                                "Canonical provider lost physics, termination, or success synchronization"
                            )
                    provider_report["physics_sync"].update(
                        {
                            "checks": provider_sync_checks,
                            "maximum_qpos_delta": provider_max_qpos_delta,
                            "maximum_qvel_delta": provider_max_qvel_delta,
                            "verified": True,
                        }
                    )
                    if not component_gs_provider_mode:
                        policy_raw_observation = _compose_canonical_policy_observation(
                            raw_observation,
                            canonical_observation,
                        )
                else:
                    policy_raw_observation = raw_observation
                newly_successful = new_success & (success_step < 0)
                success_step[newly_successful] = step + 1
                success |= new_success
                done |= np.asarray(terminated | truncated, dtype=bool)
                if step == 0 or (step + 1) % 25 == 0 or np.all(done):
                    _write_json_atomic(progress_path, {
                        "schema": "fedsplat.libero_rollout_progress.v2",
                        "status": "running",
                        "step": step + 1,
                        "max_steps": max_steps,
                        "success_count": int(success.sum()),
                        "done_count": int(done.sum()),
                        "num_episodes": num_episodes,
                        "elapsed_s": time.perf_counter() - started,
                    })
                if np.all(done):
                    break
            writer.append_data(_mosaic_frame(raw_observation, done, success))
            if policy_writer is not None:
                policy_writer.append_data(
                    _mosaic_frame(policy_raw_observation, done, success)
                )
            if presentation_recorder is not None:
                presentation_recorder.append(presentation_renderer())
            frame_count += 1
        finally:
            writer.close()
            if policy_writer is not None:
                policy_writer.close()
            if presentation_recorder is not None:
                presentation_recorder.close()

        elapsed = time.perf_counter() - started
        presentation_report = None
        if presentation_recorder is not None:
            presentation_report = presentation_recorder.manifest_entry(
                source=str(presentation_source),
                camera_name=str(presentation_config["camera_name"]),
            )
            if int(presentation_report["frame_count"]) != frame_count:
                raise RuntimeError(
                    "Presentation and rollout frame counts do not align"
                )
            presentation_indices = {
                "first": 0,
                "middle": frame_count // 2,
                "last": frame_count - 1,
            }
            presentation_frames = _decode_video_frames(
                presentation_path, list(presentation_indices.values())
            )
            from PIL import Image

            presentation_keyframes = []
            for phase, index in presentation_indices.items():
                frame = presentation_frames[index]
                if float(np.asarray(frame).std()) <= 1.0:
                    raise RuntimeError(
                        f"Presentation {phase} keyframe is blank"
                    )
                path = (
                    out_dir
                    / f"presentation__policy-domain__agentview__frame-{phase}.png"
                )
                Image.fromarray(frame).save(path)
                presentation_keyframes.append(
                    {
                        "phase": phase,
                        "frame_index": index,
                        "path": path.name,
                        "sha256": _sha256(path),
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                    }
                )
            presentation_report.update(
                {
                    "observer_only": True,
                    "policy_input_unchanged": True,
                    "policy_input_width": int(spec["observation_width"]),
                    "policy_input_height": int(spec["observation_height"]),
                    "selected_environment_index": int(
                        presentation_config["environment_index"]
                    ),
                    "selection_rule": presentation_config.get(
                        "selection_rule"
                    ),
                    "formal_cell_id": presentation_config.get(
                        "formal_cell_id"
                    ),
                    "formal_audit": presentation_config.get("formal_audit"),
                    "formal_audit_sha256": presentation_config.get(
                        "formal_audit_sha256"
                    ),
                    "dedicated_native_resolution_rendering_twin": True,
                    "rendering_twin": {
                        "environment_count": 1,
                        "observation_width": presentation_width,
                        "observation_height": presentation_height,
                        "initial_model_mirror": initial_presentation_model_mirror,
                        "initial_state_mirror": initial_presentation_state_mirror,
                        "state_mirror_checks": presentation_state_mirror_checks,
                        "maximum_qpos_delta": (
                            presentation_state_mirror_max_qpos_delta
                        ),
                        "maximum_qvel_delta": (
                            presentation_state_mirror_max_qvel_delta
                        ),
                        "source_resolution_consistency": (
                            presentation_resolution_audit
                        ),
                        "full_trajectory_source_resolution_consistency": {
                            "schema": (
                                "fedsplat.native_resolution_full_trajectory_audit.v1"
                            ),
                            "checks": presentation_resolution_checks,
                            "expected_checks": frame_count + 1,
                            "failed_checks": (
                                presentation_resolution_failed_checks
                            ),
                            "maximum_mean_absolute_uint8_error": (
                                presentation_resolution_maximum_mae
                            ),
                            "minimum_rgb_correlation": (
                                presentation_resolution_minimum_correlation
                            ),
                            "outcome_or_success_used": False,
                            "passed": bool(
                                presentation_resolution_checks == frame_count + 1
                                and presentation_resolution_failed_checks == 0
                            ),
                        },
                        "offscreen_context_ownership": {
                            "simulator_count": len(
                                presentation_context_ownership
                            ),
                            "all_enforced": all(
                                record["enforced"]
                                for record in presentation_context_ownership
                            ),
                            "records": presentation_context_ownership,
                        },
                        "verified": bool(
                            initial_presentation_model_mirror["verified"]
                            and presentation_resolution_audit["passed"]
                            and presentation_resolution_checks
                            == frame_count + 1
                            and presentation_resolution_failed_checks == 0
                            and all(
                                record["enforced"]
                                for record in presentation_context_ownership
                            )
                            and presentation_state_mirror_max_qpos_delta <= 1e-9
                            and presentation_state_mirror_max_qvel_delta <= 1e-9
                        ),
                    },
                    "keyframes": presentation_keyframes,
                }
            )
            if presentation_component_scene is not None:
                presentation_latencies = np.asarray(
                    presentation_component_scene.render_latencies_s,
                    dtype=np.float64,
                )
                presentation_report["component_gs_native_scene"] = {
                    "construction_width": int(
                        presentation_component_scene.width
                    ),
                    "construction_height": int(
                        presentation_component_scene.height
                    ),
                    "gaussian_count": int(
                        presentation_component_scene.initialization_report[
                            "gaussian_count"
                        ]
                    ),
                    "maximum_gaussian_budget": int(
                        presentation_config.get(
                            "component_gs_max_gaussians", 1_200_000
                        )
                    ),
                    "simulator_rgb_per_policy_step": False,
                    "observer_only": True,
                    "initialization_report": (
                        presentation_component_scene.initialization_report
                    ),
                    "render_latency_ms": {
                        "mean": float(presentation_latencies.mean() * 1000.0),
                        "p95": float(
                            np.percentile(presentation_latencies, 95) * 1000.0
                        ),
                    },
                }
        keyframe_specs = [(0, "first"), (frame_count // 2, "middle"), (frame_count - 1, "last")]
        if np.any(success_step >= 0):
            event_index = min(int(success_step[success_step >= 0].min()), frame_count - 1)
            keyframe_specs.append((event_index, "event-success"))
        keyframe_specs = list(dict.fromkeys(keyframe_specs))
        decoded_keyframes = _decode_video_frames(
            video_path,
            [index for index, _label in keyframe_specs],
        )
        keyframes = []
        from PIL import Image

        for index, label in keyframe_specs:
            path = out_dir / f"libero_rollout__f{index:06d}__{label}.png"
            Image.fromarray(decoded_keyframes[index]).save(path)
            keyframes.append({"path": path.name, "frame_index": index, "label": label, "sha256": _sha256(path)})

        event_index = event_index if np.any(success_step >= 0) else frame_count - 1
        collage = _write_rollout_collage(
            video_path,
            frame_count,
            out_dir,
            event_index,
            success_observed=bool(np.any(success_step >= 0)),
        )

        decoded_first = decoded_keyframes[0]
        video_nonblank = float(np.var(decoded_first)) > 1.0 and frame_count > 1
        if policy_video_path is not None:
            decoded_policy = _decode_video_frames(
                policy_video_path, [0, frame_count - 1]
            )
            policy_first = decoded_policy[0]
            provider_report["policy_video"] = {
                "path": policy_video_path.name,
                "sha256": _sha256(policy_video_path),
                "bytes": policy_video_path.stat().st_size,
                "width": int(policy_first.shape[1]),
                "height": int(policy_first.shape[0]),
                "fps": int(spec["fps"]),
                "frame_count": frame_count,
                "duration_s": frame_count / float(spec["fps"]),
                "layout": "canonical_policy_input_all_environment_mosaic",
                "environment_count": num_episodes,
                "codec": "h264",
                "pixel_format": "yuv420p",
                "nonblank": float(np.var(policy_first)) > 1.0 and frame_count > 1,
            }
        if component_gs_provider is not None:
            provider_report["component_gs"] = component_gs_provider.report()
            _write_json_atomic(
                out_dir / "component_gs_provider_audit.json",
                provider_report["component_gs"],
            )
        success_count = int(success.sum())
        evidence_label = "sim_rollout_success" if success_count > 0 else "visual_ok"
        episode_diagnostics = []
        for index in range(num_episodes):
            minimum_distance = (
                float(min_eef_target_distance[index])
                if np.isfinite(min_eef_target_distance[index]) else None
            )
            displacement = (
                float(max_target_displacement[index])
                if initial_target_positions is not None else None
            )
            close_step = int(first_close_step[index]) if first_close_step[index] >= 0 else None
            episode_diagnostics.append({
                "episode_index": index,
                "target_body": diagnostic_target_names[index] if diagnostic_target_names else None,
                "success": bool(success[index]),
                "success_step": int(success_step[index]) if success_step[index] >= 0 else None,
                "min_eef_target_distance_m": minimum_distance,
                "max_target_displacement_m": displacement,
                "first_gripper_close_step": close_step,
                "failure_class": _classify_episode_diagnostic(
                    bool(success[index]), minimum_distance, displacement, close_step
                ),
            })
        latencies_ms = np.asarray(inference_seconds, dtype=np.float64) * 1000.0
        report = {
            "schema": "fedsplat.libero_rollout.v2",
            "evidence_label": evidence_label,
            "suite": suite,
            "task_id": task_id,
            "checkpoint": str(checkpoint),
            "adapter_update": spec.get("adapter_update"),
            "adapter_dense_fingerprint_sha256": adapter_fingerprint,
            "adapter_bank_manifest": spec.get("adapter_bank_manifest"),
            "visual_router_manifest": spec.get("visual_router_manifest"),
            "adapter_routing": adapter_routing_report,
            "policy_observation_provider": provider_report,
            "presentation_capture": presentation_report,
            "task_description": environment.call("task_description"),
            "seed": seed,
            "determinism": {
                "requested": deterministic,
                "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
            "libero_runtime": libero_runtime,
            "environment_contract": environment_contract,
            "visual_domain_shift": visual_shift_report,
            "startup_observation_audit": startup_observation_audit,
            "episode_diagnostics": episode_diagnostics,
            "failure_class_counts": {
                label: sum(item["failure_class"] == label for item in episode_diagnostics)
                for label in sorted({item["failure_class"] for item in episode_diagnostics})
            },
            "num_episodes": num_episodes,
            "success_count": success_count,
            "success_rate": float(success.mean()),
            "success_flags": success.astype(int).tolist(),
            "success_step": success_step.tolist(),
            "steps_executed": len(inference_seconds),
            "environment_success_signal_observed": success_count > 0,
            "adapter_applied": bool(applied_layers),
            "adapter_layers": applied_layers,
            "rollout_completed": True,
            "video_nonblank": video_nonblank,
            "video": {
                "path": video_path.name,
                "sha256": _sha256(video_path),
                "bytes": video_path.stat().st_size,
                "width": int(decoded_first.shape[1]),
                "height": int(decoded_first.shape[0]),
                "fps": int(spec["fps"]),
                "frame_count": frame_count,
                "duration_s": frame_count / float(spec["fps"]),
                "layout": "all_environment_mosaic",
                "environment_count": num_episodes,
                "codec": "h264",
                "pixel_format": "yuv420p",
            },
            "keyframes": keyframes,
            "paper_collage": collage,
            "inference_latency_ms": {
                "mean": float(latencies_ms.mean()),
                "p50": float(np.percentile(latencies_ms, 50)),
                "p95": float(np.percentile(latencies_ms, 95)),
            },
            "wall_clock_s": elapsed,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if str(spec["device"]).startswith("cuda") else 0,
        }
        (out_dir / "rollout_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _write_json_atomic(progress_path, {
            "schema": "fedsplat.libero_rollout_progress.v2",
            "status": "complete",
            "step": len(inference_seconds),
            "max_steps": max_steps,
            "success_count": success_count,
            "done_count": int(done.sum()),
            "num_episodes": num_episodes,
            "elapsed_s": elapsed,
            "report": str((out_dir / "rollout_report.json").resolve()),
        })
        _write_rollout_report(report, out_dir / "LIBERO_ROLLOUT_REPORT.md")
        manifest = [{
            "schema": "fedsplat.media.v1",
            "kind": "sim_rollout_video",
            "evidence_label": evidence_label,
            **report["video"],
            "suite": suite,
            "task_id": task_id,
            "seed": seed,
            "environment_success": success_count > 0,
        }]
        for item in keyframes:
            manifest.append({
                "schema": "fedsplat.media.v1",
                "kind": "sim_rollout_keyframe",
                "evidence_label": evidence_label,
                "source_video": video_path.name,
                "environment_success": success_count > 0,
                **item,
            })
        for item in collage["artifacts"]:
            manifest.append({
                "schema": "fedsplat.media.v1",
                "kind": item["kind"],
                "evidence_label": evidence_label,
                "environment_success": success_count > 0,
                "paper_width": collage["paper_width"],
                "grid_columns": collage["grid_columns"],
                "grid_rows": collage["grid_rows"],
                "canonical_suffix": collage["canonical_suffix"],
                "final_physical_width_in": collage["final_physical_width_in"],
                "panel_aspect_ratio": collage["panel_aspect_ratio"],
                "frame_selection_rule": collage["frame_selection_rule"],
                **item,
            })
        if visual_shift_report.get("preview"):
            manifest.append({
                "schema": "fedsplat.media.v1",
                "kind": "visual_shift_preview",
                "evidence_label": evidence_label,
                **visual_shift_report["preview"],
            })
        if provider_report is not None:
            manifest.append({
                "schema": "fedsplat.media.v1",
                "kind": (
                    "component_gs_policy_input_video"
                    if component_gs_provider_mode
                    else "canonical_policy_input_video"
                ),
                "evidence_label": provider_report["evidence_role"],
                **provider_report["policy_video"],
            })
        if presentation_report is not None:
            manifest.append(
                {
                    "schema": "fedsplat.media.v1",
                    "kind": "native_presentation_video",
                    "evidence_label": "presentation_only_nonstatistical",
                    **{
                        key: value
                        for key, value in presentation_report.items()
                        if key != "keyframes"
                    },
                }
            )
            for item in presentation_report["keyframes"]:
                manifest.append(
                    {
                        "schema": "fedsplat.media.v1",
                        "kind": "native_presentation_keyframe",
                        "evidence_label": "presentation_only_nonstatistical",
                        "source_video": presentation_report["path"],
                        "used_for_statistical_evidence": False,
                        **item,
                    }
                )
        (out_dir / "media_manifest.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in manifest),
            encoding="utf-8",
        )
        return report
    finally:
        try:
            environment.close()
        except Exception:
            pass
        if canonical_environment is not None:
            try:
                canonical_environment.close()
            except Exception:
                pass
        if presentation_environment is not None:
            try:
                presentation_environment.close()
            except Exception:
                pass


def _write_rollout_collage(
    video_path: Path,
    frame_count: int,
    out_dir: Path,
    event_index: int,
    *,
    success_observed: bool,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from .media_contract import validate_paper_layout

    layout = validate_paper_layout("single-column", 3, 2)
    selected = np.linspace(0, event_index, 6).round().astype(int).tolist()
    labels = ["Start", "20%", "40%", "60%", "80%", "Success" if success_observed else "Final"]
    decoded = _decode_video_frames(video_path, selected)
    source_height, source_width = decoded[selected[0]].shape[:2]
    panel_width = 330
    panel_height = int(round(panel_width * source_height / source_width))
    cell_width, cell_height = 350, panel_height + 46
    canvas = Image.new("RGB", (cell_width * 3, cell_height * 2), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=25)
    except OSError:
        font = ImageFont.load_default()
    for position, (frame_index, label) in enumerate(zip(selected, labels, strict=True)):
        column = position % 3
        row = position // 3
        x = column * cell_width + 10
        y = row * cell_height + 8
        panel = Image.fromarray(decoded[frame_index]).resize(
            (panel_width, panel_height), Image.Resampling.LANCZOS
        )
        canvas.paste(panel, (x, y))
        draw.text((x, y + panel_height + 7), f"{label}  f{frame_index:03d}", fill=(25, 30, 38), font=font)

    stem = ("libero_success_phases" if success_observed else "libero_failure_phases") + layout.canonical_suffix
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    canvas.save(png_path, dpi=(300, 300))
    canvas.save(pdf_path, "PDF", resolution=300.0)
    return {
        **layout.to_dict(),
        "final_physical_width_in": 3.5,
        "panel_aspect_ratio": f"{source_width}:{source_height}",
        "frame_selection_rule": (
            "six uniformly spaced normalized phases from reset through first success"
            if success_observed else
            "six uniformly spaced normalized phases from reset through final rollout frame"
        ),
        "success_observed": success_observed,
        "source_frame_count": frame_count,
        "selected_frame_indices": selected,
        "artifacts": [
            {
                "path": png_path.name,
                "kind": "paper_rollout_collage_png",
                "bytes": png_path.stat().st_size,
                "sha256": _sha256(png_path),
                "width": canvas.width,
                "height": canvas.height,
                "dpi": 300,
            },
            {
                "path": pdf_path.name,
                "kind": "paper_rollout_collage_pdf",
                "bytes": pdf_path.stat().st_size,
                "sha256": _sha256(pdf_path),
            },
        ],
    }


def _write_rollout_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# LIBERO Single-Task Rollout Report",
        "",
        f"- Evidence: `{report['evidence_label']}`",
        f"- Suite/task: `{report['suite']}` / `{report['task_id']}`",
        f"- Success: {report['success_count']}/{report['num_episodes']}",
        f"- Visual domain: `{report.get('visual_domain_shift', {}).get('name', 'nominal')}`",
        f"- Failure classes: `{report.get('failure_class_counts', {})}`",
        f"- Environment success signal: {report['environment_success_signal_observed']}",
        f"- Adapter applied: {report['adapter_applied']}",
        f"- Steps: {report['steps_executed']}",
        f"- Mean inference latency: {report['inference_latency_ms']['mean']:.2f} ms",
        f"- Video: `{report['video']['path']}` ({report['video']['frame_count']} frames)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.worker_config).read_text(encoding="utf-8"))
    errors = validate_rollout_spec(spec)
    if errors:
        raise ValueError("Invalid worker config: " + "; ".join(errors))
    report = _worker(spec)
    print(json.dumps({"rollout_completed": report["rollout_completed"], "success_rate": report["success_rate"]}))


if __name__ == "__main__":
    main()
