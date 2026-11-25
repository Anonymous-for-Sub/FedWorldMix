"""Closed-loop LIBERO evaluation for OpenVLA-OFT and FedSplat adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any


def validate_openvla_rollout_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("checkpoint_path", "openvla_root", "suite"):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            errors.append(f"{key} must be a nonempty string")
    for key in ("task_id", "seed", "max_steps", "wait_steps", "action_horizon"):
        value = spec.get(key)
        if not isinstance(value, int) or value < (0 if key in {"task_id", "seed", "wait_steps"} else 1):
            errors.append(f"{key} must be a valid nonnegative integer")
    indices = spec.get("initial_state_indices")
    if (
        not isinstance(indices, list)
        or not indices
        or any(not isinstance(value, int) or value < 0 for value in indices)
        or len(indices) != len(set(indices))
    ):
        errors.append("initial_state_indices must contain unique nonnegative integers")
    for key in ("camera_height", "camera_width", "fps"):
        if not isinstance(spec.get(key), int) or int(spec.get(key, 0)) <= 0:
            errors.append(f"{key} must be a positive integer")
    adapter = spec.get("adapter_update")
    if adapter is not None and (not isinstance(adapter, str) or not adapter.strip()):
        errors.append("adapter_update must be null or a nonempty string")
    if spec.get("expected_adapter", adapter is not None) and adapter is None:
        errors.append("expected_adapter=true requires adapter_update")
    visual_shift = spec.get("visual_domain_shift")
    if visual_shift is not None:
        if not isinstance(visual_shift, dict):
            errors.append("visual_domain_shift must be an object")
        else:
            if not isinstance(visual_shift.get("name"), str) or not visual_shift[
                "name"
            ].strip():
                errors.append("visual_domain_shift.name must be a nonempty string")
            if not isinstance(visual_shift.get("camera_name"), str) or not visual_shift[
                "camera_name"
            ].strip():
                errors.append(
                    "visual_domain_shift.camera_name must be a nonempty string"
                )
            for key, limit in (
                ("camera_position_delta_m", 0.25),
                ("camera_rotation_delta_deg", 30.0),
            ):
                values = visual_shift.get(key, [0.0, 0.0, 0.0])
                if (
                    not isinstance(values, list)
                    or len(values) != 3
                    or any(
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or abs(float(value)) > limit
                        for value in values
                    )
                ):
                    errors.append(
                        f"visual_domain_shift.{key} must contain three finite "
                        f"values in [-{limit}, {limit}]"
                    )
            for key in (
                "light_ambient_scale",
                "light_diffuse_scale",
                "light_specular_scale",
            ):
                value = visual_shift.get(key, 1.0)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 < float(value) <= 4.0
                ):
                    errors.append(
                        f"visual_domain_shift.{key} must be finite and in (0, 4]"
                    )
    provider = spec.get("policy_observation_provider")
    if provider is not None:
        if not isinstance(provider, dict):
            errors.append("policy_observation_provider must be an object")
        else:
            if (
                provider.get("mode")
                != "component_gs_state_mirror_pose_oracle"
            ):
                errors.append(
                    "OpenVLA policy_observation_provider.mode must be "
                    "component_gs_state_mirror_pose_oracle"
                )
            if (
                provider.get("evidence_role")
                != "development_component_gs_pose_oracle_only"
            ):
                errors.append(
                    "OpenVLA component-GS provider requires the "
                    "development-only evidence role"
                )
            tolerance = provider.get("state_sync_tolerance")
            if (
                not isinstance(tolerance, (int, float))
                or not math.isfinite(float(tolerance))
                or not 0.0 < float(tolerance) <= 1e-6
            ):
                errors.append(
                    "policy_observation_provider.state_sync_tolerance must "
                    "be finite and in (0, 1e-6]"
                )
            gate = provider.get("startup_quality_gate")
            if not isinstance(gate, dict) or not {
                "minimum_psnr_db",
                "minimum_ssim",
                "minimum_mean_alpha",
                "maximum_mean_absolute_uint8",
            }.issubset(gate):
                errors.append(
                    "policy_observation_provider.startup_quality_gate is incomplete"
                )
    presentation = spec.get("presentation_capture")
    if presentation is not None:
        if not isinstance(presentation, dict):
            errors.append("presentation_capture must be an object")
        else:
            from .presentation_capture import validate_presentation_capture

            streams = presentation.get("streams")
            if (
                not isinstance(streams, list)
                or not streams
                or len(streams) > 2
                or any(not isinstance(stream, dict) for stream in streams)
            ):
                errors.append(
                    "presentation_capture.streams must contain one or two objects"
                )
            else:
                names = [stream.get("name") for stream in streams]
                if (
                    any(not isinstance(name, str) or not name.strip() for name in names)
                    or len(names) != len(set(names))
                ):
                    errors.append(
                        "presentation_capture stream names must be unique and nonempty"
                    )
                for index, stream in enumerate(streams):
                    capture = {
                        **presentation,
                        "camera_name": stream.get("camera_name"),
                        "environment_index": 0,
                    }
                    capture.pop("streams", None)
                    errors.extend(
                        f"presentation_capture.streams[{index}]: {error}"
                        for error in validate_presentation_capture(
                            capture,
                            rollout_fps=int(spec.get("fps", 0)),
                            environment_count=1,
                        )
                    )
    return errors


def choose_representative_episode(
    success_flags_by_method: dict[str, list[int]],
    *,
    proposed_method: str = "fedsplat_projected_source_pair",
    priority_controls: tuple[str, ...] = ("fedavg_real", "base_checkpoint"),
) -> dict[str, Any]:
    """Choose a paired episode that exposes a treatment/control divergence."""
    if proposed_method not in success_flags_by_method:
        raise ValueError(f"Missing proposed method {proposed_method!r}")
    lengths = {len(flags) for flags in success_flags_by_method.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("Success flags must be nonempty and equally sized")
    if any(flag not in (0, 1) for flags in success_flags_by_method.values() for flag in flags):
        raise ValueError("Success flags must be binary")

    proposed = success_flags_by_method[proposed_method]
    for control in priority_controls:
        if control not in success_flags_by_method:
            continue
        for index, (treatment, baseline) in enumerate(
            zip(proposed, success_flags_by_method[control], strict=True)
        ):
            if treatment == 1 and baseline == 0:
                return {
                    "episode_index": index,
                    "reason": f"{proposed_method}_success_{control}_failure",
                    "success_flags": {
                        method: flags[index]
                        for method, flags in success_flags_by_method.items()
                    },
                }

    for index in range(next(iter(lengths))):
        outcomes = {
            method: flags[index] for method, flags in success_flags_by_method.items()
        }
        if len(set(outcomes.values())) > 1:
            return {
                "episode_index": index,
                "reason": "first_paired_outcome_divergence",
                "success_flags": outcomes,
            }
    return {
        "episode_index": 0,
        "reason": "all_paired_outcomes_equal",
        "success_flags": {
            method: flags[0] for method, flags in success_flags_by_method.items()
        },
    }


def run_openvla_oft_rollout(
    config_path: Path,
    out_dir: Path,
    runtime_python: Path,
    *,
    pythonpath: list[Path] | None = None,
    cuda_visible_device: str | None = None,
    timeout_s: int = 7200,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = _discover_project_root(config_path)
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    errors = validate_openvla_rollout_spec(spec)
    if errors:
        raise ValueError("Invalid OpenVLA-OFT rollout config: " + "; ".join(errors))
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_spec = json.loads(json.dumps(spec))
    for key in ("checkpoint_path", "openvla_root", "adapter_update"):
        if worker_spec.get(key):
            path = Path(worker_spec[key])
            worker_spec[key] = str(
                path.resolve() if path.is_absolute() else (project_root / path).resolve()
            )
    worker_spec["out_dir"] = str(out_dir)
    worker_config = out_dir / "worker_config.json"
    worker_config.write_text(json.dumps(worker_spec, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(project_root),
            *(str(path.resolve()) for path in (pythonpath or [])),
            str(Path(worker_spec["openvla_root"]).resolve()),
        ]
    ) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # The shared OpenVLA environment can block in CPython module teardown after
    # all audited artifacts are already closed and persisted. Exit directly
    # from the worker only after its report has been written and stdout flushed.
    env["FEDSPLAT_OPENVLA_WORKER_DIRECT_EXIT"] = "1"
    if cuda_visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)
    runtime_python = runtime_python.absolute()
    if not runtime_python.exists():
        raise FileNotFoundError(runtime_python)
    started = time.perf_counter()
    completed = subprocess.run(
        [
            str(runtime_python),
            "-m",
            "fedsplat.openvla_oft_rollout",
            "--worker-config",
            str(worker_config),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
    )
    (out_dir / "worker_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "worker_stderr.log").write_text(completed.stderr, encoding="utf-8")
    report_path = out_dir / "rollout_report.json"
    if completed.returncode != 0 or not report_path.is_file():
        detail = (completed.stderr or completed.stdout or "worker produced no report")[-8000:]
        raise RuntimeError(f"OpenVLA-OFT rollout worker failed ({completed.returncode}): {detail}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_wall_s"] = time.perf_counter() - started
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _discover_project_root(config_path: Path) -> Path:
    """Resolve the source root for configs copied into nested run directories."""
    resolved = config_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "fedsplat").is_dir():
            return candidate
    return resolved.parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def quat_to_axis_angle(quaternion: Any) -> Any:
    import numpy as np

    value = np.asarray(quaternion, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - value[3] ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (value[:3] * 2.0 * math.acos(value[3]) / denominator).astype(np.float32)


def process_openvla_libero_action(action: Any) -> Any:
    """Map released OFT gripper convention back to LIBERO controls."""
    import numpy as np

    value = np.asarray(action, dtype=np.float32).copy()
    if value.shape != (7,) or not np.all(np.isfinite(value)):
        raise ValueError(f"Expected one finite 7D action, got {value.shape}")
    value[-1] = np.sign(2.0 * value[-1] - 1.0)
    value[-1] *= -1.0
    return value


def _apply_adapter(model: Any, path: Path) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.nn import functional

    from .lora_updates import (
        client_update_artifact_fingerprint,
        load_client_update,
    )

    class StaticAdapterLinear(nn.Module):
        def __init__(self, base: nn.Linear, a: Any, b: Any) -> None:
            super().__init__()
            if tuple(a.shape) != (a.shape[0], base.in_features) or tuple(b.shape) != (
                base.out_features,
                a.shape[0],
            ):
                raise ValueError("Adapter factors do not match the target linear layer")
            self.base = base
            self.register_buffer(
                "adapter_a",
                torch.as_tensor(a, dtype=torch.float32, device=base.weight.device),
            )
            self.register_buffer(
                "adapter_b",
                torch.as_tensor(b, dtype=torch.float32, device=base.weight.device),
            )

        def forward(self, inputs):
            base_output = self.base(inputs)
            update = functional.linear(
                functional.linear(inputs.float(), self.adapter_a), self.adapter_b
            )
            return base_output + update.to(base_output.dtype)

    update = load_client_update(path)
    applied = []
    for layer in update.layers:
        parent_name, attribute = layer.layer_name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        base = getattr(parent, attribute)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"OpenVLA adapter target is not linear: {layer.layer_name}")
        setattr(parent, attribute, StaticAdapterLinear(base, layer.a, layer.b))
        applied.append(layer.layer_name)
    fingerprint = client_update_artifact_fingerprint(path)
    fingerprint.update(
        {
            "source_client_id": update.client_id,
            "source_reliability": update.reliability,
            "applied_layers": applied,
        }
    )
    return fingerprint


def _policy_action(
    *,
    model: Any,
    processor: Any,
    action_head: Any,
    proprio_projector: Any,
    config: Any,
    observation: dict[str, Any],
    instruction: str,
) -> Any:
    import numpy as np
    import torch

    from .openvla_oft_local_update import _normalize_proprio, _prepare_image

    agent = np.asarray(observation["agentview_image"])[::-1, ::-1].copy()
    wrist = np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1].copy()
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, _prepare_image(agent, config.center_crop)).to(
        config.device, dtype=torch.bfloat16
    )
    wrist_inputs = processor(prompt, _prepare_image(wrist, config.center_crop)).to(
        config.device, dtype=torch.bfloat16
    )
    inputs["pixel_values"] = torch.cat(
        [inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1
    )
    proprio = np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    proprio = _normalize_proprio(
        proprio, model.norm_stats[config.unnorm_key]["proprio"]
    )
    with torch.inference_mode():
        actions, _ = model.predict_action(
            **inputs,
            unnorm_key=config.unnorm_key,
            do_sample=False,
            proprio=proprio,
            proprio_projector=proprio_projector,
            noisy_action_projector=None,
            action_head=action_head,
            use_film=False,
        )
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.all(np.isfinite(actions)):
        raise RuntimeError(f"OpenVLA returned invalid action chunk {actions.shape}")
    return actions


def component_gs_policy_observation(
    provider: Any,
    observation: dict[str, Any],
    sim: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Replace only the agent-view policy image with the GS render."""
    import numpy as np

    vector_observation = {
        "pixels": {
            "image": np.asarray(observation["agentview_image"])[None, ...]
        }
    }
    rendered, startup = provider.render_policy_observation(
        vector_observation,
        [sim],
    )
    result = dict(observation)
    result["agentview_image"] = np.asarray(
        rendered["pixels"]["image"][0],
        dtype=np.uint8,
    )
    return result, startup


def _decode_video(path: Path) -> tuple[int, Any]:
    import imageio_ffmpeg
    import numpy as np

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        first = None
        count = 0
        for payload in reader:
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
            if first is None:
                first = frame.copy()
            count += 1
    finally:
        reader.close()
    if first is None:
        raise RuntimeError(f"Video has no decoded frames: {path}")
    return count, first


def _make_rollout_collage(
    out_dir: Path, episode_record: dict[str, Any]
) -> dict[str, Any]:
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw

    def selected_frames(path: Path, wanted: set[int]) -> dict[int, Any]:
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        result = {}
        try:
            metadata = next(reader)
            width, height = metadata["size"]
            for index, payload in enumerate(reader):
                if index in wanted:
                    result[index] = np.frombuffer(payload, dtype=np.uint8).reshape(
                        height, width, 3
                    ).copy()
        finally:
            reader.close()
        return result

    frame_count = int(episode_record["frame_count"])
    indices = [0, frame_count // 2, frame_count - 1]
    streams = []
    for name in ("agentview", "wrist"):
        path = Path(episode_record["videos"][name]["absolute_path"])
        frames = selected_frames(path, set(indices))
        streams.append((name, [frames[index] for index in indices]))
    height, width = streams[0][1][0].shape[:2]
    label_height = 24
    canvas = Image.new("RGB", (width * 3, (height + label_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for row, (stream, frames) in enumerate(streams):
        for column, (frame_index, frame) in enumerate(zip(indices, frames, strict=True)):
            x = column * width
            y = row * (height + label_height)
            canvas.paste(Image.fromarray(frame), (x, y + label_height))
            draw.text((x + 5, y + 5), f"{stream} / f{frame_index}", fill="black")
    path = out_dir / "paper_singlecol_3x2.png"
    canvas.save(path)
    return {"path": str(path.resolve()), "sha256": _sha256(path), "layout": "3x2"}


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image

    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Robosuite 1.4 hard-codes a shared /tmp log during import. Isolate it per
    # method so concurrent users and presentation workers cannot collide.
    import logging

    original_file_handler = logging.FileHandler

    def redirected_file_handler(filename, *args, **kwargs):
        if str(filename) == "/tmp/robosuite.log":
            filename = out_dir / "robosuite.log"
        return original_file_handler(filename, *args, **kwargs)

    logging.FileHandler = redirected_file_handler
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    finally:
        logging.FileHandler = original_file_handler

    from .openvla_oft_local_update import _load_policy

    seed = int(spec["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    started = time.perf_counter()
    model, processor, action_head, proprio, policy_config, load_seconds = _load_policy(spec)
    adapter_audit = None
    if spec.get("adapter_update"):
        adapter_audit = _apply_adapter(model, Path(spec["adapter_update"]))
    model.eval()

    suite_name = str(spec["suite"])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task_id = int(spec["task_id"])
    task = suite.get_task(task_id)
    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    initial_states = np.asarray(suite.get_task_init_states(task_id))
    indices = list(spec["initial_state_indices"])
    if max(indices) >= len(initial_states):
        raise ValueError("Requested LIBERO initial state is unavailable")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=int(spec["camera_height"]),
        camera_widths=int(spec["camera_width"]),
    )
    env.seed(seed)
    provider_config = spec.get("policy_observation_provider")
    component_provider = None
    component_startup_quality = None
    component_state_mirror_checks = 0
    component_state_mirror_max_qpos_delta = 0.0
    component_state_mirror_max_qvel_delta = 0.0
    canonical_env = None
    if provider_config is not None:
        canonical_env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=int(spec["camera_height"]),
            camera_widths=int(spec["camera_width"]),
        )
        canonical_env.seed(seed)

    visual_shift_report = {
        "applied": False,
        "name": "nominal",
        "environments": [],
    }
    if spec.get("visual_domain_shift") is not None:
        from types import SimpleNamespace

        from .libero_rollout import _apply_visual_domain_shift

        vector_proxy = SimpleNamespace(
            envs=[SimpleNamespace(_env=SimpleNamespace(sim=env.sim))]
        )
        visual_shift_report = _apply_visual_domain_shift(
            vector_proxy,
            spec["visual_domain_shift"],
        )
        if visual_shift_report.get("mutation_verified") is not True:
            raise RuntimeError("OpenVLA visual-domain mutation failed its audit")
    presentation_config = spec.get("presentation_capture")
    presentation_env = None
    presentation_context_ownership = []
    if presentation_config is not None:
        presentation_env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=int(presentation_config["height"]),
            camera_widths=int(presentation_config["width"]),
        )
        presentation_env.seed(seed)
        from .presentation_capture import enforce_offscreen_context_ownership

        presentation_context_ownership = [
            enforce_offscreen_context_ownership(env.sim),
            enforce_offscreen_context_ownership(presentation_env.sim),
        ]

    episode_records = []
    manifest = []
    try:
        for episode_index, state_index in enumerate(indices):
            episode_dir = out_dir / f"episode_{episode_index:03d}__state_{state_index:03d}"
            frame_dir = episode_dir / "direct_frames"
            frame_dir.mkdir(parents=True)
            env.reset()
            observation = env.set_init_state(initial_states[state_index])
            for _ in range(int(spec["wait_steps"])):
                observation, _reward, _done, _info = env.step(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                )
            if canonical_env is not None:
                canonical_env.reset()
                canonical_env.set_init_state(initial_states[state_index])
                from .presentation_capture import mirror_mujoco_render_state

                mirror = mirror_mujoco_render_state(
                    env.sim,
                    canonical_env.sim,
                    tolerance=float(provider_config["state_sync_tolerance"]),
                )
                if mirror["verified"] is not True:
                    raise RuntimeError(
                        "OpenVLA Component-GS initial state mirror failed"
                    )
                component_state_mirror_checks += 1
                component_state_mirror_max_qpos_delta = max(
                    component_state_mirror_max_qpos_delta,
                    float(mirror["maximum_qpos_delta"]),
                )
                component_state_mirror_max_qvel_delta = max(
                    component_state_mirror_max_qvel_delta,
                    float(mirror["maximum_qvel_delta"]),
                )
                if component_provider is None:
                    from .libero_component_gs import LiberoComponentGSProvider

                    component_provider = LiberoComponentGSProvider.build(
                        [canonical_env.sim],
                        provider_config,
                        width=int(spec["camera_width"]),
                        height=int(spec["camera_height"]),
                        seed=seed,
                        out_dir=out_dir,
                    )
            presentation_state_mirror_checks = 0
            presentation_state_mirror_max_qpos_delta = 0.0
            presentation_state_mirror_max_qvel_delta = 0.0
            presentation_resolution_audits = {}
            presentation_full_trajectory_audits = {}
            initial_presentation_state_mirror = None
            if presentation_env is not None:
                from .presentation_capture import (
                    audit_native_resolution_rendering_twin,
                    mirror_mujoco_render_state,
                )

                presentation_env.reset()
                presentation_env.set_init_state(initial_states[state_index])
                initial_presentation_state_mirror = mirror_mujoco_render_state(
                    env.sim,
                    presentation_env.sim,
                )
                if initial_presentation_state_mirror["verified"] is not True:
                    raise RuntimeError(
                        "OpenVLA presentation rendering twin failed initial sync"
                    )
                presentation_state_mirror_checks = 1
                presentation_state_mirror_max_qpos_delta = float(
                    initial_presentation_state_mirror["maximum_qpos_delta"]
                )
                presentation_state_mirror_max_qvel_delta = float(
                    initial_presentation_state_mirror["maximum_qvel_delta"]
                )
                for stream in presentation_config["streams"]:
                    name = str(stream["name"])
                    audit = audit_native_resolution_rendering_twin(
                        env.sim,
                        presentation_env.sim,
                        camera_name=str(stream["camera_name"]),
                        source_width=int(spec["camera_width"]),
                        source_height=int(spec["camera_height"]),
                        target_width=int(presentation_config["width"]),
                        target_height=int(presentation_config["height"]),
                    )
                    if audit["passed"] is not True:
                        raise RuntimeError(
                            "OpenVLA presentation rendering twin failed "
                            f"source-resolution consistency: {name}"
                        )
                    presentation_resolution_audits[name] = audit
                    presentation_full_trajectory_audits[name] = {
                        "schema": (
                            "fedsplat.native_resolution_full_trajectory_audit.v1"
                        ),
                        "checks": 1,
                        "failed_checks": 0,
                        "maximum_mean_absolute_uint8_error": float(
                            audit["mean_absolute_uint8_error"]
                        ),
                        "minimum_rgb_correlation": float(
                            audit["rgb_correlation"]
                        ),
                        "outcome_or_success_used": False,
                        "passed": True,
                    }

            video_paths = {
                "agentview": episode_dir / "agentview.mp4",
                "wrist": episode_dir / "wrist.mp4",
            }
            if component_provider is not None:
                video_paths["policy_agentview"] = (
                    episode_dir / "policy_agentview.mp4"
                )
            writers = {
                name: imageio.get_writer(
                    path,
                    fps=int(spec["fps"]),
                    codec="libx264",
                    pixelformat="yuv420p",
                    quality=8,
                )
                for name, path in video_paths.items()
            }
            presentation_paths = {}
            presentation_recorders = {}
            presentation_streams = {}
            if presentation_config is not None:
                from .presentation_capture import DirectPresentationRecorder

                for stream in presentation_config["streams"]:
                    name = str(stream["name"])
                    presentation_streams[name] = str(stream["camera_name"])
                    path = episode_dir / f"presentation__{name}.mp4"
                    presentation_paths[name] = path
                    presentation_recorders[name] = DirectPresentationRecorder(
                        path,
                        width=int(presentation_config["width"]),
                        height=int(presentation_config["height"]),
                        fps=int(presentation_config["fps"]),
                        crf=int(presentation_config["crf"]),
                    )
            action_queue: deque[Any] = deque()
            inference_s = []
            success = False
            success_step = None
            frame_count = 0
            last_frames = None
            action_queries = 0
            policy_observation = observation
            try:
                for step in range(int(spec["max_steps"])):
                    if component_provider is not None and not action_queue:
                        from .presentation_capture import mirror_mujoco_render_state

                        mirror = mirror_mujoco_render_state(
                            env.sim,
                            canonical_env.sim,
                            tolerance=float(
                                provider_config["state_sync_tolerance"]
                            ),
                        )
                        if mirror["verified"] is not True:
                            raise RuntimeError(
                                "OpenVLA Component-GS state mirror failed"
                            )
                        component_state_mirror_checks += 1
                        component_state_mirror_max_qpos_delta = max(
                            component_state_mirror_max_qpos_delta,
                            float(mirror["maximum_qpos_delta"]),
                        )
                        component_state_mirror_max_qvel_delta = max(
                            component_state_mirror_max_qvel_delta,
                            float(mirror["maximum_qvel_delta"]),
                        )
                        policy_observation, startup = (
                            component_gs_policy_observation(
                                component_provider,
                                observation,
                                canonical_env.sim,
                            )
                        )
                        if startup is not None:
                            component_startup_quality = startup
                            if startup["passed"] is not True:
                                raise RuntimeError(
                                    "OpenVLA Component-GS startup quality gate failed"
                                )
                    frames = {
                        "agentview": np.asarray(observation["agentview_image"])[::-1, ::-1].copy(),
                        "wrist": np.asarray(observation["robot0_eye_in_hand_image"])[::-1, ::-1].copy(),
                    }
                    if component_provider is not None:
                        frames["policy_agentview"] = np.asarray(
                            policy_observation["agentview_image"]
                        )[::-1, ::-1].copy()
                    last_frames = frames
                    for name, frame in frames.items():
                        writers[name].append_data(frame)
                        if step == 0 or step % int(spec.get("raw_frame_stride", 100)) == 0:
                            path = frame_dir / f"{name}__frame_{step:06d}.png"
                            Image.fromarray(frame).save(path)
                            manifest.append(
                                {
                                    "kind": "direct_frame",
                                    "episode_index": episode_index,
                                    "stream": name,
                                    "frame_index": step,
                                    "path": str(path.resolve()),
                                    "sha256": _sha256(path),
                                }
                            )
                    if presentation_recorders:
                        from .presentation_capture import (
                            compare_native_resolution_frames,
                            mirror_mujoco_render_state,
                            render_simulator_observer,
                        )

                        mirror = mirror_mujoco_render_state(
                            env.sim,
                            presentation_env.sim,
                        )
                        presentation_state_mirror_checks += 1
                        presentation_state_mirror_max_qpos_delta = max(
                            presentation_state_mirror_max_qpos_delta,
                            float(mirror["maximum_qpos_delta"]),
                        )
                        presentation_state_mirror_max_qvel_delta = max(
                            presentation_state_mirror_max_qvel_delta,
                            float(mirror["maximum_qvel_delta"]),
                        )
                        if mirror["verified"] is not True:
                            raise RuntimeError(
                                "OpenVLA native presentation twin lost state sync"
                            )
                        for name, recorder in presentation_recorders.items():
                            source_frame = render_simulator_observer(
                                env.sim,
                                camera_name=presentation_streams[name],
                                width=int(spec["camera_width"]),
                                height=int(spec["camera_height"]),
                            )
                            target_frame = render_simulator_observer(
                                presentation_env.sim,
                                camera_name=presentation_streams[name],
                                width=int(presentation_config["width"]),
                                height=int(presentation_config["height"]),
                            )
                            resolution_audit = compare_native_resolution_frames(
                                source_frame,
                                target_frame,
                            )
                            trajectory_audit = (
                                presentation_full_trajectory_audits[name]
                            )
                            trajectory_audit["checks"] += 1
                            trajectory_audit["failed_checks"] += int(
                                resolution_audit["passed"] is not True
                            )
                            trajectory_audit[
                                "maximum_mean_absolute_uint8_error"
                            ] = max(
                                float(
                                    trajectory_audit[
                                        "maximum_mean_absolute_uint8_error"
                                    ]
                                ),
                                float(
                                    resolution_audit[
                                        "mean_absolute_uint8_error"
                                    ]
                                ),
                            )
                            trajectory_audit["minimum_rgb_correlation"] = min(
                                float(
                                    trajectory_audit[
                                        "minimum_rgb_correlation"
                                    ]
                                ),
                                float(resolution_audit["rgb_correlation"]),
                            )
                            trajectory_audit["passed"] = bool(
                                trajectory_audit["failed_checks"] == 0
                            )
                            if resolution_audit["passed"] is not True:
                                raise RuntimeError(
                                    "OpenVLA native presentation twin failed "
                                    "full-trajectory source-resolution consistency "
                                    f"for {name} at frame {frame_count}"
                                )
                            recorder.append(target_frame)
                    frame_count += 1
                    if not action_queue:
                        query_started = time.perf_counter()
                        actions = _policy_action(
                            model=model,
                            processor=processor,
                            action_head=action_head,
                            proprio_projector=proprio,
                            config=policy_config,
                            observation=policy_observation,
                            instruction=task.language,
                        )
                        inference_s.append(time.perf_counter() - query_started)
                        action_queries += 1
                        action_queue.extend(
                            actions[: min(int(spec["action_horizon"]), len(actions))]
                        )
                    action = process_openvla_libero_action(action_queue.popleft())
                    observation, _reward, done, _info = env.step(action.tolist())
                    if done:
                        success = True
                        success_step = step + 1
                        break
            finally:
                for writer in writers.values():
                    writer.close()
                for recorder in presentation_recorders.values():
                    recorder.close()
            if last_frames is None:
                raise RuntimeError("Rollout produced no frames")
            for name, frame in last_frames.items():
                path = frame_dir / f"{name}__frame_{frame_count - 1:06d}__final.png"
                Image.fromarray(frame).save(path)
                manifest.append(
                    {
                        "kind": "direct_frame",
                        "episode_index": episode_index,
                        "stream": name,
                        "frame_index": frame_count - 1,
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                    }
                )

            videos = {}
            for name, path in video_paths.items():
                decoded_count, first_frame = _decode_video(path)
                videos[name] = {
                    "absolute_path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                    "decoded_frame_count": decoded_count,
                    "nonblank": bool(float(np.var(first_frame)) > 1.0),
                }
                manifest.append(
                    {
                        "kind": "direct_video",
                        "episode_index": episode_index,
                        "stream": name,
                        "path": str(path.resolve()),
                        "sha256": videos[name]["sha256"],
                    }
                )
            presentation_videos = {}
            if presentation_recorders:
                for name, recorder in presentation_recorders.items():
                    record = recorder.manifest_entry(
                        source="simulator_policy_domain",
                        camera_name=presentation_streams[name],
                    )
                    decoded_count, first_frame = _decode_video(
                        presentation_paths[name]
                    )
                    if decoded_count != frame_count or float(first_frame.var()) <= 1.0:
                        raise RuntimeError(
                            f"Presentation stream failed decode gate: {name}"
                        )
                    record.update(
                        {
                            "absolute_path": str(
                                presentation_paths[name].resolve()
                            ),
                            "decoded_frame_count": decoded_count,
                            "nonblank": True,
                            "policy_input_width": int(spec["camera_width"]),
                            "policy_input_height": int(spec["camera_height"]),
                            "observer_only": True,
                            "fully_decoded": True,
                            "dedicated_native_resolution_rendering_twin": True,
                            "source_resolution_consistency": (
                                presentation_resolution_audits[name]
                            ),
                            "full_trajectory_source_resolution_consistency": (
                                presentation_full_trajectory_audits[name]
                            ),
                        }
                    )
                    presentation_videos[name] = record
                    manifest.append(
                        {
                            "kind": "native_presentation_video",
                            "episode_index": episode_index,
                            "stream": name,
                            **record,
                        }
                    )
            episode_records.append(
                {
                    "episode_index": episode_index,
                    "initial_state_index": state_index,
                    "initial_state_sha256": _array_sha256(initial_states[state_index]),
                    "success": success,
                    "success_step": success_step,
                    "frame_count": frame_count,
                    "action_queries": action_queries,
                    "inference_latency_ms": {
                        "mean": float(np.mean(inference_s) * 1000.0),
                        "p95": float(np.percentile(inference_s, 95) * 1000.0),
                    },
                    "videos": videos,
                    "presentation_videos": presentation_videos,
                    "presentation_rendering_twin": (
                        None
                        if presentation_env is None
                        else {
                            "environment_count": 1,
                            "observation_width": int(
                                presentation_config["width"]
                            ),
                            "observation_height": int(
                                presentation_config["height"]
                            ),
                            "initial_state_mirror": (
                                initial_presentation_state_mirror
                            ),
                            "state_mirror_checks": (
                                presentation_state_mirror_checks
                            ),
                            "maximum_qpos_delta": (
                                presentation_state_mirror_max_qpos_delta
                            ),
                            "maximum_qvel_delta": (
                                presentation_state_mirror_max_qvel_delta
                            ),
                            "source_resolution_consistency": (
                                presentation_resolution_audits
                            ),
                            "full_trajectory_source_resolution_consistency": (
                                presentation_full_trajectory_audits
                            ),
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
                                presentation_state_mirror_max_qpos_delta <= 1e-9
                                and presentation_state_mirror_max_qvel_delta
                                <= 1e-9
                                and all(
                                    audit["checks"] == frame_count + 1
                                    and audit["failed_checks"] == 0
                                    and audit["passed"] is True
                                    for audit in (
                                        presentation_full_trajectory_audits.values()
                                    )
                                )
                                and all(
                                    record["enforced"]
                                    for record in presentation_context_ownership
                                )
                                and all(
                                    audit["passed"]
                                    for audit in presentation_resolution_audits.values()
                                )
                            ),
                        }
                    ),
                }
            )
            (out_dir / "rollout_progress.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "completed_episodes": episode_index + 1,
                        "total_episodes": len(indices),
                        "success_count": sum(item["success"] for item in episode_records),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        env.close()
        if canonical_env is not None:
            canonical_env.close()
        if presentation_env is not None:
            presentation_env.close()

    with (out_dir / "media_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    collage = _make_rollout_collage(out_dir, episode_records[0])
    success_flags = [bool(item["success"]) for item in episode_records]
    video_gate = all(
        video["decoded_frame_count"] == episode["frame_count"]
        and video["nonblank"]
        and video["bytes"] > 0
        for episode in episode_records
        for video in episode["videos"].values()
    )
    presentation_gate = bool(
        presentation_config is None
        or all(
            video["decoded_frame_count"] == episode["frame_count"]
            and video["nonblank"]
            and video["used_for_statistical_evidence"] is False
            and video["observer_only"] is True
            and video["width"] >= 720
            and video["height"] >= 720
            and episode["presentation_rendering_twin"]["verified"] is True
            for episode in episode_records
            for video in episode["presentation_videos"].values()
        )
    )
    expected_adapter = bool(spec.get("expected_adapter", spec.get("adapter_update") is not None))
    adapter_gate = (adapter_audit is not None) if expected_adapter else adapter_audit is None
    component_provider_report = (
        component_provider.report() if component_provider is not None else None
    )
    component_provider_gate = bool(
        provider_config is None
        or (
            component_startup_quality is not None
            and component_startup_quality.get("passed") is True
            and component_provider_report is not None
            and int(component_provider_report.get("render_calls", 0)) > 0
            and component_state_mirror_max_qpos_delta
            <= float(provider_config["state_sync_tolerance"])
            and component_state_mirror_max_qvel_delta
            <= float(provider_config["state_sync_tolerance"])
        )
    )
    report = {
        "schema": "fedsplat.openvla_oft_closed_loop.v1",
        "evidence_label": "sim_rollout_success" if any(success_flags) else "closed_loop_execution",
        "checkpoint": str(Path(spec["checkpoint_path"]).resolve()),
        "adapter_update": spec.get("adapter_update"),
        "adapter_audit": adapter_audit,
        "visual_domain_shift": visual_shift_report,
        "policy_observation_provider": (
            None
            if provider_config is None
            else {
                "mode": provider_config["mode"],
                "evidence_role": provider_config["evidence_role"],
                "pose_oracle_development_only": True,
                "three_d_gaussian_renderer_used": True,
                "startup_quality": component_startup_quality,
                "state_mirror_checks": component_state_mirror_checks,
                "maximum_qpos_delta": component_state_mirror_max_qpos_delta,
                "maximum_qvel_delta": component_state_mirror_max_qvel_delta,
                "provider_report": component_provider_report,
                "gate_pass": component_provider_gate,
            }
        ),
        "suite": suite_name,
        "task_id": task_id,
        "task_description": task.language,
        "bddl_path": str(bddl_path.resolve()),
        "bddl_sha256": _sha256(bddl_path),
        "seed": seed,
        "initial_state_indices": indices,
        "num_episodes": len(indices),
        "success_count": sum(success_flags),
        "success_rate": float(sum(success_flags) / len(success_flags)),
        "success_flags": [int(value) for value in success_flags],
        "episodes": episode_records,
        "checkpoint_load_seconds": load_seconds,
        "wall_clock_s": time.perf_counter() - started,
        "media_manifest": str((out_dir / "media_manifest.jsonl").resolve()),
        "paper_collage": collage,
        "rollout_completed": len(episode_records) == len(indices),
        "video_gate_pass": video_gate,
        "presentation_capture": (
            None
            if presentation_config is None
            else {
                "enabled": True,
                "source": presentation_config["source"],
                "width": int(presentation_config["width"]),
                "height": int(presentation_config["height"]),
                "fps": int(presentation_config["fps"]),
                "crf": int(presentation_config["crf"]),
                "native_render_only": True,
                "dedicated_native_resolution_rendering_twin": True,
                "super_resolution_used": False,
                "frame_interpolation_used": False,
                "used_for_statistical_evidence": False,
                "policy_input_unchanged": True,
                "policy_input_width": int(spec["camera_width"]),
                "policy_input_height": int(spec["camera_height"]),
                "streams": list(presentation_streams),
                "gate_pass": presentation_gate,
            }
        ),
        "presentation_gate_pass": presentation_gate,
        "adapter_gate_pass": adapter_gate,
        "component_provider_gate_pass": component_provider_gate,
        "overall_gate_pass": bool(
            len(episode_records) == len(indices)
            and video_gate
            and presentation_gate
            and adapter_gate
            and component_provider_gate
        ),
        "determinism": {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32": False,
        },
        "evidence_scope": "real OpenVLA-OFT checkpoint in simulator closed loop",
        "not_evidence_for": [
            "statistical method superiority",
            "real-robot success",
            "presentation-rerollout success-rate evidence",
        ],
    }
    (out_dir / "rollout_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "rollout_progress.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_episodes": len(indices),
                "success_count": report["success_count"],
                "overall_gate_pass": report["overall_gate_pass"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.worker_config.read_text(encoding="utf-8"))
    errors = validate_openvla_rollout_spec(spec)
    if errors:
        raise ValueError("Invalid OpenVLA-OFT rollout worker config: " + "; ".join(errors))
    report = _worker(spec)
    print(
        json.dumps({"overall_gate_pass": report["overall_gate_pass"]}),
        flush=True,
    )
    if os.environ.get("FEDSPLAT_OPENVLA_WORKER_DIRECT_EXIT") == "1":
        os._exit(0)


if __name__ == "__main__":
    main()
