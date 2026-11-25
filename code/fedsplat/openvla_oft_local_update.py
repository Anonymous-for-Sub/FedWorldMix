"""External-runtime OpenVLA-OFT local training with FedSplat LoRA uploads."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_TARGET_MODULES = ("projector.fc1", "projector.fc2", "projector.fc3")
GRADIENT_MODES = {"real_only", "naive", "projected", "augmentation_portfolio"}
SYNTHETIC_OBJECTIVES = {"action_bc", "source_paired_consistency"}


def local_update_config_fingerprint(spec: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(spec))
    canonical.pop("out_dir", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def gradient_combination_diagnostics(result: Any) -> dict[str, Any] | None:
    """Serialize projection evidence without embedding million-value gradients."""
    if result is None:
        return None
    projection = result.projection
    projection_record = None
    if projection is not None:
        projection_record = {
            "alpha": projection.alpha,
            "real_norm": projection.real_norm,
            "synthetic_norm": projection.synthetic_norm,
            "calibrated_norm": projection.calibrated_norm,
            "real_norm_sq": projection.real_norm_sq,
            "dot_before": projection.dot_before,
            "dot_after": projection.dot_after,
            "cosine_before": projection.cosine_before,
            "cosine_after": projection.cosine_after,
            "branch": projection.branch,
            "non_conflict_guaranteed": projection.non_conflict_guaranteed,
        }
    return {
        "mode": result.mode,
        "gradient_dimension": len(result.combined_gradient),
        "synthetic_weight": result.synthetic_weight,
        "quality_score": result.quality_score,
        "effective_synthetic_weight": result.effective_synthetic_weight,
        "auxiliary_trust_scale": result.auxiliary_trust_scale,
        "weighted_synthetic_norm": result.weighted_synthetic_norm,
        "projection_applied": result.projection_applied,
        "projection": projection_record,
    }


def portfolio_gradient_diagnostics(result: Any) -> dict[str, Any] | None:
    """Serialize the 2D/GS combination without retaining raw gradient vectors."""
    if result is None:
        return None
    return {
        "gs_ratio": result.gs_ratio,
        "two_d_ratio": result.two_d_ratio,
        "base_combination": gradient_combination_diagnostics(
            result.base_combination
        ),
        "gs_combination": gradient_combination_diagnostics(
            result.gs_combination
        ),
    }


def apply_visual_proxy_rgb(value: Any, source: dict[str, Any]) -> Any:
    """Apply the frozen deterministic 2D proxy in normalized RGB space."""
    import numpy as np

    image = np.asarray(value, dtype=np.uint8)
    transformed = image.astype(np.float32) / 255.0
    transformed = (
        transformed * float(source.get("contrast", 1.0))
        + float(source.get("brightness", 0.0))
    )
    shift_x = int(source.get("horizontal_shift_px", 0))
    shift_y = int(source.get("vertical_shift_px", 0))
    if shift_x or shift_y:
        transformed = np.roll(
            transformed,
            shift=(shift_y, shift_x),
            axis=(0, 1),
        )
    return np.rint(np.clip(transformed, 0.0, 1.0) * 255.0).astype(np.uint8)


def normalize_openvla_actions(
    actions: Any, statistics: dict[str, Any]
) -> Any:
    """Map LIBERO environment actions to the released OFT normalized target space."""
    import numpy as np

    values = np.asarray(actions, dtype=np.float32).copy()
    if values.shape[-1] != 7:
        raise ValueError(f"OpenVLA-OFT actions must end in dimension 7, got {values.shape}")
    values[..., -1] = (1.0 - values[..., -1]) / 2.0
    low = np.asarray(statistics["q01"], dtype=np.float32)
    high = np.asarray(statistics["q99"], dtype=np.float32)
    mask = np.asarray(statistics.get("mask", np.ones(7, dtype=bool)), dtype=bool)
    normalized = np.where(
        mask,
        2.0 * (values - low) / (high - low + 1e-8) - 1.0,
        values,
    )
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def resolve_target_module_names(
    available_names: list[str],
    exact_targets: list[str] | None = None,
    target_patterns: list[str] | None = None,
) -> list[str]:
    exact_targets = list(exact_targets or [])
    target_patterns = list(target_patterns or [])
    missing = sorted(set(exact_targets) - set(available_names))
    if missing:
        raise ValueError(f"OpenVLA-OFT LoRA targets were not found: {missing}")
    selected = set(exact_targets)
    unmatched = []
    for pattern in target_patterns:
        matches = [
            name for name in available_names if fnmatch.fnmatchcase(name, pattern)
        ]
        if not matches:
            unmatched.append(pattern)
        selected.update(matches)
    if unmatched:
        raise ValueError(f"OpenVLA-OFT target patterns matched nothing: {unmatched}")
    resolved = [name for name in available_names if name in selected]
    if not resolved:
        raise ValueError("No OpenVLA-OFT LoRA target modules were resolved")
    return resolved


def validate_local_update_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "checkpoint_path",
        "openvla_root",
        "dataset_path",
        "repo_id",
        "client_id",
        "instruction",
    ):
        if not isinstance(spec.get(key), str) or not spec[key].strip():
            errors.append(f"{key} must be a nonempty string")
    for key in ("rank", "steps"):
        if not isinstance(spec.get(key), int) or int(spec.get(key, 0)) <= 0:
            errors.append(f"{key} must be a positive integer")
    for key in ("alpha", "learning_rate"):
        value = spec.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            errors.append(f"{key} must be finite and positive")
    for key in ("train_episodes", "validation_episodes"):
        values = spec.get(key)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, int) or value < 0 for value in values)
        ):
            errors.append(f"{key} must contain nonnegative integers")
    if set(spec.get("train_episodes", [])) & set(
        spec.get("validation_episodes", [])
    ):
        errors.append("training and validation episodes must be disjoint")
    for key in ("train_samples_per_episode", "validation_samples_per_episode"):
        value = spec.get(key, 1)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"{key} must be a positive integer")
    gradient_mode = spec.get("gradient_mode", "real_only")
    if gradient_mode not in GRADIENT_MODES:
        errors.append(f"gradient_mode must be one of {sorted(GRADIENT_MODES)}")
    objective = spec.get("synthetic_objective", "action_bc")
    if objective not in SYNTHETIC_OBJECTIVES:
        errors.append(
            f"synthetic_objective must be one of {sorted(SYNTHETIC_OBJECTIVES)}"
        )
    source = spec.get("synthetic_source", {"type": "none"})
    if not isinstance(source, dict) or source.get("type", "none") not in {
        "none",
        "lerobot_dataset",
    }:
        errors.append("synthetic_source.type must be none or lerobot_dataset")
    elif gradient_mode != "real_only":
        for key in ("dataset_path", "quality_manifest"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                errors.append(f"synthetic_source.{key} is required")
    if gradient_mode == "real_only" and source.get("type", "none") != "none":
        errors.append("real_only requires synthetic_source.type=none")
    if spec.get("update_contract", "combined") != "combined":
        errors.append("OpenVLA-OFT currently requires update_contract=combined")
    targets = spec.get("target_modules", list(DEFAULT_TARGET_MODULES))
    patterns = spec.get("target_module_patterns", [])
    if not isinstance(targets, list) or any(
        not isinstance(item, str) or not item for item in targets
    ):
        errors.append("target_modules must be a string list")
    if not isinstance(patterns, list) or any(
        not isinstance(item, str) or not item for item in patterns
    ):
        errors.append("target_module_patterns must be a string list")
    if not targets and not patterns:
        errors.append("at least one target module or pattern is required")
    portfolio = spec.get("augmentation_portfolio")
    if gradient_mode == "augmentation_portfolio":
        if not isinstance(portfolio, dict):
            errors.append(
                "augmentation_portfolio gradient mode requires an "
                "augmentation_portfolio object"
            )
        else:
            if portfolio.get("schema") != "fedsplat.augmentation_portfolio.v1":
                errors.append("augmentation_portfolio has an unexpected schema")
            try:
                from .adaptive_augmentation import validate_portfolio_ratio

                validate_portfolio_ratio(portfolio.get("gs_ratio"))
            except (TypeError, ValueError):
                errors.append(
                    "augmentation_portfolio.gs_ratio must be one of "
                    "[0.0, 0.25, 0.5]"
                )
            if portfolio.get("equal_auxiliary_budget") is not True:
                errors.append(
                    "augmentation_portfolio.equal_auxiliary_budget must be true"
                )
            two_d_source = portfolio.get("two_d_source")
            if not isinstance(two_d_source, dict):
                errors.append(
                    "augmentation_portfolio.two_d_source must be an object"
                )
            else:
                if two_d_source.get("type") != "visual_proxy":
                    errors.append(
                        "augmentation_portfolio.two_d_source.type must be "
                        "visual_proxy"
                    )
                if (
                    two_d_source.get("provenance")
                    != "proxy_visual_augmentation"
                ):
                    errors.append(
                        "augmentation_portfolio.two_d_source.provenance must "
                        "be proxy_visual_augmentation"
                    )
                for key in ("contrast", "brightness"):
                    value = two_d_source.get(
                        key, 1.0 if key == "contrast" else 0.0
                    )
                    if (
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        errors.append(
                            f"augmentation_portfolio.two_d_source.{key} must "
                            "be finite"
                        )
                contrast = two_d_source.get("contrast", 1.0)
                if isinstance(contrast, (int, float)) and float(contrast) <= 0:
                    errors.append(
                        "augmentation_portfolio.two_d_source.contrast must be "
                        "positive"
                    )
                for key in ("horizontal_shift_px", "vertical_shift_px"):
                    if not isinstance(two_d_source.get(key, 0), int):
                        errors.append(
                            f"augmentation_portfolio.two_d_source.{key} must "
                            "be an integer"
                        )
        if objective != "action_bc":
            errors.append(
                "augmentation_portfolio requires synthetic_objective=action_bc"
            )
        if source.get("type") != "lerobot_dataset":
            errors.append(
                "augmentation_portfolio requires "
                "synthetic_source.type=lerobot_dataset"
            )
        norm_cap = spec.get("max_auxiliary_to_real_norm")
        if (
            not isinstance(norm_cap, (int, float))
            or not math.isfinite(float(norm_cap))
            or float(norm_cap) <= 0.0
        ):
            errors.append(
                "augmentation_portfolio requires finite positive "
                "max_auxiliary_to_real_norm"
            )
        portfolio_targets = [*targets, *patterns]
        if any(
            not (
                item.startswith("projector.")
                or item.startswith("vision_backbone.")
            )
            for item in portfolio_targets
        ):
            errors.append(
                "augmentation_portfolio OpenVLA targets must be perception-only "
                "(projector.* or vision_backbone.*)"
            )
    elif portfolio is not None:
        errors.append(
            "augmentation_portfolio requires "
            "gradient_mode=augmentation_portfolio"
        )
    initial = spec.get("initial_adapter_update")
    if initial is not None and (not isinstance(initial, str) or not initial.strip()):
        errors.append("initial_adapter_update must be null or a nonempty string")
    threshold = spec.get("projection_cosine_threshold", 0.0)
    if (
        not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not -1.0 <= float(threshold) <= 0.0
    ):
        errors.append("projection_cosine_threshold must be in [-1, 0]")
    return errors


def run_openvla_oft_local_update(
    config_path: Path,
    out_dir: Path,
    runtime_python: Path,
    pythonpath: list[Path] | None = None,
    cuda_visible_device: str | None = None,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = _discover_project_root(config_path)
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    errors = validate_local_update_spec(spec)
    if errors:
        raise ValueError("Invalid OpenVLA-OFT local update: " + "; ".join(errors))

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_spec = json.loads(json.dumps(spec))
    for key in ("checkpoint_path", "openvla_root", "dataset_path"):
        worker_spec[key] = str(_project_path(project_root, worker_spec[key]))
    if worker_spec.get("initial_adapter_update"):
        worker_spec["initial_adapter_update"] = str(
            _project_path(project_root, worker_spec["initial_adapter_update"])
        )
    source = dict(worker_spec.get("synthetic_source", {"type": "none"}))
    for key in ("dataset_path", "quality_manifest"):
        if source.get(key):
            source[key] = str(_project_path(project_root, source[key]))
    worker_spec["synthetic_source"] = source
    worker_spec["out_dir"] = str(out_dir)
    worker_config = out_dir / "worker_config.json"
    worker_config.write_text(json.dumps(worker_spec, indent=2) + "\n", encoding="utf-8")

    runtime_python = runtime_python.absolute()
    if not runtime_python.exists():
        raise FileNotFoundError(f"Runtime Python does not exist: {runtime_python}")
    env = os.environ.copy()
    paths = [project_root, *(pythonpath or []), Path(worker_spec["openvla_root"])]
    prefix = os.pathsep.join(str(path.resolve()) for path in paths)
    env["PYTHONPATH"] = prefix + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cuda_visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)
    started = time.perf_counter()
    completed = subprocess.run(
        [
            str(runtime_python),
            "-m",
            "fedsplat.openvla_oft_local_update",
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
    report_path = out_dir / "local_update_report.json"
    if completed.returncode != 0 or not report_path.is_file():
        detail = (completed.stderr or completed.stdout or "worker produced no report")[-6000:]
        raise RuntimeError(
            f"OpenVLA-OFT local-update worker failed ({completed.returncode}): {detail}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_wall_s"] = time.perf_counter() - started
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _discover_project_root(config_path: Path) -> Path:
    """Resolve the source root for configs copied into nested run directories."""
    resolved = config_path.resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "fedsplat").is_dir():
            return candidate
    return resolved.parent.parent


def _decode_video(path: Path) -> list[Any]:
    import imageio_ffmpeg
    import numpy as np

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        return [
            np.frombuffer(payload, dtype=np.uint8)
            .reshape(height, width, 3)
            .copy()
            for payload in reader
        ]
    finally:
        reader.close()


def _load_dataset(root: Path) -> dict[str, Any]:
    import pyarrow.parquet as parquet

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "data").rglob("*.parquet")):
        rows.extend(parquet.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["index"]))
    if not rows:
        raise FileNotFoundError(f"No LeRobot rows under {root}")
    streams = {}
    for key in ("observation.images.image", "observation.images.wrist_image"):
        videos = sorted((root / "videos" / key).rglob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No videos for {key} under {root}")
        streams[key] = [frame for path in videos for frame in _decode_video(path)]
        if len(streams[key]) != len(rows):
            raise ValueError(
                f"Video/data length mismatch for {key}: {len(streams[key])} != {len(rows)}"
            )
    by_episode: dict[int, list[int]] = {}
    by_episode_frame: dict[tuple[int, int], int] = {}
    for position, row in enumerate(rows):
        episode = int(row["episode_index"])
        frame = int(row["frame_index"])
        by_episode.setdefault(episode, []).append(position)
        by_episode_frame[(episode, frame)] = position
    return {
        "root": root,
        "rows": rows,
        "streams": streams,
        "by_episode": by_episode,
        "by_episode_frame": by_episode_frame,
    }


def _sample_positions(
    dataset: dict[str, Any], episodes: list[int], count: int, seed: int
) -> list[int]:
    import numpy as np

    rng = np.random.default_rng(seed)
    sampled = []
    for episode in episodes:
        candidates = dataset["by_episode"].get(int(episode))
        if not candidates:
            raise ValueError(f"Dataset lacks requested episode {episode}")
        replace = len(candidates) < count
        sampled.extend(
            int(value)
            for value in rng.choice(candidates, size=count, replace=replace)
        )
    return sampled


def source_pair_episode_map(
    manifest: dict[str, Any],
    real_episode_ids: set[int],
) -> tuple[dict[int, int], str]:
    """Map synthetic episodes to their paired real episodes."""
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Synthetic quality manifest lacks episode records")
    records: dict[int, dict[str, Any]] = {}
    for item in episodes:
        if not isinstance(item, dict) or "episode_index" not in item:
            raise ValueError("Synthetic quality episode lacks episode_index")
        episode = int(item["episode_index"])
        if episode in records:
            raise ValueError(f"Duplicate synthetic episode {episode}")
        records[episode] = item

    aggregate = manifest.get("aggregate", {})
    exact_pair_contract = (
        set(records) == real_episode_ids
        and float(aggregate.get("state_pair_max_abs_error", float("inf")))
        <= 1e-6
        and float(aggregate.get("action_pair_max_abs_error", float("inf")))
        <= 1e-6
    )
    if exact_pair_contract:
        return {episode: episode for episode in records}, "aligned_episode_index"

    demos = list(manifest.get("train_demos", [])) + list(
        manifest.get("calibration_demos", [])
    )
    if (
        len(demos) != len(real_episode_ids)
        or len(demos) != len(set(demos))
        or real_episode_ids != set(range(len(demos)))
    ):
        raise ValueError("Legacy source-pair demo order does not match real episodes")
    real_episode_by_demo = {str(demo): index for index, demo in enumerate(demos)}
    mapping = {}
    for episode, item in records.items():
        source_demo = item.get("source_demo")
        if source_demo not in real_episode_by_demo:
            raise ValueError(
                f"Synthetic episode {episode} references unknown source demo "
                f"{source_demo}"
            )
        mapping[episode] = real_episode_by_demo[str(source_demo)]
    return mapping, "declared_demo_order"


def _action_chunk(dataset: dict[str, Any], position: int, horizon: int = 8) -> Any:
    import numpy as np

    row = dataset["rows"][position]
    episode_positions = dataset["by_episode"][int(row["episode_index"])]
    local_index = episode_positions.index(position)
    positions = [
        episode_positions[min(local_index + offset, len(episode_positions) - 1)]
        for offset in range(horizon)
    ]
    return np.asarray(
        [dataset["rows"][index]["action"] for index in positions],
        dtype=np.float32,
    )


def _register(registry: Any, *args: Any) -> None:
    try:
        registry.register(*args)
    except ValueError as exc:
        if "already used" not in str(exc) and "already registered" not in str(exc):
            raise


def _load_component(checkpoint: Path, pattern: str) -> dict[str, Any]:
    import torch

    paths = sorted(checkpoint.glob(f"*{pattern}*checkpoint*.pt"))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {pattern} checkpoint, found {paths}")
    state = torch.load(paths[0], map_location="cpu", weights_only=True)
    return {key.removeprefix("module."): value for key, value in state.items()}


def _load_policy(spec: dict[str, Any]):
    import sys
    import types
    from types import SimpleNamespace

    import torch
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    root = Path(spec["openvla_root"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

    if "prismatic.training" not in sys.modules:
        package = types.ModuleType("prismatic.training")
        package.__path__ = [str(root / "prismatic" / "training")]
        sys.modules["prismatic.training"] = package
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector

    _register(AutoConfig, "openvla", OpenVLAConfig)
    _register(AutoImageProcessor, OpenVLAConfig, PrismaticImageProcessor)
    _register(AutoProcessor, OpenVLAConfig, PrismaticProcessor)
    _register(AutoModelForVision2Seq, OpenVLAConfig, OpenVLAForActionPrediction)

    checkpoint = Path(spec["checkpoint_path"])
    device = str(spec.get("device", "cuda:0"))
    started = time.perf_counter()
    model = AutoModelForVision2Seq.from_pretrained(
        str(checkpoint),
        attn_implementation=str(spec.get("attention_implementation", "sdpa")),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    model.vision_backbone.set_num_images_in_input(2)
    model = model.to(device).eval()
    model.config.use_cache = False
    with (checkpoint / "dataset_statistics.json").open(encoding="utf-8") as handle:
        model.norm_stats = json.load(handle)
    task_suite = str(spec.get("task_suite_name", "libero_10"))
    unnorm_key = (
        task_suite
        if task_suite in model.norm_stats
        else f"{task_suite}_no_noops"
    )
    if unnorm_key not in model.norm_stats:
        raise KeyError(f"Checkpoint lacks normalization statistics for {task_suite}")
    processor = AutoProcessor.from_pretrained(
        str(checkpoint), trust_remote_code=False, local_files_only=True
    )
    action_head = L1RegressionActionHead(
        input_dim=model.llm_dim, hidden_dim=model.llm_dim, action_dim=7
    )
    action_head.load_state_dict(_load_component(checkpoint, "action_head"), strict=True)
    action_head = action_head.to(device=device, dtype=torch.bfloat16).eval()
    proprio = ProprioProjector(llm_dim=model.llm_dim, proprio_dim=8)
    proprio.load_state_dict(_load_component(checkpoint, "proprio_projector"), strict=True)
    proprio = proprio.to(device=device, dtype=torch.bfloat16).eval()
    config = SimpleNamespace(
        device=device,
        unnorm_key=unnorm_key,
        center_crop=bool(spec.get("center_crop", True)),
    )
    return model, processor, action_head, proprio, config, time.perf_counter() - started


def _prepare_image(value: Any, center_crop: bool):
    import numpy as np
    from PIL import Image

    image = Image.fromarray(np.asarray(value, dtype=np.uint8)).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    if center_crop:
        side = int(round((0.9**0.5) * 224))
        offset = (224 - side) // 2
        image = image.crop((offset, offset, offset + side, offset + side))
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
    return image


def _normalize_proprio(state: Any, statistics: dict[str, Any]) -> Any:
    import numpy as np

    value = np.asarray(state, dtype=np.float32)
    low = np.asarray(statistics["q01"], dtype=np.float32)
    high = np.asarray(statistics["q99"], dtype=np.float32)
    return np.clip(2.0 * (value - low) / (high - low + 1e-8) - 1.0, -1.0, 1.0)


def _model_inputs(
    dataset: dict[str, Any],
    position: int,
    instruction: str,
    processor: Any,
    config: Any,
    statistics: dict[str, Any],
    visual_proxy_source: dict[str, Any] | None = None,
):
    import torch

    full = dataset["streams"]["observation.images.image"][position][::-1, ::-1]
    wrist = dataset["streams"]["observation.images.wrist_image"][position][::-1, ::-1]
    if visual_proxy_source is not None:
        full = apply_visual_proxy_rgb(full, visual_proxy_source)
        wrist = apply_visual_proxy_rgb(wrist, visual_proxy_source)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    primary = processor(prompt, _prepare_image(full, config.center_crop)).to(
        config.device, dtype=torch.bfloat16
    )
    secondary = processor(prompt, _prepare_image(wrist, config.center_crop)).to(
        config.device, dtype=torch.bfloat16
    )
    primary["pixel_values"] = torch.cat(
        [primary["pixel_values"], secondary["pixel_values"]], dim=1
    )
    state = dataset["rows"][position]["observation.state"]
    return primary, _normalize_proprio(state, statistics)


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional

    from .lora_updates import (
        ClientAdapterUpdate,
        LoRALayerUpdate,
        client_update_artifact_fingerprint,
        load_client_update,
        low_rank_dense_difference_norm,
        low_rank_dense_frobenius_norm,
        save_client_update,
    )
    from .adaptive_augmentation import (
        combine_augmentation_portfolio_gradients,
        validate_portfolio_ratio,
    )
    from .method import combine_real_synthetic_gradients, heldout_real_reliability

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
            super().__init__()
            self.base = base
            self.scaling = float(alpha) / float(rank)
            self.a = nn.Parameter(
                torch.empty(rank, base.in_features, device=base.weight.device)
            )
            self.b = nn.Parameter(
                torch.zeros(base.out_features, rank, device=base.weight.device)
            )
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5.0))

        def forward(self, inputs):
            base = self.base(inputs)
            update = functional.linear(functional.linear(inputs.float(), self.a), self.b)
            return base + update.to(base.dtype) * self.scaling

    seed = int(spec.get("seed", 7))
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir = Path(spec["out_dir"])
    started = time.perf_counter()
    real = _load_dataset(Path(spec["dataset_path"]))
    source = dict(spec.get("synthetic_source", {"type": "none"}))
    synthetic = (
        _load_dataset(Path(source["dataset_path"]))
        if source.get("type", "none") == "lerobot_dataset"
        else None
    )
    model, processor, action_head, proprio, cfg, load_seconds = _load_policy(spec)
    statistics = model.norm_stats[cfg.unnorm_key]

    for module in (model, action_head, proprio):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    available = [
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    targets = resolve_target_module_names(
        available,
        spec.get("target_modules", list(DEFAULT_TARGET_MODULES)),
        spec.get("target_module_patterns", []),
    )
    adapters: dict[str, LoRALinear] = {}
    for name in targets:
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        wrapper = LoRALinear(
            getattr(parent, attribute), int(spec["rank"]), float(spec["alpha"])
        )
        setattr(parent, attribute, wrapper)
        adapters[name] = wrapper

    initial_path = spec.get("initial_adapter_update")
    initial_by_name: dict[str, LoRALayerUpdate] = {}
    initial_adapter_audit = None
    if initial_path:
        initial = load_client_update(Path(initial_path))
        initial_by_name = {layer.layer_name: layer for layer in initial.layers}
        if set(initial_by_name) != set(adapters):
            raise ValueError("Global adapter layer names do not match OpenVLA targets")
        initial_adapter_audit = client_update_artifact_fingerprint(Path(initial_path))
        initial_adapter_audit.update(
            {
                "source_client_id": initial.client_id,
                "source_layer_names": sorted(initial_by_name),
            }
        )
        with torch.no_grad():
            for name, adapter in adapters.items():
                layer = initial_by_name[name]
                if layer.a.shape != tuple(adapter.a.shape) or layer.b.shape != tuple(
                    adapter.b.shape
                ):
                    raise ValueError(f"Global adapter shape mismatch for {name}")
                adapter.a.copy_(torch.as_tensor(layer.a, device=adapter.a.device))
                adapter.b.copy_(
                    torch.as_tensor(layer.b / adapter.scaling, device=adapter.b.device)
                )

    parameters = [
        parameter
        for adapter in adapters.values()
        for parameter in (adapter.a, adapter.b)
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(spec["learning_rate"]),
        weight_decay=float(spec.get("weight_decay", 0.0)),
    )
    train_positions = _sample_positions(
        real,
        list(spec["train_episodes"]),
        int(spec.get("train_samples_per_episode", 1)),
        seed + 101,
    )
    validation_positions = _sample_positions(
        real,
        list(spec["validation_episodes"]),
        int(spec.get("validation_samples_per_episode", 1)),
        seed + 202,
    )

    source_pair_map: dict[int, int] = {}
    quality_score = 1.0
    synthetic_positions: list[int] = []
    pairing_audit = None
    if synthetic is not None:
        manifest = json.loads(Path(source["quality_manifest"]).read_text())
        episode_records = {int(item["episode_index"]): item for item in manifest["episodes"]}
        selected_episodes = list(
            source.get(
                "episode_ids",
                [
                    episode
                    for episode, item in episode_records.items()
                    if item.get("source_split") == "train"
                ],
            )
        )
        synthetic_positions = _sample_positions(
            synthetic,
            selected_episodes,
            int(source.get("samples_per_episode", 1)),
            seed + 303,
        )
        real_episode_by_synthetic, pair_mapping_mode = source_pair_episode_map(
            manifest,
            set(real["by_episode"]),
        )
        state_error = 0.0
        action_error = 0.0
        for position in synthetic_positions:
            row = synthetic["rows"][position]
            evidence = episode_records[int(row["episode_index"])]
            real_episode = real_episode_by_synthetic[int(row["episode_index"])]
            real_position = real["by_episode_frame"][
                (real_episode, int(row["frame_index"]))
            ]
            source_pair_map[position] = real_position
            state_error = max(
                state_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(row["observation.state"])
                            - np.asarray(
                                real["rows"][real_position]["observation.state"]
                            )
                        )
                    )
                ),
            )
            action_error = max(
                action_error,
                float(
                    np.max(
                        np.abs(
                            np.asarray(row["action"])
                            - np.asarray(real["rows"][real_position]["action"])
                        )
                    )
                ),
            )
        if state_error > 1e-6 or action_error > 1e-6:
            raise ValueError("Synthetic source pairs changed state or action labels")
        quality_score = float(
            np.mean(
                [
                    episode_records[
                        int(synthetic["rows"][position]["episode_index"])
                    ]["quality_score"]
                    for position in synthetic_positions
                ]
            )
        )
        pairing_audit = {
            "schema": "fedsplat.openvla_source_pair_audit.v1",
            "verified": True,
            "sample_pair_count": len(source_pair_map),
            "state_max_abs_error": state_error,
            "action_max_abs_error": action_error,
            "mean_quality_score": quality_score,
            "episode_mapping_mode": pair_mapping_mode,
        }
        (out_dir / "source_pairing_audit.json").write_text(
            json.dumps(pairing_audit, indent=2) + "\n", encoding="utf-8"
        )

    instruction = str(spec["instruction"])
    mode = str(spec.get("gradient_mode", "real_only"))
    portfolio_config = dict(spec.get("augmentation_portfolio", {}))
    portfolio_ratio = (
        validate_portfolio_ratio(portfolio_config["gs_ratio"])
        if mode == "augmentation_portfolio"
        else None
    )
    portfolio_two_d_source = (
        dict(portfolio_config["two_d_source"])
        if mode == "augmentation_portfolio"
        else None
    )

    def prediction(
        dataset: dict[str, Any],
        position: int,
        visual_proxy_source: dict[str, Any] | None = None,
    ):
        inputs, state = _model_inputs(
            dataset,
            position,
            instruction,
            processor,
            cfg,
            statistics["proprio"],
            visual_proxy_source,
        )
        _, hidden = model.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=state,
            proprio_projector=proprio,
            noisy_action_projector=None,
            action_head=action_head,
            use_film=False,
        )
        return action_head.predict_action(hidden).float()

    def target(dataset: dict[str, Any], position: int):
        normalized = normalize_openvla_actions(
            _action_chunk(dataset, position), statistics["action"]
        )
        return torch.as_tensor(normalized, device=cfg.device).unsqueeze(0)

    def bc_loss(
        dataset: dict[str, Any],
        position: int,
        visual_proxy_source: dict[str, Any] | None = None,
    ):
        return functional.l1_loss(
            prediction(dataset, position, visual_proxy_source),
            target(dataset, position),
        )

    def mean_loss(
        dataset: dict[str, Any],
        positions: list[int],
        visual_proxy_source: dict[str, Any] | None = None,
    ) -> float:
        with torch.no_grad():
            return float(
                np.mean(
                    [
                        float(bc_loss(dataset, position, visual_proxy_source))
                        for position in positions
                    ]
                )
            )

    train_loss_before = mean_loss(real, train_positions)
    validation_loss_before = mean_loss(real, validation_positions)
    two_d_loss_before = (
        mean_loss(real, train_positions, portfolio_two_d_source)
        if portfolio_two_d_source is not None
        else None
    )
    step_records = []
    objective = str(spec.get("synthetic_objective", "action_bc"))
    for step in range(int(spec["steps"])):
        optimizer.zero_grad(set_to_none=True)
        real_position = train_positions[step % len(train_positions)]
        real_loss = bc_loss(real, real_position)
        real_gradients = torch.autograd.grad(real_loss, parameters)
        synthetic_loss = None
        combination = None
        portfolio_combination = None
        two_d_loss = None
        if mode == "real_only":
            combined = real_gradients
        elif mode == "augmentation_portfolio":
            synthetic_position = synthetic_positions[step % len(synthetic_positions)]
            two_d_loss = bc_loss(
                real,
                real_position,
                portfolio_two_d_source,
            )
            two_d_gradients = torch.autograd.grad(two_d_loss, parameters)
            synthetic_loss = bc_loss(synthetic, synthetic_position)
            synthetic_gradients = torch.autograd.grad(synthetic_loss, parameters)

            def flattened(gradients):
                return np.concatenate(
                    [
                        gradient.detach().float().cpu().numpy().ravel()
                        for gradient in gradients
                    ]
                )

            portfolio_combination = combine_augmentation_portfolio_gradients(
                flattened(real_gradients),
                flattened(two_d_gradients),
                flattened(synthetic_gradients),
                gs_ratio=portfolio_ratio,
                gs_quality_score=quality_score,
                max_gs_to_base_norm=float(
                    spec["max_auxiliary_to_real_norm"]
                ),
                projection_cosine_threshold=float(
                    spec.get("projection_cosine_threshold", 0.0)
                ),
            )
            vector = np.asarray(
                portfolio_combination.combined_gradient,
                dtype=np.float32,
            )
            combined = []
            offset = 0
            for parameter in parameters:
                size = parameter.numel()
                combined.append(
                    torch.as_tensor(
                        vector[offset : offset + size],
                        device=parameter.device,
                    ).reshape_as(parameter)
                )
                offset += size
        else:
            synthetic_position = synthetic_positions[step % len(synthetic_positions)]
            synthetic_prediction = prediction(synthetic, synthetic_position)
            synthetic_loss = functional.l1_loss(
                synthetic_prediction, target(synthetic, synthetic_position)
            )
            if objective == "source_paired_consistency":
                with torch.no_grad():
                    reference = prediction(
                        real, source_pair_map[synthetic_position]
                    ).detach()
                synthetic_loss = synthetic_loss + float(
                    spec.get("paired_action_consistency_weight", 1.0)
                ) * functional.l1_loss(synthetic_prediction, reference)
            synthetic_gradients = torch.autograd.grad(synthetic_loss, parameters)
            real_vector = np.concatenate(
                [gradient.detach().float().cpu().numpy().ravel() for gradient in real_gradients]
            )
            synthetic_vector = np.concatenate(
                [
                    gradient.detach().float().cpu().numpy().ravel()
                    for gradient in synthetic_gradients
                ]
            )
            combination = combine_real_synthetic_gradients(
                real_vector,
                synthetic_vector,
                mode=mode,
                synthetic_weight=float(spec.get("synthetic_weight", 1.0)),
                quality_score=quality_score,
                max_auxiliary_to_real_norm=spec.get("max_auxiliary_to_real_norm"),
                projection_cosine_threshold=float(
                    spec.get("projection_cosine_threshold", 0.0)
                ),
            )
            vector = np.asarray(combination.combined_gradient, dtype=np.float32)
            combined = []
            offset = 0
            for parameter in parameters:
                size = parameter.numel()
                combined.append(
                    torch.as_tensor(
                        vector[offset : offset + size], device=parameter.device
                    ).reshape_as(parameter)
                )
                offset += size
        for parameter, gradient in zip(parameters, combined, strict=True):
            parameter.grad = gradient
        grad_norm = float(
            torch.sqrt(sum(torch.sum(gradient.float() ** 2) for gradient in combined))
        )
        torch.nn.utils.clip_grad_norm_(
            parameters, float(spec.get("max_grad_norm", 1.0))
        )
        optimizer.step()
        step_records.append(
            {
                "step": step + 1,
                "real_loss": float(real_loss.detach()),
                "synthetic_loss": (
                    float(synthetic_loss.detach()) if synthetic_loss is not None else None
                ),
                "two_d_loss": (
                    float(two_d_loss.detach()) if two_d_loss is not None else None
                ),
                "gradient_norm_before_clip": grad_norm,
                "combination": gradient_combination_diagnostics(combination),
                "portfolio_combination": portfolio_gradient_diagnostics(
                    portfolio_combination
                ),
            }
        )

    train_loss_after = mean_loss(real, train_positions)
    validation_loss_after = mean_loss(real, validation_positions)
    two_d_loss_after = (
        mean_loss(real, train_positions, portfolio_two_d_source)
        if portfolio_two_d_source is not None
        else None
    )
    reliability = heldout_real_reliability(
        validation_loss_before, validation_loss_after
    )
    layers = []
    dense_adapter_norm_sq = 0.0
    dense_delta_norm_sq = 0.0
    for name, adapter in adapters.items():
        layer = LoRALayerUpdate(
            name,
            adapter.a.detach().cpu().numpy().astype(np.float32),
            (
                adapter.b.detach().cpu().numpy().astype(np.float32)
                * adapter.scaling
            ),
        )
        layers.append(layer)
        dense_adapter_norm_sq += low_rank_dense_frobenius_norm(layer.a, layer.b) ** 2
        dense_delta_norm_sq += low_rank_dense_difference_norm(
            layer, initial_by_name.get(name)
        ) ** 2
    dense_adapter_norm = math.sqrt(dense_adapter_norm_sq)
    dense_delta_norm = math.sqrt(dense_delta_norm_sq)
    elapsed = time.perf_counter() - started
    update = ClientAdapterUpdate(
        client_id=str(spec["client_id"]),
        layers=tuple(layers),
        n_real=sum(
            len(real["by_episode"][int(episode)])
            for episode in spec["train_episodes"]
        ),
        reliability=reliability,
        staleness=int(spec.get("staleness", 0)),
        compute_time_s=elapsed,
    )
    upload = save_client_update(update, out_dir / "adapter_upload")
    finite = all(
        math.isfinite(value)
        for value in (
            train_loss_before,
            train_loss_after,
            validation_loss_before,
            validation_loss_after,
            dense_adapter_norm,
            dense_delta_norm,
        )
    )
    report = {
        "schema": "fedsplat.openvla_oft_local_update.v1",
        "backend": "openvla_oft",
        "evidence_label": (
            "real_model_local_update"
            if mode == "real_only"
            else (
                "adaptive_2d_gs_portfolio_model_local_update"
                if mode == "augmentation_portfolio"
                else "source_paired_gaussian_model_local_update"
            )
        ),
        "client_id": spec["client_id"],
        "training_config_sha256": local_update_config_fingerprint(spec),
        "initial_adapter": initial_adapter_audit,
        "checkpoint": spec["checkpoint_path"],
        "dataset": spec["dataset_path"],
        "repo_id": spec["repo_id"],
        "device": cfg.device,
        "train_episodes": spec["train_episodes"],
        "validation_episodes": spec["validation_episodes"],
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "gradient_mode": mode,
        "synthetic_objective": objective,
        "source_pairing": pairing_audit,
        "target_modules": targets,
        "rank": int(spec["rank"]),
        "steps": int(spec["steps"]),
        "step_records": step_records,
        "augmentation_portfolio": (
            {
                "schema": "fedsplat.augmentation_portfolio_execution.v1",
                "gs_ratio": portfolio_ratio,
                "two_d_ratio": 1.0 - portfolio_ratio,
                "equal_auxiliary_budget": True,
                "two_d_source": portfolio_two_d_source,
                "gs_quality_score": quality_score,
                "real_forward_steps": len(step_records),
                "two_d_forward_steps": len(step_records),
                "gs_forward_steps": len(step_records),
                "mean_effective_gs_weight": float(
                    np.mean(
                        [
                            record["portfolio_combination"]["gs_combination"][
                                "effective_synthetic_weight"
                            ]
                            for record in step_records
                        ]
                    )
                ),
                "two_d_loss_before": two_d_loss_before,
                "two_d_loss_after": two_d_loss_after,
            }
            if mode == "augmentation_portfolio"
            else None
        ),
        "train_loss_before": train_loss_before,
        "train_loss_after": train_loss_after,
        "validation_loss_before": validation_loss_before,
        "validation_loss_after": validation_loss_after,
        "heldout_real_reliability": reliability,
        "reliability": reliability,
        "dense_delta_norm": dense_delta_norm,
        "dense_adapter_norm": dense_adapter_norm,
        "optimization_signal_ready": train_loss_after < train_loss_before,
        "load_seconds": load_seconds,
        "compute_time_s": elapsed,
        "upload": upload,
        "local_update_ready": bool(
            finite
            and dense_adapter_norm > 0.0
            and dense_delta_norm > 0.0
            and upload["total_upload_bytes"] > 0
        ),
        "evidence_scope": "offline forward/backward OpenVLA-OFT projector LoRA update",
        "not_evidence_for": ["federated aggregation", "closed-loop task success"],
    }
    (out_dir / "local_update_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.worker_config.read_text(encoding="utf-8"))
    errors = validate_local_update_spec(spec)
    if errors:
        raise ValueError("Invalid OpenVLA-OFT worker config: " + "; ".join(errors))
    report = _worker(spec)
    print(json.dumps({"local_update_ready": report["local_update_ready"]}))


if __name__ == "__main__":
    main()
