"""External-runtime SmolVLA local training with project-owned LoRA uploads."""

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


DEFAULT_TARGET_MODULES = (
    "model.state_proj",
    "model.action_in_proj",
    "model.action_out_proj",
)

GRADIENT_MODES = {
    "real_only",
    "naive",
    "projected",
    "augmentation_portfolio",
    "crossfit",
    "factor_paired",
    "paired_distill",
    "paired_bc_distill",
}
PAIRED_DISTILL_MODES = {"paired_distill", "paired_bc_distill"}
SYNTHETIC_SOURCE_TYPES = {"none", "visual_proxy", "lerobot_dataset"}
UPDATE_CONTRACTS = {"combined", "dual_channel"}
SHARPNESS_MODES = {"none", "sam"}
SYNTHETIC_GRADIENT_SCOPES = {"all", "perception"}
POLICY_ANCHOR_MODES = {
    "none",
    "real_branch_nonperception",
    "counterfactual_branch_nonperception",
}
SYNTHETIC_OBJECTIVES = {"action_bc", "source_paired_consistency"}
REPRESENTATION_ANCHOR_MODES = {"none", "pretrained_connector"}
FACTOR_COUNTERFACTUAL_MODES = {"real_only", "gaussian_bc"}
PAIRED_TEACHER_MODES = {"current_policy", "initial_global"}
PAIRED_TEACHER_REFERENCES = {"initial_adapter", "pretrained_base"}
PAIRED_ACTION_CONSISTENCY_MODES = {"flow_velocity", "inference_trajectory"}
SYNTHETIC_SAMPLING_MODES = {
    "uniform",
    "gripper_event_balanced",
    "preselected_all",
}
ROBOT_ADAPTERS = {"none", "piper_native_7d"}


def _is_visual_proxy_image_tensor(
    key: str,
    value: Any,
    image_keys: set[str],
) -> bool:
    """Exclude LeRobot sequence-padding masks from pixel transforms."""
    return key in image_keys and int(getattr(value, "ndim", 0)) >= 3


def _evenly_spaced_offsets(
    frame_count: int,
    count: int,
    *,
    start_fraction: float = 0.1,
    stop_fraction: float = 0.9,
) -> list[int]:
    """Return deterministic frame offsets without importing the ML runtime."""
    if frame_count <= 0 or count <= 0:
        raise ValueError("frame_count and count must be positive")
    if count > frame_count:
        raise ValueError(
            f"Cannot select {count} unique frames from an episode of {frame_count}"
        )
    if count == frame_count:
        return list(range(frame_count))
    if not 0.0 <= start_fraction <= stop_fraction <= 1.0:
        raise ValueError("sampling fractions must satisfy 0 <= start <= stop <= 1")
    if count == 1:
        fractions = [(start_fraction + stop_fraction) / 2.0]
    else:
        step = (stop_fraction - start_fraction) / float(count - 1)
        fractions = [start_fraction + step * index for index in range(count)]
    offsets = [
        min(frame_count - 1, max(0, int(round(fraction * (frame_count - 1)))))
        for fraction in fractions
    ]
    if len(set(offsets)) != count:
        raise ValueError(
            f"Sampling produced duplicate offsets for frame_count={frame_count}, count={count}"
        )
    return offsets


def event_balanced_frame_offsets(
    actions: list[list[float]],
    count: int,
    *,
    gripper_action_index: int = -1,
) -> tuple[list[int], dict[str, Any]]:
    """Cover gripper events, late completion, and the full episode deterministically."""
    frame_count = len(actions)
    if frame_count <= 0:
        raise ValueError("Event-balanced sampling requires a nonempty action sequence")
    if count < 12:
        raise ValueError("Event-balanced sampling requires at least 12 frames per episode")
    if count > frame_count:
        raise ValueError(
            f"Cannot select {count} unique frames from an episode of {frame_count}"
        )
    try:
        gripper = [float(action[gripper_action_index]) for action in actions]
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("Every action must expose the configured gripper channel") from error
    if not all(math.isfinite(value) for value in gripper):
        raise ValueError("Gripper actions must be finite")

    closed = [value >= 0.0 for value in gripper]
    transitions = [
        index
        for index in range(1, frame_count)
        if closed[index] != closed[index - 1]
    ]
    if not transitions:
        raise ValueError("Event-balanced sampling found no gripper transition")
    if len(transitions) > count:
        raise ValueError(
            f"Episode has {len(transitions)} gripper transitions but only {count} samples"
        )

    selected: list[int] = []
    selected_set: set[int] = set()
    reasons: dict[str, list[int]] = {
        "gripper_event": [],
        "late_completion": [],
        "coverage_fill": [],
    }

    def add(offset: int, reason: str) -> bool:
        clipped = min(frame_count - 1, max(0, int(offset)))
        if clipped in selected_set:
            return False
        selected.append(clipped)
        selected_set.add(clipped)
        reasons[reason].append(clipped)
        return True

    event_budget = max(count // 2, len(transitions))
    event_offsets = (0, -3, 3, -6, 6, -12, 12)
    for relative_offset in event_offsets:
        for transition in transitions:
            if len(reasons["gripper_event"]) >= event_budget:
                break
            add(transition + relative_offset, "gripper_event")
        if len(reasons["gripper_event"]) >= event_budget:
            break

    late_budget = max(1, count // 4)
    late_candidates = _evenly_spaced_offsets(
        frame_count,
        late_budget,
        start_fraction=0.70,
        stop_fraction=1.0,
    )
    for offset in late_candidates:
        add(offset, "late_completion")
    if len(reasons["late_completion"]) < late_budget:
        for offset in range(int(round(0.70 * (frame_count - 1))), frame_count):
            if len(reasons["late_completion"]) >= late_budget:
                break
            add(offset, "late_completion")

    # Greedy farthest-point fill gives deterministic whole-trajectory coverage
    # after the task-critical event and completion quotas have been secured.
    while len(selected) < count:
        remaining = [offset for offset in range(frame_count) if offset not in selected_set]
        candidate = max(
            remaining,
            key=lambda offset: (
                min(abs(offset - chosen) for chosen in selected),
                -offset,
            ),
        )
        add(candidate, "coverage_fill")

    offsets = sorted(selected)
    nearest_transition_distances = [
        min(abs(offset - transition) for offset in offsets)
        for transition in transitions
    ]
    audit = {
        "sampling_mode": "gripper_event_balanced",
        "frame_count": frame_count,
        "requested_count": count,
        "selected_count": len(offsets),
        "gripper_action_index": gripper_action_index,
        "gripper_transition_offsets": transitions,
        "selected_frame_offsets": offsets,
        "selected_by_reason": {
            key: sorted(values) for key, values in reasons.items()
        },
        "event_window_offsets": list(event_offsets),
        "late_phase_start_fraction": 0.70,
        "maximum_nearest_transition_distance_frames": max(
            nearest_transition_distances
        ),
        "minimum_selected_phase_fraction": offsets[0] / float(frame_count - 1),
        "maximum_selected_phase_fraction": offsets[-1] / float(frame_count - 1),
    }
    if audit["maximum_nearest_transition_distance_frames"] != 0:
        raise RuntimeError("Event-balanced sampling failed to include every transition")
    return offsets, audit


def deterministic_inference_step(seed: int, num_steps: int) -> int:
    """Cycle deterministically through every inference denoising step."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    return int(seed) % int(num_steps)


def preselected_all_frame_offsets(
    frame_count: int,
    count: int,
) -> tuple[list[int], dict[str, Any]]:
    """Use every frame when upstream already froze the episode selection."""
    if frame_count <= 0:
        raise ValueError("preselected_all requires a nonempty episode")
    if count != frame_count:
        raise ValueError(
            "preselected_all requires samples_per_episode to equal the "
            f"stored frame count ({frame_count}), got {count}"
        )
    offsets = list(range(frame_count))
    return offsets, {
        "sampling_mode": "preselected_all",
        "frame_count": frame_count,
        "requested_count": count,
        "selected_count": len(offsets),
        "selected_frame_offsets": offsets,
        "upstream_event_selection_required": True,
        "all_preselected_frames_used": True,
        "minimum_selected_phase_fraction": 0.0,
        "maximum_selected_phase_fraction": 1.0,
    }


def source_demo_episode_map(
    quality_evidence: dict[str, Any],
    total_episodes: int,
) -> dict[str, int]:
    """Resolve source demos from per-episode provenance, preserving dataset order."""
    episodes = quality_evidence.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Source-paired quality evidence requires an episodes list")
    episode_by_demo: dict[str, int] = {}
    demo_by_episode: dict[int, str] = {}
    for item in episodes:
        if not isinstance(item, dict) or "episode_index" not in item:
            raise ValueError("Source-paired quality episode lacks episode_index")
        source_demo = item.get("source_demo", item.get("demo"))
        if not isinstance(source_demo, str) or not source_demo:
            raise ValueError("Source-paired quality episode lacks source_demo")
        episode_id = int(item["episode_index"])
        if source_demo in episode_by_demo:
            raise ValueError(f"Source-paired quality evidence duplicates demo {source_demo}")
        if episode_id in demo_by_episode:
            raise ValueError(f"Source-paired quality evidence duplicates episode {episode_id}")
        episode_by_demo[source_demo] = episode_id
        demo_by_episode[episode_id] = source_demo

    expected_episode_ids = set(range(int(total_episodes)))
    observed_episode_ids = set(demo_by_episode)
    if observed_episode_ids != expected_episode_ids:
        raise ValueError(
            "Source-paired quality episode coverage mismatch: "
            f"missing={sorted(expected_episode_ids - observed_episode_ids)}, "
            f"extra={sorted(observed_episode_ids - expected_episode_ids)}"
        )
    declared_demos = list(quality_evidence.get("train_demos", [])) + list(
        quality_evidence.get("calibration_demos", [])
    )
    if (
        len(declared_demos) != int(total_episodes)
        or len(declared_demos) != len(set(declared_demos))
        or set(declared_demos) != set(episode_by_demo)
    ):
        raise ValueError(
            "Source-paired consistency requires one unique declared source demo "
            f"per real episode: demos={len(declared_demos)}, "
            f"real_episodes={total_episodes}"
        )
    return episode_by_demo


def resolve_module_scope(module_names: list[str], patterns: list[str]) -> list[str]:
    """Resolve an explicit module subset and reject an ineffective scope."""
    if not patterns:
        raise ValueError("Module-scope patterns must not be empty")
    selected = [
        name
        for name in module_names
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    ]
    unmatched = [
        pattern
        for pattern in patterns
        if not any(fnmatch.fnmatchcase(name, pattern) for name in module_names)
    ]
    if unmatched:
        raise ValueError(f"Module-scope patterns matched no LoRA targets: {unmatched}")
    if not selected:
        raise ValueError("Module scope resolved to no LoRA targets")
    return selected


def compose_scoped_adapter_layers(real_layers, mixed_layers, mixed_module_names: set[str]):
    """Use mixed perception layers and real-only policy layers by module name."""
    real_by_name = {layer.layer_name: layer for layer in real_layers}
    mixed_by_name = {layer.layer_name: layer for layer in mixed_layers}
    if set(real_by_name) != set(mixed_by_name):
        raise ValueError("Real and mixed adapter branches have different layer sets")
    unknown = mixed_module_names - set(real_by_name)
    if unknown:
        raise ValueError(f"Mixed module scope contains unknown layers: {sorted(unknown)}")
    return tuple(
        mixed_by_name[name] if name in mixed_module_names else real_by_name[name]
        for name in real_by_name
    )


def local_update_config_fingerprint(spec: dict[str, Any]) -> str:
    """Hash every training-affecting field while allowing output relocation."""
    canonical = json.loads(json.dumps(spec))
    canonical.pop("out_dir", None)
    if canonical.get("gradient_mode", "real_only") not in {
        "augmentation_portfolio",
        "crossfit",
        *PAIRED_DISTILL_MODES,
    }:
        source = canonical.get("synthetic_source")
        if isinstance(source, dict):
            source.pop("selection", None)
            source.pop("fallback", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_target_module_names(
    available_names: list[str],
    exact_targets: list[str] | None = None,
    target_patterns: list[str] | None = None,
) -> list[str]:
    """Resolve exact and glob LoRA targets while rejecting silent misses."""
    exact_targets = list(exact_targets or [])
    target_patterns = list(target_patterns or [])
    available = set(available_names)
    missing = [name for name in exact_targets if name not in available]
    if missing:
        raise ValueError(f"LoRA targets were not found or are not linear: {missing}")
    selected = set(exact_targets)
    unmatched_patterns = []
    for pattern in target_patterns:
        matches = [name for name in available_names if fnmatch.fnmatchcase(name, pattern)]
        if not matches:
            unmatched_patterns.append(pattern)
        selected.update(matches)
    if unmatched_patterns:
        raise ValueError(f"LoRA target patterns matched no linear modules: {unmatched_patterns}")
    resolved = [name for name in available_names if name in selected]
    if not resolved:
        raise ValueError("No LoRA target modules were resolved")
    return resolved


def _episode_ids(spec: dict[str, Any], plural_key: str, singular_key: str) -> list[int]:
    values = spec.get(plural_key)
    if values is None and singular_key in spec:
        values = [spec[singular_key]]
    return list(values) if isinstance(values, list) else []


def validate_local_update_spec(spec: dict[str, Any]) -> list[str]:
    """Return contract errors without importing the external ML runtime."""
    errors = []
    robot_adapter = spec.get("robot_adapter", "none")
    if robot_adapter not in ROBOT_ADAPTERS:
        errors.append(
            f"robot_adapter must be one of {sorted(ROBOT_ADAPTERS)}"
        )
    for key in ("checkpoint_path", "dataset_path", "repo_id", "client_id"):
        if not isinstance(spec.get(key), str) or not str(spec[key]).strip():
            errors.append(f"{key} must be a nonempty string")
    for key in ("rank", "steps"):
        if not isinstance(spec.get(key), int) or int(spec[key]) <= 0:
            errors.append(f"{key} must be a positive integer")
    for key in ("learning_rate", "alpha"):
        value = spec.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
            errors.append(f"{key} must be finite and positive")
    train_episodes = _episode_ids(spec, "train_episodes", "train_episode")
    validation_episodes = _episode_ids(spec, "validation_episodes", "validation_episode")
    if not train_episodes or any(not isinstance(value, int) or value < 0 for value in train_episodes):
        errors.append("train_episodes must contain nonnegative integers")
    if not validation_episodes or any(not isinstance(value, int) or value < 0 for value in validation_episodes):
        errors.append("validation_episodes must contain nonnegative integers")
    if len(train_episodes) != len(set(train_episodes)) or len(validation_episodes) != len(set(validation_episodes)):
        errors.append("training and validation episode lists must not contain duplicates")
    if set(train_episodes) & set(validation_episodes):
        errors.append("training and validation episodes must be disjoint")
    owner_local_split = spec.get("owner_local_split")
    if owner_local_split is not None:
        if not isinstance(owner_local_split, dict):
            errors.append("owner_local_split must be an object")
        else:
            if owner_local_split.get("schema") != (
                "fedsplat.owner_local_source_split.v1"
            ):
                errors.append("owner_local_split has an unexpected schema")
            source_train = owner_local_split.get("source_train_episodes")
            source_validation = owner_local_split.get(
                "source_validation_episodes"
            )
            if (
                not isinstance(source_train, list)
                or not isinstance(source_validation, list)
                or any(
                    not isinstance(value, int) or value < 0
                    for value in [*source_train, *source_validation]
                )
            ):
                errors.append(
                    "owner_local_split source episodes must be nonnegative "
                    "integer lists"
                )
            else:
                if (
                    len(source_train) != len(set(source_train))
                    or len(source_validation) != len(set(source_validation))
                    or set(source_train) & set(source_validation)
                ):
                    errors.append(
                        "owner_local_split source train/validation episodes "
                        "must be unique and disjoint"
                    )
                if not set(train_episodes + validation_episodes).issubset(
                    set(source_train) | set(source_validation)
                ):
                    errors.append(
                        "owner-local episodes must be a subset of declared "
                        "source episodes"
                    )
                if set(train_episodes) & set(source_validation):
                    errors.append(
                        "source validation episodes cannot enter owner-local "
                        "training"
                    )
    record_marginal_utility = spec.get(
        "record_marginal_utility_episodes", False
    )
    if not isinstance(record_marginal_utility, bool):
        errors.append("record_marginal_utility_episodes must be a boolean")
    marginal_canonical_penalty = spec.get(
        "marginal_utility_canonical_penalty", 1.0
    )
    if (
        not isinstance(marginal_canonical_penalty, (int, float))
        or not math.isfinite(float(marginal_canonical_penalty))
        or float(marginal_canonical_penalty) < 0.0
    ):
        errors.append(
            "marginal_utility_canonical_penalty must be finite and nonnegative"
        )
    samples_per_episode = spec.get("samples_per_episode", 1)
    if not isinstance(samples_per_episode, int) or samples_per_episode <= 0:
        errors.append("samples_per_episode must be a positive integer")
    for key in (
        "train_samples_per_episode",
        "validation_samples_per_episode",
        "batch_size",
        "evaluation_batch_size",
    ):
        value = spec.get(key)
        if value is not None and (not isinstance(value, int) or value <= 0):
            errors.append(f"{key} must be a positive integer")
    if "resample_each_step" in spec and not isinstance(spec["resample_each_step"], bool):
        errors.append("resample_each_step must be a boolean")
    if "use_synthetic_quality_weight" in spec and not isinstance(
        spec["use_synthetic_quality_weight"], bool
    ):
        errors.append("use_synthetic_quality_weight must be a boolean")
    expected_task_index = spec.get("expected_task_index")
    if expected_task_index is not None and (not isinstance(expected_task_index, int) or not 0 <= expected_task_index <= 9):
        errors.append("expected_task_index must be an integer in [0, 9]")
    staleness = spec.get("staleness", 0)
    if not isinstance(staleness, int) or staleness < 0:
        errors.append("staleness must be a nonnegative integer")
    gradient_mode = spec.get("gradient_mode", "real_only")
    if gradient_mode not in GRADIENT_MODES:
        errors.append(f"gradient_mode must be one of {sorted(GRADIENT_MODES)}")
    if record_marginal_utility and gradient_mode not in PAIRED_DISTILL_MODES:
        errors.append(
            "record_marginal_utility_episodes requires paired distillation"
        )
    if record_marginal_utility and len(validation_episodes) < 2:
        errors.append(
            "record_marginal_utility_episodes requires at least two "
            "validation episodes"
        )
    if gradient_mode == "factor_paired":
        factor_label = spec.get("factor_label")
        if not isinstance(factor_label, str) or not factor_label.strip():
            errors.append("factor_paired requires a nonempty factor_label")
        factor_variations = spec.get("factor_variations")
        if (
            not isinstance(factor_variations, list)
            or not factor_variations
            or any(not isinstance(value, str) or not value.strip() for value in factor_variations)
            or len(factor_variations) != len(set(factor_variations))
        ):
            errors.append("factor_paired requires unique nonempty factor_variations")
        factor_counterfactual_mode = spec.get(
            "factor_counterfactual_mode", "real_only"
        )
        if factor_counterfactual_mode not in FACTOR_COUNTERFACTUAL_MODES:
            errors.append(
                "factor_counterfactual_mode must be one of "
                f"{sorted(FACTOR_COUNTERFACTUAL_MODES)}"
            )
    elif "factor_counterfactual_mode" in spec:
        errors.append("factor_counterfactual_mode requires gradient_mode=factor_paired")
    sharpness_mode = spec.get("sharpness_mode", "none")
    if sharpness_mode not in SHARPNESS_MODES:
        errors.append(f"sharpness_mode must be one of {sorted(SHARPNESS_MODES)}")
    sam_rho = spec.get("sam_rho", 0.03)
    if (
        not isinstance(sam_rho, (int, float))
        or not math.isfinite(float(sam_rho))
    ):
        errors.append("sam_rho must be finite")
    elif sharpness_mode == "sam" and float(sam_rho) <= 0.0:
        errors.append("sam_rho must be positive when sharpness_mode=sam")
    synthetic_gradient_scope = spec.get("synthetic_gradient_scope", "all")
    if synthetic_gradient_scope not in SYNTHETIC_GRADIENT_SCOPES:
        errors.append(
            "synthetic_gradient_scope must be one of "
            f"{sorted(SYNTHETIC_GRADIENT_SCOPES)}"
        )
    policy_anchor_mode = spec.get("policy_anchor_mode", "none")
    if policy_anchor_mode not in POLICY_ANCHOR_MODES:
        errors.append(f"policy_anchor_mode must be one of {sorted(POLICY_ANCHOR_MODES)}")
    if policy_anchor_mode != "none" and gradient_mode not in {"crossfit", "factor_paired"}:
        errors.append("policy_anchor_mode requires gradient_mode=crossfit or factor_paired")
    perception_patterns = spec.get("perception_module_patterns", [])
    if not isinstance(perception_patterns, list) or any(
        not isinstance(item, str) or not item for item in perception_patterns
    ):
        errors.append("perception_module_patterns must be a string list")
    if (
        synthetic_gradient_scope == "perception" or policy_anchor_mode != "none"
    ) and not perception_patterns:
        errors.append(
            "perception_module_patterns are required for perception-scoped gradients or policy anchoring"
        )
    local_trainable_patterns = spec.get("local_trainable_module_patterns", [])
    if not isinstance(local_trainable_patterns, list) or any(
        not isinstance(item, str) or not item for item in local_trainable_patterns
    ):
        errors.append("local_trainable_module_patterns must be a string list")
    synthetic_objective = spec.get("synthetic_objective", "action_bc")
    if synthetic_objective not in SYNTHETIC_OBJECTIVES:
        errors.append(f"synthetic_objective must be one of {sorted(SYNTHETIC_OBJECTIVES)}")
    representation_anchor_mode = spec.get("representation_anchor_mode", "none")
    if representation_anchor_mode not in REPRESENTATION_ANCHOR_MODES:
        errors.append(
            "representation_anchor_mode must be one of "
            f"{sorted(REPRESENTATION_ANCHOR_MODES)}"
        )
    representation_anchor_weight = spec.get("representation_anchor_weight", 1.0)
    if (
        not isinstance(representation_anchor_weight, (int, float))
        or not math.isfinite(float(representation_anchor_weight))
        or float(representation_anchor_weight) < 0.0
    ):
        errors.append("representation_anchor_weight must be finite and nonnegative")
    max_anchor_ratio = spec.get("max_anchor_to_bc_perception_norm", 0.25)
    if (
        not isinstance(max_anchor_ratio, (int, float))
        or not math.isfinite(float(max_anchor_ratio))
        or float(max_anchor_ratio) <= 0.0
    ):
        errors.append("max_anchor_to_bc_perception_norm must be finite and positive")
    if representation_anchor_mode != "none":
        if not perception_patterns:
            errors.append("representation anchoring requires perception_module_patterns")
        if isinstance(representation_anchor_weight, (int, float)) and float(representation_anchor_weight) <= 0.0:
            errors.append("representation_anchor_weight must be positive when anchoring is enabled")
    paired_weights = {
        key: spec.get(key, default)
        for key, default in (
            ("paired_action_consistency_weight", 1.0),
            ("paired_feature_consistency_weight", 0.25),
            ("paired_synthetic_bc_weight", 0.0),
            ("paired_canonical_action_identity_weight", 0.0),
            ("paired_canonical_feature_identity_weight", 0.0),
        )
    }
    for key, value in paired_weights.items():
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            errors.append(f"{key} must be finite and nonnegative")
    if synthetic_objective == "source_paired_consistency":
        if gradient_mode not in {
            "crossfit",
            "factor_paired",
            *PAIRED_DISTILL_MODES,
        }:
            errors.append(
                "source_paired_consistency requires gradient_mode=crossfit, "
                "factor_paired, paired_distill, or paired_bc_distill"
            )
        if synthetic_gradient_scope != "perception":
            errors.append("source_paired_consistency requires synthetic_gradient_scope=perception")
        if all(
            isinstance(value, (int, float)) and float(value) <= 0.0
            for value in paired_weights.values()
        ):
            errors.append("source_paired_consistency requires at least one positive objective weight")
    paired_teacher_mode = spec.get("paired_teacher_mode", "current_policy")
    if paired_teacher_mode not in PAIRED_TEACHER_MODES:
        errors.append(
            f"paired_teacher_mode must be one of {sorted(PAIRED_TEACHER_MODES)}"
        )
    identity_enabled = any(
        isinstance(paired_weights[key], (int, float))
        and float(paired_weights[key]) > 0.0
        for key in (
            "paired_canonical_action_identity_weight",
            "paired_canonical_feature_identity_weight",
        )
    )
    if identity_enabled and paired_teacher_mode != "initial_global":
        errors.append("canonical identity losses require paired_teacher_mode=initial_global")
    paired_action_consistency_mode = spec.get(
        "paired_action_consistency_mode", "flow_velocity"
    )
    if paired_action_consistency_mode not in PAIRED_ACTION_CONSISTENCY_MODES:
        errors.append(
            "paired_action_consistency_mode must be one of "
            f"{sorted(PAIRED_ACTION_CONSISTENCY_MODES)}"
        )
    if paired_action_consistency_mode == "inference_trajectory":
        if gradient_mode not in {"crossfit", *PAIRED_DISTILL_MODES}:
            errors.append(
                "inference_trajectory action consistency requires "
                "gradient_mode=paired_distill, paired_bc_distill, or crossfit"
            )
        paired_bc = paired_weights.get("paired_synthetic_bc_weight", 0.0)
        if (
            gradient_mode == "paired_distill"
            and isinstance(paired_bc, (int, float))
            and float(paired_bc) != 0.0
        ):
            errors.append(
                "pure inference-trajectory distillation requires "
                "paired_synthetic_bc_weight=0"
            )
        if (
            gradient_mode == "paired_bc_distill"
            and isinstance(paired_bc, (int, float))
            and float(paired_bc) <= 0.0
        ):
            errors.append(
                "paired_bc_distill requires paired_synthetic_bc_weight>0"
            )
    if gradient_mode in PAIRED_DISTILL_MODES:
        if synthetic_objective != "source_paired_consistency":
            errors.append(
                f"{gradient_mode} requires synthetic_objective=source_paired_consistency"
            )
        if paired_teacher_mode != "initial_global":
            errors.append(
                f"{gradient_mode} requires paired_teacher_mode=initial_global"
            )
        if not identity_enabled:
            errors.append(
                f"{gradient_mode} requires a positive canonical identity weight"
            )
        if not local_trainable_patterns:
            errors.append(
                f"{gradient_mode} requires local_trainable_module_patterns"
            )
        distillation_gate = spec.get("paired_distillation_gate", {})
        if not isinstance(distillation_gate, dict):
            errors.append("paired_distillation_gate must be an object")
        else:
            for key in (
                "minimum_train_pair_gain",
                "minimum_anchor_pair_gain",
                "real_validation_tolerance",
            ):
                value = distillation_gate.get(key, 0.0)
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    errors.append(
                        f"paired_distillation_gate.{key} must be finite and nonnegative"
                    )
    if gradient_mode == "factor_paired":
        factor_counterfactual_mode = spec.get(
            "factor_counterfactual_mode", "real_only"
        )
        if factor_counterfactual_mode == "gaussian_bc":
            if policy_anchor_mode != "counterfactual_branch_nonperception":
                errors.append(
                    "gaussian_bc factor counterfactual requires "
                    "policy_anchor_mode=counterfactual_branch_nonperception"
                )
            paired_bc = paired_weights.get("paired_synthetic_bc_weight", 0.0)
            if not isinstance(paired_bc, (int, float)) or float(paired_bc) != 1.0:
                errors.append(
                    "gaussian_bc factor counterfactual requires "
                    "paired_synthetic_bc_weight=1.0"
                )
        elif policy_anchor_mode == "counterfactual_branch_nonperception":
            errors.append(
                "counterfactual_branch_nonperception requires "
                "factor_counterfactual_mode=gaussian_bc"
            )
    synthetic_weight = spec.get("synthetic_weight", 1.0)
    if not isinstance(synthetic_weight, (int, float)) or not math.isfinite(float(synthetic_weight)) or float(synthetic_weight) < 0.0:
        errors.append("synthetic_weight must be finite and nonnegative")
    max_auxiliary_to_real_norm = spec.get("max_auxiliary_to_real_norm")
    if max_auxiliary_to_real_norm is not None and (
        not isinstance(max_auxiliary_to_real_norm, (int, float))
        or not math.isfinite(float(max_auxiliary_to_real_norm))
        or float(max_auxiliary_to_real_norm) <= 0.0
    ):
        errors.append("max_auxiliary_to_real_norm must be finite and positive")
    proximal_mu = spec.get("proximal_mu", 0.0)
    if not isinstance(proximal_mu, (int, float)) or not math.isfinite(float(proximal_mu)) or float(proximal_mu) < 0.0:
        errors.append("proximal_mu must be finite and nonnegative")
    projection_threshold = spec.get("projection_cosine_threshold", 0.0)
    if not isinstance(projection_threshold, (int, float)) or not math.isfinite(float(projection_threshold)) or not -1.0 <= float(projection_threshold) <= 0.0:
        errors.append("projection_cosine_threshold must be finite and in [-1, 0]")
    augmentation_portfolio = spec.get("augmentation_portfolio")
    if gradient_mode == "augmentation_portfolio":
        if not isinstance(augmentation_portfolio, dict):
            errors.append(
                "augmentation_portfolio gradient mode requires an "
                "augmentation_portfolio object"
            )
        else:
            if augmentation_portfolio.get("schema") != (
                "fedsplat.augmentation_portfolio.v1"
            ):
                errors.append("augmentation_portfolio has an unexpected schema")
            gs_ratio = augmentation_portfolio.get("gs_ratio")
            if (
                not isinstance(gs_ratio, (int, float))
                or not math.isfinite(float(gs_ratio))
                or float(gs_ratio) not in {0.0, 0.25, 0.5}
            ):
                errors.append(
                    "augmentation_portfolio.gs_ratio must be one of "
                    "[0.0, 0.25, 0.5]"
                )
            if augmentation_portfolio.get("equal_auxiliary_budget") is not True:
                errors.append(
                    "augmentation_portfolio.equal_auxiliary_budget must be true"
                )
            two_d_source = augmentation_portfolio.get("two_d_source")
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
                contrast = two_d_source.get("contrast", 1.0)
                brightness = two_d_source.get("brightness", 0.0)
                if (
                    not isinstance(contrast, (int, float))
                    or not math.isfinite(float(contrast))
                    or float(contrast) <= 0.0
                ):
                    errors.append(
                        "augmentation_portfolio.two_d_source.contrast must be "
                        "finite and positive"
                    )
                if (
                    not isinstance(brightness, (int, float))
                    or not math.isfinite(float(brightness))
                ):
                    errors.append(
                        "augmentation_portfolio.two_d_source.brightness must "
                        "be finite"
                    )
                for key in ("horizontal_shift_px", "vertical_shift_px"):
                    if not isinstance(two_d_source.get(key, 0), int):
                        errors.append(
                            f"augmentation_portfolio.two_d_source.{key} must "
                            "be an integer"
                        )
        if synthetic_objective != "action_bc":
            errors.append(
                "augmentation_portfolio requires synthetic_objective=action_bc"
            )
        scope_ablation = spec.get("method_ablation", {})
        all_lora_scope_ablation = (
            isinstance(scope_ablation, dict)
            and scope_ablation.get("schema")
            == "fedsplat.safemix_mechanism_ablation.v1"
            and scope_ablation.get("removed_component")
            == "perception_only_synthetic_gradient_scope"
        )
        if (
            synthetic_gradient_scope != "perception"
            and not all_lora_scope_ablation
        ):
            errors.append(
                "augmentation_portfolio requires "
                "synthetic_gradient_scope=perception unless the frozen "
                "all-LoRA scope ablation is declared"
            )
        quality_ablation = spec.get("method_ablation", {})
        quality_gate_removed = (
            isinstance(quality_ablation, dict)
            and quality_ablation.get("schema")
            == "fedsplat.safemix_mechanism_ablation.v1"
            and quality_ablation.get("removed_component") == "quality_gate"
        )
        if (
            spec.get("use_synthetic_quality_weight") is not True
            and not quality_gate_removed
        ):
            errors.append(
                "augmentation_portfolio requires "
                "use_synthetic_quality_weight=true unless the frozen "
                "SafeMix quality-gate ablation is declared"
            )
        if max_auxiliary_to_real_norm is None:
            errors.append(
                "augmentation_portfolio requires max_auxiliary_to_real_norm"
            )
        if spec.get("update_contract", "combined") != "combined":
            errors.append(
                "augmentation_portfolio requires update_contract=combined"
            )
    elif augmentation_portfolio is not None:
        errors.append(
            "augmentation_portfolio requires "
            "gradient_mode=augmentation_portfolio"
        )
    initial_adapter_update = spec.get("initial_adapter_update")
    if initial_adapter_update is not None and (
        not isinstance(initial_adapter_update, str) or not initial_adapter_update.strip()
    ):
        errors.append("initial_adapter_update must be null or a nonempty string")
    paired_teacher_reference = spec.get(
        "paired_teacher_reference",
        "initial_adapter" if initial_adapter_update else "pretrained_base",
    )
    if paired_teacher_reference not in PAIRED_TEACHER_REFERENCES:
        errors.append(
            "paired_teacher_reference must be one of "
            f"{sorted(PAIRED_TEACHER_REFERENCES)}"
        )
    if paired_teacher_mode == "initial_global":
        has_initial_adapter = bool(
            isinstance(initial_adapter_update, str) and initial_adapter_update.strip()
        )
        if paired_teacher_reference == "initial_adapter" and not has_initial_adapter:
            errors.append(
                "paired_teacher_reference=initial_adapter requires initial_adapter_update"
            )
        if paired_teacher_reference == "pretrained_base" and has_initial_adapter:
            errors.append(
                "paired_teacher_reference=pretrained_base requires initial_adapter_update=null"
            )
    update_contract = spec.get("update_contract", "combined")
    if update_contract not in UPDATE_CONTRACTS:
        errors.append(f"update_contract must be one of {sorted(UPDATE_CONTRACTS)}")
    residual_rank = spec.get("synthetic_residual_rank", spec.get("rank"))
    if not isinstance(residual_rank, int) or residual_rank <= 0:
        errors.append("synthetic_residual_rank must be a positive integer")
    synthetic_source = spec.get("synthetic_source", {"type": "none"})
    if not isinstance(synthetic_source, dict):
        errors.append("synthetic_source must be an object")
    else:
        source_type = synthetic_source.get("type", "none")
        if source_type not in SYNTHETIC_SOURCE_TYPES:
            errors.append(f"synthetic_source.type must be one of {sorted(SYNTHETIC_SOURCE_TYPES)}")
        if gradient_mode != "real_only" and source_type == "none":
            errors.append(
                "a synthetic source is required for non-real gradient modes"
            )
        if gradient_mode in {
            "augmentation_portfolio",
            "crossfit",
            "factor_paired",
            *PAIRED_DISTILL_MODES,
        } and source_type != "lerobot_dataset":
            errors.append(
                f"{gradient_mode} requires synthetic_source.type=lerobot_dataset"
            )
        if synthetic_objective == "source_paired_consistency" and source_type != "lerobot_dataset":
            errors.append("source_paired_consistency requires synthetic_source.type=lerobot_dataset")
        provenance = synthetic_source.get("provenance")
        if source_type != "none" and (not isinstance(provenance, str) or not provenance.strip()):
            errors.append("synthetic_source.provenance must be a nonempty string")
        if source_type == "visual_proxy" and provenance != "proxy_visual_augmentation":
            errors.append("visual_proxy provenance must be proxy_visual_augmentation")
        if source_type == "lerobot_dataset":
            for key in ("dataset_path", "repo_id", "provenance"):
                if not isinstance(synthetic_source.get(key), str) or not str(synthetic_source[key]).strip():
                    errors.append(f"synthetic_source.{key} must be a nonempty string")
            source_episodes = synthetic_source.get("episode_ids")
            candidate_episodes = synthetic_source.get("candidate_episode_ids", source_episodes)
            anchor_episodes = synthetic_source.get("anchor_episode_ids", [])
            if not isinstance(source_episodes, list) or not source_episodes or any(
                not isinstance(value, int) or value < 0 for value in source_episodes
            ):
                errors.append("synthetic_source.episode_ids must contain nonnegative integers")
            if candidate_episodes is not None and (
                not isinstance(candidate_episodes, list)
                or not candidate_episodes
                or any(not isinstance(value, int) or value < 0 for value in candidate_episodes)
                or len(candidate_episodes) != len(set(candidate_episodes))
            ):
                errors.append("synthetic_source.candidate_episode_ids must contain unique nonnegative integers")
            if anchor_episodes is not None and (
                not isinstance(anchor_episodes, list)
                or any(not isinstance(value, int) or value < 0 for value in anchor_episodes)
                or len(anchor_episodes) != len(set(anchor_episodes))
            ):
                errors.append("synthetic_source.anchor_episode_ids must contain unique nonnegative integers")
            if isinstance(candidate_episodes, list) and isinstance(anchor_episodes, list) and (
                set(candidate_episodes) & set(anchor_episodes)
            ):
                errors.append("synthetic candidate and anchor episodes must be disjoint")
            if isinstance(source_episodes, list) and isinstance(candidate_episodes, list) and not set(
                candidate_episodes
            ).issubset(set(source_episodes)):
                errors.append("synthetic candidate episodes must be a subset of episode_ids")
            if isinstance(source_episodes, list) and isinstance(anchor_episodes, list) and not set(
                anchor_episodes
            ).issubset(set(source_episodes)):
                errors.append("synthetic anchor episodes must be a subset of episode_ids")
            if gradient_mode == "crossfit" and not anchor_episodes:
                errors.append("crossfit requires synthetic_source.anchor_episode_ids")
            if gradient_mode in PAIRED_DISTILL_MODES and not anchor_episodes:
                errors.append(
                    f"{gradient_mode} requires synthetic_source.anchor_episode_ids"
                )
            if gradient_mode == "factor_paired" and anchor_episodes:
                errors.append("factor_paired does not use synthetic_source.anchor_episode_ids")
            global_indices = synthetic_source.get("global_indices")
            if global_indices is not None and (
                not isinstance(global_indices, list)
                or not global_indices
                or any(not isinstance(value, int) or value < 0 for value in global_indices)
                or len(global_indices) != len(set(global_indices))
            ):
                errors.append("synthetic_source.global_indices must contain unique nonnegative integers")
            quality_manifest = synthetic_source.get("quality_manifest")
            if quality_manifest is not None and (
                not isinstance(quality_manifest, str) or not quality_manifest.strip()
            ):
                errors.append("synthetic_source.quality_manifest must be null or a nonempty string")
            if gradient_mode == "crossfit" and not quality_manifest:
                errors.append("crossfit requires synthetic_source.quality_manifest")
            if gradient_mode == "augmentation_portfolio" and not quality_manifest:
                errors.append(
                    "augmentation_portfolio requires "
                    "synthetic_source.quality_manifest"
                )
            if synthetic_objective == "source_paired_consistency" and not quality_manifest:
                errors.append("source_paired_consistency requires synthetic_source.quality_manifest")
            selection = synthetic_source.get("selection", {})
            if not isinstance(selection, dict):
                errors.append("synthetic_source.selection must be an object")
            else:
                sampling_mode = selection.get("sampling_mode", "uniform")
                if sampling_mode not in SYNTHETIC_SAMPLING_MODES:
                    errors.append(
                        "synthetic_source.selection.sampling_mode must be one of "
                        f"{sorted(SYNTHETIC_SAMPLING_MODES)}"
                    )
                selected_samples_per_episode = selection.get(
                    "samples_per_episode",
                    synthetic_source.get("samples_per_episode", samples_per_episode),
                )
                if (
                    not isinstance(selected_samples_per_episode, int)
                    or selected_samples_per_episode <= 0
                ):
                    errors.append(
                        "synthetic_source.selection.samples_per_episode must be "
                        "a positive integer"
                )
                if sampling_mode == "gripper_event_balanced":
                    event_ablation = spec.get("method_ablation", {})
                    portfolio_event_ablation = (
                        gradient_mode == "augmentation_portfolio"
                        and isinstance(event_ablation, dict)
                        and event_ablation.get("schema")
                        == "fedsplat.safemix_mechanism_ablation.v1"
                        and event_ablation.get("variant") == "event_balanced"
                    )
                    if (
                        synthetic_objective != "source_paired_consistency"
                        and not portfolio_event_ablation
                    ):
                        errors.append(
                            "gripper_event_balanced sampling requires "
                            "synthetic_objective=source_paired_consistency or "
                            "the frozen SafeMix event-sampling ablation"
                        )
                    if (
                        isinstance(selected_samples_per_episode, int)
                        and selected_samples_per_episode < 12
                    ):
                        errors.append(
                            "gripper_event_balanced sampling requires at least "
                            "12 samples per episode"
                        )
                    gripper_action_index = selection.get(
                        "gripper_action_index", -1
                    )
                    if not isinstance(gripper_action_index, int):
                        errors.append(
                            "synthetic_source.selection.gripper_action_index "
                            "must be an integer"
                        )
                if (
                    sampling_mode == "preselected_all"
                    and synthetic_objective != "source_paired_consistency"
                ):
                    errors.append(
                        "preselected_all sampling requires "
                        "synthetic_objective=source_paired_consistency"
                    )
            if isinstance(selection, dict) and gradient_mode == "crossfit":
                if selection.get("combination_mode", "projected") not in {"naive", "projected"}:
                    errors.append("synthetic_source.selection.combination_mode must be naive or projected")
                for key in ("max_selected", "max_per_variation", "minimum_variations", "samples_per_episode"):
                    value = selection.get(key)
                    if value is not None and (not isinstance(value, int) or value <= 0):
                        errors.append(f"synthetic_source.selection.{key} must be a positive integer")
                for key in ("lcb_z", "minimum_lcb", "minimum_positive_fraction", "anchor_weight"):
                    value = selection.get(key)
                    if value is not None and (
                        not isinstance(value, (int, float)) or not math.isfinite(float(value))
                    ):
                        errors.append(f"synthetic_source.selection.{key} must be finite")
                fraction = selection.get("minimum_positive_fraction", 0.5)
                if isinstance(fraction, (int, float)) and not 0.0 <= float(fraction) <= 1.0:
                    errors.append("synthetic_source.selection.minimum_positive_fraction must be in [0, 1]")
                minimum_render_coverage = selection.get("minimum_render_coverage", 0.0)
                if (
                    not isinstance(minimum_render_coverage, (int, float))
                    or not math.isfinite(float(minimum_render_coverage))
                    or not 0.0 <= float(minimum_render_coverage) <= 1.0
                ):
                    errors.append("synthetic_source.selection.minimum_render_coverage must be in [0, 1]")
                anchor_weight = selection.get("anchor_weight", 0.5)
                if isinstance(anchor_weight, (int, float)) and float(anchor_weight) < 0.0:
                    errors.append("synthetic_source.selection.anchor_weight must be nonnegative")
                fallback = synthetic_source.get("fallback", {})
                if not isinstance(fallback, dict):
                    errors.append("synthetic_source.fallback must be an object")
                else:
                    for key in ("minimum_relative_anchor_gain", "real_validation_tolerance"):
                        value = fallback.get(key, 0.0)
                        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                            errors.append(f"synthetic_source.fallback.{key} must be finite and nonnegative")
    patterns = spec.get("target_module_patterns", [])
    targets = spec.get("target_modules", [] if patterns else list(DEFAULT_TARGET_MODULES))
    if not isinstance(targets, list) or any(not isinstance(item, str) or not item for item in targets):
        errors.append("target_modules must be a string list")
    if not isinstance(patterns, list) or any(not isinstance(item, str) or not item for item in patterns):
        errors.append("target_module_patterns must be a string list")
    if isinstance(targets, list) and isinstance(patterns, list) and not targets and not patterns:
        errors.append("at least one target module or target pattern is required")
    rename_map = spec.get("rename_map", {})
    if not isinstance(rename_map, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in rename_map.items()):
        errors.append("rename_map must map strings to strings")
    return errors


def run_smolvla_local_update(
    config_path: Path,
    out_dir: Path,
    runtime_python: Path,
    pythonpath: list[Path] | None = None,
    cuda_visible_device: str | None = None,
    timeout_s: int = 3600,
) -> dict[str, Any]:
    """Resolve project paths and execute the update inside a prepared LeRobot runtime."""
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    errors = validate_local_update_spec(spec)
    if errors:
        raise ValueError("Invalid SmolVLA local-update config: " + "; ".join(errors))

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_spec = dict(spec)
    worker_spec["checkpoint_path"] = str(_project_path(project_root, spec["checkpoint_path"]))
    worker_spec["dataset_path"] = str(_project_path(project_root, spec["dataset_path"]))
    if worker_spec.get("initial_adapter_update"):
        worker_spec["initial_adapter_update"] = str(_project_path(project_root, worker_spec["initial_adapter_update"]))
    synthetic_source = dict(worker_spec.get("synthetic_source", {"type": "none"}))
    if synthetic_source.get("type") == "lerobot_dataset":
        synthetic_source["dataset_path"] = str(_project_path(project_root, synthetic_source["dataset_path"]))
        if synthetic_source.get("quality_manifest"):
            synthetic_source["quality_manifest"] = str(_project_path(project_root, synthetic_source["quality_manifest"]))
    worker_spec["synthetic_source"] = synthetic_source
    worker_spec["out_dir"] = str(out_dir)
    worker_config = out_dir / "worker_config.json"
    worker_config.write_text(json.dumps(worker_spec, indent=2) + "\n", encoding="utf-8")

    runtime_python = runtime_python.absolute()
    if not runtime_python.exists():
        raise FileNotFoundError(f"Runtime Python does not exist: {runtime_python}")
    env = os.environ.copy()
    paths = [project_root, *(pythonpath or [])]
    prefix = os.pathsep.join(str(path.resolve()) for path in paths)
    env["PYTHONPATH"] = prefix + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cuda_visible_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_device)
    runtime_started = time.perf_counter()
    completed = subprocess.run(
        [str(runtime_python), "-m", "fedsplat.smolvla_local_update", "--worker-config", str(worker_config)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
    )
    (out_dir / "worker_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "worker_stderr.log").write_text(completed.stderr, encoding="utf-8")
    report_path = out_dir / "local_update_report.json"
    if completed.returncode != 0 or not report_path.exists():
        detail = (completed.stderr or completed.stdout or "worker produced no report")[-4000:]
        raise RuntimeError(f"SmolVLA local-update worker failed ({completed.returncode}): {detail}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["runtime_wall_s"] = time.perf_counter() - runtime_started
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_worker_report(report, out_dir / "SMOLVLA_LOCAL_UPDATE_REPORT.md")
    return report


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _pyav_decoder():
    import av
    import torch

    def decode(video_path, timestamps, tolerance_s, *args, **kwargs):
        del args, kwargs
        query = [float(value) for value in timestamps]
        loaded_frames = []
        loaded_timestamps = []
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            first = max(0.0, min(query) - 1.0)
            last = max(query)
            container.seek(
                int(first / float(stream.time_base)),
                stream=stream,
                backward=True,
                any_frame=False,
            )
            for frame in container.decode(stream):
                timestamp = float(frame.pts * frame.time_base)
                array = frame.to_ndarray(format="rgb24")
                loaded_frames.append(torch.from_numpy(array).permute(2, 0, 1))
                loaded_timestamps.append(timestamp)
                if timestamp >= last:
                    break
        if not loaded_frames:
            raise RuntimeError(f"No video frames decoded from {video_path}")
        query_tensor = torch.tensor(query, dtype=torch.float64)
        loaded_tensor = torch.tensor(loaded_timestamps, dtype=torch.float64)
        distances, indices = torch.cdist(query_tensor[:, None], loaded_tensor[:, None]).min(1)
        effective_tolerance = max(float(tolerance_s), 1e-3)
        if not bool((distances < effective_tolerance).all()):
            raise RuntimeError(
                f"Decoded frame timestamp exceeds tolerance: max={float(distances.max())}, "
                f"tolerance={effective_tolerance}, video={video_path}"
            )
        return torch.stack([loaded_frames[int(index)] for index in indices]).float() / 255.0

    return decode


def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as functional

    from .runtime_compat import bootstrap_smolvla_processors

    bootstrap_smolvla_processors()

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy,
        make_att_2d_masks,
    )
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    from .lora_updates import (
        ClientAdapterUpdate,
        DualChannelAdapterUpdate,
        LoRALayerUpdate,
        factorize_dense_layer,
        load_client_update,
        save_client_update,
        save_dual_channel_update,
    )
    from .factorized_routing import build_scoped_factor_residual
    from .adaptive_augmentation import (
        combine_augmentation_portfolio_gradients,
        validate_portfolio_ratio,
    )
    from .method import (
        combine_real_synthetic_gradients,
        feasibility_gate,
        geometric_quality_score,
        heldout_real_reliability,
    )
    from .trajectory_selection import TrajectoryGradientEvidence, select_trajectories

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
            super().__init__()
            self.base = base
            self.rank = rank
            self.scaling = float(alpha) / float(rank)
            self.a = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
            self.b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
            nn.init.kaiming_uniform_(self.a, a=math.sqrt(5.0))

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
            update = functional.linear(functional.linear(inputs.float(), self.a), self.b)
            return base_output + update.to(dtype=base_output.dtype) * self.scaling

    seed = int(spec.get("seed", 7))
    torch.manual_seed(seed)
    np.random.seed(seed)
    checkpoint = Path(spec["checkpoint_path"])
    dataset_root = Path(spec["dataset_path"])
    out_dir = Path(spec["out_dir"])
    device = str(spec.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    decoder = _pyav_decoder()
    import lerobot.datasets.lerobot_dataset as dataset_module
    import lerobot.datasets.video_utils as video_module

    dataset_module.decode_video_frames = decoder
    video_module.decode_video_frames = decoder

    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    robot_adapter = str(spec.get("robot_adapter", "none"))
    robot_adapter_report = None
    validate_robot_metadata = None
    if robot_adapter == "piper_native_7d":
        from .piper_adapter import (
            configure_smolvla_policy_for_piper,
            validate_piper_lerobot_metadata,
        )

        robot_adapter_report = {
            "name": robot_adapter,
            "policy": configure_smolvla_policy_for_piper(policy),
        }
        validate_robot_metadata = validate_piper_lerobot_metadata
    metadata = LeRobotDatasetMetadata(spec["repo_id"], root=dataset_root)
    if validate_robot_metadata is not None:
        robot_adapter_report["canonical_dataset"] = (
            validate_robot_metadata(metadata)
        )
    delta_timestamps = resolve_delta_timestamps(policy.config, metadata)
    dataset = LeRobotDataset(
        spec["repo_id"],
        root=dataset_root,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    def sample_indices_by_episode(
        metadata_obj,
        episode_ids: list[int],
        count: int,
        *,
        dataset_obj=None,
        sampling_mode: str = "uniform",
        gripper_action_index: int = -1,
    ) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
        by_episode: dict[int, list[int]] = {}
        sampling_records: list[dict[str, Any]] = []
        for episode_id in episode_ids:
            if episode_id >= int(metadata_obj.total_episodes):
                raise IndexError(f"Episode {episode_id} exceeds dataset size {metadata_obj.total_episodes}")
            episode = metadata_obj.episodes[episode_id]
            start = int(episode["dataset_from_index"])
            stop = int(episode["dataset_to_index"])
            if stop <= start:
                raise ValueError(f"Episode {episode_id} has no frames")
            frame_count = stop - start
            if sampling_mode == "uniform":
                frame_offsets = _evenly_spaced_offsets(
                    frame_count,
                    count,
                )
            elif sampling_mode == "gripper_event_balanced":
                if dataset_obj is None:
                    raise ValueError(
                        "gripper_event_balanced sampling requires a loaded dataset"
                    )
                actions = [
                    list(dataset_obj.hf_dataset[start + offset]["action"])
                    for offset in range(frame_count)
                ]
                frame_offsets, record = event_balanced_frame_offsets(
                    actions,
                    count,
                    gripper_action_index=gripper_action_index,
                )
                sampling_records.append(
                    {
                        "episode_index": episode_id,
                        "dataset_from_index": start,
                        "dataset_to_index": stop,
                        "selected_global_indices": [
                            start + offset for offset in frame_offsets
                        ],
                        **record,
                    }
                )
            elif sampling_mode == "preselected_all":
                frame_offsets, record = preselected_all_frame_offsets(
                    frame_count,
                    count,
                )
                sampling_records.append(
                    {
                        "episode_index": episode_id,
                        "dataset_from_index": start,
                        "dataset_to_index": stop,
                        "selected_global_indices": [
                            start + offset for offset in frame_offsets
                        ],
                        **record,
                    }
                )
            else:
                raise ValueError(f"Unknown synthetic sampling mode: {sampling_mode}")
            if len(frame_offsets) != count or len(set(frame_offsets)) != count:
                raise RuntimeError(
                    f"Episode {episode_id} sampling did not produce {count} unique frames"
                )
            by_episode[episode_id] = [start + offset for offset in frame_offsets]
        return by_episode, sampling_records

    def flatten_episode_indices(by_episode: dict[int, list[int]]) -> list[int]:
        return [index for values in by_episode.values() for index in values]

    def collate(dataset_obj, indices: list[int]):
        return torch.utils.data.default_collate([dataset_obj[index] for index in indices])

    gradient_mode = str(spec.get("gradient_mode", "real_only"))
    dynamic_batches = bool(spec.get("resample_each_step", False)) or gradient_mode in {
        "augmentation_portfolio",
        "crossfit",
        "factor_paired",
        *PAIRED_DISTILL_MODES,
    }
    samples_per_episode = int(spec.get("samples_per_episode", 1))
    train_samples_per_episode = int(spec.get("train_samples_per_episode", samples_per_episode))
    validation_samples_per_episode = int(spec.get("validation_samples_per_episode", samples_per_episode))
    train_episodes = _episode_ids(spec, "train_episodes", "train_episode")
    validation_episodes = _episode_ids(spec, "validation_episodes", "validation_episode")
    train_indices_by_episode, _ = sample_indices_by_episode(
        metadata,
        train_episodes,
        train_samples_per_episode,
    )
    validation_indices_by_episode, _ = sample_indices_by_episode(
        metadata,
        validation_episodes,
        validation_samples_per_episode,
    )
    train_indices = flatten_episode_indices(train_indices_by_episode)
    validation_indices = flatten_episode_indices(validation_indices_by_episode)

    expected_task_index = spec.get("expected_task_index")
    if expected_task_index is not None:
        observed = {
            int(dataset.hf_dataset[index]["task_index"])
            for index in (*train_indices, *validation_indices)
        }
        if observed != {int(expected_task_index)}:
            raise ValueError(f"Expected task {expected_task_index}, observed task indices {sorted(observed)}")

    policy.config.device = device
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": "cpu" if dynamic_batches else device},
            "normalizer_processor": {
                "stats": metadata.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": spec.get("rename_map", {})},
        },
    )
    train_batch = preprocessor(collate(dataset, train_indices)) if not dynamic_batches else None
    validation_batch = preprocessor(collate(dataset, validation_indices)) if not dynamic_batches else None

    synthetic_source = dict(spec.get("synthetic_source", {"type": "none"}))
    source_type = str(synthetic_source.get("type", "none"))
    source_provenance = str(synthetic_source.get("provenance", "none"))
    synthetic_indices: list[int] = []
    synthetic_batch = None
    synthetic_dataset = None
    synthetic_metadata = None
    synthetic_indices_by_episode: dict[int, list[int]] = {}
    candidate_episode_ids: list[int] = []
    anchor_episode_ids: list[int] = []
    source_pair_sampling_audit: dict[str, Any] | None = None
    augmentation_portfolio_config = dict(
        spec.get("augmentation_portfolio", {})
    )
    portfolio_gs_ratio = (
        validate_portfolio_ratio(augmentation_portfolio_config["gs_ratio"])
        if gradient_mode == "augmentation_portfolio"
        else None
    )
    portfolio_two_d_ratio = (
        1.0 - portfolio_gs_ratio
        if portfolio_gs_ratio is not None
        else None
    )
    portfolio_two_d_dataset = None

    def visual_proxy_transform(source_config, image_keys):
        contrast = float(source_config.get("contrast", 1.0))
        brightness = float(source_config.get("brightness", 0.0))
        shift_x = int(source_config.get("horizontal_shift_px", 0))
        shift_y = int(source_config.get("vertical_shift_px", 0))

        def transform(item):
            transformed_item = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in item.items()
            }
            for key, value in list(transformed_item.items()):
                if (
                    not torch.is_tensor(value)
                    or not _is_visual_proxy_image_tensor(
                        key,
                        value,
                        image_keys,
                    )
                ):
                    continue
                transformed = value.float() * contrast + brightness
                if shift_x or shift_y:
                    transformed = torch.roll(
                        transformed,
                        shifts=(shift_y, shift_x),
                        dims=(-2, -1),
                    )
                transformed_item[key] = transformed.clamp(0.0, 1.0).to(
                    dtype=value.dtype
                )
            return transformed_item
        return transform

    class VisualProxyDataset:
        """Apply deterministic 2D augmentation before preprocessing."""

        def __init__(self, source, transform):
            self.source = source
            self.transform = transform
            self.hf_dataset = source.hf_dataset

        def __getitem__(self, index):
            return self.transform(self.source[index])

        def __len__(self):
            return len(self.source)

    if gradient_mode == "augmentation_portfolio":
        portfolio_two_d_dataset = VisualProxyDataset(
            dataset,
            visual_proxy_transform(
                augmentation_portfolio_config["two_d_source"],
                set(metadata.video_keys),
            ),
        )

    if source_type == "visual_proxy":
        transform_visual_proxy_item = visual_proxy_transform(
            synthetic_source,
            set(metadata.video_keys),
        )

        if dynamic_batches:
            synthetic_dataset = VisualProxyDataset(
                dataset,
                transform_visual_proxy_item,
            )
            synthetic_metadata = metadata
            synthetic_indices_by_episode = {
                episode_id: list(indices)
                for episode_id, indices in train_indices_by_episode.items()
            }
            candidate_episode_ids = list(train_episodes)
            synthetic_indices = flatten_episode_indices(
                synthetic_indices_by_episode
            )
        else:
            synthetic_batch = transform_visual_proxy_item(train_batch)
    elif source_type == "lerobot_dataset":
        synthetic_root = Path(synthetic_source["dataset_path"])
        synthetic_metadata = LeRobotDatasetMetadata(synthetic_source["repo_id"], root=synthetic_root)
        if validate_robot_metadata is not None:
            robot_adapter_report["synthetic_dataset"] = (
                validate_robot_metadata(synthetic_metadata)
            )
        synthetic_delta_timestamps = resolve_delta_timestamps(policy.config, synthetic_metadata)
        synthetic_dataset = LeRobotDataset(
            synthetic_source["repo_id"],
            root=synthetic_root,
            delta_timestamps=synthetic_delta_timestamps,
            video_backend="pyav",
        )
        source_episode_ids = [int(value) for value in synthetic_source["episode_ids"]]
        candidate_episode_ids = [
            int(value) for value in synthetic_source.get("candidate_episode_ids", source_episode_ids)
        ]
        anchor_episode_ids = [int(value) for value in synthetic_source.get("anchor_episode_ids", [])]
        explicit_indices = synthetic_source.get("global_indices")
        if explicit_indices is not None:
            synthetic_indices = [int(value) for value in explicit_indices]
            if any(index >= int(synthetic_metadata.total_frames) for index in synthetic_indices):
                raise IndexError(
                    f"Synthetic global index exceeds dataset size {synthetic_metadata.total_frames}: "
                    f"{synthetic_indices}"
                )
            allowed_episodes = {int(value) for value in synthetic_source["episode_ids"]}
            observed_episodes = {
                int(synthetic_dataset.hf_dataset[index]["episode_index"])
                for index in synthetic_indices
            }
            if not observed_episodes.issubset(allowed_episodes):
                raise ValueError(
                    f"Explicit synthetic indices came from episodes {sorted(observed_episodes)}, "
                    f"outside declared episodes {sorted(allowed_episodes)}"
                )
        else:
            selection_config = synthetic_source.get("selection", {})
            sampling_mode = str(selection_config.get("sampling_mode", "uniform"))
            selected_samples_per_episode = int(
                selection_config.get(
                    "samples_per_episode",
                    synthetic_source.get("samples_per_episode", samples_per_episode),
                )
            )
            synthetic_indices_by_episode, sampling_records = sample_indices_by_episode(
                synthetic_metadata,
                source_episode_ids,
                selected_samples_per_episode,
                dataset_obj=synthetic_dataset,
                sampling_mode=sampling_mode,
                gripper_action_index=int(
                    selection_config.get("gripper_action_index", -1)
                ),
            )
            synthetic_indices = flatten_episode_indices(synthetic_indices_by_episode)
            if sampling_mode in {
                "gripper_event_balanced",
                "preselected_all",
            }:
                source_pair_sampling_audit = {
                    "schema": "fedsplat.source_pair_sampling_audit.v1",
                    "sampling_mode": sampling_mode,
                    "episode_count": len(sampling_records),
                    "samples_per_episode": selected_samples_per_episode,
                    "sample_count": len(synthetic_indices),
                    "all_indices_unique_within_episode": all(
                        len(record["selected_global_indices"])
                        == len(set(record["selected_global_indices"]))
                        for record in sampling_records
                    ),
                    "all_gripper_transitions_exactly_covered": (
                        all(
                            record[
                                "maximum_nearest_transition_distance_frames"
                            ]
                            == 0
                            for record in sampling_records
                        )
                        if sampling_mode == "gripper_event_balanced"
                        else None
                    ),
                    "all_preselected_frames_used": (
                        all(
                            record["all_preselected_frames_used"]
                            for record in sampling_records
                        )
                        if sampling_mode == "preselected_all"
                        else None
                    ),
                    "all_episodes_cover_start_and_end": all(
                        record["minimum_selected_phase_fraction"] == 0.0
                        and record["maximum_selected_phase_fraction"] == 1.0
                        for record in sampling_records
                    ),
                    "episodes": sampling_records,
                }
                canonical_sampling = json.dumps(
                    source_pair_sampling_audit,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                source_pair_sampling_audit["audit_sha256"] = hashlib.sha256(
                    canonical_sampling
                ).hexdigest()
                (out_dir / "source_pair_sampling_audit.json").write_text(
                    json.dumps(source_pair_sampling_audit, indent=2) + "\n",
                    encoding="utf-8",
                )
        if not dynamic_batches:
            synthetic_batch = preprocessor(collate(synthetic_dataset, synthetic_indices))

    quality_evidence_path = synthetic_source.get("quality_manifest")
    quality_evidence = None
    trajectory_evidence: dict[int, dict[str, Any]] = {}
    if quality_evidence_path:
        quality_evidence = json.loads(Path(quality_evidence_path).read_text(encoding="utf-8"))
        aggregate_evidence = quality_evidence.get("aggregate", quality_evidence)
        synthetic_source["feasibility_checks"] = aggregate_evidence.get("feasibility_checks", {})
        synthetic_source["quality_scores"] = aggregate_evidence.get("quality_scores", {})
        if not synthetic_source["feasibility_checks"] or not synthetic_source["quality_scores"]:
            raise ValueError("Automatic synthetic quality manifest lacks feasibility_checks or quality_scores")
        for item in quality_evidence.get("episodes", []):
            if "episode_index" not in item:
                continue
            episode_id = int(item["episode_index"])
            episode_gate = feasibility_gate(item.get("feasibility_checks", {}))
            minimum_render_coverage = float(
                synthetic_source.get("selection", {}).get("minimum_render_coverage", 0.0)
            )
            observed_render_coverage = float(item.get("minimum_coverage", 1.0))
            render_coverage_pass = observed_render_coverage >= minimum_render_coverage
            episode_quality = item.get("quality_score")
            if episode_quality is None:
                episode_quality = geometric_quality_score(
                    item.get("quality_scores", {"declared": 1.0}),
                    weights=synthetic_source.get("quality_weights"),
                    feasibility=episode_gate,
                )
            trajectory_evidence[episode_id] = {
                "episode_id": episode_id,
                "source_demo": item.get("source_demo", item.get("demo")),
                "source_split": item.get("source_split"),
                "variation": str(item.get("variation", "unspecified")),
                "feasible": bool(episode_gate.accepted and render_coverage_pass),
                "quality": float(episode_quality) if render_coverage_pass else 0.0,
                "minimum_render_coverage": observed_render_coverage,
                "render_coverage_pass": render_coverage_pass,
            }
        if gradient_mode in {
            "augmentation_portfolio",
            "crossfit",
            "factor_paired",
            *PAIRED_DISTILL_MODES,
        }:
            missing = sorted(
                set(
                    candidate_episode_ids
                    + (
                        anchor_episode_ids
                        if gradient_mode in {"crossfit", *PAIRED_DISTILL_MODES}
                        else []
                    )
                )
                - set(trajectory_evidence)
            )
            if missing:
                raise ValueError(
                    f"{gradient_mode} quality manifest lacks episode metadata: {missing}"
                )
        if gradient_mode == "factor_paired":
            expected_variations = set(spec["factor_variations"])
            observed_variations = {
                trajectory_evidence[episode_id]["variation"]
                for episode_id in candidate_episode_ids
            }
            if observed_variations != expected_variations:
                raise ValueError(
                    "factor_paired variation contract mismatch: "
                    f"expected={sorted(expected_variations)}, "
                    f"observed={sorted(observed_variations)}"
                )
    synthetic_objective = str(spec.get("synthetic_objective", "action_bc"))
    paired_real_index_by_synthetic_index: dict[int, int] = {}
    source_pairing_audit: dict[str, Any] | None = None
    if synthetic_objective == "source_paired_consistency":
        if synthetic_dataset is None or synthetic_metadata is None or quality_evidence is None:
            raise ValueError("Source-paired consistency requires loaded real/synthetic datasets and quality evidence")
        real_episode_by_demo = source_demo_episode_map(
            quality_evidence,
            int(metadata.total_episodes),
        )
        state_max_abs_error = 0.0
        action_max_abs_error = 0.0
        verified_episode_pairs: set[tuple[int, int]] = set()
        for synthetic_episode_id, sampled_indices in synthetic_indices_by_episode.items():
            evidence = trajectory_evidence.get(synthetic_episode_id)
            if evidence is None or not evidence.get("source_demo"):
                raise ValueError(f"Synthetic episode {synthetic_episode_id} lacks source-demo provenance")
            source_demo = str(evidence["source_demo"])
            if source_demo not in real_episode_by_demo:
                raise ValueError(
                    f"Synthetic episode {synthetic_episode_id} references unknown source demo {source_demo}"
                )
            real_episode_id = real_episode_by_demo[source_demo]
            owner_local_split = spec.get("owner_local_split")
            if owner_local_split is None:
                expected_real_episodes = (
                    set(train_episodes)
                    if evidence.get("source_split") == "train"
                    else set(validation_episodes)
                )
            else:
                source_split = evidence.get("source_split")
                if source_split == "train":
                    expected_real_episodes = set(
                        owner_local_split["source_train_episodes"]
                    )
                elif source_split == "calibration":
                    expected_real_episodes = set(
                        owner_local_split["source_validation_episodes"]
                    )
                else:
                    raise ValueError(
                        f"Unsupported source split for owner-local partition: "
                        f"{source_split}"
                    )
            if real_episode_id not in expected_real_episodes:
                raise ValueError(
                    f"Synthetic episode {synthetic_episode_id} source split does not match real episode "
                    f"{real_episode_id}"
                )
            if real_episode_id not in set(
                train_episodes + validation_episodes
            ):
                raise ValueError(
                    f"Synthetic episode {synthetic_episode_id} is outside "
                    f"owner-local episode membership"
                )
            real_episode = metadata.episodes[real_episode_id]
            real_start = int(real_episode["dataset_from_index"])
            real_stop = int(real_episode["dataset_to_index"])
            synthetic_episode = synthetic_metadata.episodes[synthetic_episode_id]
            synthetic_frames = int(synthetic_episode["dataset_to_index"]) - int(
                synthetic_episode["dataset_from_index"]
            )
            if synthetic_frames != real_stop - real_start:
                raise ValueError(
                    f"Source-paired episode length mismatch: synthetic={synthetic_episode_id} "
                    f"({synthetic_frames}), real={real_episode_id} ({real_stop - real_start})"
                )
            verified_episode_pairs.add((synthetic_episode_id, real_episode_id))
            for synthetic_index in sampled_indices:
                synthetic_row = synthetic_dataset.hf_dataset[synthetic_index]
                frame_index = int(synthetic_row["frame_index"])
                real_index = real_start + frame_index
                if not real_start <= real_index < real_stop:
                    raise ValueError(
                        f"Synthetic frame {frame_index} exceeds source real episode {real_episode_id}"
                    )
                real_row = dataset.hf_dataset[real_index]
                if int(real_row["frame_index"]) != frame_index:
                    raise ValueError("Source-paired real/synthetic frame indices do not match")
                state_error = float(np.max(np.abs(
                    np.asarray(synthetic_row["observation.state"], dtype=np.float64)
                    - np.asarray(real_row["observation.state"], dtype=np.float64)
                )))
                action_error = float(np.max(np.abs(
                    np.asarray(synthetic_row["action"], dtype=np.float64)
                    - np.asarray(real_row["action"], dtype=np.float64)
                )))
                state_max_abs_error = max(state_max_abs_error, state_error)
                action_max_abs_error = max(action_max_abs_error, action_error)
                if state_error > 1e-6 or action_error > 1e-6:
                    raise ValueError(
                        "Source-paired real/synthetic state-action mismatch: "
                        f"synthetic_index={synthetic_index}, state_error={state_error}, "
                        f"action_error={action_error}"
                    )
                paired_real_index_by_synthetic_index[synthetic_index] = real_index
        if set(paired_real_index_by_synthetic_index) != set(synthetic_indices):
            raise ValueError("Source-paired consistency did not map every sampled synthetic frame")
        source_pairing_audit = {
            "schema": "fedsplat.source_paired_gaussian_audit.v1",
            "verified": True,
            "source_demo_count": len(real_episode_by_demo),
            "episode_pair_count": len(verified_episode_pairs),
            "sample_pair_count": len(paired_real_index_by_synthetic_index),
            "state_max_abs_error": state_max_abs_error,
            "action_max_abs_error": action_max_abs_error,
            "sampling_mode": (
                source_pair_sampling_audit["sampling_mode"]
                if source_pair_sampling_audit is not None
                else "uniform"
            ),
            "sampling_audit_sha256": (
                source_pair_sampling_audit["audit_sha256"]
                if source_pair_sampling_audit is not None
                else None
            ),
            "episode_pairs": [
                {"synthetic_episode": synthetic_id, "real_episode": real_id}
                for synthetic_id, real_id in sorted(verified_episode_pairs)
            ],
        }
        (out_dir / "source_pairing_audit.json").write_text(
            json.dumps(source_pairing_audit, indent=2) + "\n", encoding="utf-8"
        )
    if source_type == "none":
        gate = feasibility_gate({name: True for name in ("schema", "joint_limits", "ik", "collision", "visibility", "instruction_action")})
        quality_score = 1.0
    else:
        gate = feasibility_gate(synthetic_source.get("feasibility_checks", {}))
        quality_score = geometric_quality_score(
            synthetic_source.get("quality_scores", {"declared": 1.0}),
            weights=synthetic_source.get("quality_weights"),
            feasibility=gate,
        )
    policy = policy.to(device)
    for parameter in policy.parameters():
        parameter.requires_grad = False

    available_linear_names = [name for name, module in policy.named_modules() if isinstance(module, nn.Linear)]
    patterns = spec.get("target_module_patterns", [])
    exact_targets = spec.get("target_modules", [] if patterns else list(DEFAULT_TARGET_MODULES))
    resolved_targets = resolve_target_module_names(available_linear_names, exact_targets, patterns)
    adapters: dict[str, LoRALinear] = {}
    for module_name in resolved_targets:
        parent_name, attribute = module_name.rsplit(".", 1)
        parent = policy.get_submodule(parent_name)
        base = getattr(parent, attribute)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRA target is not torch.nn.Linear: {module_name}")
        wrapper = LoRALinear(base, rank=int(spec["rank"]), alpha=float(spec["alpha"]))
        setattr(parent, attribute, wrapper)
        adapters[module_name] = wrapper

    initial_adapter_path = spec.get("initial_adapter_update")
    if initial_adapter_path:
        initial_update = load_client_update(Path(initial_adapter_path))
        initial_layers = {layer.layer_name: layer for layer in initial_update.layers}
        missing_initial_layers = set(adapters) - set(initial_layers)
        if missing_initial_layers:
            raise ValueError(
                "Initial/global adapter lacks local target layers: "
                f"missing={sorted(missing_initial_layers)}, observed={sorted(initial_layers)}"
            )
        for module_name, adapter in adapters.items():
            layer = initial_layers[module_name]
            if layer.a.shape != tuple(adapter.a.shape) or layer.b.shape != tuple(adapter.b.shape):
                raise ValueError(
                    f"Initial adapter shape mismatch for {module_name}: "
                    f"A{layer.a.shape}/B{layer.b.shape} versus A{tuple(adapter.a.shape)}/B{tuple(adapter.b.shape)}"
                )
            with torch.no_grad():
                adapter.a.copy_(torch.as_tensor(layer.a, dtype=adapter.a.dtype, device=adapter.a.device))
                adapter.b.copy_(
                    torch.as_tensor(layer.b / adapter.scaling, dtype=adapter.b.dtype, device=adapter.b.device)
                )

    local_trainable_patterns = list(spec.get("local_trainable_module_patterns", []))
    local_trainable_modules = (
        set(resolve_module_scope(list(adapters), local_trainable_patterns))
        if local_trainable_patterns
        else set(adapters)
    )
    for module_name, adapter in adapters.items():
        enabled = module_name in local_trainable_modules
        adapter.a.requires_grad_(enabled)
        adapter.b.requires_grad_(enabled)

    trainable_entries = [
        (module_name, factor_name, parameter)
        for module_name, adapter in adapters.items()
        if module_name in local_trainable_modules
        for factor_name, parameter in (("a", adapter.a), ("b", adapter.b))
    ]
    trainable = [parameter for _module_name, _factor_name, parameter in trainable_entries]
    unexpected_trainable = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and all(parameter is not target for target in trainable)
    ]
    if unexpected_trainable:
        raise RuntimeError(f"Unexpected trainable parameters outside LoRA adapters: {unexpected_trainable}")
    initial_trainable = [parameter.detach().clone() for parameter in trainable]
    initial_adapter_parameters = {
        module_name: (
            adapter.a.detach().clone(),
            adapter.b.detach().clone(),
        )
        for module_name, adapter in adapters.items()
    }

    synthetic_gradient_scope = str(spec.get("synthetic_gradient_scope", "all"))
    policy_anchor_mode = str(spec.get("policy_anchor_mode", "none"))
    representation_anchor_mode = str(spec.get("representation_anchor_mode", "none"))
    representation_anchor_weight = float(spec.get("representation_anchor_weight", 1.0))
    max_anchor_to_bc_perception_norm = float(
        spec.get("max_anchor_to_bc_perception_norm", 0.25)
    )
    paired_action_consistency_weight = float(
        spec.get("paired_action_consistency_weight", 1.0)
    )
    paired_feature_consistency_weight = float(
        spec.get("paired_feature_consistency_weight", 0.25)
    )
    paired_synthetic_bc_weight = float(spec.get("paired_synthetic_bc_weight", 0.0))
    paired_canonical_action_identity_weight = float(
        spec.get("paired_canonical_action_identity_weight", 0.0)
    )
    paired_canonical_feature_identity_weight = float(
        spec.get("paired_canonical_feature_identity_weight", 0.0)
    )
    paired_teacher_mode = str(spec.get("paired_teacher_mode", "current_policy"))
    paired_teacher_reference = str(
        spec.get(
            "paired_teacher_reference",
            "initial_adapter" if initial_adapter_path else "pretrained_base",
        )
    )
    paired_action_consistency_mode = str(
        spec.get("paired_action_consistency_mode", "flow_velocity")
    )
    factor_counterfactual_mode = str(
        spec.get("factor_counterfactual_mode", "real_only")
    )
    perception_patterns = list(spec.get("perception_module_patterns", []))
    if (
        synthetic_gradient_scope == "perception"
        or policy_anchor_mode != "none"
        or representation_anchor_mode != "none"
    ):
        perception_modules = set(resolve_module_scope(list(adapters), perception_patterns))
    else:
        perception_modules = set(adapters)
    synthetic_parameter_scope = tuple(
        synthetic_gradient_scope == "all" or module_name in perception_modules
        for module_name, _factor_name, _parameter in trainable_entries
    )
    if not any(synthetic_parameter_scope):
        raise RuntimeError("Synthetic-gradient scope contains no trainable parameters")
    perception_parameter_scope = tuple(
        module_name in perception_modules
        for module_name, _factor_name, _parameter in trainable_entries
    )
    if representation_anchor_mode != "none" and not any(perception_parameter_scope):
        raise RuntimeError("Representation-anchor scope contains no trainable parameters")
    if paired_teacher_mode == "initial_global":
        if paired_teacher_reference == "initial_adapter" and not initial_adapter_path:
            raise RuntimeError(
                "paired_teacher_reference=initial_adapter requires an initial adapter"
            )
        if paired_teacher_reference == "pretrained_base" and initial_adapter_path:
            raise RuntimeError(
                "paired_teacher_reference=pretrained_base requires no initial adapter"
            )
    if (
        gradient_mode in PAIRED_DISTILL_MODES
        and not local_trainable_modules.issubset(perception_modules)
    ):
        raise RuntimeError(
            "Frozen-teacher paired distillation must restrict local trainable "
            "modules to the declared perception scope"
        )
    sharpness_mode = str(spec.get("sharpness_mode", "none"))
    sam_rho = float(spec.get("sam_rho", 0.03))

    batch_size = int(spec.get("batch_size", samples_per_episode))
    evaluation_batch_size = int(spec.get("evaluation_batch_size", batch_size))
    raw_item_caches: dict[int, dict[int, Any]] = {}

    def move_to_device(value):
        if torch.is_tensor(value):
            return value.to(device=device, non_blocking=True)
        if isinstance(value, dict):
            return {key: move_to_device(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(move_to_device(item) for item in value)
        if isinstance(value, list):
            return [move_to_device(item) for item in value]
        return value

    def prepared_batch(dataset_obj, indices: list[int]):
        cache = raw_item_caches.setdefault(id(dataset_obj), {})
        items = []
        for index in indices:
            if index not in cache:
                cache[index] = dataset_obj[index]
            items.append(cache[index])
        processed = preprocessor(torch.utils.data.default_collate(items))
        return move_to_device(processed) if dynamic_batches else processed

    def deterministic_loss(batch, evaluation_seed: int) -> float:
        policy.eval()
        torch.manual_seed(evaluation_seed)
        with torch.no_grad():
            loss, _ = policy(batch)
        return float(loss.detach().cpu())

    def loss_for_indices(dataset_obj, indices: list[int], evaluation_seed: int) -> float:
        if not indices:
            raise ValueError("Loss evaluation requires at least one sample")
        weighted_loss = 0.0
        for offset in range(0, len(indices), evaluation_batch_size):
            chunk = indices[offset : offset + evaluation_batch_size]
            weighted_loss += deterministic_loss(
                prepared_batch(dataset_obj, chunk),
                evaluation_seed + offset,
            ) * len(chunk)
        return weighted_loss / len(indices)

    connector_module = policy.model.vlm_with_expert.get_vlm_model().connector
    action_output_module = policy.model.action_out_proj

    def first_tensor(value):
        if torch.is_tensor(value):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, dict):
            for item in value.values():
                tensor = first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def captured_policy_forward(batch, gradient_seed: int, *, noise=None, flow_time=None, no_grad=False):
        visual_features = []
        action_outputs = []

        def capture_visual(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None:
                raise RuntimeError("Visual connector produced no tensor output")
            visual_features.append(tensor)

        def capture_action(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None:
                raise RuntimeError("Action projection produced no tensor output")
            action_outputs.append(tensor)

        handles = [
            connector_module.register_forward_hook(capture_visual),
            action_output_module.register_forward_hook(capture_action),
        ]
        try:
            torch.manual_seed(gradient_seed)
            if no_grad:
                with torch.no_grad():
                    loss, _ = policy(batch, noise=noise, time=flow_time)
            else:
                loss, _ = policy(batch, noise=noise, time=flow_time)
        finally:
            for handle in handles:
                handle.remove()
        if not visual_features or len(action_outputs) != 1:
            raise RuntimeError(
                "Policy capture contract failed: "
                f"visual={len(visual_features)}, action={len(action_outputs)}"
            )
        return loss, visual_features, action_outputs[0]

    def frozen_initial_policy_forward(batch, gradient_seed: int, *, noise, flow_time):
        current = {
            module_name: (
                adapter.a.detach().clone(),
                adapter.b.detach().clone(),
            )
            for module_name, adapter in adapters.items()
        }
        try:
            with torch.no_grad():
                for module_name, adapter in adapters.items():
                    initial_a, initial_b = initial_adapter_parameters[module_name]
                    adapter.a.copy_(initial_a)
                    adapter.b.copy_(initial_b)
            _loss, features, action = captured_policy_forward(
                batch,
                gradient_seed,
                noise=noise,
                flow_time=flow_time,
                no_grad=True,
            )
            return [value.detach() for value in features], action.detach()
        finally:
            with torch.no_grad():
                for module_name, adapter in adapters.items():
                    current_a, current_b = current[module_name]
                    adapter.a.copy_(current_a)
                    adapter.b.copy_(current_b)

    def pretrained_visual_features(batch):
        scaling_snapshot = {name: adapter.scaling for name, adapter in adapters.items()}
        try:
            for adapter in adapters.values():
                adapter.scaling = 0.0
            images, _image_masks = policy.prepare_images(batch)
            with torch.no_grad():
                return [
                    policy.model.vlm_with_expert.embed_image(image).detach()
                    for image in images
                ]
        finally:
            for name, scaling in scaling_snapshot.items():
                adapters[name].scaling = scaling

    def feature_cosine_loss(student_features, target_features, *, pooled: bool):
        if len(student_features) != len(target_features) or not student_features:
            raise RuntimeError("Feature-consistency views do not match")
        losses = []
        for student, target in zip(student_features, target_features, strict=True):
            if student.shape != target.shape:
                raise RuntimeError(
                    f"Feature-consistency shape mismatch: {tuple(student.shape)} != {tuple(target.shape)}"
                )
            student_value = student.float()
            target_value = target.detach().float()
            if pooled:
                student_value = student_value.mean(dim=-2)
                target_value = target_value.mean(dim=-2)
            student_value = functional.normalize(student_value, dim=-1)
            target_value = functional.normalize(target_value, dim=-1)
            losses.append(1.0 - (student_value * target_value).sum(dim=-1).mean())
        return torch.stack(losses).mean()

    def normalized_action_consistency_loss(student, target):
        if student.shape != target.shape:
            raise RuntimeError(
                f"Action-consistency shape mismatch: {tuple(student.shape)} != {tuple(target.shape)}"
            )
        student_value = student.float()
        target_value = target.detach().float()
        target_energy = target_value.square().mean().clamp_min(1e-4)
        return (student_value - target_value).square().mean() / target_energy

    def executed_and_chunk_consistency_losses(student, target):
        original_action_dim = int(policy.config.action_feature.shape[0])
        student_actions = student[..., :original_action_dim]
        target_actions = target[..., :original_action_dim]
        executed = normalized_action_consistency_loss(
            student_actions[:, 0], target_actions[:, 0]
        )
        full_chunk = normalized_action_consistency_loss(
            student_actions, target_actions
        )
        return executed, full_chunk

    def captured_inference_prefix(batch, *, no_grad: bool):
        visual_features = []

        def capture_visual(_module, _inputs, output):
            tensor = first_tensor(output)
            if tensor is None:
                raise RuntimeError("Visual connector produced no tensor output")
            visual_features.append(tensor)

        handle = connector_module.register_forward_hook(capture_visual)

        def build_prefix():
            images, image_masks = policy.prepare_images(batch)
            state = policy.prepare_state(batch)
            language_tokens = batch[OBS_LANGUAGE_TOKENS]
            language_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
            prefix_embs, prefix_pad_masks, prefix_att_masks = (
                policy.model.embed_prefix(
                    images,
                    image_masks,
                    language_tokens,
                    language_masks,
                    state=state,
                )
            )
            prefix_attention = make_att_2d_masks(
                prefix_pad_masks, prefix_att_masks
            )
            prefix_positions = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = policy.model.vlm_with_expert.forward(
                attention_mask=prefix_attention,
                position_ids=prefix_positions,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=policy.model.config.use_cache,
                fill_kv_cache=True,
            )
            return prefix_pad_masks, past_key_values

        try:
            if no_grad:
                with torch.no_grad():
                    prefix_pad_masks, past_key_values = build_prefix()
            else:
                prefix_pad_masks, past_key_values = build_prefix()
        finally:
            handle.remove()
        if not visual_features:
            raise RuntimeError("Inference prefix produced no visual features")
        return prefix_pad_masks, past_key_values, visual_features

    def captured_inference_trajectory(batch, *, noise, no_grad: bool):
        prefix_pad_masks, past_key_values, visual_features = (
            captured_inference_prefix(batch, no_grad=no_grad)
        )
        num_steps = int(policy.model.config.num_steps)
        dt = -1.0 / num_steps

        def denoise_trajectory():
            x_t = noise
            states = []
            velocities = []
            for step in range(num_steps):
                time_value = 1.0 + step * dt
                time_tensor = torch.tensor(
                    time_value,
                    dtype=torch.float32,
                    device=x_t.device,
                ).expand(x_t.shape[0])
                states.append(x_t.detach())
                velocity = policy.model.denoise_step(
                    x_t=x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=time_tensor,
                )
                velocities.append(velocity.detach() if no_grad else velocity)
                x_t = x_t + dt * velocity
            return states, velocities, x_t

        if no_grad:
            with torch.no_grad():
                states, velocities, final_action = denoise_trajectory()
        else:
            states, velocities, final_action = denoise_trajectory()
        return {
            "visual_features": [
                value.detach() if no_grad else value for value in visual_features
            ],
            "states": states,
            "velocities": velocities,
            "final_action": final_action.detach() if no_grad else final_action,
        }

    def captured_teacher_forced_denoise(
        batch,
        *,
        teacher_state,
        denoise_step_index: int,
    ):
        prefix_pad_masks, past_key_values, visual_features = (
            captured_inference_prefix(batch, no_grad=False)
        )
        num_steps = int(policy.model.config.num_steps)
        time_value = 1.0 - float(denoise_step_index) / float(num_steps)
        time_tensor = torch.tensor(
            time_value,
            dtype=torch.float32,
            device=teacher_state.device,
        ).expand(teacher_state.shape[0])
        velocity = policy.model.denoise_step(
            x_t=teacher_state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            timestep=time_tensor,
        )
        return visual_features, velocity

    def frozen_initial_inference_trajectory(batch, *, noise):
        current = {
            module_name: (
                adapter.a.detach().clone(),
                adapter.b.detach().clone(),
            )
            for module_name, adapter in adapters.items()
        }
        try:
            with torch.no_grad():
                for module_name, adapter in adapters.items():
                    initial_a, initial_b = initial_adapter_parameters[module_name]
                    adapter.a.copy_(initial_a)
                    adapter.b.copy_(initial_b)
            return captured_inference_trajectory(
                batch,
                noise=noise,
                no_grad=True,
            )
        finally:
            with torch.no_grad():
                for module_name, adapter in adapters.items():
                    current_a, current_b = current[module_name]
                    adapter.a.copy_(current_a)
                    adapter.b.copy_(current_b)

    def gradient_values(loss, parameter_scope: tuple[bool, ...], *, retain_graph: bool = False):
        selected_positions = [index for index, enabled in enumerate(parameter_scope) if enabled]
        if not selected_positions:
            raise ValueError("Gradient parameter scope must not be empty")
        selected_values = torch.autograd.grad(
            loss,
            [trainable[index] for index in selected_positions],
            allow_unused=True,
            retain_graph=retain_graph,
        )
        selected_by_position = dict(zip(selected_positions, selected_values, strict=True))
        return tuple(
            torch.zeros_like(parameter)
            if index not in selected_by_position or selected_by_position[index] is None
            else selected_by_position[index]
            for index, parameter in enumerate(trainable)
        )

    def scoped_gradient_norm(values, parameter_scope: tuple[bool, ...]) -> float:
        return math.sqrt(sum(
            float(value.detach().float().square().sum().cpu())
            for value, enabled in zip(values, parameter_scope, strict=True)
            if enabled
        ))

    def gradients(
        batch,
        gradient_seed: int,
        parameter_scope: tuple[bool, ...] | None = None,
        *,
        synthetic: bool = False,
        paired_reference_batch=None,
        synthetic_objective_override: str | None = None,
    ):
        policy.train()
        scope = parameter_scope or tuple(True for _ in trainable)
        active_synthetic_objective = (
            synthetic_objective_override or synthetic_objective
        )
        metrics: dict[str, Any] = {
            "objective": "action_bc",
            "bc_loss": None,
            "representation_anchor_loss": None,
            "representation_anchor_gradient_norm": 0.0,
            "bc_perception_gradient_norm": 0.0,
            "effective_representation_anchor_weight": 0.0,
            "paired_action_consistency_loss": None,
            "paired_full_chunk_consistency_loss": None,
            "paired_inference_denoise_step": None,
            "paired_feature_consistency_loss": None,
            "paired_canonical_action_identity_loss": None,
            "paired_canonical_full_chunk_identity_loss": None,
            "paired_canonical_feature_identity_loss": None,
        }

        if synthetic and active_synthetic_objective == "source_paired_consistency":
            if paired_reference_batch is None:
                raise ValueError("Source-paired synthetic gradients require a paired real batch")
            with torch.no_grad():
                actions = policy.prepare_action(batch)
                torch.manual_seed(gradient_seed)
                shared_noise = policy.model.sample_noise(actions.shape, actions.device)
            if paired_action_consistency_mode == "inference_trajectory":
                policy.eval()
                if paired_teacher_mode != "initial_global":
                    raise RuntimeError(
                        "Inference-trajectory distillation requires the frozen initial teacher"
                    )
                teacher = frozen_initial_inference_trajectory(
                    paired_reference_batch,
                    noise=shared_noise,
                )
                with torch.no_grad():
                    shared_time = policy.model.sample_time(
                        actions.shape[0], actions.device
                    )
                denoise_step_index = deterministic_inference_step(
                    gradient_seed, int(policy.model.config.num_steps)
                )
                student_visual, student_velocity = captured_teacher_forced_denoise(
                    batch,
                    teacher_state=teacher["states"][denoise_step_index],
                    denoise_step_index=denoise_step_index,
                )
                target_velocity = teacher["velocities"][denoise_step_index]
                action_consistency, full_chunk_consistency = (
                    executed_and_chunk_consistency_losses(
                        student_velocity, target_velocity
                    )
                )
                feature_consistency = feature_cosine_loss(
                    student_visual,
                    teacher["visual_features"],
                    pooled=True,
                )
                canonical_action_identity = torch.zeros(
                    (), device=student_velocity.device
                )
                canonical_full_chunk_identity = torch.zeros(
                    (), device=student_velocity.device
                )
                canonical_feature_identity = torch.zeros(
                    (), device=student_velocity.device
                )
                if (
                    paired_canonical_action_identity_weight > 0.0
                    or paired_canonical_feature_identity_weight > 0.0
                ):
                    canonical_visual, canonical_velocity = (
                        captured_teacher_forced_denoise(
                            paired_reference_batch,
                            teacher_state=teacher["states"][denoise_step_index],
                            denoise_step_index=denoise_step_index,
                        )
                    )
                    (
                        canonical_action_identity,
                        canonical_full_chunk_identity,
                    ) = executed_and_chunk_consistency_losses(
                        canonical_velocity, target_velocity
                    )
                    canonical_feature_identity = feature_cosine_loss(
                        canonical_visual,
                        teacher["visual_features"],
                        pooled=True,
                    )
                if paired_synthetic_bc_weight > 0.0:
                    bc_loss, _bc_visual, _bc_action = captured_policy_forward(
                        batch,
                        gradient_seed,
                        noise=shared_noise,
                        flow_time=shared_time,
                    )
                else:
                    bc_loss = torch.zeros((), device=student_velocity.device)
                objective_loss = (
                    paired_action_consistency_weight * action_consistency
                    + paired_feature_consistency_weight * feature_consistency
                    + paired_synthetic_bc_weight * bc_loss
                    + paired_canonical_action_identity_weight
                    * canonical_action_identity
                    + paired_canonical_feature_identity_weight
                    * canonical_feature_identity
                )
                values = gradient_values(objective_loss, scope)
                metrics.update(
                    {
                        "objective": "source_paired_inference_trajectory",
                        "bc_loss": float(bc_loss.detach().cpu()),
                        "paired_action_consistency_loss": float(
                            action_consistency.detach().cpu()
                        ),
                        "paired_full_chunk_consistency_loss": float(
                            full_chunk_consistency.detach().cpu()
                        ),
                        "paired_inference_denoise_step": denoise_step_index,
                        "paired_feature_consistency_loss": float(
                            feature_consistency.detach().cpu()
                        ),
                        "paired_canonical_action_identity_loss": float(
                            canonical_action_identity.detach().cpu()
                        ),
                        "paired_canonical_full_chunk_identity_loss": float(
                            canonical_full_chunk_identity.detach().cpu()
                        ),
                        "paired_canonical_feature_identity_loss": float(
                            canonical_feature_identity.detach().cpu()
                        ),
                    }
                )
                return {
                    "loss": objective_loss,
                    "gradients": values,
                    "metrics": metrics,
                }
            with torch.no_grad():
                shared_time = policy.model.sample_time(actions.shape[0], actions.device)
            if paired_teacher_mode == "initial_global":
                target_visual, target_action = frozen_initial_policy_forward(
                    paired_reference_batch,
                    gradient_seed,
                    noise=shared_noise,
                    flow_time=shared_time,
                )
            else:
                _target_loss, target_visual, target_action = captured_policy_forward(
                    paired_reference_batch,
                    gradient_seed,
                    noise=shared_noise,
                    flow_time=shared_time,
                    no_grad=True,
                )
            bc_loss, student_visual, student_action = captured_policy_forward(
                batch,
                gradient_seed,
                noise=shared_noise,
                flow_time=shared_time,
            )
            action_consistency = normalized_action_consistency_loss(student_action, target_action)
            feature_consistency = feature_cosine_loss(
                student_visual, target_visual, pooled=True
            )
            canonical_action_identity = torch.zeros((), device=student_action.device)
            canonical_feature_identity = torch.zeros((), device=student_action.device)
            if (
                paired_canonical_action_identity_weight > 0.0
                or paired_canonical_feature_identity_weight > 0.0
            ):
                (
                    _canonical_loss,
                    canonical_student_visual,
                    canonical_student_action,
                ) = captured_policy_forward(
                    paired_reference_batch,
                    gradient_seed,
                    noise=shared_noise,
                    flow_time=shared_time,
                )
                canonical_action_identity = normalized_action_consistency_loss(
                    canonical_student_action, target_action
                )
                canonical_feature_identity = feature_cosine_loss(
                    canonical_student_visual, target_visual, pooled=True
                )
            objective_loss = (
                paired_action_consistency_weight * action_consistency
                + paired_feature_consistency_weight * feature_consistency
                + paired_synthetic_bc_weight * bc_loss
                + paired_canonical_action_identity_weight
                * canonical_action_identity
                + paired_canonical_feature_identity_weight
                * canonical_feature_identity
            )
            values = gradient_values(objective_loss, scope)
            metrics.update({
                "objective": active_synthetic_objective,
                "bc_loss": float(bc_loss.detach().cpu()),
                "paired_action_consistency_loss": float(action_consistency.detach().cpu()),
                "paired_feature_consistency_loss": float(feature_consistency.detach().cpu()),
                "paired_canonical_action_identity_loss": float(
                    canonical_action_identity.detach().cpu()
                ),
                "paired_canonical_feature_identity_loss": float(
                    canonical_feature_identity.detach().cpu()
                ),
            })
            return {"loss": objective_loss, "gradients": values, "metrics": metrics}

        teacher_features = None
        if not synthetic and representation_anchor_mode == "pretrained_connector":
            teacher_features = pretrained_visual_features(batch)
        loss, student_features, _action_output = captured_policy_forward(batch, gradient_seed)
        representation_loss = None
        if teacher_features is not None:
            representation_loss = feature_cosine_loss(
                student_features, teacher_features, pooled=False
            )
        primary_values = gradient_values(
            loss,
            scope,
            retain_graph=representation_loss is not None,
        )
        values = primary_values
        effective_anchor_weight = 0.0
        anchor_gradient_norm = 0.0
        bc_perception_norm = scoped_gradient_norm(primary_values, perception_parameter_scope)
        if representation_loss is not None:
            anchor_scope = tuple(
                enabled and perception_enabled
                for enabled, perception_enabled in zip(scope, perception_parameter_scope, strict=True)
            )
            anchor_values = gradient_values(representation_loss, anchor_scope)
            anchor_gradient_norm = scoped_gradient_norm(anchor_values, anchor_scope)
            weighted_anchor_norm = representation_anchor_weight * anchor_gradient_norm
            maximum_anchor_norm = max_anchor_to_bc_perception_norm * bc_perception_norm
            trust_scale = min(1.0, maximum_anchor_norm / (weighted_anchor_norm + 1e-12))
            effective_anchor_weight = representation_anchor_weight * trust_scale
            values = tuple(
                primary + effective_anchor_weight * anchor
                for primary, anchor in zip(primary_values, anchor_values, strict=True)
            )
        objective_loss = (
            loss + effective_anchor_weight * representation_loss
            if representation_loss is not None else loss
        )
        metrics.update({
            "bc_loss": float(loss.detach().cpu()),
            "representation_anchor_loss": (
                float(representation_loss.detach().cpu())
                if representation_loss is not None else None
            ),
            "representation_anchor_gradient_norm": anchor_gradient_norm,
            "bc_perception_gradient_norm": bc_perception_norm,
            "effective_representation_anchor_weight": effective_anchor_weight,
        })
        return {"loss": objective_loss, "gradients": values, "metrics": metrics}

    def gradients_for_indices(
        dataset_obj,
        indices: list[int],
        gradient_seed: int,
        parameter_scope: tuple[bool, ...] | None = None,
        *,
        synthetic: bool = False,
        paired_reference_indices: list[int] | None = None,
    ):
        paired_reference_batch = (
            prepared_batch(dataset, paired_reference_indices)
            if paired_reference_indices is not None else None
        )
        return gradients(
            prepared_batch(dataset_obj, indices),
            gradient_seed,
            parameter_scope,
            synthetic=synthetic,
            paired_reference_batch=paired_reference_batch,
        )

    def flatten(values, parameter_scope: tuple[bool, ...] | None = None) -> np.ndarray:
        scope = parameter_scope or tuple(True for _ in trainable)
        selected = [
            value.detach().reshape(-1).float().cpu()
            for value, enabled in zip(values, scope, strict=True)
            if enabled
        ]
        if not selected:
            raise ValueError("Cannot flatten an empty gradient scope")
        return torch.cat(selected).numpy()

    def paired_source_indices(indices: list[int]) -> list[int] | None:
        if synthetic_objective != "source_paired_consistency":
            return None
        missing = [index for index in indices if index not in paired_real_index_by_synthetic_index]
        if missing:
            raise ValueError(f"Synthetic samples have no source-real pair: {missing}")
        return [paired_real_index_by_synthetic_index[index] for index in indices]

    def synthetic_objective_loss_for_indices(indices: list[int], evaluation_seed: int) -> float:
        if not indices:
            raise ValueError("Synthetic objective evaluation requires at least one sample")
        if synthetic_objective != "source_paired_consistency":
            return loss_for_indices(synthetic_dataset, indices, evaluation_seed)

        policy.eval()
        weighted_loss = 0.0
        for offset in range(0, len(indices), evaluation_batch_size):
            chunk = indices[offset : offset + evaluation_batch_size]
            synthetic_batch_value = prepared_batch(synthetic_dataset, chunk)
            reference_batch_value = prepared_batch(dataset, paired_source_indices(chunk))
            seed_value = evaluation_seed + offset
            with torch.no_grad():
                actions = policy.prepare_action(synthetic_batch_value)
                torch.manual_seed(seed_value)
                shared_noise = policy.model.sample_noise(actions.shape, actions.device)
            if paired_action_consistency_mode == "inference_trajectory":
                teacher = frozen_initial_inference_trajectory(
                    reference_batch_value,
                    noise=shared_noise,
                )
                student = captured_inference_trajectory(
                    synthetic_batch_value,
                    noise=shared_noise,
                    no_grad=True,
                )
                action_consistency, _full_chunk_consistency = (
                    executed_and_chunk_consistency_losses(
                        student["final_action"], teacher["final_action"]
                    )
                )
                feature_consistency = feature_cosine_loss(
                    student["visual_features"],
                    teacher["visual_features"],
                    pooled=True,
                )
                canonical_action_identity = torch.zeros(
                    (), device=student["final_action"].device
                )
                canonical_feature_identity = torch.zeros(
                    (), device=student["final_action"].device
                )
                if (
                    paired_canonical_action_identity_weight > 0.0
                    or paired_canonical_feature_identity_weight > 0.0
                ):
                    canonical_student = captured_inference_trajectory(
                        reference_batch_value,
                        noise=shared_noise,
                        no_grad=True,
                    )
                    canonical_action_identity, _canonical_full_chunk = (
                        executed_and_chunk_consistency_losses(
                            canonical_student["final_action"],
                            teacher["final_action"],
                        )
                    )
                    canonical_feature_identity = feature_cosine_loss(
                        canonical_student["visual_features"],
                        teacher["visual_features"],
                        pooled=True,
                    )
                if paired_synthetic_bc_weight > 0.0:
                    with torch.no_grad():
                        shared_time = policy.model.sample_time(
                            actions.shape[0], actions.device
                        )
                    bc_loss, _bc_visual, _bc_action = captured_policy_forward(
                        synthetic_batch_value,
                        seed_value,
                        noise=shared_noise,
                        flow_time=shared_time,
                        no_grad=True,
                    )
                else:
                    bc_loss = torch.zeros(
                        (), device=student["final_action"].device
                    )
                objective = (
                    paired_action_consistency_weight * action_consistency
                    + paired_feature_consistency_weight * feature_consistency
                    + paired_synthetic_bc_weight * bc_loss
                    + paired_canonical_action_identity_weight
                    * canonical_action_identity
                    + paired_canonical_feature_identity_weight
                    * canonical_feature_identity
                )
                weighted_loss += float(objective.detach().cpu()) * len(chunk)
                continue
            with torch.no_grad():
                shared_time = policy.model.sample_time(actions.shape[0], actions.device)
            if paired_teacher_mode == "initial_global":
                target_visual, target_action = frozen_initial_policy_forward(
                    reference_batch_value,
                    seed_value,
                    noise=shared_noise,
                    flow_time=shared_time,
                )
            else:
                _target_loss, target_visual, target_action = captured_policy_forward(
                    reference_batch_value,
                    seed_value,
                    noise=shared_noise,
                    flow_time=shared_time,
                    no_grad=True,
                )
            bc_loss, student_visual, student_action = captured_policy_forward(
                synthetic_batch_value,
                seed_value,
                noise=shared_noise,
                flow_time=shared_time,
                no_grad=True,
            )
            action_consistency = normalized_action_consistency_loss(
                student_action, target_action
            )
            feature_consistency = feature_cosine_loss(
                student_visual, target_visual, pooled=True
            )
            canonical_action_identity = torch.zeros((), device=student_action.device)
            canonical_feature_identity = torch.zeros((), device=student_action.device)
            if (
                paired_canonical_action_identity_weight > 0.0
                or paired_canonical_feature_identity_weight > 0.0
            ):
                (
                    _canonical_loss,
                    canonical_student_visual,
                    canonical_student_action,
                ) = captured_policy_forward(
                    reference_batch_value,
                    seed_value,
                    noise=shared_noise,
                    flow_time=shared_time,
                    no_grad=True,
                )
                canonical_action_identity = normalized_action_consistency_loss(
                    canonical_student_action, target_action
                )
                canonical_feature_identity = feature_cosine_loss(
                    canonical_student_visual, target_visual, pooled=True
                )
            objective = (
                paired_action_consistency_weight * action_consistency
                + paired_feature_consistency_weight * feature_consistency
                + paired_synthetic_bc_weight * bc_loss
                + paired_canonical_action_identity_weight
                * canonical_action_identity
                + paired_canonical_feature_identity_weight
                * canonical_feature_identity
            )
            weighted_loss += float(objective.detach().cpu()) * len(chunk)
        return weighted_loss / len(indices)

    record_marginal_utility = bool(
        spec.get("record_marginal_utility_episodes", False)
    )
    marginal_canonical_penalty = float(
        spec.get("marginal_utility_canonical_penalty", 1.0)
    )

    def marginal_utility_episode_losses() -> dict[int, dict[str, float]]:
        if not record_marginal_utility:
            return {}
        if set(validation_episodes) != set(anchor_episode_ids):
            raise ValueError(
                "Marginal-utility recording requires identical canonical "
                "validation and paired-Gaussian anchor episodes"
            )
        missing = [
            episode_id
            for episode_id in validation_episodes
            if episode_id not in synthetic_indices_by_episode
        ]
        if missing:
            raise ValueError(
                "Marginal-utility episodes are missing Gaussian pairs: "
                f"{missing}"
            )
        records = {}
        for episode_id in sorted(validation_episodes):
            evaluation_seed = seed + 120_000 + episode_id * 101
            records[episode_id] = {
                "canonical_loss": loss_for_indices(
                    dataset,
                    validation_indices_by_episode[episode_id],
                    evaluation_seed,
                ),
                "paired_gaussian_loss": synthetic_objective_loss_for_indices(
                    synthetic_indices_by_episode[episode_id],
                    evaluation_seed + 10_000,
                ),
            }
        return records

    def replace_scope(
        full_vector: np.ndarray,
        scoped_vector: np.ndarray,
        parameter_scope: tuple[bool, ...],
    ) -> np.ndarray:
        result = np.asarray(full_vector, dtype=np.float32).copy()
        full_offset = 0
        scoped_offset = 0
        for parameter, enabled in zip(trainable, parameter_scope, strict=True):
            size = parameter.numel()
            if enabled:
                result[full_offset : full_offset + size] = scoped_vector[
                    scoped_offset : scoped_offset + size
                ]
                scoped_offset += size
            full_offset += size
        if full_offset != result.size or scoped_offset != scoped_vector.size:
            raise ValueError("Scoped gradient size mismatch")
        return result

    def assign_gradient(vector: np.ndarray) -> None:
        offset = 0
        for parameter in trainable:
            size = parameter.numel()
            value = torch.as_tensor(vector[offset : offset + size], dtype=parameter.dtype, device=parameter.device)
            parameter.grad = value.reshape_as(parameter)
            offset += size
        if offset != vector.size:
            raise ValueError(f"Combined gradient size mismatch: assigned {offset}, received {vector.size}")

    def perturb_for_sam(vector: np.ndarray) -> tuple[list[Any], float]:
        norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))
        scale = sam_rho / (norm + 1e-12)
        snapshots = [parameter.detach().clone() for parameter in trainable]
        offset = 0
        with torch.no_grad():
            for parameter in trainable:
                size = parameter.numel()
                perturbation = torch.as_tensor(
                    vector[offset : offset + size] * scale,
                    dtype=parameter.dtype,
                    device=parameter.device,
                ).reshape_as(parameter)
                parameter.add_(perturbation)
                offset += size
        if offset != vector.size:
            raise ValueError("SAM perturbation gradient size mismatch")
        return snapshots, norm

    def restore_snapshots(snapshots: list[Any]) -> None:
        with torch.no_grad():
            for parameter, snapshot in zip(trainable, snapshots, strict=True):
                parameter.copy_(snapshot)

    def restore_initial_adapter() -> None:
        with torch.no_grad():
            for parameter, initial in zip(trainable, initial_trainable, strict=True):
                parameter.copy_(initial)

    def load_adapter_layers(candidate_layers: tuple[LoRALayerUpdate, ...]) -> None:
        by_name = {layer.layer_name: layer for layer in candidate_layers}
        if set(by_name) != set(adapters):
            raise ValueError("Candidate adapter layers do not match local targets")
        with torch.no_grad():
            for module_name, adapter in adapters.items():
                layer = by_name[module_name]
                if layer.a.shape != tuple(adapter.a.shape) or layer.b.shape != tuple(adapter.b.shape):
                    raise ValueError(
                        f"Candidate adapter shape mismatch for {module_name}: "
                        f"A{layer.a.shape}/B{layer.b.shape} versus "
                        f"A{tuple(adapter.a.shape)}/B{tuple(adapter.b.shape)}"
                    )
                adapter.a.copy_(
                    torch.as_tensor(layer.a, dtype=adapter.a.dtype, device=adapter.a.device)
                )
                adapter.b.copy_(
                    torch.as_tensor(
                        layer.b / adapter.scaling,
                        dtype=adapter.b.dtype,
                        device=adapter.b.device,
                    )
                )

    def capture_layers() -> tuple[tuple[LoRALayerUpdate, ...], float]:
        captured = []
        dense_norm_sq = 0.0
        for module_name, adapter in adapters.items():
            a = adapter.a.detach().cpu().numpy().astype(np.float32)
            b = (adapter.b.detach().cpu().numpy() * adapter.scaling).astype(np.float32)
            layer = LoRALayerUpdate(layer_name=module_name, a=a, b=b)
            captured.append(layer)
            dense_norm_sq += float(np.sum(layer.dense().astype(np.float64) ** 2))
        return tuple(captured), math.sqrt(dense_norm_sq)

    synthetic_weight = float(spec.get("synthetic_weight", 1.0))
    max_auxiliary_to_real_norm = spec.get("max_auxiliary_to_real_norm")
    proximal_mu = float(spec.get("proximal_mu", 0.0))
    projection_cosine_threshold = float(spec.get("projection_cosine_threshold", 0.0))
    update_contract = str(spec.get("update_contract", "combined"))
    synthetic_residual_rank = int(spec.get("synthetic_residual_rank", spec["rank"]))
    selection_config = dict(synthetic_source.get("selection", {}))
    selected_synthetic_episode_ids = list(candidate_episode_ids)
    crossfit_selection_report = None
    if gradient_mode == "crossfit":
        real_anchor_vectors = []
        for fold, episode_id in enumerate(validation_episodes):
            fold_result = gradients_for_indices(
                dataset,
                validation_indices_by_episode[episode_id],
                seed + 20_000 + fold,
                synthetic_parameter_scope,
            )
            real_anchor_vectors.append(flatten(fold_result["gradients"], synthetic_parameter_scope))
        real_anchor_mean = np.mean(np.stack(real_anchor_vectors), axis=0)
        anchor_weight = float(selection_config.get("anchor_weight", 0.5))
        anchor_vectors = list(real_anchor_vectors)
        if anchor_weight > 0.0:
            for fold, episode_id in enumerate(anchor_episode_ids):
                synthetic_fold_indices = synthetic_indices_by_episode[episode_id]
                fold_result = gradients_for_indices(
                    synthetic_dataset,
                    synthetic_fold_indices,
                    seed + 21_000 + fold,
                    synthetic_parameter_scope,
                    synthetic=True,
                    paired_reference_indices=paired_source_indices(synthetic_fold_indices),
                )
                anchor_vectors.append(
                    real_anchor_mean
                    + anchor_weight * flatten(fold_result["gradients"], synthetic_parameter_scope)
                )
        candidate_evidence = []
        for candidate_index, episode_id in enumerate(candidate_episode_ids):
            candidate_indices = synthetic_indices_by_episode[episode_id]
            candidate_result = gradients_for_indices(
                synthetic_dataset,
                candidate_indices,
                seed + 22_000 + candidate_index,
                synthetic_parameter_scope,
                synthetic=True,
                paired_reference_indices=paired_source_indices(candidate_indices),
            )
            evidence = trajectory_evidence[episode_id]
            candidate_evidence.append(TrajectoryGradientEvidence(
                episode_id=episode_id,
                variation=evidence["variation"],
                quality=evidence["quality"],
                feasible=evidence["feasible"],
                gradient=flatten(candidate_result["gradients"], synthetic_parameter_scope),
                source_demo=evidence["source_demo"],
            ))
        selection = select_trajectories(
            candidate_evidence,
            anchor_vectors,
            max_selected=int(selection_config.get("max_selected", len(candidate_evidence))),
            max_per_variation=int(selection_config.get("max_per_variation", len(candidate_evidence))),
            minimum_variations=int(selection_config.get("minimum_variations", 1)),
            lcb_z=float(selection_config.get("lcb_z", 1.0)),
            minimum_lcb=float(selection_config.get("minimum_lcb", 0.0)),
            minimum_positive_fraction=float(selection_config.get("minimum_positive_fraction", 0.5)),
        )
        selected_synthetic_episode_ids = list(selection.selected_episode_ids)
        crossfit_selection_report = {
            "schema": "fedsplat.crossfit_trajectory_selection.v1",
            "real_anchor_episode_ids": validation_episodes,
            "synthetic_anchor_episode_ids": anchor_episode_ids,
            "anchor_weight": anchor_weight,
            "gradient_signature_scope": synthetic_gradient_scope,
            "gradient_signature_objective": synthetic_objective,
            "gradient_signature_modules": sorted(
                perception_modules if synthetic_gradient_scope == "perception" else adapters
            ),
            "gradient_signature_parameters": sum(
                parameter.numel()
                for parameter, enabled in zip(trainable, synthetic_parameter_scope, strict=True)
                if enabled
            ),
            **selection.to_dict(),
        }
        (out_dir / "trajectory_selection_report.json").write_text(
            json.dumps(crossfit_selection_report, indent=2) + "\n",
            encoding="utf-8",
        )

    selected_synthetic_indices_by_episode = {
        episode_id: synthetic_indices_by_episode[episode_id]
        for episode_id in selected_synthetic_episode_ids
        if episode_id in synthetic_indices_by_episode
    }
    selected_synthetic_indices = flatten_episode_indices(selected_synthetic_indices_by_episode)
    measured_selected_quality_score = (
        float(np.mean([trajectory_evidence[episode_id]["quality"] for episode_id in selected_synthetic_episode_ids]))
        if selected_synthetic_episode_ids and trajectory_evidence
        else quality_score
    )
    use_synthetic_quality_weight = bool(spec.get("use_synthetic_quality_weight", True))
    selected_quality_score = measured_selected_quality_score if use_synthetic_quality_weight else 1.0

    marginal_episode_losses_before = marginal_utility_episode_losses()
    portfolio_two_d_loss_before = (
        loss_for_indices(
            portfolio_two_d_dataset,
            train_indices,
            seed + 2400,
        )
        if portfolio_two_d_dataset is not None
        else None
    )
    if dynamic_batches:
        train_loss_before = loss_for_indices(dataset, train_indices, seed + 1000)
        validation_loss_before = loss_for_indices(dataset, validation_indices, seed + 2000)
        synthetic_loss_before = (
            synthetic_objective_loss_for_indices(selected_synthetic_indices, seed + 2500)
            if selected_synthetic_indices else None
        )
        synthetic_anchor_loss_before = (
            synthetic_objective_loss_for_indices(
                flatten_episode_indices({key: synthetic_indices_by_episode[key] for key in anchor_episode_ids}),
                seed + 2600,
            )
            if anchor_episode_ids else None
        )
    else:
        train_loss_before = deterministic_loss(train_batch, seed + 1000)
        validation_loss_before = deterministic_loss(validation_batch, seed + 2000)
        synthetic_loss_before = deterministic_loss(synthetic_batch, seed + 2500) if synthetic_batch is not None else None
        synthetic_anchor_loss_before = None

    def scheduled_minibatch(
        by_episode: dict[int, list[int]],
        step: int,
        salt: int,
    ) -> tuple[list[int], list[int]]:
        if not by_episode:
            return [], []
        episode_ids = sorted(by_episode)
        rng = np.random.default_rng(seed * 1_000_003 + step * 1009 + salt)
        chosen_episodes = [episode_ids[int(value)] for value in rng.integers(0, len(episode_ids), size=batch_size)]
        chosen_indices = [
            by_episode[episode_id][int(rng.integers(0, len(by_episode[episode_id])))]
            for episode_id in chosen_episodes
        ]
        return chosen_indices, chosen_episodes

    def train_branch(
        branch_mode: str,
        *,
        synthetic_objective_override: str | None = None,
    ) -> list[dict[str, Any]]:
        if (
            synthetic_objective_override is not None
            and synthetic_objective_override not in SYNTHETIC_OBJECTIVES
        ):
            raise ValueError(
                f"Unknown branch synthetic objective: {synthetic_objective_override}"
            )
        optimizer = torch.optim.AdamW(trainable, lr=float(spec["learning_rate"]), weight_decay=0.0)
        records = []
        for step in range(int(spec["steps"])):
            optimizer.zero_grad(set_to_none=True)
            # Both branches must see the same real-gradient trajectory so their
            # difference isolates the optional synthetic contribution.
            gradient_seed = seed + 3000 + step
            if dynamic_batches:
                real_step_indices, real_step_episodes = scheduled_minibatch(
                    train_indices_by_episode,
                    step,
                    17,
                )
                real_step_batch = prepared_batch(dataset, real_step_indices)
            else:
                real_step_indices, real_step_episodes = train_indices, train_episodes
                real_step_batch = train_batch
            synthetic_step_indices: list[int] = []
            synthetic_step_episodes: list[int] = []
            synthetic_step_batch = None
            synthetic_reference_indices: list[int] = []
            synthetic_reference_batch = None
            two_d_step_indices: list[int] = []
            two_d_step_episodes: list[int] = []
            two_d_step_batch = None
            step_quality = selected_quality_score
            if branch_mode == "augmentation_portfolio":
                two_d_step_indices, two_d_step_episodes = scheduled_minibatch(
                    train_indices_by_episode,
                    step,
                    23,
                )
                two_d_step_batch = prepared_batch(
                    portfolio_two_d_dataset,
                    two_d_step_indices,
                )
            has_synthetic = (
                bool(selected_synthetic_indices_by_episode)
                if dynamic_batches
                else synthetic_batch is not None
            )
            # All portfolio candidates execute the same real/2D/GS forward
            # budget. The lambda=0 candidate keeps an exactly zero GS weight.
            needs_gs_batch = branch_mode != "real_only"
            if (
                branch_mode != "real_only"
                and needs_gs_batch
                and has_synthetic
                and selected_quality_score > 0.0
            ):
                if dynamic_batches:
                    synthetic_step_indices, synthetic_step_episodes = scheduled_minibatch(
                        selected_synthetic_indices_by_episode,
                        step,
                        31,
                    )
                    synthetic_step_batch = prepared_batch(synthetic_dataset, synthetic_step_indices)
                    paired_indices = paired_source_indices(synthetic_step_indices)
                    if paired_indices is not None:
                        synthetic_reference_indices = paired_indices
                        synthetic_reference_batch = prepared_batch(dataset, paired_indices)
                    if trajectory_evidence and use_synthetic_quality_weight:
                        step_quality = float(np.mean([
                            trajectory_evidence[episode_id]["quality"]
                            for episode_id in synthetic_step_episodes
                        ]))
                else:
                    synthetic_step_batch = synthetic_batch

            def branch_gradient():
                real_result = gradients(real_step_batch, gradient_seed)
                two_d_result = None
                if two_d_step_batch is not None:
                    two_d_result = gradients(
                        two_d_step_batch,
                        gradient_seed,
                        synthetic_parameter_scope,
                        synthetic=True,
                    )
                synthetic_result = None
                if synthetic_step_batch is not None:
                    synthetic_result = gradients(
                        synthetic_step_batch,
                        gradient_seed,
                        synthetic_parameter_scope,
                        synthetic=True,
                        paired_reference_batch=synthetic_reference_batch,
                        synthetic_objective_override=synthetic_objective_override,
                    )
                real_gradients = real_result["gradients"]
                synthetic_gradients = (
                    synthetic_result["gradients"] if synthetic_result is not None else None
                )
                real_vector = flatten(real_gradients)
                real_scope_vector = flatten(real_gradients, synthetic_parameter_scope)
                synthetic_scope_vector = (
                    flatten(synthetic_gradients, synthetic_parameter_scope)
                    if synthetic_gradients is not None
                    else np.zeros_like(real_scope_vector)
                )
                portfolio_combination = None
                two_d_scope_vector = (
                    flatten(
                        two_d_result["gradients"],
                        synthetic_parameter_scope,
                    )
                    if two_d_result is not None
                    else None
                )
                if branch_mode == "augmentation_portfolio":
                    if two_d_scope_vector is None:
                        raise RuntimeError(
                            "Augmentation portfolio received no 2D batch"
                        )
                    portfolio_combination = (
                        combine_augmentation_portfolio_gradients(
                            real_scope_vector,
                            two_d_scope_vector,
                            synthetic_scope_vector,
                            gs_ratio=portfolio_gs_ratio,
                            gs_quality_score=step_quality,
                            max_gs_to_base_norm=float(
                                max_auxiliary_to_real_norm
                            ),
                            projection_cosine_threshold=(
                                projection_cosine_threshold
                            ),
                        )
                    )
                    combination = portfolio_combination.gs_combination
                    combined_gradient = replace_scope(
                        real_vector,
                        np.asarray(
                            portfolio_combination.combined_gradient,
                            dtype=np.float32,
                        ),
                        synthetic_parameter_scope,
                    )
                elif branch_mode == "synthetic_only":
                    if synthetic_result is None:
                        raise RuntimeError(
                            "Synthetic-only distillation received no paired batch"
                        )
                    combination = combine_real_synthetic_gradients(
                        np.zeros_like(real_scope_vector),
                        synthetic_scope_vector,
                        mode="naive",
                        synthetic_weight=1.0,
                        quality_score=step_quality,
                        max_auxiliary_to_real_norm=None,
                        projection_cosine_threshold=projection_cosine_threshold,
                    )
                    combined_gradient = replace_scope(
                        np.zeros_like(real_vector),
                        np.asarray(combination.combined_gradient, dtype=np.float32),
                        synthetic_parameter_scope,
                    )
                else:
                    combination = combine_real_synthetic_gradients(
                        real_scope_vector,
                        synthetic_scope_vector if branch_mode != "real_only" else None,
                        mode=branch_mode,
                        synthetic_weight=synthetic_weight,
                        quality_score=step_quality,
                        max_auxiliary_to_real_norm=max_auxiliary_to_real_norm,
                        projection_cosine_threshold=projection_cosine_threshold,
                    )
                    combined_gradient = replace_scope(
                        real_vector,
                        np.asarray(combination.combined_gradient, dtype=np.float32),
                        synthetic_parameter_scope,
                    )
                if proximal_mu > 0.0:
                    proximal_vector = flatten(
                        tuple(
                            proximal_mu * (parameter - reference)
                            for parameter, reference in zip(trainable, initial_trainable, strict=True)
                        )
                    )
                    combined_gradient = combined_gradient + proximal_vector
                else:
                    proximal_vector = np.zeros_like(combined_gradient)
                return {
                    "real_loss": real_result["loss"],
                    "synthetic_loss": (
                        synthetic_result["loss"] if synthetic_result is not None else None
                    ),
                    "two_d_loss": (
                        two_d_result["loss"] if two_d_result is not None else None
                    ),
                    "real_metrics": real_result["metrics"],
                    "synthetic_metrics": (
                        synthetic_result["metrics"] if synthetic_result is not None else None
                    ),
                    "two_d_metrics": (
                        two_d_result["metrics"]
                        if two_d_result is not None
                        else None
                    ),
                    "real_vector": real_vector,
                    "synthetic_scope_vector": synthetic_scope_vector,
                    "two_d_scope_vector": two_d_scope_vector,
                    "combined_gradient": combined_gradient,
                    "proximal_vector": proximal_vector,
                    "combination": combination,
                    "portfolio_combination": portfolio_combination,
                }

            base_result = branch_gradient()
            applied_result = base_result
            sam_base_gradient_norm = float(np.linalg.norm(base_result["combined_gradient"]))
            if sharpness_mode == "sam":
                snapshots, sam_base_gradient_norm = perturb_for_sam(base_result["combined_gradient"])
                try:
                    applied_result = branch_gradient()
                finally:
                    restore_snapshots(snapshots)
            assign_gradient(applied_result["combined_gradient"])
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0).detach().cpu())
            optimizer.step()
            combination = applied_result["combination"]
            projection = combination.projection
            base_real_loss = float(base_result["real_loss"].detach().cpu())
            applied_real_loss = float(applied_result["real_loss"].detach().cpu())
            base_synthetic_loss = base_result["synthetic_loss"]
            applied_synthetic_loss = applied_result["synthetic_loss"]
            base_two_d_loss = base_result["two_d_loss"]
            applied_two_d_loss = applied_result["two_d_loss"]
            portfolio_combination = applied_result[
                "portfolio_combination"
            ]
            records.append({
                "step": step + 1,
                "loss": base_real_loss,
                "real_loss": base_real_loss,
                "synthetic_loss": (
                    float(base_synthetic_loss.detach().cpu())
                    if base_synthetic_loss is not None else None
                ),
                "two_d_loss": (
                    float(base_two_d_loss.detach().cpu())
                    if base_two_d_loss is not None
                    else None
                ),
                "grad_norm": grad_norm,
                "gradient_mode": branch_mode,
                "sharpness_mode": sharpness_mode,
                "sam_rho": sam_rho if sharpness_mode == "sam" else 0.0,
                "sam_base_gradient_norm": sam_base_gradient_norm,
                "sam_perturbed_gradient_norm": (
                    float(np.linalg.norm(applied_result["combined_gradient"]))
                    if sharpness_mode == "sam" else None
                ),
                "sam_real_loss_increase": (
                    applied_real_loss - base_real_loss if sharpness_mode == "sam" else None
                ),
                "sam_synthetic_loss_increase": (
                    float(applied_synthetic_loss.detach().cpu())
                    - float(base_synthetic_loss.detach().cpu())
                    if sharpness_mode == "sam"
                    and applied_synthetic_loss is not None
                    and base_synthetic_loss is not None
                    else None
                ),
                "sam_two_d_loss_increase": (
                    float(applied_two_d_loss.detach().cpu())
                    - float(base_two_d_loss.detach().cpu())
                    if sharpness_mode == "sam"
                    and applied_two_d_loss is not None
                    and base_two_d_loss is not None
                    else None
                ),
                "synthetic_gradient_scope": synthetic_gradient_scope,
                "synthetic_objective": synthetic_objective,
                "active_synthetic_objective": (
                    base_result["synthetic_metrics"]["objective"]
                    if base_result["synthetic_metrics"] is not None
                    else None
                ),
                "real_bc_loss": base_result["real_metrics"]["bc_loss"],
                "real_representation_anchor_loss": base_result["real_metrics"][
                    "representation_anchor_loss"
                ],
                "real_representation_anchor_gradient_norm": base_result["real_metrics"][
                    "representation_anchor_gradient_norm"
                ],
                "real_bc_perception_gradient_norm": base_result["real_metrics"][
                    "bc_perception_gradient_norm"
                ],
                "effective_representation_anchor_weight": base_result["real_metrics"][
                    "effective_representation_anchor_weight"
                ],
                "synthetic_bc_loss": (
                    base_result["synthetic_metrics"]["bc_loss"]
                    if base_result["synthetic_metrics"] is not None else None
                ),
                "two_d_bc_loss": (
                    base_result["two_d_metrics"]["bc_loss"]
                    if base_result["two_d_metrics"] is not None
                    else None
                ),
                "paired_action_consistency_loss": (
                    base_result["synthetic_metrics"]["paired_action_consistency_loss"]
                    if base_result["synthetic_metrics"] is not None else None
                ),
                "paired_full_chunk_consistency_loss": (
                    base_result["synthetic_metrics"][
                        "paired_full_chunk_consistency_loss"
                    ]
                    if base_result["synthetic_metrics"] is not None else None
                ),
                "paired_inference_denoise_step": (
                    base_result["synthetic_metrics"][
                        "paired_inference_denoise_step"
                    ]
                    if base_result["synthetic_metrics"] is not None else None
                ),
                "paired_feature_consistency_loss": (
                    base_result["synthetic_metrics"]["paired_feature_consistency_loss"]
                    if base_result["synthetic_metrics"] is not None else None
                ),
                "paired_canonical_action_identity_loss": (
                    base_result["synthetic_metrics"][
                        "paired_canonical_action_identity_loss"
                    ]
                    if base_result["synthetic_metrics"] is not None else None
                ),
                "paired_canonical_full_chunk_identity_loss": (
                    base_result["synthetic_metrics"][
                        "paired_canonical_full_chunk_identity_loss"
                    ]
                    if base_result["synthetic_metrics"] is not None else None
                ),
                "paired_canonical_feature_identity_loss": (
                    base_result["synthetic_metrics"][
                        "paired_canonical_feature_identity_loss"
                    ]
                    if base_result["synthetic_metrics"] is not None else None
                ),
                "effective_synthetic_weight": combination.effective_synthetic_weight,
                "portfolio_gs_ratio": (
                    portfolio_combination.gs_ratio
                    if portfolio_combination is not None
                    else None
                ),
                "portfolio_two_d_ratio": (
                    portfolio_combination.two_d_ratio
                    if portfolio_combination is not None
                    else None
                ),
                "portfolio_base_gradient_norm": (
                    float(np.linalg.norm(
                        portfolio_combination.base_combination.combined_gradient
                    ))
                    if portfolio_combination is not None
                    else None
                ),
                "two_d_gradient_norm": (
                    float(np.linalg.norm(base_result["two_d_scope_vector"]))
                    if base_result["two_d_scope_vector"] is not None
                    else None
                ),
                "auxiliary_trust_scale": combination.auxiliary_trust_scale,
                "weighted_synthetic_gradient_norm": combination.weighted_synthetic_norm,
                "projection_applied": combination.projection_applied,
                "projection_branch": projection.branch if projection else None,
                "dot_before": projection.dot_before if projection else None,
                "dot_after": projection.dot_after if projection else None,
                "cosine_before": projection.cosine_before if projection else None,
                "cosine_after": projection.cosine_after if projection else None,
                "real_gradient_norm": (
                    projection.real_norm
                    if projection else float(np.linalg.norm(base_result["real_vector"]))
                ),
                "synthetic_gradient_norm": projection.synthetic_norm if projection else 0.0,
                "calibrated_synthetic_gradient_norm": projection.calibrated_norm if projection else 0.0,
                "proximal_gradient_norm": float(np.linalg.norm(applied_result["proximal_vector"])),
                "real_batch_indices": real_step_indices,
                "real_batch_episode_ids": real_step_episodes,
                "synthetic_batch_indices": synthetic_step_indices,
                "synthetic_batch_episode_ids": synthetic_step_episodes,
                "synthetic_source_real_indices": synthetic_reference_indices,
                "synthetic_batch_quality": step_quality if synthetic_step_indices else None,
                "two_d_batch_indices": two_d_step_indices,
                "two_d_batch_episode_ids": two_d_step_episodes,
            })
        return records

    real_branch = None
    needs_counterfactual_branch = (
        (update_contract == "dual_channel" and gradient_mode != "real_only")
        or gradient_mode in {"crossfit", "factor_paired"}
    )
    counterfactual_training_mode = (
        "naive"
        if gradient_mode == "factor_paired"
        and factor_counterfactual_mode == "gaussian_bc"
        else "real_only"
    )
    counterfactual_synthetic_objective = (
        "action_bc" if counterfactual_training_mode == "naive" else None
    )
    if needs_counterfactual_branch:
        restore_initial_adapter()
        real_branch_started = time.perf_counter()
        real_step_records = train_branch(
            counterfactual_training_mode,
            synthetic_objective_override=counterfactual_synthetic_objective,
        )
        real_train_loss_after = (
            loss_for_indices(dataset, train_indices, seed + 1000)
            if dynamic_batches else deterministic_loss(train_batch, seed + 1000)
        )
        real_validation_loss_after = (
            loss_for_indices(dataset, validation_indices, seed + 2000)
            if dynamic_batches else deterministic_loss(validation_batch, seed + 2000)
        )
        real_synthetic_loss_after = (
            (
                loss_for_indices(
                    synthetic_dataset, selected_synthetic_indices, seed + 2500
                )
                if counterfactual_synthetic_objective == "action_bc"
                else synthetic_objective_loss_for_indices(
                    selected_synthetic_indices, seed + 2500
                )
            )
            if dynamic_batches and selected_synthetic_indices
            else None
        )
        real_synthetic_anchor_loss_after = (
            synthetic_objective_loss_for_indices(
                flatten_episode_indices({key: synthetic_indices_by_episode[key] for key in anchor_episode_ids}),
                seed + 2600,
            )
            if dynamic_batches and anchor_episode_ids else None
        )
        real_layers, real_dense_norm = capture_layers()
        real_branch = {
            "step_records": real_step_records,
            "train_loss_after": real_train_loss_after,
            "validation_loss_after": real_validation_loss_after,
            "synthetic_loss_after": real_synthetic_loss_after,
            "synthetic_anchor_loss_after": real_synthetic_anchor_loss_after,
            "layers": real_layers,
            "dense_delta_norm": real_dense_norm,
            "compute_time_s": time.perf_counter() - real_branch_started,
            "counterfactual_mode": factor_counterfactual_mode,
            "training_mode": counterfactual_training_mode,
            "synthetic_objective": counterfactual_synthetic_objective,
        }
        restore_initial_adapter()

    combined_branch_started = time.perf_counter()
    effective_training_mode = (
        str(selection_config.get("combination_mode", "projected"))
        if gradient_mode == "crossfit"
        else "synthetic_only" if gradient_mode in PAIRED_DISTILL_MODES
        else "naive" if gradient_mode == "factor_paired"
        else gradient_mode
    )
    if effective_training_mode not in {
        "real_only",
        "naive",
        "projected",
        "synthetic_only",
        "augmentation_portfolio",
    }:
        raise ValueError(f"Unknown effective training mode: {effective_training_mode}")
    if gradient_mode == "crossfit" and not selected_synthetic_episode_ids:
        effective_training_mode = "real_only"
    step_records = train_branch(effective_training_mode)

    portfolio_two_d_loss_after = (
        loss_for_indices(
            portfolio_two_d_dataset,
            train_indices,
            seed + 2400,
        )
        if portfolio_two_d_dataset is not None
        else None
    )
    train_loss_after = (
        loss_for_indices(dataset, train_indices, seed + 1000)
        if dynamic_batches else deterministic_loss(train_batch, seed + 1000)
    )
    validation_loss_after = (
        loss_for_indices(dataset, validation_indices, seed + 2000)
        if dynamic_batches else deterministic_loss(validation_batch, seed + 2000)
    )
    synthetic_loss_after = (
        synthetic_objective_loss_for_indices(selected_synthetic_indices, seed + 2500)
        if dynamic_batches and selected_synthetic_indices
        else deterministic_loss(synthetic_batch, seed + 2500) if synthetic_batch is not None else None
    )
    synthetic_anchor_loss_after = (
        synthetic_objective_loss_for_indices(
            flatten_episode_indices({key: synthetic_indices_by_episode[key] for key in anchor_episode_ids}),
            seed + 2600,
        )
        if dynamic_batches and anchor_episode_ids else None
    )
    layers, dense_delta_norm = capture_layers()
    raw_mixed_snapshot = {
        "train_loss_after": train_loss_after,
        "validation_loss_after": validation_loss_after,
        "synthetic_loss_after": synthetic_loss_after,
        "synthetic_anchor_loss_after": synthetic_anchor_loss_after,
        "dense_delta_norm": dense_delta_norm,
    }
    if gradient_mode in {"crossfit", "factor_paired"} and policy_anchor_mode in {
        "real_branch_nonperception",
        "counterfactual_branch_nonperception",
    }:
        layers = compose_scoped_adapter_layers(
            real_branch["layers"],
            layers,
            perception_modules,
        )
        load_adapter_layers(layers)
        train_loss_after = (
            loss_for_indices(dataset, train_indices, seed + 1000)
            if dynamic_batches else deterministic_loss(train_batch, seed + 1000)
        )
        validation_loss_after = (
            loss_for_indices(dataset, validation_indices, seed + 2000)
            if dynamic_batches else deterministic_loss(validation_batch, seed + 2000)
        )
        synthetic_loss_after = (
            synthetic_objective_loss_for_indices(selected_synthetic_indices, seed + 2500)
            if dynamic_batches and selected_synthetic_indices
            else deterministic_loss(synthetic_batch, seed + 2500)
            if synthetic_batch is not None else None
        )
        synthetic_anchor_loss_after = (
            synthetic_objective_loss_for_indices(
                flatten_episode_indices({
                    key: synthetic_indices_by_episode[key] for key in anchor_episode_ids
                }),
                seed + 2600,
            )
            if dynamic_batches and anchor_episode_ids else None
        )
        layers, dense_delta_norm = capture_layers()
    combined_branch_time = time.perf_counter() - combined_branch_started
    crossfit_report = None
    if gradient_mode == "crossfit":
        fallback = dict(synthetic_source.get("fallback", {}))
        anchor_weight = float(selection_config.get("anchor_weight", 0.5))
        real_anchor_objective = float(real_branch["validation_loss_after"])
        mixed_anchor_objective = float(validation_loss_after)
        if real_branch["synthetic_anchor_loss_after"] is not None:
            real_anchor_objective += anchor_weight * float(real_branch["synthetic_anchor_loss_after"])
        if synthetic_anchor_loss_after is not None:
            mixed_anchor_objective += anchor_weight * float(synthetic_anchor_loss_after)
        minimum_gain = float(fallback.get("minimum_relative_anchor_gain", 0.0))
        real_tolerance = float(fallback.get("real_validation_tolerance", 0.0))
        anchor_gain = (real_anchor_objective - mixed_anchor_objective) / (
            abs(real_anchor_objective) + 1e-12
        )
        anchor_pass = anchor_gain >= minimum_gain
        real_validation_pass = validation_loss_after <= (
            float(real_branch["validation_loss_after"]) * (1.0 + real_tolerance)
        )
        mixed_committed = bool(selected_synthetic_episode_ids) and anchor_pass and real_validation_pass
        mixed_snapshot = {
            "train_loss_after": train_loss_after,
            "validation_loss_after": validation_loss_after,
            "synthetic_loss_after": synthetic_loss_after,
            "synthetic_anchor_loss_after": synthetic_anchor_loss_after,
            "dense_delta_norm": dense_delta_norm,
            "policy_anchor_mode": policy_anchor_mode,
            "raw_unanchored_mixed_branch": raw_mixed_snapshot,
            "layers": layers,
            "step_records": step_records,
        }
        if not mixed_committed:
            load_adapter_layers(real_branch["layers"])
            train_loss_after = float(real_branch["train_loss_after"])
            validation_loss_after = float(real_branch["validation_loss_after"])
            synthetic_loss_after = real_branch["synthetic_loss_after"]
            synthetic_anchor_loss_after = real_branch["synthetic_anchor_loss_after"]
            layers = real_branch["layers"]
            dense_delta_norm = float(real_branch["dense_delta_norm"])
            step_records = real_branch["step_records"]
        crossfit_report = {
            "schema": "fedsplat.crossfit_local_fallback.v3",
            "anchor_objective": synthetic_objective,
            "selected_branch": (
                "mixed_perception_real_policy" if mixed_committed and policy_anchor_mode != "none"
                else "mixed" if mixed_committed else "real_only"
            ),
            "mixed_committed": mixed_committed,
            "policy_anchor_mode": policy_anchor_mode,
            "mixed_perception_modules": sorted(perception_modules),
            "real_anchored_modules": sorted(set(adapters) - perception_modules),
            "minimum_relative_anchor_gain": minimum_gain,
            "real_validation_tolerance": real_tolerance,
            "anchor_gain": anchor_gain,
            "anchor_pass": anchor_pass,
            "real_validation_pass": real_validation_pass,
            "real_branch": {
                key: value for key, value in real_branch.items() if key not in {"layers", "step_records"}
            },
            "mixed_branch": {
                key: value for key, value in mixed_snapshot.items() if key not in {"layers", "step_records"}
            },
            "selection": crossfit_selection_report,
        }
    factor_paired_report = None
    if gradient_mode == "factor_paired":
        factor_label = str(spec.get("factor_label", "unspecified"))
        factor_paired_report = {
            "schema": "fedsplat.factor_paired_local_update.v2",
            "factor_label": factor_label,
            "factor_counterfactual_mode": factor_counterfactual_mode,
            "counterfactual_training_mode": counterfactual_training_mode,
            "counterfactual_synthetic_objective": counterfactual_synthetic_objective,
            "selected_episode_ids": selected_synthetic_episode_ids,
            "selected_episode_variations": [
                trajectory_evidence[episode_id]["variation"]
                for episode_id in selected_synthetic_episode_ids
            ],
            "source_pairing_verified": bool(
                source_pairing_audit and source_pairing_audit.get("verified")
            ),
            "synthetic_gradient_scope": synthetic_gradient_scope,
            "mixed_perception_modules": sorted(perception_modules),
            "real_anchored_modules": sorted(set(adapters) - perception_modules),
            "counterfactual_anchored_modules": sorted(
                set(adapters) - perception_modules
            ),
            "counterfactual_training_seed_matched": True,
            "counterfactual_minibatch_schedule_matched": True,
            "real_counterfactual_training_seed_matched": (
                factor_counterfactual_mode == "real_only"
            ),
            "real_counterfactual_minibatch_schedule_matched": (
                factor_counterfactual_mode == "real_only"
            ),
            "gaussian_bc_counterfactual_training_seed_matched": (
                factor_counterfactual_mode == "gaussian_bc"
            ),
            "gaussian_bc_counterfactual_minibatch_schedule_matched": (
                factor_counterfactual_mode == "gaussian_bc"
            ),
            "counterfactual_branch": {
                key: value
                for key, value in real_branch.items()
                if key not in {"layers", "step_records"}
            },
            "real_branch": {
                key: value
                for key, value in real_branch.items()
                if key not in {"layers", "step_records"}
            }
            if factor_counterfactual_mode == "real_only"
            else None,
            "gaussian_bc_counterfactual_branch": {
                key: value
                for key, value in real_branch.items()
                if key not in {"layers", "step_records"}
            }
            if factor_counterfactual_mode == "gaussian_bc"
            else None,
            "factor_branch": {
                "train_loss_after": train_loss_after,
                "validation_loss_after": validation_loss_after,
                "synthetic_loss_after": synthetic_loss_after,
                "dense_delta_norm": dense_delta_norm,
                "raw_unanchored_mixed_branch": raw_mixed_snapshot,
            },
        }
    marginal_utility_report = None
    if record_marginal_utility:
        marginal_episode_losses_after = marginal_utility_episode_losses()
        episode_records = []
        for episode_id in sorted(marginal_episode_losses_before):
            before = marginal_episode_losses_before[episode_id]
            after = marginal_episode_losses_after[episode_id]
            paired_gain = (
                before["paired_gaussian_loss"]
                - after["paired_gaussian_loss"]
            ) / (abs(before["paired_gaussian_loss"]) + 1e-12)
            canonical_regression = max(
                0.0,
                (
                    after["canonical_loss"] - before["canonical_loss"]
                )
                / (abs(before["canonical_loss"]) + 1e-12),
            )
            utility = (
                paired_gain
                - marginal_canonical_penalty * canonical_regression
            )
            episode_records.append(
                {
                    "episode_id": episode_id,
                    "paired_gaussian_loss_before": before[
                        "paired_gaussian_loss"
                    ],
                    "paired_gaussian_loss_after": after[
                        "paired_gaussian_loss"
                    ],
                    "canonical_loss_before": before["canonical_loss"],
                    "canonical_loss_after": after["canonical_loss"],
                    "paired_relative_gain": paired_gain,
                    "canonical_relative_regression": canonical_regression,
                    "marginal_utility": utility,
                }
            )
        utilities = [row["marginal_utility"] for row in episode_records]
        marginal_utility_report = {
            "schema": "fedsplat.source_paired_marginal_utility.v1",
            "common_baseline": paired_teacher_reference,
            "canonical_penalty": marginal_canonical_penalty,
            "episode_grouping": "heldout_source_episode",
            "episode_count": len(episode_records),
            "episode_records": episode_records,
            "mean_marginal_utility": float(np.mean(utilities)),
            "minimum_episode_marginal_utility": float(min(utilities)),
            "all_episode_utilities_finite": bool(
                all(math.isfinite(value) for value in utilities)
            ),
        }
    paired_distillation_report = None
    if gradient_mode in PAIRED_DISTILL_MODES:
        gate_spec = dict(spec.get("paired_distillation_gate", {}))
        minimum_train_gain = float(gate_spec.get("minimum_train_pair_gain", 0.0))
        minimum_anchor_gain = float(gate_spec.get("minimum_anchor_pair_gain", 0.0))
        real_tolerance = float(gate_spec.get("real_validation_tolerance", 0.0))
        if synthetic_loss_before is None or synthetic_loss_after is None:
            raise RuntimeError("Paired distillation lacks training-pair objectives")
        if synthetic_anchor_loss_before is None or synthetic_anchor_loss_after is None:
            raise RuntimeError("Paired distillation lacks held-out pair objectives")
        train_pair_gain = (
            float(synthetic_loss_before) - float(synthetic_loss_after)
        ) / (abs(float(synthetic_loss_before)) + 1e-12)
        anchor_pair_gain = (
            float(synthetic_anchor_loss_before)
            - float(synthetic_anchor_loss_after)
        ) / (abs(float(synthetic_anchor_loss_before)) + 1e-12)
        train_pair_pass = train_pair_gain >= minimum_train_gain
        anchor_pair_pass = anchor_pair_gain >= minimum_anchor_gain
        real_validation_pass = validation_loss_after <= (
            validation_loss_before * (1.0 + real_tolerance)
        )
        paired_distillation_report = {
            "schema": (
                "fedsplat.frozen_teacher_paired_distillation.v2"
                if paired_action_consistency_mode == "inference_trajectory"
                else "fedsplat.frozen_teacher_paired_distillation.v1"
            ),
            "teacher": paired_teacher_mode,
            "teacher_reference": paired_teacher_reference,
            "training_variant": gradient_mode,
            "action_consistency_mode": paired_action_consistency_mode,
            "inference_contract": (
                {
                    "num_denoise_steps": int(policy.model.config.num_steps),
                    "policy_action_steps": int(policy.config.n_action_steps),
                    "training_estimator": (
                        "deterministic_cyclic_teacher_forced_denoise_step"
                    ),
                    "primary_action_target": "first_executed_action_token",
                    "evaluation": "complete_denoising_trajectory",
                    "shared_noise": True,
                }
                if paired_action_consistency_mode == "inference_trajectory"
                else None
            ),
            "train_pair_gain": train_pair_gain,
            "heldout_pair_gain": anchor_pair_gain,
            "minimum_train_pair_gain": minimum_train_gain,
            "minimum_heldout_pair_gain": minimum_anchor_gain,
            "real_validation_tolerance": real_tolerance,
            "train_pair_pass": train_pair_pass,
            "heldout_pair_pass": anchor_pair_pass,
            "real_validation_pass": real_validation_pass,
            "deployment_gate_passed": bool(
                train_pair_pass and anchor_pair_pass and real_validation_pass
            ),
        }
    elapsed = time.perf_counter() - started
    reliability = heldout_real_reliability(validation_loss_before, validation_loss_after)
    update = ClientAdapterUpdate(
        client_id=spec["client_id"],
        layers=layers,
        n_real=len(train_indices),
        reliability=reliability,
        staleness=int(spec.get("staleness", 0)),
        compute_time_s=elapsed,
    )
    upload = save_client_update(update, out_dir / "adapter_upload")
    factor_residual_upload = None
    factor_real_reference = None
    factor_counterfactual_reference = None
    if gradient_mode == "factor_paired":
        real_reliability = heldout_real_reliability(
            validation_loss_before,
            float(real_branch["validation_loss_after"]),
        )
        matched_counterfactual_update = ClientAdapterUpdate(
            client_id=spec["client_id"],
            layers=real_branch["layers"],
            n_real=len(train_indices),
            reliability=real_reliability,
            staleness=int(spec.get("staleness", 0)),
            compute_time_s=float(real_branch["compute_time_s"]),
        )
        factor_residual, factor_residual_audit = build_scoped_factor_residual(
            update,
            matched_counterfactual_update,
            perception_modules,
        )
        reference_name = (
            "factor_gaussian_bc_reference_local"
            if factor_counterfactual_mode == "gaussian_bc"
            else "factor_real_reference_local"
        )
        factor_counterfactual_reference = save_client_update(
            matched_counterfactual_update,
            out_dir / reference_name,
        )
        factor_real_reference = (
            factor_counterfactual_reference
            if factor_counterfactual_mode == "real_only"
            else None
        )
        factor_residual_upload = save_client_update(
            factor_residual,
            out_dir / "factor_residual_upload",
        )
        residual_definition = (
            "factor_adapter_minus_internal_matched_gaussian_bc_counterfactual"
            if factor_counterfactual_mode == "gaussian_bc"
            else "factor_adapter_minus_internal_matched_real_counterfactual"
        )
        factor_paired_report["residual_contract"] = {
            "schema": "fedsplat.factor_residual_upload.v1",
            "definition": residual_definition,
            "counterfactual_mode": factor_counterfactual_mode,
            "server_aggregation_unit": "factor_residual_upload",
            "counterfactual_reference_is_local_audit_artifact": True,
            "real_reference_is_local_audit_artifact": (
                factor_counterfactual_mode == "real_only"
            ),
            "perception_only": True,
            "nonperception_residual_dense_norm": factor_residual_audit[
                "nonperception_residual_dense_norm"
            ],
            "reconstruction_relative_error": factor_residual_audit[
                "reconstruction_relative_error"
            ],
            "residual_upload_bytes": int(factor_residual_upload["total_upload_bytes"]),
            "residual_upload_sha256": factor_residual_upload["payload_sha256"],
            "counterfactual_reference_sha256": factor_counterfactual_reference[
                "payload_sha256"
            ],
            "real_reference_sha256": (
                factor_real_reference["payload_sha256"]
                if factor_real_reference is not None
                else None
            ),
        }
    dual_channel_upload = None
    dual_channel_report = None
    if update_contract == "dual_channel" and real_branch is not None:
        uncompressed_marginal_signed = (
            float(real_branch["validation_loss_after"]) - validation_loss_after
        ) / (abs(float(real_branch["validation_loss_after"])) + 1e-12)
        residual_layers = []
        residual_error_sq = 0.0
        residual_reference_sq = 0.0
        for real_layer, combined_layer in zip(real_branch["layers"], layers, strict=True):
            residual_dense = combined_layer.dense().astype(np.float64) - real_layer.dense().astype(np.float64)
            residual_layer = factorize_dense_layer(
                real_layer.layer_name,
                residual_dense,
                synthetic_residual_rank,
            )
            residual_layers.append(residual_layer)
            residual_error_sq += float(np.sum((residual_layer.dense() - residual_dense) ** 2))
            residual_reference_sq += float(np.sum(residual_dense**2))
        deployed_layers = tuple(
            factorize_dense_layer(
                real_layer.layer_name,
                real_layer.dense().astype(np.float64) + residual_layer.dense().astype(np.float64),
                int(spec["rank"]),
            )
            for real_layer, residual_layer in zip(
                real_branch["layers"], residual_layers, strict=True
            )
        )
        load_adapter_layers(deployed_layers)
        deployed_validation_loss_after = deterministic_loss(validation_batch, seed + 2000)
        marginal_signed = (
            float(real_branch["validation_loss_after"]) - deployed_validation_loss_after
        ) / (abs(float(real_branch["validation_loss_after"])) + 1e-12)
        marginal_reliability = min(max(marginal_signed, 0.0), 1.0)
        dual_compute_time_s = time.perf_counter() - started
        real_reliability = heldout_real_reliability(
            validation_loss_before,
            float(real_branch["validation_loss_after"]),
        )
        real_update = ClientAdapterUpdate(
            client_id=spec["client_id"],
            layers=real_branch["layers"],
            n_real=len(train_indices),
            reliability=real_reliability,
            staleness=int(spec.get("staleness", 0)),
            compute_time_s=float(real_branch["compute_time_s"]),
        )
        residual_update = ClientAdapterUpdate(
            client_id=spec["client_id"],
            layers=tuple(residual_layers),
            n_real=len(train_indices),
            reliability=marginal_reliability,
            staleness=int(spec.get("staleness", 0)),
            compute_time_s=combined_branch_time,
        )
        dual_package = DualChannelAdapterUpdate(
            client_id=spec["client_id"],
            real=real_update,
            synthetic_residual=residual_update,
            marginal_reliability=marginal_reliability,
            compute_time_s=dual_compute_time_s,
        )
        dual_channel_upload = save_dual_channel_update(dual_package, out_dir / "dual_channel_upload")
        dual_channel_report = {
            "real_train_loss_after": float(real_branch["train_loss_after"]),
            "real_validation_loss_after": float(real_branch["validation_loss_after"]),
            "real_reliability": real_reliability,
            "uncompressed_marginal_reliability_signed": uncompressed_marginal_signed,
            "deployed_candidate_validation_loss_after": deployed_validation_loss_after,
            "marginal_reliability_signed": marginal_signed,
            "marginal_reliability": marginal_reliability,
            "real_dense_delta_norm": float(real_branch["dense_delta_norm"]),
            "synthetic_residual_rank": synthetic_residual_rank,
            "deployed_adapter_rank": int(spec["rank"]),
            "dual_channel_compute_time_s": dual_compute_time_s,
            "synthetic_residual_relative_error": math.sqrt(residual_error_sq / max(residual_reference_sq, 1e-24)),
            "real_step_records": real_branch["step_records"],
        }
    elif update_contract == "dual_channel":
        dual_package = DualChannelAdapterUpdate(
            client_id=spec["client_id"],
            real=update,
            synthetic_residual=None,
            marginal_reliability=0.0,
            compute_time_s=elapsed,
        )
        dual_channel_upload = save_dual_channel_update(dual_package, out_dir / "dual_channel_upload")
        dual_channel_report = {
            "real_train_loss_after": train_loss_after,
            "real_validation_loss_after": validation_loss_after,
            "real_reliability": reliability,
            "marginal_reliability_signed": 0.0,
            "marginal_reliability": 0.0,
            "real_dense_delta_norm": dense_delta_norm,
            "synthetic_residual_rank": None,
            "synthetic_residual_relative_error": 0.0,
            "real_step_records": step_records,
        }
    finite = all(
        math.isfinite(value)
        for value in (train_loss_before, train_loss_after, validation_loss_before, validation_loss_after, dense_delta_norm)
    )
    optimization_signal_ready = (
        synthetic_loss_after < synthetic_loss_before
        if gradient_mode in PAIRED_DISTILL_MODES
        and synthetic_loss_before is not None
        and synthetic_loss_after is not None
        else train_loss_after < train_loss_before
    )
    local_update_ready = finite and dense_delta_norm > 0.0 and upload["total_upload_bytes"] > 0
    if gradient_mode in PAIRED_DISTILL_MODES:
        local_update_ready = bool(
            local_update_ready
            and paired_distillation_report is not None
            and paired_distillation_report["deployment_gate_passed"]
        )
    if record_marginal_utility:
        local_update_ready = bool(
            local_update_ready
            and marginal_utility_report is not None
            and marginal_utility_report["episode_count"] >= 2
            and marginal_utility_report["all_episode_utilities_finite"]
        )
    if gradient_mode == "factor_paired":
        residual_contract = factor_paired_report["residual_contract"]
        local_update_ready = bool(
            local_update_ready
            and factor_residual_upload is not None
            and residual_contract["nonperception_residual_dense_norm"] <= 1e-12
            and residual_contract["reconstruction_relative_error"] <= 1e-6
        )
    if update_contract == "dual_channel":
        local_update_ready = local_update_ready and dual_channel_upload is not None
    augmentation_portfolio_report = None
    if gradient_mode == "augmentation_portfolio":
        portfolio_records = [
            record
            for record in step_records
            if record.get("portfolio_gs_ratio") is not None
        ]
        if len(portfolio_records) != int(spec["steps"]):
            raise RuntimeError(
                "Augmentation portfolio did not produce one audit record per "
                "training step"
            )
        augmentation_portfolio_report = {
            "schema": "fedsplat.augmentation_portfolio_execution.v1",
            "gs_ratio": portfolio_gs_ratio,
            "two_d_ratio": portfolio_two_d_ratio,
            "equal_auxiliary_budget": bool(
                augmentation_portfolio_config["equal_auxiliary_budget"]
            ),
            "two_d_source": augmentation_portfolio_config["two_d_source"],
            "gs_source_provenance": source_provenance,
            "gs_quality_score": selected_quality_score,
            "gs_quality_gate_active": True,
            "gs_projection_base": "real_plus_weighted_2d",
            "gs_projection_rate": float(np.mean([
                bool(record["projection_applied"])
                for record in portfolio_records
            ])),
            "mean_effective_gs_weight": float(np.mean([
                float(record["effective_synthetic_weight"])
                for record in portfolio_records
            ])),
            "mean_two_d_gradient_norm": float(np.mean([
                float(record["two_d_gradient_norm"])
                for record in portfolio_records
            ])),
            "two_d_loss_before": portfolio_two_d_loss_before,
            "two_d_loss_after": portfolio_two_d_loss_after,
            "gs_forward_steps": sum(
                bool(record["synthetic_batch_indices"])
                for record in portfolio_records
            ),
            "two_d_forward_steps": sum(
                bool(record["two_d_batch_indices"])
                for record in portfolio_records
            ),
        }
    report = {
        "schema": "fedsplat.smolvla_local_update.v6",
        "evidence_label": (
            "real_model_local_update"
            if gradient_mode == "real_only"
            else f"{source_provenance}_model_local_update"
        ),
        "client_id": spec["client_id"],
        "training_config_sha256": local_update_config_fingerprint(spec),
        "seed": seed,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "repo_id": spec["repo_id"],
        "robot_adapter": robot_adapter_report,
        "device": device,
        "train_episodes": train_episodes,
        "validation_episodes": validation_episodes,
        "owner_local_split": spec.get("owner_local_split"),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "samples_per_episode": samples_per_episode,
        "train_samples_per_episode": train_samples_per_episode,
        "validation_samples_per_episode": validation_samples_per_episode,
        "batch_size": batch_size,
        "evaluation_batch_size": evaluation_batch_size,
        "resample_each_step": dynamic_batches,
        "expected_task_index": expected_task_index,
        "gradient_mode": gradient_mode,
        "effective_training_mode": effective_training_mode,
        "factor_label": spec.get("factor_label"),
        "factor_variations": spec.get("factor_variations", []),
        "factor_counterfactual_mode": (
            factor_counterfactual_mode if gradient_mode == "factor_paired" else None
        ),
        "sharpness_mode": sharpness_mode,
        "sam_rho": sam_rho if sharpness_mode == "sam" else 0.0,
        "synthetic_gradient_scope": synthetic_gradient_scope,
        "synthetic_objective": synthetic_objective,
        "policy_anchor_mode": policy_anchor_mode,
        "representation_anchor_mode": representation_anchor_mode,
        "representation_anchor_weight": representation_anchor_weight,
        "max_anchor_to_bc_perception_norm": max_anchor_to_bc_perception_norm,
        "paired_action_consistency_weight": paired_action_consistency_weight,
        "paired_feature_consistency_weight": paired_feature_consistency_weight,
        "paired_synthetic_bc_weight": paired_synthetic_bc_weight,
        "paired_teacher_mode": paired_teacher_mode,
        "paired_teacher_reference": paired_teacher_reference,
        "paired_action_consistency_mode": paired_action_consistency_mode,
        "paired_canonical_action_identity_weight": (
            paired_canonical_action_identity_weight
        ),
        "paired_canonical_feature_identity_weight": (
            paired_canonical_feature_identity_weight
        ),
        "source_pairing_audit": source_pairing_audit,
        "source_pair_sampling_audit": source_pair_sampling_audit,
        "perception_module_patterns": perception_patterns,
        "perception_modules": sorted(perception_modules),
        "nonperception_modules": sorted(set(adapters) - perception_modules),
        "local_trainable_module_patterns": local_trainable_patterns,
        "local_trainable_modules": sorted(local_trainable_modules),
        "locally_frozen_modules": sorted(set(adapters) - local_trainable_modules),
        "update_contract": update_contract,
        "synthetic_weight": synthetic_weight,
        "max_auxiliary_to_real_norm": max_auxiliary_to_real_norm,
        "projection_cosine_threshold": projection_cosine_threshold,
        "synthetic_source": {
            "type": source_type,
            "provenance": source_provenance,
            "indices": synthetic_indices,
            "candidate_episode_ids": candidate_episode_ids,
            "anchor_episode_ids": anchor_episode_ids,
            "selected_episode_ids": selected_synthetic_episode_ids,
            "selected_indices": selected_synthetic_indices,
            "feasibility": gate.to_dict(),
            "quality_score": quality_score,
            "measured_selected_quality_score": measured_selected_quality_score,
            "selected_quality_score": selected_quality_score,
            "quality_weighting_enabled": use_synthetic_quality_weight,
            "quality_manifest": quality_evidence_path,
        },
        "augmentation_portfolio": augmentation_portfolio_report,
        "trajectory_selection": crossfit_selection_report,
        "crossfit": crossfit_report,
        "factor_paired": factor_paired_report,
        "paired_distillation": paired_distillation_report,
        "marginal_utility": marginal_utility_report,
        "steps_completed": len(step_records),
        "target_modules": list(adapters),
        "rank": int(spec["rank"]),
        "alpha": float(spec["alpha"]),
        "learning_rate": float(spec["learning_rate"]),
        "initial_adapter_update": initial_adapter_path,
        "proximal_mu": proximal_mu,
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable_adapter_parameters": sum(parameter.numel() for parameter in trainable),
        "train_loss_before": train_loss_before,
        "train_loss_after": train_loss_after,
        "synthetic_loss_before": synthetic_loss_before,
        "synthetic_loss_after": synthetic_loss_after,
        "synthetic_anchor_loss_before": synthetic_anchor_loss_before,
        "synthetic_anchor_loss_after": synthetic_anchor_loss_after,
        "validation_loss_before": validation_loss_before,
        "validation_loss_after": validation_loss_after,
        "heldout_real_reliability": reliability,
        "dense_delta_norm": dense_delta_norm,
        "step_records": step_records,
        "compute_time_s": elapsed,
        "n_real_samples": len(train_indices),
        "staleness": int(spec.get("staleness", 0)),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0,
        "upload": upload,
        "factor_residual_upload": factor_residual_upload,
        "factor_counterfactual_reference_local": factor_counterfactual_reference,
        "factor_real_reference_local": factor_real_reference,
        "dual_channel_upload": dual_channel_upload,
        "dual_channel": dual_channel_report,
        "optimization_signal_ready": optimization_signal_ready,
        "local_update_ready": local_update_ready,
        "closed_loop_ready": False,
    }
    (out_dir / "local_update_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_worker_report(report, out_dir / "SMOLVLA_LOCAL_UPDATE_REPORT.md")
    return report


def _write_worker_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# SmolVLA Local Update Report",
        "",
        f"- Evidence: `{report['evidence_label']}`",
        f"- Local update ready: {report['local_update_ready']}",
        f"- Closed-loop ready: {report['closed_loop_ready']}",
        f"- Train episodes: {report['train_episodes']}",
        f"- Validation episodes: {report['validation_episodes']}",
        f"- Real samples: {report['n_real_samples']}",
        f"- Gradient mode: `{report['gradient_mode']}`",
        f"- Effective training mode: `{report.get('effective_training_mode', report['gradient_mode'])}`",
        f"- Sharpness mode: `{report.get('sharpness_mode', 'none')}` (rho={report.get('sam_rho', 0.0)})",
        f"- Synthetic-gradient scope: `{report.get('synthetic_gradient_scope', 'all')}`",
        f"- Synthetic objective: `{report.get('synthetic_objective', 'action_bc')}`",
        f"- Paired teacher: `{report.get('paired_teacher_mode', 'current_policy')}`",
        f"- Locally trainable modules: {len(report.get('local_trainable_modules', report.get('target_modules', [])))}",
        f"- Policy anchor: `{report.get('policy_anchor_mode', 'none')}`",
        f"- Representation anchor: `{report.get('representation_anchor_mode', 'none')}`",
        f"- Dynamic minibatches: {report.get('resample_each_step', False)}",
        f"- Update contract: `{report.get('update_contract', 'combined')}`",
        f"- Auxiliary/real norm cap: {report['max_auxiliary_to_real_norm']}",
        f"- Auxiliary provenance: `{report['synthetic_source']['provenance']}`",
        f"- Auxiliary quality: {report['synthetic_source']['quality_score']:.6f}",
        f"- Selected synthetic episodes: {report['synthetic_source'].get('selected_episode_ids', [])}",
        f"- Steps: {report['steps_completed']}",
        f"- Train loss: {report['train_loss_before']:.6f} -> {report['train_loss_after']:.6f}",
        f"- Validation loss: {report['validation_loss_before']:.6f} -> {report['validation_loss_after']:.6f}",
        f"- Held-out-real reliability: {report['heldout_real_reliability']:.6f}",
        f"- Dense update norm: {report['dense_delta_norm']:.6f}",
        f"- Upload bytes: {report['upload']['total_upload_bytes']}",
        f"- Runtime wall time: {report.get('runtime_wall_s', report['compute_time_s']):.3f} s",
        "",
        "This report records a SmolVLA forward/backward/update and serialized",
        "adapter upload completed on disjoint real train/validation episodes.",
        "Auxiliary provenance is reported literally; a visual proxy is not",
        "Gaussian-demonstration evidence. This is not task-success evidence.",
    ]
    if report.get("crossfit"):
        crossfit = report["crossfit"]
        lines.extend([
            "",
            "## Cross-Fit Gaussian Replay",
            "",
            f"- Committed branch: `{crossfit['selected_branch']}`",
            f"- Anchor relative gain: {crossfit['anchor_gain']:.6f}",
            f"- Anchor gate passed: {crossfit['anchor_pass']}",
            f"- Real-validation gate passed: {crossfit['real_validation_pass']}",
            f"- Eligible trajectories: {report['trajectory_selection']['eligible_count']}",
            f"- Selected trajectories: {len(report['trajectory_selection']['selected_episode_ids'])}",
        ])
    if report.get("augmentation_portfolio"):
        portfolio = report["augmentation_portfolio"]
        lines.extend([
            "",
            "## Adaptive Augmentation Portfolio",
            "",
            f"- 2D/GS ratio: {portfolio['two_d_ratio']:.2f}/{portfolio['gs_ratio']:.2f}",
            f"- GS quality score: {portfolio['gs_quality_score']:.6f}",
            f"- Mean effective GS weight: {portfolio['mean_effective_gs_weight']:.6f}",
            f"- GS projection rate: {portfolio['gs_projection_rate']:.6f}",
            f"- GS projection base: `{portfolio['gs_projection_base']}`",
            f"- 2D loss: {portfolio['two_d_loss_before']:.6f} -> "
            f"{portfolio['two_d_loss_after']:.6f}",
            f"- 2D/GS forward steps: {portfolio['two_d_forward_steps']}/"
            f"{portfolio['gs_forward_steps']}",
        ])
    if report.get("paired_distillation"):
        distillation = report["paired_distillation"]
        lines.extend([
            "",
            "## Frozen-Teacher Paired Distillation",
            "",
            f"- Train-pair relative gain: {distillation['train_pair_gain']:.6f}",
            f"- Held-out-pair relative gain: {distillation['heldout_pair_gain']:.6f}",
            f"- Canonical validation gate passed: {distillation['real_validation_pass']}",
            f"- Deployment gate passed: {distillation['deployment_gate_passed']}",
        ])
    if report.get("dual_channel"):
        dual = report["dual_channel"]
        lines.extend([
            "",
            "## Dual-Channel Update",
            "",
            f"- Real-channel reliability: {dual['real_reliability']:.6f}",
            f"- Synthetic marginal reliability: {dual['marginal_reliability_signed']:.6f}",
            f"- Synthetic residual rank: {dual['synthetic_residual_rank']}",
            f"- Synthetic residual relative error: {dual['synthetic_residual_relative_error']:.6f}",
            f"- Beacon bytes: {report['dual_channel_upload']['beacon_bytes']}",
            f"- Mandatory real upload bytes: {report['dual_channel_upload']['real_upload_bytes']}",
            "- Optional synthetic residual upload bytes: "
            f"{report['dual_channel_upload']['synthetic_residual_upload_bytes']}",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-config", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.worker_config).read_text(encoding="utf-8"))
    errors = validate_local_update_spec(spec)
    if errors:
        raise ValueError("Invalid worker config: " + "; ".join(errors))
    runtime_started = time.perf_counter()
    report = _worker(spec)
    report["runtime_wall_s"] = time.perf_counter() - runtime_started
    out_dir = Path(spec["out_dir"])
    (out_dir / "local_update_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_worker_report(report, out_dir / "SMOLVLA_LOCAL_UPDATE_REPORT.md")
    print(json.dumps({"local_update_ready": report["local_update_ready"], "report": str(Path(spec["out_dir"]) / "local_update_report.json")}))


if __name__ == "__main__":
    main()
