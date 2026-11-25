"""Native PiPER state/action contract for FedSplat VLA backends."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


ROBOT_TYPE = "piper_single_d455"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_DIMENSION = 7
ACTION_DIMENSION = 7
STATE_NAMES = (
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "gripper_open_fraction",
)
ACTION_NAMES = (
    "joint_target_1_rad",
    "joint_target_2_rad",
    "joint_target_3_rad",
    "joint_target_4_rad",
    "joint_target_5_rad",
    "joint_target_6_rad",
    "gripper_target_open_fraction",
)


def aligned_state_action(
    state: np.ndarray,
    action: np.ndarray,
    alignment: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the manifest supervision convention without changing dimensions."""
    state = np.asarray(state)
    action = np.asarray(action)
    if state.ndim != 2 or state.shape[1] != STATE_DIMENSION:
        raise ValueError(f"PiPER state must have shape [T, 7], got {state.shape}")
    if action.ndim != 2 or action.shape[1] != ACTION_DIMENSION:
        raise ValueError(f"PiPER action must have shape [T, 7], got {action.shape}")
    if len(state) != len(action):
        raise ValueError("PiPER state and action lengths must match")
    if alignment == "same_index":
        return state, action
    if alignment == "next_index":
        if len(state) < 2:
            raise ValueError("next_index alignment requires at least two frames")
        return state[:-1], action[1:]
    raise ValueError(f"Unknown PiPER action alignment: {alignment}")


def normalize_native(
    values: np.ndarray,
    statistics: dict[str, Any],
) -> np.ndarray:
    """Apply train-only mean/std statistics to a native 7D vector."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] != STATE_DIMENSION:
        raise ValueError(f"Expected native 7D values, got {values.shape}")
    mean = np.asarray(statistics["mean"], dtype=np.float32)
    std = np.asarray(statistics["std"], dtype=np.float32)
    if mean.shape != (STATE_DIMENSION,) or std.shape != (STATE_DIMENSION,):
        raise ValueError("PiPER normalization statistics must be 7D")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("PiPER normalization statistics must be finite")
    if np.any(std <= 0):
        raise ValueError("PiPER normalization standard deviations must be positive")
    return (values - mean) / std


def _feature_with_shape(feature: Any, shape: tuple[int, ...]) -> Any:
    if isinstance(feature, dict):
        updated = copy.deepcopy(feature)
        updated["shape"] = list(shape)
        return updated
    feature_type = type(feature)
    try:
        return feature_type(type=feature.type, shape=shape)
    except (AttributeError, TypeError) as error:
        raise TypeError(f"Unsupported policy feature object: {feature!r}") from error


def configure_smolvla_policy_for_piper(policy: Any) -> dict[str, Any]:
    """Set native 7D feature metadata while preserving model-internal padding."""
    config = policy.config
    if int(config.max_state_dim) < STATE_DIMENSION:
        raise ValueError("SmolVLA max_state_dim is smaller than PiPER state")
    if int(config.max_action_dim) < ACTION_DIMENSION:
        raise ValueError("SmolVLA max_action_dim is smaller than PiPER action")
    if STATE_KEY not in config.input_features:
        raise KeyError(f"SmolVLA checkpoint lacks {STATE_KEY}")
    if ACTION_KEY not in config.output_features:
        raise KeyError(f"SmolVLA checkpoint lacks {ACTION_KEY}")
    config.input_features[STATE_KEY] = _feature_with_shape(
        config.input_features[STATE_KEY],
        (STATE_DIMENSION,),
    )
    config.output_features[ACTION_KEY] = _feature_with_shape(
        config.output_features[ACTION_KEY],
        (ACTION_DIMENSION,),
    )
    return {
        "robot_type": ROBOT_TYPE,
        "native_state_dimension": STATE_DIMENSION,
        "native_action_dimension": ACTION_DIMENSION,
        "model_internal_state_dimension": int(config.max_state_dim),
        "model_internal_action_dimension": int(config.max_action_dim),
        "dataset_dummy_dimensions_added": 0,
    }


def validate_piper_lerobot_metadata(metadata: Any) -> dict[str, Any]:
    """Reject a mislabeled or padded PiPER LeRobot dataset."""
    if metadata.robot_type != ROBOT_TYPE:
        raise ValueError(
            f"PiPER dataset robot_type must be {ROBOT_TYPE}, "
            f"got {metadata.robot_type}"
        )
    shapes = metadata.shapes
    if tuple(shapes.get(STATE_KEY, ())) != (STATE_DIMENSION,):
        raise ValueError(
            f"PiPER dataset {STATE_KEY} must be native 7D, "
            f"got {shapes.get(STATE_KEY)}"
        )
    if tuple(shapes.get(ACTION_KEY, ())) != (ACTION_DIMENSION,):
        raise ValueError(
            f"PiPER dataset {ACTION_KEY} must be native 7D, "
            f"got {shapes.get(ACTION_KEY)}"
        )
    names = metadata.names
    state_names = tuple((names.get(STATE_KEY) or {}).get("motors", ()))
    action_names = tuple((names.get(ACTION_KEY) or {}).get("motors", ()))
    if state_names != STATE_NAMES:
        raise ValueError(
            f"PiPER dataset state names differ from the frozen contract: "
            f"{state_names}"
        )
    if action_names != ACTION_NAMES:
        raise ValueError(
            f"PiPER dataset action names differ from the frozen contract: "
            f"{action_names}"
        )
    return {
        "robot_type": metadata.robot_type,
        "native_state_dimension": STATE_DIMENSION,
        "native_action_dimension": ACTION_DIMENSION,
        "state_names": list(state_names),
        "action_names": list(action_names),
        "dataset_dummy_dimensions_added": 0,
    }
