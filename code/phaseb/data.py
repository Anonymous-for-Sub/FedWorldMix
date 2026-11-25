"""Meta-World task split and data interface for Phase B."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ClientDatasetRef:
    client_id: str
    task_name: str
    split: str
    path: str


@dataclass
class SampleRecord:
    instruction: str
    proprio: List[float]
    action: List[float]
    success: float | None
    trajectory_id: str | None
    step_in_trajectory: int | None
    foreground_objects: List[str]
    background_objects: List[str]
    image_tokens: List[List[float]]
    camera_views: List[str] = field(default_factory=list)
    rgb_npz_path: str | None = None
    rgb_step: int | None = None
    rgb_views: List[str] = field(default_factory=list)


def _as_float_list(value: object) -> List[float]:
    if not isinstance(value, list):
        return []
    out: List[float] = []
    for item in value:
        if isinstance(item, (int, float)):
            out.append(float(item))
    return out


def _as_token_matrix(value: object) -> List[List[float]]:
    if not isinstance(value, list):
        return []
    matrix: List[List[float]] = []
    for row in value:
        parsed = _as_float_list(row)
        if parsed:
            matrix.append(parsed)
    return matrix


def _as_string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


def _first_present(payload: Dict[str, object], keys: List[str]) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _extract_nested(payload: Dict[str, object], path: str) -> object | None:
    current: object = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _extract_instruction(payload: Dict[str, object]) -> str:
    direct = _first_present(payload, ["instruction", "lang", "language", "text", "task_instruction"])
    if isinstance(direct, str):
        return direct
    nested = _extract_nested(payload, "meta.instruction")
    return str(nested) if isinstance(nested, str) else ""


def _extract_proprio(payload: Dict[str, object]) -> List[float]:
    for key in ["proprio", "state", "robot_state"]:
        value = payload.get(key, None)
        parsed = _as_float_list(value)
        if parsed:
            return parsed
    nested = _extract_nested(payload, "obs.proprio")
    parsed_nested = _as_float_list(nested)
    if parsed_nested:
        return parsed_nested
    return []


def _extract_action(payload: Dict[str, object]) -> List[float]:
    for key in ["action", "actions", "target_action", "expert_action", "act"]:
        parsed = _as_float_list(payload.get(key, None))
        if parsed:
            return parsed
    nested = _extract_nested(payload, "target.action")
    parsed_nested = _as_float_list(nested)
    if parsed_nested:
        return parsed_nested
    return []


def _extract_success(payload: Dict[str, object]) -> float | None:
    value = _first_present(payload, ["success", "success_rate", "label_success"])
    if isinstance(value, (int, float)):
        return float(value)
    nested = _extract_nested(payload, "metrics.success")
    if isinstance(nested, (int, float)):
        return float(nested)
    return None


def _extract_trajectory_id(payload: Dict[str, object]) -> str | None:
    value = _first_present(payload, ["trajectory_id", "traj_id", "episode_id", "demo_id"])
    if isinstance(value, (str, int)):
        return str(value)
    nested = _extract_nested(payload, "meta.trajectory_id")
    if isinstance(nested, (str, int)):
        return str(nested)
    return None


def _extract_step_in_trajectory(payload: Dict[str, object]) -> int | None:
    value = _first_present(payload, ["step_in_trajectory", "step_id", "step", "t"])
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    nested = _extract_nested(payload, "meta.step_in_trajectory")
    if isinstance(nested, int):
        return int(nested)
    if isinstance(nested, float):
        return int(nested)
    return None


def _extract_image_tokens(payload: Dict[str, object]) -> List[List[float]]:
    direct = _as_token_matrix(payload.get("image_tokens", None))
    if direct:
        return direct

    alt = _as_token_matrix(payload.get("vision_tokens", None))
    if alt:
        return alt

    observations = payload.get("observations", None)
    if isinstance(observations, dict):
        tokens = _as_token_matrix(observations.get("image_tokens", None))
        if tokens:
            return tokens
        cameras = observations.get("cameras", None)
        if isinstance(cameras, dict):
            merged: List[List[float]] = []
            for camera_value in cameras.values():
                merged.extend(_as_token_matrix(camera_value))
            if merged:
                return merged
    return []


def _extract_rgb_npz_path(payload: Dict[str, object]) -> str | None:
    value = payload.get("rgb_npz_path", None)
    if isinstance(value, str) and value:
        return value
    return None


def _extract_rgb_step(payload: Dict[str, object]) -> int | None:
    value = payload.get("rgb_step", None)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return None


def build_task_clients(tasks: List[str], clients_per_task: int = 1) -> Dict[str, str]:
    """Return mapping: client_id -> task_name."""
    mapping: Dict[str, str] = {}
    client_index = 0
    for task in tasks:
        for _ in range(clients_per_task):
            mapping[f"client{client_index}"] = task
            client_index += 1
    return mapping


def build_dataset_layout(root: str, client_to_task: Dict[str, str]) -> Dict[str, Dict[str, ClientDatasetRef]]:
    """Build expected train/val/test paths for each client."""
    out: Dict[str, Dict[str, ClientDatasetRef]] = {}
    for client_id, task_name in client_to_task.items():
        out[client_id] = {}
        for split in ("train", "val", "test"):
            path = f"{root}/{task_name}/{split}.jsonl"
            out[client_id][split] = ClientDatasetRef(
                client_id=client_id,
                task_name=task_name,
                split=split,
                path=path,
            )
    return out


def missing_paths(layout: Dict[str, Dict[str, ClientDatasetRef]]) -> List[str]:
    """Return paths that are expected but currently absent."""
    missing: List[str] = []
    for client_splits in layout.values():
        for ref in client_splits.values():
            if not os.path.exists(ref.path):
                missing.append(ref.path)
    return missing


def read_jsonl_records(path: str, limit: int = 0) -> List[SampleRecord]:
    """Read sample records from a JSONL file.

    Expected line schema keys:
    - instruction (str)
    - proprio (list[float])
    - action (list[float])
    - foreground_objects (list[str], optional)
    - background_objects (list[str], optional)
    - image_tokens (list[list[float]], optional)
    - camera_views (list[str], optional)
    - rgb_npz_path (str, optional)
    - rgb_step (int, optional)
    - rgb_views (list[str], optional)
    """
    records: List[SampleRecord] = []
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(
                SampleRecord(
                    instruction=_extract_instruction(payload),
                    proprio=_extract_proprio(payload),
                    action=_extract_action(payload),
                    success=_extract_success(payload),
                    trajectory_id=_extract_trajectory_id(payload),
                    step_in_trajectory=_extract_step_in_trajectory(payload),
                    foreground_objects=_as_string_list(payload.get("foreground_objects", [])),
                    background_objects=_as_string_list(payload.get("background_objects", [])),
                    image_tokens=_extract_image_tokens(payload),
                    camera_views=_as_string_list(payload.get("camera_views", [])),
                    rgb_npz_path=_extract_rgb_npz_path(payload),
                    rgb_step=_extract_rgb_step(payload),
                    rgb_views=_as_string_list(payload.get("rgb_views", [])),
                )
            )
            if limit > 0 and len(records) >= limit:
                break
    return records


def summarize_layout(layout: Dict[str, Dict[str, ClientDatasetRef]], max_samples: int = 0) -> Dict[str, Dict[str, int]]:
    """Return per-client split sample counts."""
    summary: Dict[str, Dict[str, int]] = {}
    for client_id, client_splits in layout.items():
        summary[client_id] = {}
        for split, ref in client_splits.items():
            summary[client_id][split] = len(read_jsonl_records(ref.path, limit=max_samples))
    return summary
