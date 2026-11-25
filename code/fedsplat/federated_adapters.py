"""Project-owned aggregation of serialized client LoRA updates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .lora_updates import (
    ClientAdapterUpdate,
    aggregate_lora_updates,
    load_client_update,
    save_client_update,
)
from .method import reliability_raw_weight


AGGREGATION_MODES = {"fedavg", "reliability"}


def validate_federated_adapter_spec(spec: dict[str, Any]) -> list[str]:
    errors = []
    if spec.get("aggregation_mode") not in AGGREGATION_MODES:
        errors.append(f"aggregation_mode must be one of {sorted(AGGREGATION_MODES)}")
    if not isinstance(spec.get("server_rank"), int) or int(spec.get("server_rank", 0)) <= 0:
        errors.append("server_rank must be a positive integer")
    if not isinstance(spec.get("global_update_id"), str) or not str(spec.get("global_update_id", "")).strip():
        errors.append("global_update_id must be a nonempty string")
    updates = spec.get("updates")
    if not isinstance(updates, list) or len(updates) < 2:
        errors.append("updates must contain at least two client update entries")
    else:
        paths = []
        for index, item in enumerate(updates):
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"].strip():
                errors.append(f"updates[{index}].path must be a nonempty string")
                continue
            paths.append(item["path"])
        if len(paths) != len(set(paths)):
            errors.append("client update paths must be unique")
    gamma = spec.get("staleness_gamma", 0.1)
    floor = spec.get("reliability_floor", 0.05)
    if not isinstance(gamma, (int, float)) or not math.isfinite(float(gamma)) or float(gamma) < 0.0:
        errors.append("staleness_gamma must be finite and nonnegative")
    if not isinstance(floor, (int, float)) or not math.isfinite(float(floor)) or not 0.0 < float(floor) <= 1.0:
        errors.append("reliability_floor must be finite and in (0, 1]")
    return errors


def aggregate_adapter_manifest(config_path: Path, out_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = _resolve_project_root(config_path)
    spec = json.loads(config_path.read_text(encoding="utf-8"))
    errors = validate_federated_adapter_spec(spec)
    if errors:
        raise ValueError("Invalid federated adapter config: " + "; ".join(errors))

    loaded = []
    metadata_rows = []
    for item in spec["updates"]:
        update_path = _project_path(project_root, item["path"])
        update = load_client_update(update_path)
        loaded.append(update)
        metadata_path = update_path if update_path.suffix == ".json" else update_path / "adapter_update.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_rows.append({
            "client_id": update.client_id,
            "path": str(update_path),
            "n_real": update.n_real,
            "reliability": update.reliability,
            "staleness": update.staleness,
            "compute_time_s": update.compute_time_s,
            "upload_bytes": int(metadata["total_upload_bytes"]),
        })
    client_ids = [update.client_id for update in loaded]
    if len(client_ids) != len(set(client_ids)):
        raise ValueError("Loaded client update IDs must be unique")

    mode = str(spec["aggregation_mode"])
    raw_weights = {}
    for update in loaded:
        if mode == "fedavg":
            raw_weights[update.client_id] = float(update.n_real)
        else:
            raw_weights[update.client_id] = reliability_raw_weight(
                update.n_real,
                update.reliability,
                update.staleness,
                reliability_floor=float(spec.get("reliability_floor", 0.05)),
                staleness_gamma=float(spec.get("staleness_gamma", 0.1)),
            )
    total_weight = sum(raw_weights.values())
    if total_weight <= 0.0:
        raise ValueError("Client aggregation weights have no positive mass")
    weights = {client_id: value / total_weight for client_id, value in raw_weights.items()}
    layers = aggregate_lora_updates(loaded, weights, server_rank=int(spec["server_rank"]))

    layer_errors = []
    for layer_index, layer in enumerate(layers):
        exact = sum(
            weights[update.client_id] * update.layers[layer_index].dense().astype(np.float64)
            for update in loaded
        )
        reconstructed = layer.dense().astype(np.float64)
        exact_norm = float(np.linalg.norm(exact))
        error_norm = float(np.linalg.norm(exact - reconstructed))
        layer_errors.append({
            "layer_name": layer.layer_name,
            "exact_dense_norm": exact_norm,
            "compression_error_norm": error_norm,
            "relative_compression_error": error_norm / max(exact_norm, 1e-12),
            "server_rank": layer.rank,
        })

    global_update = ClientAdapterUpdate(
        client_id=str(spec["global_update_id"]),
        layers=layers,
        n_real=sum(update.n_real for update in loaded),
        reliability=sum(weights[update.client_id] * update.reliability for update in loaded),
        staleness=0,
        compute_time_s=sum(update.compute_time_s for update in loaded),
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    download = save_client_update(global_update, out_dir / "adapter_upload")
    finite = all(
        math.isfinite(item["relative_compression_error"])
        for item in layer_errors
    )
    report = {
        "schema": "fedsplat.federated_adapter_aggregation.v1",
        "evidence_label": "federated_adapter_aggregation",
        "aggregation_mode": mode,
        "server_rank": int(spec["server_rank"]),
        "global_update_id": global_update.client_id,
        "num_clients": len(loaded),
        "weights": weights,
        "clients": metadata_rows,
        "total_client_upload_bytes": sum(item["upload_bytes"] for item in metadata_rows),
        "global_download": download,
        "layer_compression": layer_errors,
        "aggregation_ready": finite and download["total_upload_bytes"] > 0,
        "closed_loop_ready": False,
    }
    (out_dir / "federated_aggregation_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_report(report, out_dir / "FEDERATED_AGGREGATION_REPORT.md")
    return report


def _resolve_project_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "fedsplat").is_dir() and (candidate / "runs").is_dir():
            return candidate
    return config_path.parent.parent


def _project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _write_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Federated Adapter Aggregation Report",
        "",
        f"- Evidence: `{report['evidence_label']}`",
        f"- Aggregation ready: {report['aggregation_ready']}",
        f"- Closed-loop ready: {report['closed_loop_ready']}",
        f"- Mode: `{report['aggregation_mode']}`",
        f"- Clients: {report['num_clients']}",
        f"- Client upload bytes: {report['total_client_upload_bytes']}",
        f"- Global download bytes: {report['global_download']['total_upload_bytes']}",
        "",
        "Aggregation evidence does not imply policy task success. The emitted",
        "adapter must be reapplied and evaluated in closed loop.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
