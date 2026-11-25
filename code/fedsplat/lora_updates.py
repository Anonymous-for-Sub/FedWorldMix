"""Serializable LoRA update contract and heterogeneous-rank aggregation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoRALayerUpdate:
    layer_name: str
    a: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        a = np.asarray(self.a, dtype=np.float32)
        b = np.asarray(self.b, dtype=np.float32)
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("LoRA A and B must be matrices")
        if b.shape[1] != a.shape[0]:
            raise ValueError(f"LoRA rank mismatch for {self.layer_name}: B{b.shape} and A{a.shape}")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError("LoRA matrices must contain only finite values")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    @property
    def rank(self) -> int:
        return int(self.a.shape[0])

    @property
    def dense_shape(self) -> tuple[int, int]:
        return int(self.b.shape[0]), int(self.a.shape[1])

    def dense(self) -> np.ndarray:
        return self.b @ self.a


@dataclass(frozen=True)
class ClientAdapterUpdate:
    client_id: str
    layers: tuple[LoRALayerUpdate, ...]
    n_real: int
    reliability: float
    staleness: int = 0
    compute_time_s: float = 0.0

    def __post_init__(self) -> None:
        names = [layer.layer_name for layer in self.layers]
        if not self.client_id or not self.layers or len(names) != len(set(names)):
            raise ValueError("Client update requires an ID and uniquely named layers")
        if self.n_real < 0 or self.staleness < 0 or self.compute_time_s < 0.0:
            raise ValueError("Client update counts and costs must be nonnegative")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("Client reliability must be in [0, 1]")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def client_update_artifact_fingerprint(path: Path) -> dict[str, Any]:
    """Return stable hashes for an adapter package used across FL rounds."""
    metadata_path = path if path.suffix == ".json" else path / "adapter_update.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload_path = metadata_path.parent / metadata["payload_file"]
    metadata_sha256 = _sha256_file(metadata_path)
    payload_sha256 = _sha256_file(payload_path)
    digest = hashlib.sha256(
        f"{metadata_sha256}:{payload_sha256}".encode("ascii")
    ).hexdigest()
    return {
        "schema": "fedsplat.client_adapter_artifact_fingerprint.v1",
        "metadata_path": str(metadata_path.resolve()),
        "payload_path": str(payload_path.resolve()),
        "metadata_sha256": metadata_sha256,
        "payload_sha256": payload_sha256,
        "package_sha256": digest,
        "serialized_bytes": payload_path.stat().st_size,
        "metadata_bytes": metadata_path.stat().st_size,
    }


def low_rank_dense_frobenius_norm(
    a: np.ndarray, b: np.ndarray
) -> float:
    """Compute ``||B A||_F`` through rank-sized Gram matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError(f"Expected compatible rank factors, got B{b.shape} and A{a.shape}")
    squared = float(np.sum((b.T @ b) * (a @ a.T)))
    return float(np.sqrt(max(0.0, squared)))


def low_rank_dense_difference_norm(
    current: LoRALayerUpdate, previous: LoRALayerUpdate | None
) -> float:
    """Compute ``||B_c A_c - B_p A_p||_F`` without materializing dense weights."""
    if previous is None:
        return low_rank_dense_frobenius_norm(current.a, current.b)
    if current.layer_name != previous.layer_name or current.dense_shape != previous.dense_shape:
        raise ValueError("LoRA layers must name the same dense parameter for differencing")
    a = np.concatenate((current.a, previous.a), axis=0)
    b = np.concatenate((current.b, -previous.b), axis=1)
    return low_rank_dense_frobenius_norm(a, b)


@dataclass(frozen=True)
class DualChannelAdapterUpdate:
    """A mandatory real adapter plus an optional synthetic residual."""

    client_id: str
    real: ClientAdapterUpdate
    synthetic_residual: ClientAdapterUpdate | None
    marginal_reliability: float
    compute_time_s: float | None = None

    def __post_init__(self) -> None:
        if not self.client_id or self.real.client_id != self.client_id:
            raise ValueError("Dual-channel client IDs must match")
        if not 0.0 <= self.marginal_reliability <= 1.0:
            raise ValueError("Marginal reliability must be in [0, 1]")
        if self.compute_time_s is not None and self.compute_time_s < 0.0:
            raise ValueError("Dual-channel compute time must be nonnegative")
        residual = self.synthetic_residual
        if residual is None:
            if self.marginal_reliability != 0.0:
                raise ValueError("A missing synthetic residual must have zero marginal reliability")
            return
        if residual.client_id != self.client_id:
            raise ValueError("Synthetic residual client ID must match the real channel")
        real_names = [layer.layer_name for layer in self.real.layers]
        residual_names = [layer.layer_name for layer in residual.layers]
        if real_names != residual_names:
            raise ValueError("Real and synthetic-residual layers must match")


def save_client_update(update: ClientAdapterUpdate, out_dir: Path) -> dict[str, Any]:
    """Serialize arrays and metadata and report actual payload bytes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    layer_meta = []
    for index, layer in enumerate(update.layers):
        a_key = f"layer_{index}_a"
        b_key = f"layer_{index}_b"
        arrays[a_key] = layer.a
        arrays[b_key] = layer.b
        layer_meta.append(
            {
                "layer_name": layer.layer_name,
                "a_key": a_key,
                "b_key": b_key,
                "rank": layer.rank,
                "dense_shape": list(layer.dense_shape),
            }
        )

    payload_path = out_dir / "adapter_update.npz"
    np.savez_compressed(payload_path, **arrays)
    metadata = {
        "schema": "fedsplat.client_adapter_update.v1",
        "client_id": update.client_id,
        "n_real": update.n_real,
        "reliability": update.reliability,
        "staleness": update.staleness,
        "compute_time_s": update.compute_time_s,
        "layers": layer_meta,
        "payload_file": payload_path.name,
        "serialized_bytes": payload_path.stat().st_size,
        "payload_sha256": _sha256_file(payload_path),
    }
    metadata_path = out_dir / "adapter_update.json"
    metadata["metadata_bytes"] = 0
    metadata["total_upload_bytes"] = metadata["serialized_bytes"]
    for _ in range(8):
        encoded = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
        size = len(encoded)
        total = metadata["serialized_bytes"] + size
        if size == metadata["metadata_bytes"] and total == metadata["total_upload_bytes"]:
            break
        metadata["metadata_bytes"] = size
        metadata["total_upload_bytes"] = total
    metadata_path.write_bytes((json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
    return metadata


def load_client_update(path: Path) -> ClientAdapterUpdate:
    metadata_path = path if path.suffix == ".json" else path / "adapter_update.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload_path = metadata_path.parent / metadata["payload_file"]
    with np.load(payload_path) as arrays:
        layers = tuple(
            LoRALayerUpdate(
                layer_name=item["layer_name"],
                a=arrays[item["a_key"]],
                b=arrays[item["b_key"]],
            )
            for item in metadata["layers"]
        )
    return ClientAdapterUpdate(
        client_id=metadata["client_id"],
        layers=layers,
        n_real=int(metadata["n_real"]),
        reliability=float(metadata["reliability"]),
        staleness=int(metadata.get("staleness", 0)),
        compute_time_s=float(metadata.get("compute_time_s", 0.0)),
    )


def clip_layers_by_dense_norm(
    layers: tuple[LoRALayerUpdate, ...],
    max_norm: float | None,
) -> tuple[tuple[LoRALayerUpdate, ...], dict[str, float | bool | None]]:
    """Apply one global trust-region scale while preserving every LoRA direction."""
    if not layers:
        raise ValueError("At least one LoRA layer is required")
    if max_norm is not None and (not np.isfinite(max_norm) or max_norm <= 0.0):
        raise ValueError("max_norm must be finite and positive when provided")
    norm_before = float(np.sqrt(sum(
        np.sum(layer.dense().astype(np.float64) ** 2) for layer in layers
    )))
    scale = 1.0 if max_norm is None or norm_before <= max_norm else float(max_norm / norm_before)
    clipped = tuple(
        LoRALayerUpdate(layer.layer_name, layer.a.copy(), layer.b * scale)
        for layer in layers
    )
    norm_after = float(np.sqrt(sum(
        np.sum(layer.dense().astype(np.float64) ** 2) for layer in clipped
    )))
    return clipped, {
        "enabled": max_norm is not None,
        "max_dense_update_norm": max_norm,
        "dense_update_norm_before": norm_before,
        "dense_update_norm_after": norm_after,
        "scale": scale,
        "clipped": scale < 1.0,
    }


def _combine_layer_terms(
    layer_name: str,
    terms: list[tuple[float, LoRALayerUpdate]],
    rank: int,
) -> LoRALayerUpdate:
    active = [(float(weight), layer) for weight, layer in terms if weight != 0.0]
    if not active or rank <= 0:
        raise ValueError("Low-rank combination requires a nonzero term and positive rank")
    dense_shape = active[0][1].dense_shape
    if any(layer.layer_name != layer_name or layer.dense_shape != dense_shape for _, layer in active):
        raise ValueError("Low-rank combination layers must share name and dense shape")
    b_cat = np.concatenate(
        [layer.b.astype(np.float64) * weight for weight, layer in active], axis=1
    )
    a_cat = np.concatenate([layer.a.astype(np.float64) for _, layer in active], axis=0)
    q_b, r_b = np.linalg.qr(b_cat, mode="reduced")
    q_a, r_a = np.linalg.qr(a_cat.T, mode="reduced")
    u, singular_values, vt = np.linalg.svd(r_b @ r_a.T, full_matrices=False)
    effective_rank = min(rank, len(singular_values))
    b = (q_b @ u[:, :effective_rank]) * singular_values[:effective_rank]
    a = vt[:effective_rank, :] @ q_a.T
    return LoRALayerUpdate(layer_name, a, b)


def combine_lora_layer_terms(
    layer_name: str,
    terms: list[tuple[float, LoRALayerUpdate]],
    rank: int,
) -> LoRALayerUpdate:
    """Public low-rank sum used to compose a base adapter and routed residuals."""
    return _combine_layer_terms(layer_name, terms, rank)


def clip_layers_by_dense_delta_norm(
    layers: tuple[LoRALayerUpdate, ...],
    previous_layers: tuple[LoRALayerUpdate, ...] | None,
    max_norm: float | None,
    server_rank: int,
) -> tuple[tuple[LoRALayerUpdate, ...], dict[str, Any]]:
    """Clip the global model change relative to the preceding FL round."""
    if not layers or server_rank <= 0:
        raise ValueError("Delta trust region requires layers and a positive server rank")
    if max_norm is not None and (not np.isfinite(max_norm) or max_norm <= 0.0):
        raise ValueError("max_norm must be finite and positive when provided")
    previous_by_name = (
        {layer.layer_name: layer for layer in previous_layers}
        if previous_layers is not None
        else {}
    )
    if previous_layers is not None and set(previous_by_name) != {
        layer.layer_name for layer in layers
    }:
        raise ValueError("Current and previous global adapters must contain the same layers")

    norm_before = float(
        np.sqrt(
            sum(
                low_rank_dense_difference_norm(
                    layer, previous_by_name.get(layer.layer_name)
                )
                ** 2
                for layer in layers
            )
        )
    )
    if max_norm is None or norm_before <= max_norm:
        return layers, {
            "enabled": max_norm is not None,
            "reference": "previous_global_adapter" if previous_layers else "zero_adapter",
            "max_dense_update_norm": max_norm,
            "dense_update_norm_before": norm_before,
            "dense_update_norm_after": norm_before,
            "scale": 1.0,
            "clipped": False,
            "server_rank": server_rank,
        }

    if previous_layers is None:
        scale = float(max_norm / norm_before)
        clipped = tuple(
            LoRALayerUpdate(layer.layer_name, layer.a.copy(), layer.b * scale)
            for layer in layers
        )
    else:
        current_by_name = {layer.layer_name: layer for layer in layers}

        def interpolate(scale: float) -> tuple[LoRALayerUpdate, ...]:
            return tuple(
                _combine_layer_terms(
                    layer.layer_name,
                    [
                        (scale, current_by_name[layer.layer_name]),
                        (1.0 - scale, layer),
                    ],
                    server_rank,
                )
                for layer in previous_layers
            )

        lower, upper = 0.0, 1.0
        clipped = previous_layers
        for _ in range(28):
            scale = (lower + upper) / 2.0
            candidate = interpolate(scale)
            candidate_norm = float(
                np.sqrt(
                    sum(
                        low_rank_dense_difference_norm(layer, previous_by_name[layer.layer_name])
                        ** 2
                        for layer in candidate
                    )
                )
            )
            if candidate_norm <= max_norm:
                lower = scale
                clipped = candidate
            else:
                upper = scale
        scale = lower

    norm_after = float(
        np.sqrt(
            sum(
                low_rank_dense_difference_norm(
                    layer, previous_by_name.get(layer.layer_name)
                )
                ** 2
                for layer in clipped
            )
        )
    )
    if norm_after > float(max_norm) * (1.0 + 1e-6):
        raise ArithmeticError("Server delta trust region failed to enforce its norm cap")
    return clipped, {
        "enabled": True,
        "reference": "previous_global_adapter" if previous_layers else "zero_adapter",
        "max_dense_update_norm": max_norm,
        "dense_update_norm_before": norm_before,
        "dense_update_norm_after": norm_after,
        "scale": scale,
        "clipped": True,
        "server_rank": server_rank,
    }


def save_dual_channel_update(update: DualChannelAdapterUpdate, out_dir: Path) -> dict[str, Any]:
    """Serialize a two-phase upload package and report each channel separately."""
    out_dir.mkdir(parents=True, exist_ok=True)
    real_metadata = save_client_update(update.real, out_dir / "real_adapter")
    residual_metadata = None
    if update.synthetic_residual is not None:
        residual_metadata = save_client_update(update.synthetic_residual, out_dir / "synthetic_residual")
    metadata = {
        "schema": "fedsplat.dual_channel_adapter_update.v1",
        "client_id": update.client_id,
        "n_real": update.real.n_real,
        "real_reliability": update.real.reliability,
        "marginal_reliability": update.marginal_reliability,
        "staleness": update.real.staleness,
        "compute_time_s": (
            update.real.compute_time_s if update.compute_time_s is None else update.compute_time_s
        ),
        "real_adapter_path": "real_adapter",
        "synthetic_residual_path": "synthetic_residual" if residual_metadata is not None else None,
        "real_upload_bytes": int(real_metadata["total_upload_bytes"]),
        "synthetic_residual_upload_bytes": (
            int(residual_metadata["total_upload_bytes"]) if residual_metadata is not None else 0
        ),
        "beacon_bytes": 0,
    }
    metadata_path = out_dir / "dual_channel_update.json"
    for _ in range(8):
        encoded = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
        if len(encoded) == metadata["beacon_bytes"]:
            break
        metadata["beacon_bytes"] = len(encoded)
    metadata_path.write_bytes((json.dumps(metadata, indent=2) + "\n").encode("utf-8"))
    return metadata


def load_dual_channel_update(path: Path, *, load_residual: bool = True) -> DualChannelAdapterUpdate:
    """Load a dual package; callers may defer the optional residual payload."""
    metadata_path = path if path.suffix == ".json" else path / "dual_channel_update.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "fedsplat.dual_channel_adapter_update.v1":
        raise ValueError(f"Unsupported dual-channel schema: {metadata.get('schema')}")
    real = load_client_update(metadata_path.parent / metadata["real_adapter_path"])
    residual_path = metadata.get("synthetic_residual_path")
    residual = (
        load_client_update(metadata_path.parent / residual_path)
        if load_residual and residual_path is not None
        else None
    )
    marginal_reliability = float(metadata["marginal_reliability"]) if residual is not None else 0.0
    return DualChannelAdapterUpdate(
        client_id=metadata["client_id"],
        real=real,
        synthetic_residual=residual,
        marginal_reliability=marginal_reliability,
        compute_time_s=float(metadata.get("compute_time_s", real.compute_time_s)),
    )


def aggregate_lora_updates(
    updates: list[ClientAdapterUpdate],
    weights: dict[str, float],
    server_rank: int,
) -> tuple[LoRALayerUpdate, ...]:
    """Aggregate updates and compute the exact truncated SVD in low-rank space."""
    if server_rank <= 0:
        raise ValueError("Server rank must be positive")
    return _aggregate_low_rank_updates(updates, weights, server_rank)


def _aggregate_low_rank_updates(
    updates: list[ClientAdapterUpdate],
    weights: dict[str, float],
    server_rank: int,
) -> tuple[LoRALayerUpdate, ...]:
    if not updates:
        raise ValueError("At least one update is required")
    layer_names = [layer.layer_name for layer in updates[0].layers]
    for update in updates:
        if [layer.layer_name for layer in update.layers] != layer_names:
            raise ValueError("All client updates must contain the same ordered layer names")
    positive_weights = {
        update.client_id: max(float(weights.get(update.client_id, 0.0)), 0.0)
        for update in updates
    }
    weight_total = sum(positive_weights.values())
    if weight_total <= 0.0:
        raise ValueError("Aggregation weights must have positive mass")

    aggregated = []
    for layer_index, layer_name in enumerate(layer_names):
        b_blocks = []
        a_blocks = []
        dense_shape = updates[0].layers[layer_index].dense_shape
        for update in updates:
            layer = update.layers[layer_index]
            if layer.dense_shape != dense_shape:
                raise ValueError(f"Dense shape mismatch for layer {layer_name}")
            coefficient = positive_weights[update.client_id] / weight_total
            if coefficient == 0.0:
                continue
            b_blocks.append(layer.b.astype(np.float64) * coefficient)
            a_blocks.append(layer.a.astype(np.float64))
        b_cat = np.concatenate(b_blocks, axis=1)
        a_cat = np.concatenate(a_blocks, axis=0)
        q_b, r_b = np.linalg.qr(b_cat, mode="reduced")
        q_a, r_a = np.linalg.qr(a_cat.T, mode="reduced")
        u, singular_values, vt = np.linalg.svd(r_b @ r_a.T, full_matrices=False)
        effective_rank = min(server_rank, len(singular_values))
        b = (q_b @ u[:, :effective_rank]) * singular_values[:effective_rank]
        a = vt[:effective_rank, :] @ q_a.T
        aggregated.append(LoRALayerUpdate(layer_name, a, b))
    return tuple(aggregated)


def aggregate_dual_channel_updates(
    updates: list[DualChannelAdapterUpdate],
    real_weights: dict[str, float],
    residual_weights: dict[str, float],
    server_rank: int,
) -> tuple[LoRALayerUpdate, ...]:
    """Aggregate real and selected residual channels against one client mass."""
    if not updates:
        raise ValueError("At least one dual-channel update is required")
    if server_rank <= 0:
        raise ValueError("Server rank must be positive")
    real_dense = _aggregate_dense_updates([update.real for update in updates], real_weights)
    selected = [
        update.synthetic_residual
        for update in updates
        if update.synthetic_residual is not None and float(residual_weights.get(update.client_id, 0.0)) > 0.0
    ]
    if not selected:
        return tuple(
            factorize_dense_layer(layer_name, dense, server_rank)
            for layer_name, dense in real_dense.items()
        )
    selected_updates = [update for update in selected if update is not None]
    real_weight_total = sum(
        max(float(real_weights.get(update.client_id, 0.0)), 0.0) for update in updates
    )
    residual_dense = _aggregate_dense_updates(
        selected_updates,
        residual_weights,
        normalization_total=real_weight_total,
    )
    combined = []
    for layer_name, real_value in real_dense.items():
        if layer_name not in residual_dense:
            raise ValueError("Real and synthetic-residual aggregation layers must match")
        combined.append(
            factorize_dense_layer(
                layer_name,
                real_value + residual_dense[layer_name],
                server_rank,
            )
        )
    return tuple(combined)


def _aggregate_dense_updates(
    updates: list[ClientAdapterUpdate],
    weights: dict[str, float],
    *,
    normalization_total: float | None = None,
) -> dict[str, np.ndarray]:
    if not updates:
        raise ValueError("At least one update is required")
    layer_names = [layer.layer_name for layer in updates[0].layers]
    for update in updates:
        if [layer.layer_name for layer in update.layers] != layer_names:
            raise ValueError("All client updates must contain the same ordered layer names")
    selected_weight_total = sum(
        max(float(weights.get(update.client_id, 0.0)), 0.0) for update in updates
    )
    weight_total = selected_weight_total if normalization_total is None else float(normalization_total)
    if weight_total <= 0.0 or selected_weight_total > weight_total + 1e-12:
        raise ValueError("Aggregation weights must have positive mass")
    aggregated: dict[str, np.ndarray] = {}
    for layer_index, layer_name in enumerate(layer_names):
        dense = None
        for update in updates:
            value = update.layers[layer_index].dense().astype(np.float64)
            contribution = max(float(weights.get(update.client_id, 0.0)), 0.0) / weight_total
            if dense is None:
                dense = np.zeros_like(value)
            if dense.shape != value.shape:
                raise ValueError(f"Dense shape mismatch for layer {layer_name}")
            dense += contribution * value
        assert dense is not None
        aggregated[layer_name] = dense
    return aggregated


def factorize_dense_layer(layer_name: str, dense: np.ndarray, rank: int) -> LoRALayerUpdate:
    """Compress a dense adapter delta into the repository LoRA convention."""
    dense = np.asarray(dense, dtype=np.float64)
    if dense.ndim != 2 or not np.all(np.isfinite(dense)) or rank <= 0:
        raise ValueError("Dense adapter must be a finite matrix and rank must be positive")
    u, singular_values, vt = np.linalg.svd(dense, full_matrices=False)
    effective_rank = min(rank, len(singular_values))
    b = u[:, :effective_rank] * singular_values[:effective_rank]
    a = vt[:effective_rank, :]
    return LoRALayerUpdate(layer_name=layer_name, a=a, b=b)


def dense_adapter_layers(update: ClientAdapterUpdate) -> dict[str, np.ndarray]:
    """Return named float64 dense deltas for server optimizers and diagnostics."""
    return {layer.layer_name: layer.dense().astype(np.float64) for layer in update.layers}
