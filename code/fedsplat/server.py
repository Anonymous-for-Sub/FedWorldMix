"""Server-side aggregation utilities for FedSplat."""

from __future__ import annotations

import enum
import math
from collections.abc import Sequence

from .schemas import ClientStats
from .method import reliability_raw_weight


class AggregationMode(str, enum.Enum):
    FEDAVG = "fedavg"
    QUALITY = "quality"
    CONSISTENCY = "consistency"
    DOMAIN_AWARE = "domain_aware"
    UTILITY = "utility"
    RELIABILITY = "reliability"


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(float(value), 0.0) for key, value in raw.items()}
    total = sum(clipped.values())
    if total <= 1e-12:
        uniform = 1.0 / float(max(len(clipped), 1))
        return {key: uniform for key in clipped}
    return {key: value / total for key, value in clipped.items()}


def compute_client_weights(
    stats: Sequence[ClientStats],
    mode: AggregationMode | str = AggregationMode.FEDAVG,
    synthetic_weight: float = 0.5,
    divergence_beta: float = 1.0,
    novelty_weight: float = 0.25,
    reliability_floor: float = 0.05,
    staleness_gamma: float = 0.1,
) -> dict[str, float]:
    """Compute server aggregation weights from upload-safe client statistics."""
    mode = AggregationMode(mode)
    raw: dict[str, float] = {}
    for item in stats:
        base_n = max(item.n_real + synthetic_weight * item.n_aug, 1.0)
        if mode == AggregationMode.FEDAVG:
            raw[item.client_id] = base_n
        elif mode == AggregationMode.QUALITY:
            raw[item.client_id] = base_n * max(item.q_aug, 0.0)
        elif mode == AggregationMode.CONSISTENCY:
            raw[item.client_id] = base_n * max(item.q_aug, 0.0) * math.exp(
                -divergence_beta * max(item.real_synthetic_divergence, 0.0)
            )
        elif mode == AggregationMode.DOMAIN_AWARE:
            consistency = math.exp(-divergence_beta * max(item.real_synthetic_divergence, 0.0))
            domain_term = (1.0 - novelty_weight) + novelty_weight * max(item.domain_novelty, 0.0)
            raw[item.client_id] = base_n * max(item.q_aug, 0.0) * consistency * domain_term
        elif mode == AggregationMode.UTILITY:
            consistency = math.exp(-divergence_beta * max(item.real_synthetic_divergence, 0.0))
            utility = max(item.q_aug, 0.0) * consistency * max(item.domain_novelty, 0.0)
            raw[item.client_id] = base_n * utility * max(item.validation_score, 0.0) / item.upload_cost_mb
        elif mode == AggregationMode.RELIABILITY:
            raw[item.client_id] = reliability_raw_weight(
                n_real=item.n_real,
                reliability=item.reliability,
                staleness=item.staleness,
                reliability_floor=reliability_floor,
                staleness_gamma=staleness_gamma,
            )
        else:
            raise ValueError(f"Unsupported aggregation mode: {mode}")
    return _normalize(raw)


def aggregate_layer_vectors(
    client_to_layers: dict[str, list[list[float]]],
    weights: dict[str, float],
) -> list[list[float]]:
    """Aggregate layer vectors with global client weights.

    This is intentionally simple. FedVLA-style layer-wise weights can be added
    later by passing a per-layer matrix or by reusing its EDA implementation.
    """
    if not client_to_layers:
        return []
    client_ids = list(client_to_layers)
    num_layers = len(client_to_layers[client_ids[0]])
    dim = len(client_to_layers[client_ids[0]][0])
    output: list[list[float]] = []
    for layer_index in range(num_layers):
        layer = [0.0 for _ in range(dim)]
        for client_id in client_ids:
            weight = weights.get(client_id, 0.0)
            source = client_to_layers[client_id][layer_index]
            for idx, value in enumerate(source):
                layer[idx] += weight * value
        output.append(layer)
    return output
