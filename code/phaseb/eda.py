"""Expert-Driven Aggregation utilities for Phase B."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Iterable[float]) -> float:
    return math.sqrt(_dot(a, a))


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    aa = list(a)
    bb = list(b)
    denom = _norm(aa) * _norm(bb)
    if denom <= 1e-12:
        return 0.0
    return _dot(aa, bb) / denom


def compute_layer_weights(client_to_stats: Dict[str, List[List[float]]]) -> Dict[str, List[float]]:
    """Compute EDA layer weights from expert selection counts."""
    if not client_to_stats:
        return {}

    client_ids = list(client_to_stats.keys())
    num_clients = len(client_ids)
    num_layers = len(next(iter(client_to_stats.values())))

    weights: Dict[str, List[float]] = {cid: [0.0 for _ in range(num_layers)] for cid in client_ids}
    for layer_index in range(num_layers):
        similarity = [[0.0 for _ in range(num_clients)] for _ in range(num_clients)]
        for i, cid_i in enumerate(client_ids):
            vec_i = client_to_stats[cid_i][layer_index]
            for j, cid_j in enumerate(client_ids):
                vec_j = client_to_stats[cid_j][layer_index]
                similarity[i][j] = cosine(vec_i, vec_j)

        row_sums = [sum(row) for row in similarity]
        denom = sum(row_sums)
        if denom <= 1e-12:
            uniform = 1.0 / float(num_clients)
            for cid in client_ids:
                weights[cid][layer_index] = uniform
            continue
        for i, cid in enumerate(client_ids):
            weights[cid][layer_index] = row_sums[i] / denom
    return weights


def aggregate_trunk_vectors(
    client_to_layers: Dict[str, List[List[float]]],
    weights: Dict[str, List[float]],
) -> List[List[float]]:
    """Aggregate trunk layer vectors using EDA weights."""
    if not client_to_layers:
        return []
    client_ids = list(client_to_layers.keys())
    num_layers = len(client_to_layers[client_ids[0]])
    vector_dim = len(client_to_layers[client_ids[0]][0])

    output: List[List[float]] = []
    for layer_index in range(num_layers):
        layer = [0.0 for _ in range(vector_dim)]
        for cid in client_ids:
            coeff = weights[cid][layer_index]
            source = client_to_layers[cid][layer_index]
            for dim in range(vector_dim):
                layer[dim] += coeff * source[dim]
        output.append(layer)
    return output

