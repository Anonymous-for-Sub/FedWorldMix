"""Factor-residual construction, directional consensus, and adapter-bank routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .lora_updates import (
    ClientAdapterUpdate,
    LoRALayerUpdate,
    aggregate_lora_updates,
    clip_layers_by_dense_norm,
    combine_lora_layer_terms,
)


def _ordered_layers(update: ClientAdapterUpdate) -> dict[str, LoRALayerUpdate]:
    return {layer.layer_name: layer for layer in update.layers}


def subtract_adapter_updates(
    factor: ClientAdapterUpdate,
    base: ClientAdapterUpdate,
    *,
    client_id: str | None = None,
) -> ClientAdapterUpdate:
    """Represent ``factor - base`` exactly by concatenating low-rank terms."""
    factor_layers = _ordered_layers(factor)
    base_layers = _ordered_layers(base)
    if set(factor_layers) != set(base_layers):
        raise ValueError("Factor and base adapters must contain identical layers")
    layers = []
    for layer in factor.layers:
        reference = base_layers[layer.layer_name]
        if layer.dense_shape != reference.dense_shape:
            raise ValueError(f"Adapter dense shape mismatch for {layer.layer_name}")
        layers.append(
            LoRALayerUpdate(
                layer_name=layer.layer_name,
                a=np.concatenate((layer.a, reference.a), axis=0),
                b=np.concatenate((layer.b, -reference.b), axis=1),
            )
        )
    return ClientAdapterUpdate(
        client_id=client_id or factor.client_id,
        layers=tuple(layers),
        n_real=factor.n_real,
        reliability=factor.reliability,
        staleness=factor.staleness,
        compute_time_s=factor.compute_time_s,
    )


def build_scoped_factor_residual(
    factor: ClientAdapterUpdate,
    matched_real: ClientAdapterUpdate,
    perception_modules: set[str],
) -> tuple[ClientAdapterUpdate, dict[str, Any]]:
    """Build an exact matched residual and zero every non-perception layer."""
    layer_names = {layer.layer_name for layer in factor.layers}
    if not perception_modules or not perception_modules.issubset(layer_names):
        raise ValueError("Perception residual scope must be a nonempty adapter subset")
    raw = subtract_adapter_updates(factor, matched_real, client_id=factor.client_id)
    layers = tuple(
        layer
        if layer.layer_name in perception_modules
        else LoRALayerUpdate(
            layer_name=layer.layer_name,
            a=np.zeros((1, layer.dense_shape[1]), dtype=np.float32),
            b=np.zeros((layer.dense_shape[0], 1), dtype=np.float32),
        )
        for layer in raw.layers
    )
    residual = ClientAdapterUpdate(
        client_id=factor.client_id,
        layers=layers,
        n_real=factor.n_real,
        reliability=factor.reliability,
        staleness=factor.staleness,
        compute_time_s=factor.compute_time_s,
    )
    real_by_name = _ordered_layers(matched_real)
    factor_by_name = _ordered_layers(factor)
    reconstruction_error_sq = 0.0
    reference_sq = 0.0
    nonperception_norm_sq = 0.0
    for layer in residual.layers:
        reconstructed = (
            real_by_name[layer.layer_name].dense().astype(np.float64)
            + layer.dense().astype(np.float64)
        )
        target = factor_by_name[layer.layer_name].dense().astype(np.float64)
        reconstruction_error_sq += float(np.sum((reconstructed - target) ** 2))
        reference_sq += float(np.sum(target**2))
        if layer.layer_name not in perception_modules:
            nonperception_norm_sq += float(
                np.sum(layer.dense().astype(np.float64) ** 2)
            )
    return residual, {
        "perception_modules": sorted(perception_modules),
        "nonperception_residual_dense_norm": float(np.sqrt(nonperception_norm_sq)),
        "reconstruction_relative_error": float(
            np.sqrt(reconstruction_error_sq / max(reference_sq, 1e-24))
        ),
    }


def low_rank_layer_inner_product(
    first: LoRALayerUpdate,
    second: LoRALayerUpdate,
) -> float:
    """Compute ``<B1 A1, B2 A2>_F`` through rank-sized products."""
    if first.layer_name != second.layer_name or first.dense_shape != second.dense_shape:
        raise ValueError("LoRA inner products require matching layer names and shapes")
    b_gram = first.b.astype(np.float64).T @ second.b.astype(np.float64)
    a_gram = second.a.astype(np.float64) @ first.a.astype(np.float64).T
    return float(np.sum(b_gram * a_gram.T))


def adapter_inner_product(
    first: ClientAdapterUpdate,
    second: ClientAdapterUpdate,
) -> float:
    first_layers = _ordered_layers(first)
    second_layers = _ordered_layers(second)
    if set(first_layers) != set(second_layers):
        raise ValueError("Adapter inner products require identical layers")
    return float(
        sum(
            low_rank_layer_inner_product(layer, second_layers[layer.layer_name])
            for layer in first.layers
        )
    )


def adapter_dense_norm(update: ClientAdapterUpdate) -> float:
    return float(np.sqrt(max(adapter_inner_product(update, update), 0.0)))


def adapter_linear_combination_error(
    approximation: ClientAdapterUpdate,
    terms: list[tuple[float, ClientAdapterUpdate]],
) -> dict[str, float]:
    """Measure low-rank composition error without materializing dense adapters."""
    if not terms:
        raise ValueError("At least one target adapter term is required")
    if any(not np.isfinite(weight) for weight, _update in terms):
        raise ValueError("Adapter combination weights must be finite")
    approximation_norm_sq = adapter_inner_product(
        approximation, approximation
    )
    target_norm_sq = 0.0
    for left_weight, left in terms:
        for right_weight, right in terms:
            target_norm_sq += (
                left_weight
                * right_weight
                * adapter_inner_product(left, right)
            )
    cross = sum(
        weight * adapter_inner_product(approximation, update)
        for weight, update in terms
    )
    error_norm_sq = max(
        approximation_norm_sq + target_norm_sq - 2.0 * cross,
        0.0,
    )
    return {
        "approximation_dense_norm": float(
            np.sqrt(max(approximation_norm_sq, 0.0))
        ),
        "target_dense_norm": float(np.sqrt(max(target_norm_sq, 0.0))),
        "absolute_dense_error": float(np.sqrt(error_norm_sq)),
        "relative_dense_error": float(
            np.sqrt(error_norm_sq / max(target_norm_sq, 1e-24))
        ),
    }


@dataclass(frozen=True)
class DirectionalConsensus:
    weights: dict[str, float]
    cosine_matrix: list[list[float]]
    mean_peer_cosine: dict[str, float]
    positive_peer_consensus: dict[str, float]
    raw_scores: dict[str, float]
    fallback_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fedsplat.directional_consensus.v1",
            "weights": self.weights,
            "cosine_matrix": self.cosine_matrix,
            "mean_peer_cosine": self.mean_peer_cosine,
            "positive_peer_consensus": self.positive_peer_consensus,
            "raw_scores": self.raw_scores,
            "fallback_used": self.fallback_used,
            "rule": "n_real * mean(max(cosine to each nonzero peer residual, 0)); FedAvg fallback if all scores are zero",
        }


def directional_consensus_weights(
    residuals: list[ClientAdapterUpdate],
    *,
    epsilon: float = 1e-12,
) -> DirectionalConsensus:
    if len(residuals) < 2:
        raise ValueError("Directional consensus requires at least two clients")
    ids = [update.client_id for update in residuals]
    if len(ids) != len(set(ids)):
        raise ValueError("Directional consensus requires unique client IDs")
    norms = [adapter_dense_norm(update) for update in residuals]
    cosine = np.eye(len(residuals), dtype=np.float64)
    for left in range(len(residuals)):
        for right in range(left + 1, len(residuals)):
            denominator = norms[left] * norms[right]
            value = (
                adapter_inner_product(residuals[left], residuals[right]) / denominator
                if denominator > epsilon
                else 0.0
            )
            value = float(np.clip(value, -1.0, 1.0))
            cosine[left, right] = cosine[right, left] = value
    peer_means = {}
    positive_consensus = {}
    raw_scores = {}
    for index, update in enumerate(residuals):
        peers = [
            float(cosine[index, peer])
            for peer in range(len(residuals))
            if peer != index and norms[peer] > epsilon
        ]
        mean = float(np.mean(peers)) if peers and norms[index] > epsilon else 0.0
        positive_mean = (
            float(np.mean([max(value, 0.0) for value in peers]))
            if peers and norms[index] > epsilon
            else 0.0
        )
        peer_means[update.client_id] = mean
        positive_consensus[update.client_id] = positive_mean
        raw_scores[update.client_id] = float(max(update.n_real, 1) * positive_mean)
    total = sum(raw_scores.values())
    fallback = total <= epsilon
    if fallback:
        raw_scores = {
            update.client_id: float(max(update.n_real, 1)) for update in residuals
        }
        total = sum(raw_scores.values())
    weights = {client_id: score / total for client_id, score in raw_scores.items()}
    return DirectionalConsensus(
        weights=weights,
        cosine_matrix=cosine.tolist(),
        mean_peer_cosine=peer_means,
        positive_peer_consensus=positive_consensus,
        raw_scores=raw_scores,
        fallback_used=fallback,
    )


def aggregate_factor_residuals(
    residuals: list[ClientAdapterUpdate],
    *,
    server_rank: int,
    max_dense_norm: float | None,
    client_id: str,
) -> tuple[ClientAdapterUpdate, dict[str, Any]]:
    consensus = directional_consensus_weights(residuals)
    layers = aggregate_lora_updates(residuals, consensus.weights, server_rank)
    layers, trust_region = clip_layers_by_dense_norm(layers, max_dense_norm)
    update = ClientAdapterUpdate(
        client_id=client_id,
        layers=layers,
        n_real=sum(item.n_real for item in residuals),
        reliability=sum(
            consensus.weights[item.client_id] * item.reliability for item in residuals
        ),
    )
    return update, {
        "directional_consensus": consensus.to_dict(),
        "residual_trust_region": trust_region,
    }


def compose_base_and_residuals(
    base: ClientAdapterUpdate,
    residuals: list[tuple[float, ClientAdapterUpdate]],
    *,
    rank: int,
    client_id: str,
) -> ClientAdapterUpdate:
    base_layers = _ordered_layers(base)
    residual_layers = [(weight, _ordered_layers(update)) for weight, update in residuals]
    for _weight, layers in residual_layers:
        if set(layers) != set(base_layers):
            raise ValueError("Every routed residual must match the base adapter layers")
    composed = []
    for base_layer in base.layers:
        terms = [(1.0, base_layer)] + [
            (weight, layers[base_layer.layer_name])
            for weight, layers in residual_layers
            if weight != 0.0
        ]
        composed.append(combine_lora_layer_terms(base_layer.layer_name, terms, rank))
    return ClientAdapterUpdate(
        client_id=client_id,
        layers=tuple(composed),
        n_real=base.n_real,
        reliability=base.reliability,
    )
