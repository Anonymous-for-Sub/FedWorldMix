"""Image-only factor routing from federated visual sufficient statistics."""

from __future__ import annotations

from typing import Any
import itertools

import numpy as np


FEATURE_SCHEMA = "rgb8x8_gray_edge_hist_global.v1"
GRID_SIZE = 8
HISTOGRAM_BINS = 8
SCORE_BRIDGE_SCHEMA = "source_scores_poly2_abs_nominal.v1"


def _as_rgb_batch(images: Any) -> np.ndarray:
    values = np.asarray(images)
    if values.ndim == 3:
        values = values[None]
    if values.ndim != 4:
        raise ValueError(f"Router images must be rank-4, observed {values.shape}")
    if values.shape[-1] != 3 and values.shape[1] == 3:
        values = np.transpose(values, (0, 2, 3, 1))
    if values.shape[-1] != 3:
        raise ValueError(f"Router images must have three RGB channels, observed {values.shape}")
    values = values.astype(np.float64)
    if values.size and float(values.max()) <= 1.0 + 1e-6:
        values *= 255.0
    return np.clip(values, 0.0, 255.0)


def extract_visual_descriptors(images: Any) -> np.ndarray:
    """Extract deterministic image-only descriptors at resolution-independent sites."""
    values = _as_rgb_batch(images)
    rows = []
    for image in values:
        height, width = image.shape[:2]
        y = np.linspace(0, height - 1, GRID_SIZE).round().astype(int)
        x = np.linspace(0, width - 1, GRID_SIZE).round().astype(int)
        thumbnail = image[np.ix_(y, x)] / 255.0
        gray = thumbnail @ np.asarray([0.299, 0.587, 0.114])
        normalized_gray = (gray - float(gray.mean())) / max(float(gray.std()), 1e-3)
        dx = np.zeros_like(gray)
        dy = np.zeros_like(gray)
        dx[:, 1:] = gray[:, 1:] - gray[:, :-1]
        dy[1:, :] = gray[1:, :] - gray[:-1, :]
        histograms = []
        for channel in range(3):
            histogram, _ = np.histogram(
                image[..., channel],
                bins=HISTOGRAM_BINS,
                range=(0.0, 255.0),
            )
            histograms.extend((histogram / max(int(histogram.sum()), 1)).tolist())
        global_values = np.concatenate(
            (
                image.mean(axis=(0, 1)) / 255.0,
                image.std(axis=(0, 1)) / 255.0,
            )
        )
        rows.append(
            np.concatenate(
                (
                    thumbnail.reshape(-1),
                    normalized_gray.reshape(-1),
                    dx.reshape(-1),
                    dy.reshape(-1),
                    np.asarray(histograms, dtype=np.float64),
                    global_values,
                )
            )
        )
    return np.asarray(rows, dtype=np.float64)


def visual_descriptor_dimension() -> int:
    return GRID_SIZE * GRID_SIZE * (3 + 1 + 1 + 1) + 3 * HISTOGRAM_BINS + 6


def source_visual_scores(
    descriptors: np.ndarray,
    manifest: dict[str, Any],
) -> np.ndarray:
    mean = np.asarray(manifest["mean"], dtype=np.float64)
    scale = np.asarray(manifest["scale"], dtype=np.float64)
    weights = np.asarray(manifest["weights"], dtype=np.float64)
    bias = np.asarray(manifest["bias"], dtype=np.float64)
    return ((np.asarray(descriptors, dtype=np.float64) - mean) / scale) @ weights + bias


def score_bridge_features(
    source_scores: np.ndarray,
    nominal_center: np.ndarray,
) -> np.ndarray:
    """Map source factor scores to a compact cross-renderer calibration basis."""
    scores = np.asarray(source_scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[None]
    center = np.asarray(nominal_center, dtype=np.float64)
    if scores.ndim != 2 or center.shape != (scores.shape[1],):
        raise ValueError("Score bridge expects [N,F] scores and an [F] nominal center")
    pairwise = [
        scores[:, left] * scores[:, right]
        for left, right in itertools.combinations(range(scores.shape[1]), 2)
    ]
    blocks = [scores, scores**2, np.abs(scores - center)]
    if pairwise:
        blocks.append(np.stack(pairwise, axis=1))
    return np.concatenate(blocks, axis=1)


def score_bridge_feature_dimension(num_factors: int) -> int:
    if num_factors <= 0:
        raise ValueError("Score bridge requires at least one factor")
    return 3 * num_factors + num_factors * (num_factors - 1) // 2


def visual_router_scores(
    descriptors: np.ndarray,
    manifest: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return deployed factor scores and the source-router scores they derive from."""
    source_scores = source_visual_scores(descriptors, manifest)
    bridge = manifest.get("score_bridge")
    if bridge is None:
        return source_scores, source_scores
    factors = list(manifest["factors"])
    if bridge.get("schema") != SCORE_BRIDGE_SCHEMA:
        raise ValueError(f"Unsupported visual score bridge: {bridge.get('schema')}")
    if list(bridge.get("source_factor_order", [])) != factors:
        raise ValueError("Score bridge factor ordering does not match the source router")
    features = score_bridge_features(source_scores, bridge["nominal_center"])
    expected_dimension = score_bridge_feature_dimension(len(factors))
    if features.shape[1] != expected_dimension:
        raise ValueError("Score bridge feature dimension mismatch")
    mean = np.asarray(bridge["mean"], dtype=np.float64)
    scale = np.asarray(bridge["scale"], dtype=np.float64)
    weights = np.asarray(bridge["weights"], dtype=np.float64)
    bias = np.asarray(bridge["bias"], dtype=np.float64)
    scores = ((features - mean) / scale) @ weights + bias
    return scores, source_scores


def _balanced_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels > 0.5]
    negatives = scores[labels <= 0.5]
    if not len(positives) or not len(negatives):
        raise ValueError("Router calibration requires positive and negative examples")
    unique = np.unique(scores)
    candidates = np.concatenate(
        (
            [unique[0] - 1e-9],
            (unique[:-1] + unique[1:]) / 2.0,
            [unique[-1] + 1e-9],
        )
    )
    best = None
    for threshold in candidates:
        predicted = scores >= threshold
        tpr = float(predicted[labels > 0.5].mean())
        tnr = float((~predicted[labels <= 0.5]).mean())
        objective = (tpr + tnr) / 2.0
        # A higher threshold is the deterministic conservative tie-break.
        candidate = (objective, float(threshold))
        if best is None or candidate > best:
            best = candidate
    return float(best[1])


def _factor_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = scores >= threshold
    positive = labels > 0.5
    negative = ~positive
    tp = int((predicted & positive).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & negative).sum())
    fp = int((predicted & negative).sum())
    return {
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "tpr": tp / max(tp + fn, 1),
        "tnr": tn / max(tn + fp, 1),
        "balanced_accuracy": 0.5
        * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
    }


def fit_federated_visual_router(
    clients: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    factors: list[str],
    fit_clients: list[str],
    calibration_client: str,
    audit_client: str,
    ridge: float = 1.0,
    margin_fraction: float = 0.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a two-round ridge router using only per-client sufficient statistics."""
    if not factors or len(factors) != len(set(factors)):
        raise ValueError("Router factors must be nonempty and unique")
    required = set(fit_clients) | {calibration_client, audit_client}
    if len(required) != len(fit_clients) + 2 or not required.issubset(clients):
        raise ValueError("Fit, calibration, and audit client partitions must be disjoint and present")
    if ridge <= 0.0 or margin_fraction < 0.0:
        raise ValueError("Router ridge must be positive and margin fraction nonnegative")
    dimension = visual_descriptor_dimension()
    for client_id, (features, labels) in clients.items():
        if features.ndim != 2 or features.shape[1] != dimension:
            raise ValueError(f"Unexpected router feature shape for {client_id}: {features.shape}")
        if labels.shape != (features.shape[0], len(factors)):
            raise ValueError(f"Unexpected router label shape for {client_id}: {labels.shape}")

    count = sum(clients[client_id][0].shape[0] for client_id in fit_clients)
    feature_sum = sum(
        (clients[client_id][0].sum(axis=0) for client_id in fit_clients),
        np.zeros(dimension, dtype=np.float64),
    )
    feature_square_sum = sum(
        ((clients[client_id][0] ** 2).sum(axis=0) for client_id in fit_clients),
        np.zeros(dimension, dtype=np.float64),
    )
    mean = feature_sum / count
    variance = np.maximum(feature_square_sum / count - mean**2, 0.0)
    scale = np.sqrt(variance + 1e-6)

    xtx = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
    xty = np.zeros((dimension + 1, len(factors)), dtype=np.float64)
    client_statistics = []
    for client_id in fit_clients:
        features, labels = clients[client_id]
        standardized = (features - mean) / scale
        design = np.concatenate(
            (standardized, np.ones((len(standardized), 1), dtype=np.float64)), axis=1
        )
        local_xtx = design.T @ design
        local_xty = design.T @ labels
        xtx += local_xtx
        xty += local_xty
        client_statistics.append(
            {
                "client_id": client_id,
                "samples": int(len(features)),
                "moment_upload_scalars": int(2 * dimension + 1),
                "ridge_upload_scalars": int(local_xtx.size + local_xty.size),
            }
        )
    regularizer = np.eye(dimension + 1, dtype=np.float64) * float(ridge)
    regularizer[-1, -1] = 0.0
    solution = np.linalg.solve(xtx + regularizer, xty)
    weights = solution[:-1]
    bias = solution[-1]

    def scores_for(client_id: str) -> tuple[np.ndarray, np.ndarray]:
        features, labels = clients[client_id]
        scores = ((features - mean) / scale) @ weights + bias
        return scores, labels

    calibration_scores, calibration_labels = scores_for(calibration_client)
    thresholds = {}
    margins = {}
    calibration_metrics = {}
    for index, factor in enumerate(factors):
        scores = calibration_scores[:, index]
        labels = calibration_labels[:, index]
        threshold = _balanced_threshold(scores, labels)
        positive_median = float(np.median(scores[labels > 0.5]))
        negative_median = float(np.median(scores[labels <= 0.5]))
        separation = max(positive_median - negative_median, 0.0)
        margin = float(margin_fraction * separation)
        thresholds[factor] = threshold
        margins[factor] = margin
        calibration_metrics[factor] = {
            **_factor_metrics(scores, labels, threshold),
            "threshold": threshold,
            "abstention_margin": margin,
            "positive_median": positive_median,
            "negative_median": negative_median,
        }
    audit_scores, audit_labels = scores_for(audit_client)
    audit_metrics = {
        factor: _factor_metrics(
            audit_scores[:, index], audit_labels[:, index], thresholds[factor]
        )
        for index, factor in enumerate(factors)
    }
    nominal_mask = audit_labels.sum(axis=1) == 0
    nominal_false_positive = (
        float(
            np.any(
                audit_scores[nominal_mask]
                >= np.asarray([thresholds[factor] for factor in factors]),
                axis=1,
            ).mean()
        )
        if nominal_mask.any()
        else None
    )
    audit_passed = bool(
        all(row["balanced_accuracy"] >= 0.80 for row in audit_metrics.values())
        and nominal_false_positive is not None
        and nominal_false_positive <= 0.10
    )
    manifest = {
        "schema": "fedsplat.federated_visual_router.v1",
        "router_input": "startup_agentview_rgb_only",
        "feature_schema": FEATURE_SCHEMA,
        "feature_dimension": dimension,
        "factors": factors,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "thresholds": thresholds,
        "abstention_margins": margins,
        "fallback_route": "base",
        "factor_ordering": "lexicographic_route_join_with_plus",
        "fit_clients": fit_clients,
        "calibration_client": calibration_client,
        "audit_client": audit_client,
        "ridge": float(ridge),
        "margin_fraction": float(margin_fraction),
        "audit_passed": audit_passed,
    }
    report = {
        "schema": "fedsplat.federated_visual_router_audit.v1",
        "fit_samples": count,
        "client_sufficient_statistics": client_statistics,
        "raw_images_uploaded": 0,
        "calibration_metrics": calibration_metrics,
        "audit_metrics": audit_metrics,
        "audit_nominal_false_positive_rate": nominal_false_positive,
        "audit_passed": audit_passed,
    }
    return manifest, report


def fit_federated_score_bridge(
    clients: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    factors: list[str],
    fit_clients: list[str],
    calibration_client: str,
    audit_client: str,
    ridge: float = 1.0,
    margin_fraction: float = 0.05,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float], dict[str, Any]]:
    """Fit a federated nonlinear bridge over compact source-router scores."""
    if not factors or len(factors) != len(set(factors)):
        raise ValueError("Score bridge factors must be nonempty and unique")
    required = set(fit_clients) | {calibration_client, audit_client}
    if len(required) != len(fit_clients) + 2 or not required.issubset(clients):
        raise ValueError("Score bridge client partitions must be disjoint and present")
    if ridge <= 0.0 or margin_fraction < 0.0:
        raise ValueError("Score bridge ridge must be positive and margin nonnegative")
    factor_count = len(factors)
    for client_id, (scores, labels) in clients.items():
        if scores.ndim != 2 or scores.shape[1] != factor_count:
            raise ValueError(f"Unexpected source-score shape for {client_id}: {scores.shape}")
        if labels.shape != scores.shape:
            raise ValueError(f"Unexpected bridge-label shape for {client_id}: {labels.shape}")

    nominal_rows = [
        clients[client_id][0][clients[client_id][1].sum(axis=1) == 0]
        for client_id in fit_clients
    ]
    if not all(len(rows) for rows in nominal_rows):
        raise ValueError("Every bridge fit client requires nominal examples")
    nominal_center = np.concatenate(nominal_rows).mean(axis=0)
    client_features = {
        client_id: score_bridge_features(scores, nominal_center)
        for client_id, (scores, _labels) in clients.items()
    }
    dimension = score_bridge_feature_dimension(factor_count)
    fit_count = sum(len(client_features[client_id]) for client_id in fit_clients)
    feature_sum = sum(
        (client_features[client_id].sum(axis=0) for client_id in fit_clients),
        np.zeros(dimension, dtype=np.float64),
    )
    feature_square_sum = sum(
        ((client_features[client_id] ** 2).sum(axis=0) for client_id in fit_clients),
        np.zeros(dimension, dtype=np.float64),
    )
    mean = feature_sum / fit_count
    variance = np.maximum(feature_square_sum / fit_count - mean**2, 0.0)
    scale = np.sqrt(variance + 1e-6)
    xtx = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
    xty = np.zeros((dimension + 1, factor_count), dtype=np.float64)
    client_statistics = []
    for client_id in fit_clients:
        features = client_features[client_id]
        labels = clients[client_id][1]
        design = np.concatenate(
            ((features - mean) / scale, np.ones((len(features), 1))), axis=1
        )
        xtx += design.T @ design
        xty += design.T @ labels
        client_statistics.append(
            {
                "client_id": client_id,
                "samples": int(len(features)),
                "round1_nominal_center_upload_scalars": factor_count + 1,
                "round2_feature_moment_upload_scalars": 2 * dimension + 1,
                "round3_ridge_upload_scalars": int(design.shape[1] ** 2 + design.shape[1] * factor_count),
            }
        )
    regularizer = np.eye(dimension + 1, dtype=np.float64) * float(ridge)
    regularizer[-1, -1] = 0.0
    solution = np.linalg.solve(xtx + regularizer, xty)
    weights = solution[:-1]
    bias = solution[-1]

    def scores_for(client_id: str) -> tuple[np.ndarray, np.ndarray]:
        features = client_features[client_id]
        labels = clients[client_id][1]
        return ((features - mean) / scale) @ weights + bias, labels

    calibration_scores, calibration_labels = scores_for(calibration_client)
    thresholds = {}
    margins = {}
    calibration_metrics = {}
    for index, factor in enumerate(factors):
        scores = calibration_scores[:, index]
        labels = calibration_labels[:, index]
        threshold = _balanced_threshold(scores, labels)
        positive_median = float(np.median(scores[labels > 0.5]))
        negative_median = float(np.median(scores[labels <= 0.5]))
        margin = float(margin_fraction * max(positive_median - negative_median, 0.0))
        thresholds[factor] = threshold
        margins[factor] = margin
        calibration_metrics[factor] = {
            **_factor_metrics(scores, labels, threshold + margin),
            "threshold": threshold,
            "abstention_margin": margin,
        }
    audit_scores, audit_labels = scores_for(audit_client)
    activation_thresholds = np.asarray(
        [thresholds[factor] + margins[factor] for factor in factors]
    )
    predictions = audit_scores >= activation_thresholds
    uncertain = np.abs(
        audit_scores - np.asarray([thresholds[factor] for factor in factors])
    ) < np.asarray([margins[factor] for factor in factors])
    fallback = np.any(uncertain, axis=1)
    predictions[fallback] = False
    audit_metrics = {
        factor: _factor_metrics(
            audit_scores[:, index],
            audit_labels[:, index],
            activation_thresholds[index],
        )
        for index, factor in enumerate(factors)
    }
    nominal_mask = audit_labels.sum(axis=1) == 0
    nominal_false_positive = (
        float(np.any(predictions[nominal_mask], axis=1).mean())
        if nominal_mask.any()
        else None
    )
    exact_route_accuracy = float(np.all(predictions == (audit_labels > 0.5), axis=1).mean())
    audit_passed = bool(
        all(row["balanced_accuracy"] >= 0.80 for row in audit_metrics.values())
        and nominal_false_positive is not None
        and nominal_false_positive <= 0.10
        and exact_route_accuracy >= 0.80
    )
    bridge = {
        "schema": SCORE_BRIDGE_SCHEMA,
        "source_factor_order": factors,
        "feature_dimension": dimension,
        "nominal_center": nominal_center.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "ridge": float(ridge),
        "margin_fraction": float(margin_fraction),
        "fit_clients": fit_clients,
        "calibration_client": calibration_client,
        "audit_client": audit_client,
    }
    report = {
        "schema": "fedsplat.federated_score_bridge_audit.v1",
        "feature_schema": SCORE_BRIDGE_SCHEMA,
        "fit_samples": fit_count,
        "client_sufficient_statistics": client_statistics,
        "raw_images_uploaded": 0,
        "calibration_metrics": calibration_metrics,
        "audit_metrics": audit_metrics,
        "audit_nominal_false_positive_rate": nominal_false_positive,
        "audit_exact_route_accuracy": exact_route_accuracy,
        "audit_fallback_count": int(fallback.sum()),
        "audit_passed": audit_passed,
    }
    return bridge, thresholds, margins, report


def route_visual_descriptors(
    descriptors: np.ndarray,
    manifest: dict[str, Any],
    *,
    available_routes: set[str] | None = None,
) -> dict[str, Any]:
    factors = list(manifest["factors"])
    scores, source_scores = visual_router_scores(descriptors, manifest)
    rows = []
    for score_row in scores:
        selected = []
        uncertain = []
        factor_rows = {}
        for index, factor in enumerate(factors):
            threshold = float(manifest["thresholds"][factor])
            margin = float(manifest["abstention_margins"][factor])
            score = float(score_row[index])
            active = score >= threshold + margin
            is_uncertain = abs(score - threshold) < margin
            if active:
                selected.append(factor)
            if is_uncertain:
                uncertain.append(factor)
            factor_rows[factor] = {
                "score": score,
                "threshold": threshold,
                "abstention_margin": margin,
                "active": active,
                "uncertain": is_uncertain,
            }
        requested_route = "+".join(sorted(selected)) if selected else "base"
        supported = available_routes is None or requested_route in available_routes
        fallback = bool(uncertain or not supported)
        rows.append(
            {
                "requested_route": requested_route,
                "selected_route": manifest.get("fallback_route", "base")
                if fallback
                else requested_route,
                "selected_factors": [] if fallback else sorted(selected),
                "uncertain_factors": uncertain,
                "fallback_used": fallback,
                "requested_route_supported": supported,
                "factor_scores": factor_rows,
                "source_factor_scores": {
                    factor: float(source_scores[len(rows), index])
                    for index, factor in enumerate(factors)
                },
            }
        )
    return {
        "schema": "fedsplat.visual_router_decisions.v1",
        "feature_schema": manifest["feature_schema"],
        "rows": rows,
        "fallback_count": sum(row["fallback_used"] for row in rows),
        "selected_route_counts": {
            route: sum(row["selected_route"] == route for row in rows)
            for route in sorted({row["selected_route"] for row in rows})
        },
    }


def route_visual_images(
    images: Any,
    manifest: dict[str, Any],
    *,
    available_routes: set[str] | None = None,
) -> dict[str, Any]:
    return route_visual_descriptors(
        extract_visual_descriptors(images),
        manifest,
        available_routes=available_routes,
    )
