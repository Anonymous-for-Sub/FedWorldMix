"""Calibrated RGB-D Gaussian utilities for the RoboTwin Sim-GS track."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


SH_C0 = 0.28209479177387814


@dataclass
class GaussianCloud:
    means: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    quaternions: np.ndarray
    opacities: np.ndarray
    actor_ids: np.ndarray
    sh_coefficients: np.ndarray | None = None

    def validate(self) -> None:
        count = int(self.means.shape[0])
        expected = {
            "means": (count, 3),
            "colors": (count, 3),
            "scales": (count, 3),
            "quaternions": (count, 4),
            "opacities": (count,),
            "actor_ids": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if self.sh_coefficients is not None:
            if self.sh_coefficients.ndim != 3:
                raise ValueError("SH coefficients must have shape [N, K, 3]")
            if self.sh_coefficients.shape[0] != count or self.sh_coefficients.shape[2] != 3:
                raise ValueError(
                    "SH coefficients must have the same count as the Gaussian cloud"
                )
            bands = int(round(math.sqrt(self.sh_coefficients.shape[1])))
            if bands * bands != self.sh_coefficients.shape[1]:
                raise ValueError("SH coefficient count must be a square number")
            if not np.all(np.isfinite(self.sh_coefficients)):
                raise ValueError("SH coefficients contain non-finite values")
        if count == 0:
            raise ValueError("Gaussian cloud is empty")
        if np.any(self.scales <= 0):
            raise ValueError("Gaussian scales must be positive")
        if np.any((self.opacities <= 0) | (self.opacities >= 1)):
            raise ValueError("Gaussian opacities must lie strictly inside (0, 1)")


def look_at_entity_pose(
    position: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray | None = None,
) -> np.ndarray:
    """Return a SAPIEN camera-entity pose with +x forward, +y left, +z up."""
    position = np.asarray(position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    world_up = np.asarray(
        [0.0, 0.0, 1.0] if world_up is None else world_up, dtype=np.float64
    )
    forward = target - position
    forward /= np.linalg.norm(forward)
    left = np.cross(world_up, forward)
    if np.linalg.norm(left) < 1e-8:
        raise ValueError("Camera forward direction is parallel to world up")
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    up /= np.linalg.norm(up)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.stack([forward, left, up], axis=1)
    matrix[:3, 3] = position
    return matrix


def homogeneous_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape == (3, 4):
        extrinsic = np.vstack([extrinsic, np.array([0.0, 0.0, 0.0, 1.0])])
    if extrinsic.shape != (4, 4):
        raise ValueError("Expected a 3x4 or 4x4 extrinsic matrix")
    return extrinsic


def validate_camera_matrices(intrinsic: np.ndarray, extrinsic: np.ndarray) -> dict[str, float]:
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    extrinsic = homogeneous_extrinsic(extrinsic)
    if intrinsic.shape != (3, 3):
        raise ValueError("Expected a 3x3 intrinsic matrix")
    if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(extrinsic)):
        raise ValueError("Camera matrices contain non-finite values")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    det_rotation = float(np.linalg.det(extrinsic[:3, :3]))
    condition = float(np.linalg.cond(extrinsic))
    if abs(det_rotation - 1.0) > 1e-3:
        raise ValueError(f"Extrinsic rotation determinant is {det_rotation:.6f}")
    if not np.isfinite(condition) or condition > 1e8:
        raise ValueError(f"Extrinsic matrix is ill-conditioned: {condition:.3e}")
    return {
        "rotation_determinant": det_rotation,
        "extrinsic_condition_number": condition,
    }


def unproject_rgbd(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    actor_ids: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    stride: int = 1,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Unproject an OpenCV RGB-D frame into metric world coordinates."""
    if stride <= 0:
        raise ValueError("stride must be positive")
    rgb = np.asarray(rgb)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    actor_ids = np.asarray(actor_ids)
    world_to_camera = homogeneous_extrinsic(world_to_camera)
    height, width = depth_m.shape
    if rgb.shape[:2] != (height, width) or actor_ids.shape != (height, width):
        raise ValueError("RGB, depth, and actor-ID images must share spatial shape")
    camera_audit = validate_camera_matrices(intrinsic, world_to_camera)

    rows, cols = np.mgrid[0:height:stride, 0:width:stride]
    sampled_depth = depth_m[::stride, ::stride]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= min_depth_m)
        & (sampled_depth <= max_depth_m)
    )
    z = sampled_depth[valid]
    u = cols[valid].astype(np.float64)
    v = rows[valid].astype(np.float64)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    camera_points = np.stack(
        [(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)], axis=1
    )
    camera_to_world = np.linalg.inv(world_to_camera)
    world_points = (camera_to_world @ camera_points.T).T[:, :3]
    colors = rgb[::stride, ::stride][valid, :3].astype(np.float32) / 255.0
    ids = actor_ids[::stride, ::stride][valid].astype(np.int64)

    projected = (world_to_camera @ np.c_[world_points, np.ones(len(world_points))].T).T
    reproj_u = fx * projected[:, 0] / projected[:, 2] + cx
    reproj_v = fy * projected[:, 1] / projected[:, 2] + cy
    reprojection = np.sqrt((reproj_u - u) ** 2 + (reproj_v - v) ** 2)
    audit = {
        **camera_audit,
        "sampled_pixels": int(sampled_depth.size),
        "valid_points": int(len(world_points)),
        "valid_depth_fraction": float(valid.mean()),
        "median_reprojection_px": float(np.median(reprojection)) if len(reprojection) else math.inf,
        "max_reprojection_px": float(np.max(reprojection)) if len(reprojection) else math.inf,
    }
    return (
        world_points.astype(np.float32),
        colors.astype(np.float32),
        ids,
        audit,
    )


def crop_points(
    points: np.ndarray,
    colors: np.ndarray,
    actor_ids: np.ndarray,
    bounds_min: Iterable[float],
    bounds_max: Iterable[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds_min = np.asarray(list(bounds_min), dtype=np.float32)
    bounds_max = np.asarray(list(bounds_max), dtype=np.float32)
    keep = np.all((points >= bounds_min) & (points <= bounds_max), axis=1)
    return points[keep], colors[keep], actor_ids[keep]


def filter_unbound_actor_points(
    points: np.ndarray,
    colors: np.ndarray,
    actor_ids: np.ndarray,
    *,
    ignored_actor_ids: Iterable[int] = (0,),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove valid-depth pixels that segmentation did not bind to an actor."""
    points = np.asarray(points)
    colors = np.asarray(colors)
    actor_ids = np.asarray(actor_ids)
    if len(points) != len(colors) or len(points) != len(actor_ids):
        raise ValueError("Points, colors, and actor IDs must have equal length")
    ignored = sorted(set(int(value) for value in ignored_actor_ids))
    keep = ~np.isin(actor_ids, ignored)
    if not np.any(keep):
        raise ValueError("Actor filtering removed every point")
    return (
        points[keep],
        colors[keep],
        actor_ids[keep],
        {
            "ignored_actor_ids": ignored,
            "input_points": int(len(points)),
            "kept_points": int(np.count_nonzero(keep)),
            "removed_points": int(np.count_nonzero(~keep)),
            "removed_fraction": float(np.mean(~keep)),
        },
    )


def voxel_fuse(
    points: np.ndarray,
    colors: np.ndarray,
    actor_ids: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse points by metric voxel and actor ID so components never bleed."""
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if len(points) == 0:
        raise ValueError("Cannot fuse an empty point set")
    voxels = np.floor(points / voxel_size_m).astype(np.int64)
    keys = np.c_[voxels, actor_ids.astype(np.int64)]
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    fused_points = np.stack(
        [np.bincount(inverse, weights=points[:, dim]) / counts for dim in range(3)],
        axis=1,
    )
    fused_colors = np.stack(
        [np.bincount(inverse, weights=colors[:, dim]) / counts for dim in range(3)],
        axis=1,
    )
    return (
        fused_points.astype(np.float32),
        np.clip(fused_colors, 0.0, 1.0).astype(np.float32),
        unique_keys[:, 3].astype(np.int64),
        counts.astype(np.int32),
    )


def voxel_fuse_multires(
    points: np.ndarray,
    colors: np.ndarray,
    actor_ids: np.ndarray,
    voxel_size_m: float,
    *,
    voxel_size_by_actor_id: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse actor groups at different resolutions without crossing boundaries."""
    overrides = {
        int(actor_id): float(size)
        for actor_id, size in (voxel_size_by_actor_id or {}).items()
    }
    if voxel_size_m <= 0 or any(size <= 0 for size in overrides.values()):
        raise ValueError("All voxel sizes must be positive")
    if len(points) == 0:
        raise ValueError("Cannot fuse an empty point set")

    outputs = []
    overridden = np.zeros(len(actor_ids), dtype=bool)
    for actor_id, size in sorted(overrides.items()):
        selected = actor_ids == actor_id
        if not np.any(selected):
            continue
        overridden |= selected
        outputs.append(
            voxel_fuse(points[selected], colors[selected], actor_ids[selected], size)
        )
    if np.any(~overridden):
        outputs.append(
            voxel_fuse(
                points[~overridden],
                colors[~overridden],
                actor_ids[~overridden],
                voxel_size_m,
            )
        )
    if not outputs:
        raise ValueError("Multi-resolution fusion produced no points")
    return tuple(np.concatenate(items, axis=0) for items in zip(*outputs))


def build_gaussian_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    actor_ids: np.ndarray,
    *,
    voxel_size_m: float,
    scale_multiplier: float = 0.85,
    opacity: float = 0.82,
    max_gaussians: int = 180_000,
    seed: int = 0,
    preserve_actor_id: int | None = None,
    voxel_size_by_actor_id: dict[int, float] | None = None,
) -> GaussianCloud:
    if max_gaussians <= 0:
        raise ValueError("max_gaussians must be positive")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(points))
    if len(indices) > max_gaussians:
        if preserve_actor_id is None:
            indices = np.sort(rng.choice(indices, max_gaussians, replace=False))
        else:
            preserve = indices[actor_ids == preserve_actor_id]
            remaining = indices[actor_ids != preserve_actor_id]
            keep_remaining = max(0, max_gaussians - len(preserve))
            if len(preserve) > max_gaussians:
                preserve = rng.choice(preserve, max_gaussians, replace=False)
                remaining = np.empty(0, dtype=np.int64)
                keep_remaining = 0
            elif len(remaining) > keep_remaining:
                remaining = rng.choice(remaining, keep_remaining, replace=False)
            indices = np.sort(np.r_[preserve, remaining])
    count = len(indices)
    selected_actor_ids = actor_ids[indices].astype(np.int64)
    scales = np.full(
        (count, 3), float(voxel_size_m * scale_multiplier), dtype=np.float32
    )
    for actor_id, actor_voxel_size in (voxel_size_by_actor_id or {}).items():
        if float(actor_voxel_size) <= 0:
            raise ValueError("All actor-specific voxel sizes must be positive")
        scales[selected_actor_ids == int(actor_id)] = float(
            actor_voxel_size * scale_multiplier
        )
    cloud = GaussianCloud(
        means=points[indices].astype(np.float32),
        colors=np.clip(colors[indices], 1e-4, 1.0 - 1e-4).astype(np.float32),
        scales=scales,
        quaternions=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
        opacities=np.full(count, opacity, dtype=np.float32),
        actor_ids=selected_actor_ids,
    )
    cloud.validate()
    return cloud


def initialize_surface_gaussians(
    cloud: GaussianCloud,
    *,
    neighbor_count: int = 8,
    normal_scale_multiplier: float = 0.4,
    tangent_scale_multiplier: float = 1.35,
    excluded_actor_ids: Iterable[int] = (),
) -> tuple[GaussianCloud, dict[str, float | int]]:
    """Initialize anisotropic Gaussian frames from actor-local surface neighborhoods."""
    if neighbor_count < 3:
        raise ValueError("neighbor_count must be at least 3")
    if normal_scale_multiplier <= 0 or tangent_scale_multiplier <= 0:
        raise ValueError("Surface scale multipliers must be positive")
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    result = GaussianCloud(
        means=cloud.means.copy(),
        colors=cloud.colors.copy(),
        scales=cloud.scales.copy(),
        quaternions=cloud.quaternions.copy(),
        opacities=cloud.opacities.copy(),
        actor_ids=cloud.actor_ids.copy(),
        sh_coefficients=(
            None
            if cloud.sh_coefficients is None
            else cloud.sh_coefficients.copy()
        ),
    )
    initialized = 0
    actor_groups = 0
    excluded_actor_ids = set(int(value) for value in excluded_actor_ids)
    for actor_id in np.unique(result.actor_ids):
        if int(actor_id) in excluded_actor_ids:
            continue
        indices = np.flatnonzero(result.actor_ids == actor_id)
        if len(indices) < 3:
            continue
        actor_groups += 1
        points = result.means[indices].astype(np.float64)
        count = min(neighbor_count, len(points))
        _, neighbors = cKDTree(points).query(points, k=count, workers=-1)
        if count == 1:
            neighbors = neighbors[:, None]
        neighborhoods = points[neighbors]
        centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / max(1, count - 1)
        _, eigenvectors = np.linalg.eigh(covariance)
        frames = eigenvectors
        negative = np.linalg.det(frames) < 0.0
        frames[negative, :, 2] *= -1.0
        xyzw = Rotation.from_matrix(frames).as_quat()
        result.quaternions[indices] = np.c_[
            xyzw[:, 3], xyzw[:, 0], xyzw[:, 1], xyzw[:, 2]
        ].astype(np.float32)
        base_scale = np.median(result.scales[indices], axis=1)
        result.scales[indices, 0] = base_scale * normal_scale_multiplier
        result.scales[indices, 1] = base_scale * tangent_scale_multiplier
        result.scales[indices, 2] = base_scale * tangent_scale_multiplier
        initialized += len(indices)
    result.validate()
    return result, {
        "initialized_gaussians": int(initialized),
        "actor_groups": int(actor_groups),
        "neighbor_count": int(neighbor_count),
        "normal_scale_multiplier": float(normal_scale_multiplier),
        "tangent_scale_multiplier": float(tangent_scale_multiplier),
    }


def complete_support_appearance(
    cloud: GaussianCloud,
    *,
    target_actor_id: int,
    support_actor_id: int,
    xy_padding_m: float = 0.015,
    z_below_m: float = 0.06,
    z_above_m: float = 0.012,
    neighbor_count: int = 24,
) -> tuple[GaussianCloud, dict[str, Any]]:
    """Fill a target-occluded support patch from nearby support Gaussians."""
    if xy_padding_m < 0 or z_below_m <= 0 or z_above_m <= 0:
        raise ValueError("Support completion distances must be positive")
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be positive")
    from scipy.spatial import cKDTree

    target = cloud.actor_ids == int(target_actor_id)
    support = cloud.actor_ids == int(support_actor_id)
    if not np.any(target):
        raise ValueError(f"No Gaussians found for target actor ID {target_actor_id}")
    if not np.any(support):
        raise ValueError(f"No Gaussians found for support actor ID {support_actor_id}")

    target_min = cloud.means[target].min(axis=0).astype(np.float64)
    target_max = cloud.means[target].max(axis=0).astype(np.float64)
    inside_xy = (
        (cloud.means[:, 0] >= target_min[0] - xy_padding_m)
        & (cloud.means[:, 0] <= target_max[0] + xy_padding_m)
        & (cloud.means[:, 1] >= target_min[1] - xy_padding_m)
        & (cloud.means[:, 1] <= target_max[1] + xy_padding_m)
    )
    height_band = (
        (cloud.means[:, 2] >= target_min[2] - z_below_m)
        & (cloud.means[:, 2] <= target_min[2] + z_above_m)
    )
    candidates = np.flatnonzero(support & inside_xy & height_band)
    donors = np.flatnonzero(support & ~inside_xy & height_band)
    if len(candidates) == 0:
        raise ValueError("No support Gaussians found beneath the target footprint")
    if len(donors) == 0:
        raise ValueError("No nearby support Gaussians available for completion")

    count = min(int(neighbor_count), len(donors))
    _, nearest = cKDTree(cloud.means[donors]).query(
        cloud.means[candidates], k=count, workers=-1
    )
    if count == 1:
        nearest = nearest[:, None]
    donor_neighbors = donors[nearest]
    result = GaussianCloud(
        means=cloud.means.copy(),
        colors=cloud.colors.copy(),
        scales=cloud.scales.copy(),
        quaternions=cloud.quaternions.copy(),
        opacities=cloud.opacities.copy(),
        actor_ids=cloud.actor_ids.copy(),
        sh_coefficients=(
            None
            if cloud.sh_coefficients is None
            else cloud.sh_coefficients.copy()
        ),
    )
    result.colors[candidates] = np.median(
        cloud.colors[donor_neighbors], axis=1
    ).astype(np.float32)
    if result.sh_coefficients is not None:
        result.sh_coefficients[candidates] = np.median(
            cloud.sh_coefficients[donor_neighbors], axis=1
        ).astype(np.float32)
        result.colors[candidates] = np.clip(
            result.sh_coefficients[candidates, 0] * SH_C0 + 0.5,
            1e-4,
            1.0 - 1e-4,
        ).astype(np.float32)
    result.validate()
    return result, {
        "target_actor_id": int(target_actor_id),
        "support_actor_id": int(support_actor_id),
        "completed_gaussians": int(len(candidates)),
        "donor_gaussians": int(len(donors)),
        "neighbor_count": int(count),
        "xy_padding_m": float(xy_padding_m),
        "z_below_m": float(z_below_m),
        "z_above_m": float(z_above_m),
        "target_bbox_min": target_min.tolist(),
        "target_bbox_max": target_max.tolist(),
    }


def project_protected_component_geometry(
    initial_cloud: GaussianCloud,
    optimized_cloud: GaussianCloud,
    *,
    actor_ids: Iterable[int],
    preserve_opacity: bool = True,
) -> tuple[GaussianCloud, dict[str, Any]]:
    """Restore calibrated rigid geometry while retaining optimized appearance.

    Densification and photometric optimization are useful for the static scene,
    but millimetre-scale drift makes a small rigid component inconsistent with
    its simulator pose. Each protected component is therefore projected back to
    its RGB-D initialization. Optimized color/SH values are transferred from the
    nearest optimized Gaussian of the same actor.
    """
    from scipy.spatial import cKDTree

    initial_cloud.validate()
    optimized_cloud.validate()
    protected_ids = sorted(set(int(value) for value in actor_ids))
    if not protected_ids:
        return optimized_cloud, {
            "enabled": False,
            "protected_actor_ids": [],
            "components": [],
        }

    keep_optimized = ~np.isin(optimized_cloud.actor_ids, protected_ids)
    means = [optimized_cloud.means[keep_optimized]]
    colors = [optimized_cloud.colors[keep_optimized]]
    scales = [optimized_cloud.scales[keep_optimized]]
    quaternions = [optimized_cloud.quaternions[keep_optimized]]
    opacities = [optimized_cloud.opacities[keep_optimized]]
    output_actor_ids = [optimized_cloud.actor_ids[keep_optimized]]
    sh_coefficients = (
        [optimized_cloud.sh_coefficients[keep_optimized]]
        if optimized_cloud.sh_coefficients is not None
        else None
    )
    components: list[dict[str, Any]] = []

    for actor_id in protected_ids:
        initial_indices = np.flatnonzero(initial_cloud.actor_ids == actor_id)
        optimized_indices = np.flatnonzero(optimized_cloud.actor_ids == actor_id)
        if len(initial_indices) == 0:
            raise ValueError(
                f"Initial cloud contains no protected actor ID {actor_id}"
            )
        if len(optimized_indices) == 0:
            raise ValueError(
                f"Optimized cloud contains no protected actor ID {actor_id}"
            )

        initial_means = initial_cloud.means[initial_indices]
        optimized_means = optimized_cloud.means[optimized_indices]
        distances, nearest_local = cKDTree(optimized_means).query(
            initial_means, k=1, workers=-1
        )
        nearest = optimized_indices[np.asarray(nearest_local, dtype=np.int64)]

        means.append(initial_means.copy())
        colors.append(optimized_cloud.colors[nearest].copy())
        scales.append(initial_cloud.scales[initial_indices].copy())
        quaternions.append(initial_cloud.quaternions[initial_indices].copy())
        opacities.append(
            initial_cloud.opacities[initial_indices].copy()
            if preserve_opacity
            else optimized_cloud.opacities[nearest].copy()
        )
        output_actor_ids.append(initial_cloud.actor_ids[initial_indices].copy())
        if sh_coefficients is not None:
            sh_coefficients.append(
                optimized_cloud.sh_coefficients[nearest].copy()
            )

        centroid_delta = optimized_means.mean(axis=0) - initial_means.mean(axis=0)
        components.append(
            {
                "actor_id": int(actor_id),
                "initial_gaussians": int(len(initial_indices)),
                "optimized_gaussians": int(len(optimized_indices)),
                "discarded_densified_gaussians": int(
                    max(0, len(optimized_indices) - len(initial_indices))
                ),
                "centroid_drift_m": centroid_delta.astype(float).tolist(),
                "centroid_drift_norm_m": float(np.linalg.norm(centroid_delta)),
                "appearance_transfer_mean_distance_m": float(
                    np.mean(distances)
                ),
                "appearance_transfer_max_distance_m": float(np.max(distances)),
            }
        )

    result = GaussianCloud(
        means=np.concatenate(means).astype(np.float32),
        colors=np.concatenate(colors).astype(np.float32),
        scales=np.concatenate(scales).astype(np.float32),
        quaternions=np.concatenate(quaternions).astype(np.float32),
        opacities=np.concatenate(opacities).astype(np.float32),
        actor_ids=np.concatenate(output_actor_ids).astype(np.int64),
        sh_coefficients=(
            None
            if sh_coefficients is None
            else np.concatenate(sh_coefficients).astype(np.float32)
        ),
    )
    result.validate()
    return result, {
        "enabled": True,
        "protected_actor_ids": protected_ids,
        "preserve_opacity": bool(preserve_opacity),
        "input_gaussians": int(len(optimized_cloud.means)),
        "output_gaussians": int(len(result.means)),
        "components": components,
    }


def _rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def transform_component(
    cloud: GaussianCloud,
    actor_id: int,
    transform: np.ndarray,
) -> GaussianCloud:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must be 4x4")
    result = GaussianCloud(
        means=cloud.means.copy(),
        colors=cloud.colors.copy(),
        scales=cloud.scales.copy(),
        quaternions=cloud.quaternions.copy(),
        opacities=cloud.opacities.copy(),
        actor_ids=cloud.actor_ids.copy(),
        sh_coefficients=(
            None
            if cloud.sh_coefficients is None
            else cloud.sh_coefficients.copy()
        ),
    )
    mask = result.actor_ids == int(actor_id)
    if not np.any(mask):
        raise ValueError(f"No Gaussians found for actor ID {actor_id}")
    homogeneous = np.c_[result.means[mask], np.ones(int(mask.sum()))]
    result.means[mask] = (transform @ homogeneous.T).T[:, :3].astype(np.float32)
    delta_quaternion = _rotation_matrix_to_wxyz(transform[:3, :3])
    delta_quaternions = np.repeat(delta_quaternion[None], int(mask.sum()), axis=0)
    rotated = _quaternion_multiply_wxyz(delta_quaternions, result.quaternions[mask])
    rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
    result.quaternions[mask] = rotated.astype(np.float32)
    result.validate()
    return result


def transform_components(
    cloud: GaussianCloud,
    transforms_by_actor: dict[int, np.ndarray],
) -> GaussianCloud:
    """Apply independent rigid transforms to actor-tagged Gaussian components."""
    result = GaussianCloud(
        means=cloud.means.copy(),
        colors=cloud.colors.copy(),
        scales=cloud.scales.copy(),
        quaternions=cloud.quaternions.copy(),
        opacities=cloud.opacities.copy(),
        actor_ids=cloud.actor_ids.copy(),
        sh_coefficients=(
            None
            if cloud.sh_coefficients is None
            else cloud.sh_coefficients.copy()
        ),
    )
    for actor_id, transform in transforms_by_actor.items():
        indices = np.flatnonzero(result.actor_ids == int(actor_id))
        if len(indices) == 0:
            continue
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("Every component transform must be 4x4")
        homogeneous = np.c_[result.means[indices], np.ones(len(indices))]
        result.means[indices] = (transform @ homogeneous.T).T[:, :3].astype(
            np.float32
        )
        delta_quaternion = _rotation_matrix_to_wxyz(transform[:3, :3])
        rotated = _quaternion_multiply_wxyz(
            np.repeat(delta_quaternion[None], len(indices), axis=0),
            result.quaternions[indices],
        )
        rotated /= np.linalg.norm(rotated, axis=1, keepdims=True)
        result.quaternions[indices] = rotated.astype(np.float32)
    result.validate()
    return result


def render_gaussians(
    cloud: GaussianCloud,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    width: int,
    height: int,
    *,
    device: str = "cuda",
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    actor_id: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from gsplat import rasterization

    cloud.validate()
    select = np.ones(len(cloud.means), dtype=bool)
    if actor_id is not None:
        select = cloud.actor_ids == int(actor_id)
        if not np.any(select):
            raise ValueError(f"No Gaussians found for actor ID {actor_id}")
    means = torch.as_tensor(cloud.means[select], dtype=torch.float32, device=device)
    quaternions = torch.as_tensor(
        cloud.quaternions[select], dtype=torch.float32, device=device
    )
    scales = torch.as_tensor(cloud.scales[select], dtype=torch.float32, device=device)
    opacities = torch.as_tensor(
        cloud.opacities[select], dtype=torch.float32, device=device
    )
    sh_degree = None
    if cloud.sh_coefficients is None:
        colors = torch.as_tensor(
            cloud.colors[select], dtype=torch.float32, device=device
        )
    else:
        colors = torch.as_tensor(
            cloud.sh_coefficients[select], dtype=torch.float32, device=device
        )
        sh_degree = int(round(math.sqrt(colors.shape[1]))) - 1
    viewmat = torch.as_tensor(
        homogeneous_extrinsic(world_to_camera), dtype=torch.float32, device=device
    )[None]
    camera_matrix = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)[None]
    backgrounds = torch.as_tensor(background, dtype=torch.float32, device=device)[None]
    with torch.inference_mode():
        rendered, alpha, _ = rasterization(
            means,
            quaternions,
            scales,
            opacities,
            colors,
            viewmat,
            camera_matrix,
            int(width),
            int(height),
            backgrounds=backgrounds,
            packed=False,
            render_mode="RGB",
            sh_degree=sh_degree,
            rasterize_mode="antialiased",
        )
    rgb = (
        rendered[0].detach().clamp(0, 1).mul(255).byte().cpu().numpy()
    )
    return rgb, alpha[0, ..., 0].detach().cpu().numpy()


def render_actor_visibility(
    cloud: GaussianCloud,
    actor_id: int,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    width: int,
    height: int,
    *,
    device: str = "cuda",
) -> np.ndarray:
    """Render an actor contribution map while retaining full-scene occlusion."""
    import torch
    from gsplat import rasterization

    cloud.validate()
    actor_mask = cloud.actor_ids == int(actor_id)
    if not np.any(actor_mask):
        raise ValueError(f"No Gaussians found for actor ID {actor_id}")
    means = torch.as_tensor(cloud.means, dtype=torch.float32, device=device)
    quaternions = torch.as_tensor(
        cloud.quaternions, dtype=torch.float32, device=device
    )
    scales = torch.as_tensor(cloud.scales, dtype=torch.float32, device=device)
    opacities = torch.as_tensor(cloud.opacities, dtype=torch.float32, device=device)
    labels = torch.as_tensor(
        np.repeat(actor_mask[:, None], 3, axis=1),
        dtype=torch.float32,
        device=device,
    )
    viewmat = torch.as_tensor(
        homogeneous_extrinsic(world_to_camera), dtype=torch.float32, device=device
    )[None]
    camera_matrix = torch.as_tensor(intrinsic, dtype=torch.float32, device=device)[None]
    background = torch.zeros((1, 3), dtype=torch.float32, device=device)
    with torch.inference_mode():
        rendered, _, _ = rasterization(
            means,
            quaternions,
            scales,
            opacities,
            labels,
            viewmat,
            camera_matrix,
            int(width),
            int(height),
            backgrounds=background,
            packed=False,
            render_mode="RGB",
            rasterize_mode="antialiased",
        )
    return rendered[0, ..., 0].detach().clamp(0, 1).cpu().numpy()


def optimize_appearance(
    cloud: GaussianCloud,
    train_views: list[dict[str, np.ndarray]],
    *,
    steps: int,
    device: str = "cuda",
    learning_rate_color: float = 0.02,
    learning_rate_opacity: float = 0.005,
    learning_rate_scale: float = 0.001,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[GaussianCloud, list[float]]:
    """Optimize SH0-equivalent colors, opacity, and scales with fixed geometry."""
    if steps <= 0:
        return cloud, []
    import torch
    from gsplat import rasterization

    cloud.validate()
    means = torch.as_tensor(cloud.means, dtype=torch.float32, device=device)
    quaternions = torch.as_tensor(cloud.quaternions, dtype=torch.float32, device=device)
    color_logits = torch.nn.Parameter(
        torch.logit(torch.as_tensor(cloud.colors, dtype=torch.float32, device=device))
    )
    opacity_logits = torch.nn.Parameter(
        torch.logit(torch.as_tensor(cloud.opacities, dtype=torch.float32, device=device))
    )
    log_scales = torch.nn.Parameter(
        torch.log(torch.as_tensor(cloud.scales, dtype=torch.float32, device=device))
    )
    initial_log_scales = log_scales.detach().clone()
    optimizer = torch.optim.Adam(
        [
            {"params": [color_logits], "lr": learning_rate_color},
            {"params": [opacity_logits], "lr": learning_rate_opacity},
            {"params": [log_scales], "lr": learning_rate_scale},
        ]
    )
    backgrounds = torch.as_tensor(background, dtype=torch.float32, device=device)[None]
    losses: list[float] = []
    for step in range(steps):
        view = train_views[step % len(train_views)]
        target = (
            torch.as_tensor(view["rgb"], dtype=torch.float32, device=device)[None] / 255.0
        )
        intrinsic = torch.as_tensor(
            view["intrinsic"], dtype=torch.float32, device=device
        )[None]
        extrinsic = torch.as_tensor(
            homogeneous_extrinsic(view["extrinsic"]),
            dtype=torch.float32,
            device=device,
        )[None]
        rendered, _, _ = rasterization(
            means,
            quaternions,
            torch.exp(log_scales),
            torch.sigmoid(opacity_logits),
            torch.sigmoid(color_logits),
            extrinsic,
            intrinsic,
            int(view["width"]),
            int(view["height"]),
            backgrounds=backgrounds,
            packed=False,
            render_mode="RGB",
        )
        l1 = torch.mean(torch.abs(rendered - target))
        mse = torch.mean((rendered - target) ** 2)
        scale_regularizer = torch.mean((log_scales - initial_log_scales) ** 2)
        loss = 0.8 * l1 + 0.2 * mse + 1e-3 * scale_regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            color_logits.clamp_(-9.0, 9.0)
            opacity_logits.clamp_(-6.0, 6.0)
            log_scales.clamp_(math.log(5e-4), math.log(5e-2))
        losses.append(float(loss.detach().cpu()))

    optimized = GaussianCloud(
        means=cloud.means.copy(),
        colors=torch.sigmoid(color_logits).detach().cpu().numpy().astype(np.float32),
        scales=torch.exp(log_scales).detach().cpu().numpy().astype(np.float32),
        quaternions=cloud.quaternions.copy(),
        opacities=torch.sigmoid(opacity_logits).detach().cpu().numpy().astype(np.float32),
        actor_ids=cloud.actor_ids.copy(),
        sh_coefficients=(
            None
            if cloud.sh_coefficients is None
            else cloud.sh_coefficients.copy()
        ),
    )
    optimized.validate()
    return optimized, losses


def _differentiable_ssim(x, y):
    import torch
    import torch.nn.functional as functional

    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    mu_x = functional.avg_pool2d(x, 11, stride=1, padding=5)
    mu_y = functional.avg_pool2d(y, 11, stride=1, padding=5)
    sigma_x = functional.avg_pool2d(x * x, 11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = functional.avg_pool2d(y * y, 11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = functional.avg_pool2d(x * y, 11, stride=1, padding=5) - mu_x * mu_y
    c1 = torch.as_tensor(0.01**2, dtype=x.dtype, device=x.device)
    c2 = torch.as_tensor(0.03**2, dtype=x.dtype, device=x.device)
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (
        sigma_x + sigma_y + c2
    )
    return (numerator / denominator.clamp_min(1e-8)).mean()


def _gaussian_snapshot(params: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in params.items()
        if key
        not in {
            "anchors",
            "initial_scales",
            "initial_quats",
            "initial_sh0",
            "initial_opacities",
        }
    }


def optimize_gaussians(
    cloud: GaussianCloud,
    train_views: list[dict[str, np.ndarray]],
    *,
    steps: int,
    device: str = "cuda",
    seed: int = 0,
    validation_indices: Iterable[int] = (),
    sh_degree: int = 3,
    learning_rate_mean: float = 8e-5,
    learning_rate_color_dc: float = 0.01,
    learning_rate_color_rest: float = 5e-4,
    learning_rate_opacity: float = 0.01,
    learning_rate_scale: float = 0.002,
    learning_rate_rotation: float = 0.001,
    learning_rate_final_factor: float = 0.15,
    l1_weight: float = 0.75,
    ssim_weight: float = 0.25,
    depth_weight: float = 0.02,
    geometry_weight: float = 0.002,
    scale_weight: float = 0.0002,
    color_anchor_weight: float = 0.005,
    sh_rest_weight: float = 0.001,
    opacity_anchor_weight: float = 0.001,
    batch_size: int = 1,
    max_position_delta_m: float = 0.015,
    min_scale_m: float = 2e-4,
    max_scale_m: float = 3e-2,
    background: tuple[float, float, float] = (0.817, 0.817, 0.817),
    eval_every: int = 50,
    min_steps: int = 500,
    patience: int = 5,
    densification: dict[str, Any] | None = None,
    protected_actor_ids: Iterable[int] = (),
    frozen_actor_ids: Iterable[int] = (),
) -> tuple[GaussianCloud, dict[str, Any]]:
    """Optimize a calibrated RGB-D initialization with a bounded 3DGS schedule.

    The four external test views remain outside this function. Optional validation
    indices refer only to the supplied training-view list and are excluded from
    gradient updates.
    """
    if steps <= 0:
        return cloud, {
            "mode": "full_3dgs",
            "steps_completed": 0,
            "losses": [],
            "initial_gaussians": int(len(cloud.means)),
            "final_gaussians": int(len(cloud.means)),
        }
    if not train_views:
        raise ValueError("At least one training view is required")
    if sh_degree < 0 or sh_degree > 3:
        raise ValueError("sh_degree must lie in [0, 3]")
    if max_position_delta_m <= 0:
        raise ValueError("max_position_delta_m must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    import torch
    from gsplat import DefaultStrategy, rasterization
    from gsplat.strategy.ops import remove

    cloud.validate()
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    validation = sorted(set(int(index) for index in validation_indices))
    if any(index < 0 or index >= len(train_views) for index in validation):
        raise ValueError("validation_indices contains an invalid training-view index")
    fit_indices = [index for index in range(len(train_views)) if index not in validation]
    if not fit_indices:
        raise ValueError("Validation split leaves no views for optimization")

    prepared_views = []
    for view in train_views:
        prepared = {
            "rgb": torch.as_tensor(view["rgb"], dtype=torch.float32, device=device)[None]
            / 255.0,
            "intrinsic": torch.as_tensor(
                view["intrinsic"], dtype=torch.float32, device=device
            )[None],
            "extrinsic": torch.as_tensor(
                homogeneous_extrinsic(view["extrinsic"]),
                dtype=torch.float32,
                device=device,
            )[None],
            "width": int(view["width"]),
            "height": int(view["height"]),
        }
        if "depth_m" in view:
            prepared["depth_m"] = torch.as_tensor(
                view["depth_m"], dtype=torch.float32, device=device
            )[None]
        prepared_views.append(prepared)

    coefficient_count = (sh_degree + 1) ** 2
    sh0 = (cloud.colors - 0.5)[:, None, :] / SH_C0
    sh_rest = np.zeros((len(cloud.means), coefficient_count - 1, 3), dtype=np.float32)
    if cloud.sh_coefficients is not None:
        available = min(coefficient_count, cloud.sh_coefficients.shape[1])
        sh0 = cloud.sh_coefficients[:, :1].copy()
        if available > 1:
            sh_rest[:, : available - 1] = cloud.sh_coefficients[:, 1:available]

    def parameter(array: np.ndarray, requires_grad: bool = True):
        return torch.nn.Parameter(
            torch.as_tensor(array, dtype=torch.float32, device=device),
            requires_grad=requires_grad,
        )

    params: dict[str, Any] = {
        "means": parameter(cloud.means),
        "scales": parameter(np.log(cloud.scales)),
        "quats": parameter(cloud.quaternions),
        "opacities": parameter(np.log(cloud.opacities / (1.0 - cloud.opacities))),
        "sh0": parameter(sh0.astype(np.float32)),
        "shN": parameter(sh_rest),
        "actor_ids": parameter(cloud.actor_ids.astype(np.float32), False),
        "anchors": parameter(cloud.means, False),
        "initial_scales": parameter(np.log(cloud.scales), False),
        "initial_quats": parameter(cloud.quaternions, False),
        "initial_sh0": parameter(sh0.astype(np.float32), False),
        "initial_opacities": parameter(
            np.log(cloud.opacities / (1.0 - cloud.opacities)), False
        ),
    }
    learning_rates = {
        "means": learning_rate_mean,
        "scales": learning_rate_scale,
        "quats": learning_rate_rotation,
        "opacities": learning_rate_opacity,
        "sh0": learning_rate_color_dc,
        "shN": learning_rate_color_rest,
    }
    optimizers = {
        name: torch.optim.Adam([params[name]], lr=rate, eps=1e-15)
        for name, rate in learning_rates.items()
    }

    densification = dict(densification or {})
    densification_enabled = bool(densification.get("enabled", True))
    refine_start = int(densification.get("refine_start_iter", 150))
    refine_stop = int(densification.get("refine_stop_iter", 401))
    max_gaussians = int(densification.get("max_gaussians", 450_000))
    strategy = DefaultStrategy(
        prune_opa=float(densification.get("prune_opacity", 0.005)),
        grow_grad2d=float(densification.get("grow_grad2d", 8e-4)),
        grow_scale3d=float(densification.get("grow_scale3d", 0.001)),
        prune_scale3d=float(densification.get("prune_scale3d", 0.05)),
        refine_start_iter=refine_start if densification_enabled else steps + 1,
        refine_stop_iter=refine_stop if densification_enabled else steps + 1,
        reset_every=int(densification.get("reset_every", steps + 1)),
        refine_every=int(densification.get("refine_every", 100)),
        absgrad=bool(densification.get("absgrad", True)),
        verbose=bool(densification.get("verbose", False)),
    )
    strategy.check_sanity(params, optimizers)
    scene_extent = np.ptp(cloud.means, axis=0)
    strategy_state = strategy.initialize_state(scene_scale=float(np.max(scene_extent)))
    protected_actor_ids = set(int(value) for value in protected_actor_ids)
    frozen_actor_ids = set(int(value) for value in frozen_actor_ids)
    backgrounds = torch.as_tensor(background, dtype=torch.float32, device=device)[None]

    def active_sh_degree(step: int) -> int:
        if sh_degree == 0:
            return 0
        progress = (step + 1) / max(1, steps)
        return min(sh_degree, max(0, int(progress * (sh_degree + 1))))

    @torch.no_grad()
    def validation_psnr() -> float:
        indices = validation if validation else fit_indices
        values = []
        for index in indices:
            view = prepared_views[index]
            rendered, _, _ = rasterization(
                params["means"],
                torch.nn.functional.normalize(params["quats"], dim=-1),
                torch.exp(params["scales"]),
                torch.sigmoid(params["opacities"]),
                torch.cat([params["sh0"], params["shN"]], dim=1),
                view["extrinsic"],
                view["intrinsic"],
                view["width"],
                view["height"],
                backgrounds=backgrounds,
                packed=False,
                render_mode="RGB",
                sh_degree=sh_degree,
                rasterize_mode="antialiased",
            )
            mse = torch.mean((rendered - view["rgb"]) ** 2).clamp_min(1e-12)
            values.append(float((-10.0 * torch.log10(mse)).cpu()))
        return float(np.mean(values))

    losses: list[float] = []
    validation_history: list[dict[str, float | int]] = []
    best_validation_psnr = validation_psnr()
    best_snapshot: dict[str, np.ndarray] | None = _gaussian_snapshot(params)
    initial_validation_psnr = best_validation_psnr
    stale_evaluations = 0
    capped_gaussians = 0
    stopped_early = False
    skipped_nonfinite_losses = 0
    repaired_gradient_values = 0
    repaired_parameter_values = 0
    fit_order = np.asarray(fit_indices, dtype=np.int64)
    np_rng.shuffle(fit_order)
    fit_cursor = 0

    for step in range(steps):
        if fit_cursor + batch_size > len(fit_order):
            np_rng.shuffle(fit_order)
            fit_cursor = 0
        batch_indices = fit_order[fit_cursor : fit_cursor + batch_size]
        fit_cursor += len(batch_indices)
        selected_views = [prepared_views[int(index)] for index in batch_indices]
        if len({(view["width"], view["height"]) for view in selected_views}) != 1:
            raise ValueError("All views in an optimization batch must share resolution")
        view = {
            "rgb": torch.cat([item["rgb"] for item in selected_views], dim=0),
            "intrinsic": torch.cat(
                [item["intrinsic"] for item in selected_views], dim=0
            ),
            "extrinsic": torch.cat(
                [item["extrinsic"] for item in selected_views], dim=0
            ),
            "width": selected_views[0]["width"],
            "height": selected_views[0]["height"],
        }
        if all("depth_m" in item for item in selected_views):
            view["depth_m"] = torch.cat(
                [item["depth_m"] for item in selected_views], dim=0
            )
        decay = learning_rate_final_factor ** (step / max(1, steps - 1))
        optimizers["means"].param_groups[0]["lr"] = learning_rate_mean * decay
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
        rendered, alpha, info = rasterization(
            params["means"],
            torch.nn.functional.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]),
            colors,
            view["extrinsic"],
            view["intrinsic"],
            view["width"],
            view["height"],
            backgrounds=backgrounds.expand(len(selected_views), -1),
            packed=False,
            render_mode="RGB+ED" if "depth_m" in view else "RGB",
            sh_degree=active_sh_degree(step),
            absgrad=strategy.absgrad,
            rasterize_mode="antialiased",
        )
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        rgb = rendered[..., :3]
        l1 = torch.mean(torch.abs(rgb - view["rgb"]))
        dssim = 1.0 - _differentiable_ssim(rgb, view["rgb"])
        loss = l1_weight * l1 + ssim_weight * dssim
        if "depth_m" in view:
            target_depth = view["depth_m"]
            valid_depth = (target_depth > 0.05) & (target_depth < 4.0)
            valid_depth &= alpha[..., 0] > 0.05
            if torch.any(valid_depth):
                predicted_depth = rendered[..., 3]
                relative_depth = torch.abs(predicted_depth - target_depth) / target_depth
                loss = loss + depth_weight * relative_depth[valid_depth].clamp_max(1.0).mean()
        displacement = (params["means"] - params["anchors"]) / max_position_delta_m
        geometry_regularizer = displacement.square().mean()
        scale_regularizer = (params["scales"] - params["initial_scales"]).square().mean()
        color_regularizer = (params["sh0"] - params["initial_sh0"]).square().mean()
        rest_regularizer = params["shN"].square().mean()
        opacity_regularizer = (
            torch.sigmoid(params["opacities"])
            - torch.sigmoid(params["initial_opacities"])
        ).square().mean()
        loss = (
            loss
            + geometry_weight * geometry_regularizer
            + scale_weight * scale_regularizer
            + color_anchor_weight * color_regularizer
            + sh_rest_weight * rest_regularizer
            + opacity_anchor_weight * opacity_regularizer
        )

        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        if not bool(torch.isfinite(loss)):
            skipped_nonfinite_losses += 1
            if skipped_nonfinite_losses >= 3:
                stopped_early = True
                break
            continue
        loss.backward()
        with torch.no_grad():
            frozen = torch.zeros(
                len(params["actor_ids"]), dtype=torch.bool, device=device
            )
            actor_values = params["actor_ids"].round().to(torch.int64)
            for actor_id in frozen_actor_ids:
                frozen |= actor_values == actor_id
            for name in optimizers:
                gradient = params[name].grad
                if gradient is None:
                    continue
                if torch.any(frozen):
                    gradient[frozen] = 0.0
                invalid_gradient = ~torch.isfinite(gradient)
                repaired_gradient_values += int(invalid_gradient.sum().item())
                if torch.any(invalid_gradient):
                    gradient.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        for optimizer in optimizers.values():
            optimizer.step()
        strategy.step_post_backward(
            params, optimizers, strategy_state, step, info, packed=False
        )
        with torch.no_grad():
            invalid_means = ~torch.isfinite(params["means"])
            repaired_parameter_values += int(invalid_means.sum().item())
            params["means"].copy_(
                torch.where(invalid_means, params["anchors"], params["means"])
            )
            invalid_scales = ~torch.isfinite(params["scales"])
            repaired_parameter_values += int(invalid_scales.sum().item())
            params["scales"].copy_(
                torch.where(
                    invalid_scales, params["initial_scales"], params["scales"]
                )
            )
            invalid_quats = ~torch.isfinite(params["quats"])
            repaired_parameter_values += int(invalid_quats.sum().item())
            params["quats"].copy_(
                torch.where(
                    invalid_quats, params["initial_quats"], params["quats"]
                )
            )
            for name in ("opacities", "sh0", "shN"):
                invalid = ~torch.isfinite(params[name])
                repaired_parameter_values += int(invalid.sum().item())
                if torch.any(invalid):
                    params[name].nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            delta = (params["means"] - params["anchors"]).clamp(
                -max_position_delta_m, max_position_delta_m
            )
            params["means"].copy_(params["anchors"] + delta)
            params["scales"].clamp_(math.log(min_scale_m), math.log(max_scale_m))
            params["quats"].copy_(
                torch.nn.functional.normalize(params["quats"], dim=-1)
            )
            params["opacities"].clamp_(-8.0, 8.0)
            params["sh0"].clamp_(-5.0, 5.0)
            params["shN"].clamp_(-3.0, 3.0)

        if len(params["means"]) > max_gaussians:
            with torch.no_grad():
                excess = len(params["means"]) - max_gaussians
                actor_values = params["actor_ids"].round().to(torch.int64)
                protected = torch.zeros_like(actor_values, dtype=torch.bool)
                for actor_id in protected_actor_ids:
                    protected |= actor_values == actor_id
                candidates = torch.where(~protected)[0]
                if len(candidates) < excess:
                    candidates = torch.arange(len(actor_values), device=device)
                candidate_opacity = torch.sigmoid(params["opacities"][candidates])
                prune_indices = candidates[
                    torch.topk(candidate_opacity, excess, largest=False).indices
                ]
                prune_mask = torch.zeros(
                    len(params["means"]), dtype=torch.bool, device=device
                )
                prune_mask[prune_indices] = True
                remove(params, optimizers, strategy_state, prune_mask)
                capped_gaussians += int(excess)
        losses.append(float(loss.detach().cpu()))

        should_evaluate = (step + 1) % eval_every == 0 or step + 1 == steps
        evaluation_ready = step + 1 >= min(steps, max(min_steps, refine_stop + 1))
        if should_evaluate and evaluation_ready:
            score = validation_psnr()
            validation_history.append(
                {
                    "step": step + 1,
                    "psnr_db": score,
                    "gaussians": int(len(params["means"])),
                }
            )
            if math.isfinite(score) and score > best_validation_psnr + 1e-4:
                best_validation_psnr = score
                best_snapshot = _gaussian_snapshot(params)
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            if step + 1 >= min_steps and stale_evaluations >= patience:
                stopped_early = True
                break

    if best_snapshot is None:
        best_validation_psnr = validation_psnr()
        best_snapshot = _gaussian_snapshot(params)
    actor_ids_np = np.rint(best_snapshot["actor_ids"]).astype(np.int64)
    sh_coefficients = np.concatenate(
        [best_snapshot["sh0"], best_snapshot["shN"]], axis=1
    ).astype(np.float32)
    colors_np = np.clip(
        sh_coefficients[:, 0] * SH_C0 + 0.5, 1e-4, 1.0 - 1e-4
    ).astype(np.float32)
    optimized = GaussianCloud(
        means=best_snapshot["means"].astype(np.float32),
        colors=colors_np,
        scales=np.exp(best_snapshot["scales"]).astype(np.float32),
        quaternions=(
            best_snapshot["quats"]
            / np.linalg.norm(best_snapshot["quats"], axis=1, keepdims=True)
        ).astype(np.float32),
        opacities=(1.0 / (1.0 + np.exp(-best_snapshot["opacities"]))).astype(
            np.float32
        ),
        actor_ids=actor_ids_np,
        sh_coefficients=sh_coefficients,
    )
    optimized.validate()
    summary = {
        "mode": "full_3dgs",
        "steps_requested": int(steps),
        "steps_completed": int(len(losses)),
        "stopped_early": stopped_early,
        "losses": losses,
        "initial_gaussians": int(len(cloud.means)),
        "final_gaussians": int(len(optimized.means)),
        "capped_gaussians": int(capped_gaussians),
        "best_validation_psnr_db": float(best_validation_psnr),
        "initial_validation_psnr_db": float(initial_validation_psnr),
        "validation_indices": validation,
        "fit_view_count": int(len(fit_indices)),
        "validation_view_count": int(len(validation)),
        "batch_size": int(batch_size),
        "validation_history": validation_history,
        "sh_degree": int(sh_degree),
        "densification_enabled": densification_enabled,
        "frozen_actor_ids": sorted(frozen_actor_ids),
        "background": list(background),
        "skipped_nonfinite_losses": int(skipped_nonfinite_losses),
        "repaired_gradient_values": int(repaired_gradient_values),
        "repaired_parameter_values": int(repaired_parameter_values),
    }
    return optimized, summary


def write_graphdeco_ply(cloud: GaussianCloud, output_path: Path, sh_degree: int = 3) -> None:
    from plyfile import PlyData, PlyElement

    cloud.validate()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rest_count = 3 * ((sh_degree + 1) ** 2 - 1)
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names.extend(["f_dc_0", "f_dc_1", "f_dc_2"])
    names.extend([f"f_rest_{index}" for index in range(rest_count)])
    names.append("opacity")
    names.extend(["scale_0", "scale_1", "scale_2"])
    names.extend(["rot_0", "rot_1", "rot_2", "rot_3"])
    dtype = [(name, "f4") for name in names] + [("actor_id", "i4")]
    vertices = np.empty(len(cloud.means), dtype=dtype)
    if cloud.sh_coefficients is None:
        features_dc = (cloud.colors - 0.5) / SH_C0
        features_rest = np.zeros((len(cloud.means), rest_count), dtype=np.float32)
    else:
        features_dc = cloud.sh_coefficients[:, 0]
        available_rest = cloud.sh_coefficients[:, 1:]
        padded_rest = np.zeros(
            (len(cloud.means), (sh_degree + 1) ** 2 - 1, 3), dtype=np.float32
        )
        count = min(padded_rest.shape[1], available_rest.shape[1])
        padded_rest[:, :count] = available_rest[:, :count]
        features_rest = padded_rest.transpose(0, 2, 1).reshape(len(cloud.means), -1)
    opacity_logits = np.log(cloud.opacities / (1.0 - cloud.opacities))
    attributes = np.c_[
        cloud.means,
        np.zeros_like(cloud.means),
        features_dc,
        features_rest,
        opacity_logits,
        np.log(cloud.scales),
        cloud.quaternions,
    ]
    for index, name in enumerate(names):
        vertices[name] = attributes[:, index].astype(np.float32)
    vertices["actor_id"] = cloud.actor_ids.astype(np.int32)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(output_path)


def read_graphdeco_ply(input_path: Path) -> GaussianCloud:
    """Read a Graphdeco-compatible PLY, including FedSplat actor IDs and SH."""
    from plyfile import PlyData

    vertices = PlyData.read(Path(input_path))["vertex"].data
    names = set(vertices.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "actor_id",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"PLY is missing required Gaussian properties: {missing}")
    means = np.stack([vertices[name] for name in ("x", "y", "z")], axis=1)
    features_dc = np.stack(
        [vertices[f"f_dc_{index}"] for index in range(3)], axis=1
    )
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if rest_names:
        flat_rest = np.stack([vertices[name] for name in rest_names], axis=1)
        if flat_rest.shape[1] % 3 != 0:
            raise ValueError("PLY SH-rest property count must be divisible by 3")
        rest = flat_rest.reshape(len(vertices), 3, -1).transpose(0, 2, 1)
        sh_coefficients = np.concatenate([features_dc[:, None, :], rest], axis=1)
    else:
        sh_coefficients = features_dc[:, None, :]
    cloud = GaussianCloud(
        means=means.astype(np.float32),
        colors=np.clip(features_dc * SH_C0 + 0.5, 1e-4, 1.0 - 1e-4).astype(
            np.float32
        ),
        scales=np.exp(
            np.stack([vertices[f"scale_{index}"] for index in range(3)], axis=1)
        ).astype(np.float32),
        quaternions=np.stack(
            [vertices[f"rot_{index}"] for index in range(4)], axis=1
        ).astype(np.float32),
        opacities=(1.0 / (1.0 + np.exp(-vertices["opacity"]))).astype(np.float32),
        actor_ids=vertices["actor_id"].astype(np.int64),
        sh_coefficients=sh_coefficients.astype(np.float32),
    )
    cloud.validate()
    return cloud


def psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    reference_f = reference.astype(np.float64) / 255.0
    prediction_f = prediction.astype(np.float64) / 255.0
    error = (reference_f - prediction_f) ** 2
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return math.nan
        error = error[mask]
    mse = float(np.mean(error))
    return 100.0 if mse <= 1e-12 else float(-10.0 * math.log10(mse))


def ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    import cv2

    x = reference.astype(np.float64) / 255.0
    y = prediction.astype(np.float64) / 255.0
    c1, c2 = 0.01**2, 0.03**2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return float(np.mean(score))


def mask_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    union = np.logical_or(reference, prediction).sum()
    return 1.0 if union == 0 else float(np.logical_and(reference, prediction).sum() / union)


def soft_mask_iou(reference: np.ndarray, score: np.ndarray) -> float:
    """Compute IoU against a continuous contribution map without thresholding."""
    reference = np.asarray(reference, dtype=np.float64)
    score = np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0)
    if reference.shape != score.shape:
        raise ValueError("Reference and score arrays must share shape")
    intersection = float(np.sum(reference * score))
    union = float(np.sum(reference) + np.sum(score) - intersection)
    return 1.0 if union == 0.0 else intersection / union


def binary_average_precision(reference: np.ndarray, score: np.ndarray) -> float | None:
    """Compute threshold-free pixel average precision for a binary mask."""
    reference = np.asarray(reference, dtype=bool).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if reference.shape != score.shape:
        raise ValueError("Reference and score arrays must share shape")
    positive_count = int(reference.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-score, kind="mergesort")
    ordered_truth = reference[order]
    ordered_score = score[order]
    true_positive = np.cumsum(ordered_truth)
    false_positive = np.cumsum(~ordered_truth)
    distinct_ends = np.r_[
        np.flatnonzero(np.diff(ordered_score)), len(ordered_score) - 1
    ]
    precision = true_positive[distinct_ends] / (
        true_positive[distinct_ends] + false_positive[distinct_ends]
    )
    recall = true_positive[distinct_ends] / positive_count
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def camera_center_from_extrinsic(world_to_camera: np.ndarray) -> np.ndarray:
    """Return the camera center in world coordinates."""
    return np.linalg.inv(homogeneous_extrinsic(world_to_camera))[:3, 3]


def fit_visibility_threshold_calibration(
    records: list[dict[str, Any]],
    *,
    candidate_thresholds: Iterable[float],
    distance_bin_count: int = 4,
    min_visible_pixels: int = 20,
    fallback_threshold: float = 0.5,
) -> dict[str, Any]:
    """Fit distance-conditioned alpha thresholds using calibration views only."""
    thresholds = sorted(set(float(value) for value in candidate_thresholds))
    if not thresholds or any(value <= 0.0 or value >= 1.0 for value in thresholds):
        raise ValueError("Visibility thresholds must lie strictly inside (0, 1)")
    if distance_bin_count <= 0:
        raise ValueError("distance_bin_count must be positive")
    if min_visible_pixels <= 0:
        raise ValueError("min_visible_pixels must be positive")

    eligible = []
    rejected = 0
    for record in records:
        truth = np.asarray(record["truth_mask"], dtype=bool)
        visibility = np.asarray(record["visibility"], dtype=np.float32)
        if truth.shape != visibility.shape:
            raise ValueError("Truth masks and visibility maps must share shape")
        if int(truth.sum()) < min_visible_pixels:
            rejected += 1
            continue
        eligible.append(
            {
                "name": str(record.get("name", f"view_{len(eligible):03d}")),
                "distance_m": float(record["distance_m"]),
                "truth_mask": truth,
                "visibility": visibility,
            }
        )
    if not eligible:
        return {
            "enabled": False,
            "policy": "fixed_fallback",
            "fallback_threshold": float(fallback_threshold),
            "candidate_thresholds": thresholds,
            "eligible_views": 0,
            "rejected_small_target_views": int(rejected),
            "distance_bins": [],
        }

    ordered = sorted(eligible, key=lambda item: item["distance_m"])
    groups = np.array_split(
        np.arange(len(ordered)), min(distance_bin_count, len(ordered))
    )
    bins = []
    for group_indices in groups:
        group = [ordered[int(index)] for index in group_indices]
        scores = {}
        for threshold in thresholds:
            scores[threshold] = float(
                np.mean(
                    [
                        mask_iou(
                            item["truth_mask"], item["visibility"] > threshold
                        )
                        for item in group
                    ]
                )
            )
        selected = max(
            thresholds,
            key=lambda value: (scores[value], -abs(value - fallback_threshold)),
        )
        distances = [item["distance_m"] for item in group]
        bins.append(
            {
                "distance_center_m": float(np.mean(distances)),
                "distance_min_m": float(np.min(distances)),
                "distance_max_m": float(np.max(distances)),
                "selected_threshold": float(selected),
                "calibration_mean_iou": float(scores[selected]),
                "views": int(len(group)),
                "view_names": [item["name"] for item in group],
            }
        )
    return {
        "enabled": True,
        "policy": "nearest_distance_bin",
        "fallback_threshold": float(fallback_threshold),
        "candidate_thresholds": thresholds,
        "distance_bin_count": int(len(bins)),
        "eligible_views": int(len(eligible)),
        "rejected_small_target_views": int(rejected),
        "min_visible_pixels": int(min_visible_pixels),
        "distance_bins": bins,
    }


def visibility_threshold_for_distance(
    calibration: dict[str, Any], distance_m: float
) -> float:
    """Select the threshold from the nearest source-calibration distance bin."""
    bins = calibration.get("distance_bins", [])
    if not calibration.get("enabled", False) or not bins:
        return float(calibration.get("fallback_threshold", 0.5))
    selected = min(
        bins,
        key=lambda item: abs(float(item["distance_center_m"]) - float(distance_m)),
    )
    return float(selected["selected_threshold"])


def error_heatmap(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    import cv2

    error = np.mean(
        np.abs(reference.astype(np.float32) - prediction.astype(np.float32)), axis=2
    )
    normalized = np.clip(error * 2.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def labeled_comparison(
    reference: np.ndarray,
    prediction: np.ndarray,
    title: str,
    *,
    canvas_size: tuple[int, int] = (1280, 720),
) -> np.ndarray:
    width, height = canvas_size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    images = [
        ("RoboTwin truth", reference),
        ("Gaussian render", prediction),
        ("Absolute error", error_heatmap(reference, prediction)),
    ]
    tile_width = width // 3
    for index, (label, array) in enumerate(images):
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
        image.thumbnail((tile_width - 20, height - 90), Image.Resampling.LANCZOS)
        x = index * tile_width + (tile_width - image.width) // 2
        y = 56 + (height - 76 - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((index * tile_width + 10, 34), label, fill="black")
    draw.text((12, 10), title, fill="black")
    return np.asarray(canvas)


def write_h264(
    output_path: Path,
    frames: Iterable[np.ndarray],
    *,
    fps: float,
) -> int:
    import imageio_ffmpeg

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = iter(frames)
    try:
        first = next(frames)
    except StopIteration as error:
        raise ValueError("Cannot encode an empty frame sequence") from error
    height, width = first.shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18", "-movflags", "+faststart"],
    )
    writer.send(None)
    count = 0
    try:
        for frame in itertools.chain((first,), frames):
            if frame.shape[:2] != (height, width):
                raise ValueError("All video frames must have the same size")
            writer.send(np.ascontiguousarray(frame[..., :3], dtype=np.uint8).tobytes())
            count += 1
    finally:
        writer.close()
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
