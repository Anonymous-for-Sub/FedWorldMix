"""Persistent component-Gaussian observations for LIBERO development runs."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .robotwin_3dgs import GaussianCloud, psnr, ssim, unproject_rgbd


MODE = "component_gs_state_mirror_pose_oracle"
EVIDENCE_ROLE = "development_component_gs_pose_oracle_only"
POLICY_PIXEL_KEY = "image"


def policy_image_to_camera_coordinates(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value)[::-1])


def camera_image_to_policy_coordinates(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value)[::-1])


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _masked_image_metrics(
    reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(selected))
    if count == 0:
        return {"pixel_count": 0, "psnr_db": None, "mean_absolute_uint8": None}
    delta = reference.astype(np.float64)[selected] - prediction.astype(np.float64)[
        selected
    ]
    mse = float(np.mean(delta**2))
    return {
        "pixel_count": count,
        "psnr_db": float("inf") if mse == 0.0 else float(10.0 * math.log10(255.0**2 / mse)),
        "mean_absolute_uint8": float(np.mean(np.abs(delta))),
    }


def _quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    result = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def _quaternion_from_euler_xyz_deg(values: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(list(values), dtype=np.float64)) / 2.0
    qx = np.array([math.cos(roll), math.sin(roll), 0.0, 0.0])
    qy = np.array([math.cos(pitch), 0.0, math.sin(pitch), 0.0])
    qz = np.array([math.cos(yaw), 0.0, 0.0, math.sin(yaw)])
    return _quaternion_multiply_wxyz(_quaternion_multiply_wxyz(qx, qy), qz)


def dynamic_body_mask(model: Any) -> np.ndarray:
    """Mark bodies whose pose depends on any articulated ancestor."""
    parent = np.asarray(model.body_parentid, dtype=np.int64)
    joint_count = np.asarray(model.body_jntnum, dtype=np.int64)
    dynamic = np.zeros(len(parent), dtype=bool)
    for body_id in range(1, len(parent)):
        cursor = body_id
        visited = set()
        while cursor > 0 and cursor not in visited:
            visited.add(cursor)
            if joint_count[cursor] > 0:
                dynamic[body_id] = True
                break
            cursor = int(parent[cursor])
    return dynamic


def geom_ids_from_segmentation(model: Any, segmentation: np.ndarray) -> np.ndarray:
    """Extract visible, non-transparent MuJoCo geom IDs."""
    import mujoco

    segmentation = np.asarray(segmentation)
    if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
        raise ValueError("MuJoCo segmentation must have shape [H,W,2]")
    geom_ids = np.full(segmentation.shape[:2], -1, dtype=np.int64)
    element_type = segmentation[..., 0]
    element_id = segmentation[..., 1]

    geom = element_type == int(mujoco.mjtObj.mjOBJ_GEOM)
    valid_geom = geom & (element_id >= 0) & (element_id < int(model.ngeom))
    visible_alpha = np.zeros(segmentation.shape[:2], dtype=bool)
    if np.any(valid_geom):
        visible_alpha[valid_geom] = np.asarray(model.geom_rgba, dtype=np.float64)[
            element_id[valid_geom], 3
        ] > 0.05
    valid_geom &= visible_alpha
    if np.any(valid_geom):
        geom_ids[valid_geom] = element_id[valid_geom]
    return geom_ids


def actor_ids_from_segmentation(model: Any, segmentation: np.ndarray) -> np.ndarray:
    """Convert visible MuJoCo geom IDs to owning body IDs."""
    geom_ids = geom_ids_from_segmentation(model, segmentation)
    actor_ids = np.zeros(geom_ids.shape, dtype=np.int64)
    valid = geom_ids >= 0
    if np.any(valid):
        actor_ids[valid] = np.asarray(model.geom_bodyid, dtype=np.int64)[
            geom_ids[valid]
        ]
    return actor_ids


def geom_material_albedo(
    model: Any,
    geom_ids: np.ndarray,
    fallback_colors: np.ndarray,
    *,
    scale: float,
    blend: float,
) -> np.ndarray:
    """Resolve stable material colors for articulated appearance canonicalization."""
    ids = np.asarray(geom_ids, dtype=np.int64)
    fallback = np.asarray(fallback_colors, dtype=np.float32)
    result = fallback.copy()
    valid = ids >= 0
    if not np.any(valid):
        return result
    selected = ids[valid]
    material_ids = np.asarray(model.geom_matid, dtype=np.int64)[selected]
    albedo = np.asarray(model.geom_rgba, dtype=np.float32)[selected, :3].copy()
    has_material = material_ids >= 0
    if np.any(has_material):
        albedo[has_material] = np.asarray(model.mat_rgba, dtype=np.float32)[
            material_ids[has_material], :3
        ]
    albedo = np.clip(albedo * float(scale), 0.0, 1.0)
    result[valid] = (
        (1.0 - float(blend)) * fallback[valid] + float(blend) * albedo
    )
    return result.astype(np.float32)


def body_name_mask(model: Any, patterns: Iterable[str]) -> np.ndarray:
    patterns = [str(pattern) for pattern in patterns]
    result = np.zeros(int(model.nbody), dtype=bool)
    for body_id in range(int(model.nbody)):
        name = model.body_id2name(body_id) or ""
        result[body_id] = any(fnmatchcase(name, pattern) for pattern in patterns)
    return result


def matching_scalar_joint_qpos(model: Any, patterns: Iterable[str]) -> list[dict[str, Any]]:
    """Resolve ordered one-DoF joints used by an articulation calibration sweep."""
    import mujoco

    patterns = [str(pattern) for pattern in patterns]
    records = []
    for joint_id in range(int(model.njnt)):
        name = model.joint_id2name(joint_id) or ""
        if not any(fnmatchcase(name, pattern) for pattern in patterns):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in {
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        }:
            raise ValueError(f"Calibration joint must be scalar: {name}")
        records.append(
            {
                "joint_id": joint_id,
                "joint_name": name,
                "qpos_address": int(model.jnt_qposadr[joint_id]),
                "limited": bool(model.jnt_limited[joint_id]),
                "range": np.asarray(model.jnt_range[joint_id], dtype=np.float64).tolist(),
            }
        )
    return sorted(records, key=lambda record: record["qpos_address"])


def body_local_coordinates(
    world_points: np.ndarray,
    actor_ids: np.ndarray,
    body_positions: np.ndarray,
    body_rotations: np.ndarray,
) -> np.ndarray:
    world_points = np.asarray(world_points, dtype=np.float64)
    actor_ids = np.asarray(actor_ids, dtype=np.int64)
    positions = np.asarray(body_positions, dtype=np.float64)
    rotations = np.asarray(body_rotations, dtype=np.float64)
    if world_points.shape != (len(actor_ids), 3):
        raise ValueError("World points and actor IDs have incompatible shapes")
    if np.any(actor_ids < 0) or np.any(actor_ids >= len(positions)):
        raise ValueError("Actor IDs must index body poses")
    centered = world_points - positions[actor_ids]
    return np.einsum("nij,nj->ni", rotations[actor_ids].transpose(0, 2, 1), centered)


def world_coordinates_from_body_local(
    local_points: np.ndarray,
    actor_ids: np.ndarray,
    body_positions: np.ndarray,
    body_rotations: np.ndarray,
) -> np.ndarray:
    local_points = np.asarray(local_points, dtype=np.float64)
    actor_ids = np.asarray(actor_ids, dtype=np.int64)
    positions = np.asarray(body_positions, dtype=np.float64)
    rotations = np.asarray(body_rotations, dtype=np.float64)
    if local_points.shape != (len(actor_ids), 3):
        raise ValueError("Local points and actor IDs have incompatible shapes")
    return (
        np.einsum("nij,nj->ni", rotations[actor_ids], local_points)
        + positions[actor_ids]
    )


def body_local_directions(
    world_directions: np.ndarray,
    actor_ids: np.ndarray,
    body_rotations: np.ndarray,
) -> np.ndarray:
    directions = np.asarray(world_directions, dtype=np.float64)
    actor_ids = np.asarray(actor_ids, dtype=np.int64)
    rotations = np.asarray(body_rotations, dtype=np.float64)
    local = np.einsum(
        "nij,nj->ni", rotations[actor_ids].transpose(0, 2, 1), directions
    )
    return local / np.maximum(np.linalg.norm(local, axis=1, keepdims=True), 1e-12)


def dynamic_geom_support_mask(
    world_points: np.ndarray,
    actor_ids: np.ndarray,
    geom_ids: np.ndarray,
    model: Any,
    data: Any,
    dynamic_bodies: np.ndarray,
    *,
    margin_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reject dynamic RGB-D points outside their segmented geom support sphere."""
    points = np.asarray(world_points, dtype=np.float64)
    actors = np.asarray(actor_ids, dtype=np.int64)
    geoms = np.asarray(geom_ids, dtype=np.int64)
    dynamic = np.asarray(dynamic_bodies, dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world_points must have shape [N, 3]")
    if actors.shape != (len(points),) or geoms.shape != (len(points),):
        raise ValueError("actor_ids and geom_ids must align with world_points")
    if not math.isfinite(float(margin_m)) or float(margin_m) < 0.0:
        raise ValueError("margin_m must be finite and nonnegative")
    if np.any(actors < 0) or np.any(actors >= len(dynamic)):
        raise ValueError("actor_ids contain an invalid body index")

    geom_centers = np.asarray(data.geom_xpos, dtype=np.float64)
    geom_radii = np.asarray(model.geom_rbound, dtype=np.float64)
    if geom_centers.shape != (len(geom_radii), 3):
        raise ValueError("geom_xpos and geom_rbound have incompatible shapes")
    valid_geom = (geoms >= 0) & (geoms < len(geom_radii))
    positive_support = np.zeros(len(points), dtype=bool)
    positive_support[valid_geom] = geom_radii[geoms[valid_geom]] > 0.0
    dynamic_points = dynamic[actors]
    candidates = dynamic_points & valid_geom & positive_support
    keep = np.ones(len(points), dtype=bool)
    distances = np.empty(0, dtype=np.float64)
    thresholds = np.empty(0, dtype=np.float64)
    if np.any(candidates):
        selected_geoms = geoms[candidates]
        distances = np.linalg.norm(
            points[candidates] - geom_centers[selected_geoms], axis=1
        )
        thresholds = geom_radii[selected_geoms] + float(margin_m)
        keep[candidates] = distances <= thresholds
    rejected = candidates & ~keep
    audit = {
        "schema": "fedsplat.dynamic_geom_support_filter.v1",
        "margin_m": float(margin_m),
        "point_count": int(len(points)),
        "dynamic_point_count": int(np.count_nonzero(dynamic_points)),
        "evaluated_dynamic_point_count": int(np.count_nonzero(candidates)),
        "dynamic_points_without_valid_geom": int(
            np.count_nonzero(dynamic_points & ~valid_geom)
        ),
        "rejected_dynamic_point_count": int(np.count_nonzero(rejected)),
        "rejected_dynamic_fraction": float(
            np.count_nonzero(rejected) / max(np.count_nonzero(candidates), 1)
        ),
        "maximum_evaluated_distance_m": float(
            np.max(distances, initial=0.0)
        ),
        "maximum_support_threshold_m": float(
            np.max(thresholds, initial=0.0)
        ),
    }
    return keep, audit


def quaternions_from_z_axis(normals: np.ndarray) -> np.ndarray:
    """Return wxyz rotations whose local z-axis follows each normal."""
    values = np.asarray(normals, dtype=np.float64)
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    quaternions = np.c_[
        1.0 + values[:, 2],
        -values[:, 1],
        values[:, 0],
        np.zeros(len(values), dtype=np.float64),
    ]
    opposite = values[:, 2] < -0.999999
    quaternions[opposite] = np.array([0.0, 1.0, 0.0, 0.0])
    quaternions /= np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-12
    )
    return quaternions.astype(np.float32)


def surfel_scales(
    depth_m: np.ndarray,
    focal_px: float,
    tangent_scale_factor: float,
    normal_scale_ratio: float,
) -> np.ndarray:
    tangent = (
        np.asarray(depth_m, dtype=np.float32)
        / float(focal_px)
        * float(tangent_scale_factor)
    )
    return np.stack(
        [tangent, tangent, tangent * float(normal_scale_ratio)], axis=1
    ).astype(np.float32)


def select_novel_component_voxels(
    local_points: np.ndarray,
    actor_ids: np.ndarray,
    *,
    voxel_size_m: float,
    seen: set[tuple[int, int, int, int]] | None = None,
) -> tuple[np.ndarray, set[tuple[int, int, int, int]]]:
    """Select the first point in each actor-local voxel."""
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    local_points = np.asarray(local_points, dtype=np.float64)
    actor_ids = np.asarray(actor_ids, dtype=np.int64)
    quantized = np.floor(local_points / float(voxel_size_m)).astype(np.int64)
    observed = set() if seen is None else set(seen)
    keep = np.zeros(len(local_points), dtype=bool)
    for index, (actor_id, xyz) in enumerate(zip(actor_ids, quantized, strict=True)):
        key = (int(actor_id), int(xyz[0]), int(xyz[1]), int(xyz[2]))
        if key in observed:
            continue
        observed.add(key)
        keep[index] = True
    return keep, observed


def select_occluded_auxiliary_points(
    world_points: np.ndarray,
    actor_ids: np.ndarray,
    dynamic_bodies: np.ndarray,
    reference_depth_m: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    occlusion_margin_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep points hidden behind the base view or dynamic points outside it."""
    if occlusion_margin_m <= 0:
        raise ValueError("occlusion_margin_m must be positive")
    points = np.asarray(world_points, dtype=np.float64)
    actor_ids = np.asarray(actor_ids, dtype=np.int64)
    dynamic_bodies = np.asarray(dynamic_bodies, dtype=bool)
    depth = np.asarray(reference_depth_m, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    camera = (world_to_camera @ np.c_[points, np.ones(len(points))].T).T
    z = camera[:, 2]
    u = intrinsic[0, 0] * camera[:, 0] / np.maximum(z, 1e-8) + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera[:, 1] / np.maximum(z, 1e-8) + intrinsic[1, 2]
    columns = np.rint(u).astype(np.int64)
    rows = np.rint(v).astype(np.int64)
    inside = (
        (z > 0.0)
        & (rows >= 0)
        & (rows < depth.shape[0])
        & (columns >= 0)
        & (columns < depth.shape[1])
    )
    behind = np.zeros(len(points), dtype=bool)
    if np.any(inside):
        indices = np.flatnonzero(inside)
        reference = depth[rows[indices], columns[indices]]
        behind[indices] = z[indices] > reference + float(occlusion_margin_m)
    outside_dynamic = (~inside) & dynamic_bodies[actor_ids]
    keep = behind | outside_dynamic
    return keep, {
        "candidate_points": int(len(points)),
        "projected_inside_base_view": int(np.count_nonzero(inside)),
        "occluded_behind_base_depth": int(np.count_nonzero(behind)),
        "dynamic_outside_base_view": int(np.count_nonzero(outside_dynamic)),
        "selected_points": int(np.count_nonzero(keep)),
        "occlusion_margin_m": float(occlusion_margin_m),
    }


def select_component_auxiliary_points(
    world_points: np.ndarray,
    actor_ids: np.ndarray,
    dynamic_bodies: np.ndarray,
    reference_depth_m: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    *,
    occlusion_margin_m: float,
    dynamic_visibility_mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Complete moving components while limiting static-view duplication."""
    occluded, audit = select_occluded_auxiliary_points(
        world_points,
        actor_ids,
        dynamic_bodies,
        reference_depth_m,
        intrinsic,
        world_to_camera,
        occlusion_margin_m=occlusion_margin_m,
    )
    dynamic_points = np.asarray(dynamic_bodies, dtype=bool)[
        np.asarray(actor_ids, dtype=np.int64)
    ]
    if dynamic_visibility_mode == "all_visible":
        keep = occluded | dynamic_points
    elif dynamic_visibility_mode == "occluded_only":
        keep = occluded
    else:
        raise ValueError(
            "dynamic_visibility_mode must be 'all_visible' or 'occluded_only'"
        )
    audit.update(
        {
            "dynamic_visibility_mode": dynamic_visibility_mode,
            "visible_dynamic_candidates": int(np.count_nonzero(dynamic_points)),
            "selected_dynamic_points": int(np.count_nonzero(keep & dynamic_points)),
            "selected_static_points": int(np.count_nonzero(keep & ~dynamic_points)),
            "selected_points": int(np.count_nonzero(keep)),
        }
    )
    return keep, audit


def select_novel_voxels_by_motion_class(
    local_points: np.ndarray,
    actor_ids: np.ndarray,
    dynamic_bodies: np.ndarray,
    *,
    static_voxel_size_m: float,
    dynamic_voxel_size_m: float,
    seen_static: set[tuple[int, int, int, int]] | None = None,
    seen_dynamic: set[tuple[int, int, int, int]] | None = None,
) -> tuple[
    np.ndarray,
    set[tuple[int, int, int, int]],
    set[tuple[int, int, int, int]],
]:
    """Deduplicate static and articulated surfaces at independent resolutions."""
    points = np.asarray(local_points)
    ids = np.asarray(actor_ids, dtype=np.int64)
    is_dynamic = np.asarray(dynamic_bodies, dtype=bool)[ids]
    keep = np.zeros(len(points), dtype=bool)
    static_seen = set() if seen_static is None else set(seen_static)
    dynamic_seen = set() if seen_dynamic is None else set(seen_dynamic)
    if np.any(~is_dynamic):
        selected, static_seen = select_novel_component_voxels(
            points[~is_dynamic],
            ids[~is_dynamic],
            voxel_size_m=static_voxel_size_m,
            seen=static_seen,
        )
        keep[np.flatnonzero(~is_dynamic)] = selected
    if np.any(is_dynamic):
        selected, dynamic_seen = select_novel_component_voxels(
            points[is_dynamic],
            ids[is_dynamic],
            voxel_size_m=dynamic_voxel_size_m,
            seen=dynamic_seen,
        )
        keep[np.flatnonzero(is_dynamic)] = selected
    return keep, static_seen, dynamic_seen


def fuse_actor_local_gaussian_voxels(
    local_points: np.ndarray,
    local_normals: np.ndarray,
    colors: np.ndarray,
    actor_ids: np.ndarray,
    geom_ids: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    source_is_base: np.ndarray,
    *,
    voxel_size_m: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Robustly fuse repeated observations of actor-local surface voxels."""
    if voxel_size_m <= 0.0:
        raise ValueError("voxel_size_m must be positive")
    points = np.asarray(local_points, dtype=np.float64)
    normals = np.asarray(local_normals, dtype=np.float64)
    rgb = np.asarray(colors, dtype=np.float64)
    actors = np.asarray(actor_ids, dtype=np.int64)
    geoms = np.asarray(geom_ids, dtype=np.int64)
    gaussian_scales = np.asarray(scales, dtype=np.float64)
    gaussian_opacities = np.asarray(opacities, dtype=np.float64)
    from_base = np.asarray(source_is_base, dtype=bool)
    count = len(points)
    expected_shapes = {
        "local_points": (count, 3),
        "local_normals": (count, 3),
        "colors": (count, 3),
        "actor_ids": (count,),
        "geom_ids": (count,),
        "scales": (count, 3),
        "opacities": (count,),
        "source_is_base": (count,),
    }
    observed_shapes = {
        "local_points": points.shape,
        "local_normals": normals.shape,
        "colors": rgb.shape,
        "actor_ids": actors.shape,
        "geom_ids": geoms.shape,
        "scales": gaussian_scales.shape,
        "opacities": gaussian_opacities.shape,
        "source_is_base": from_base.shape,
    }
    for name, expected in expected_shapes.items():
        if observed_shapes[name] != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if count == 0:
        report = {
            "schema": "fedsplat.actor_local_voxel_fusion.v1",
            "method": "median_position_color_scale_max_opacity_aligned_mean_normal",
            "voxel_size_m": float(voxel_size_m),
            "input_observations": 0,
            "fused_gaussians": 0,
            "multi_observation_voxels": 0,
            "base_supported_voxels": 0,
            "auxiliary_only_voxels": 0,
            "reduction_fraction": 0.0,
            "observations_per_voxel": {"minimum": 0, "median": 0.0, "p95": 0.0, "maximum": 0},
        }
        return (
            points.astype(np.float32),
            normals.astype(np.float32),
            rgb.astype(np.float32),
            actors,
            geoms,
            gaussian_scales.astype(np.float32),
            gaussian_opacities.astype(np.float32),
            from_base,
            report,
        )

    quantized = np.floor(points / float(voxel_size_m)).astype(np.int64)
    keys = np.column_stack([actors, geoms, quantized])
    order = np.lexsort(
        (
            np.arange(count, dtype=np.int64),
            keys[:, 4],
            keys[:, 3],
            keys[:, 2],
            keys[:, 1],
            keys[:, 0],
        )
    )
    sorted_keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0), axis=1)) + 1]
    stops = np.r_[starts[1:], count]
    fused_count = len(starts)
    fused_points = np.empty((fused_count, 3), dtype=np.float64)
    fused_normals = np.empty((fused_count, 3), dtype=np.float64)
    fused_colors = np.empty((fused_count, 3), dtype=np.float64)
    fused_actors = np.empty(fused_count, dtype=np.int64)
    fused_geoms = np.empty(fused_count, dtype=np.int64)
    fused_scales = np.empty((fused_count, 3), dtype=np.float64)
    fused_opacities = np.empty(fused_count, dtype=np.float64)
    fused_from_base = np.empty(fused_count, dtype=bool)
    observation_counts = stops - starts

    for output_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        members = order[start:stop]
        fused_points[output_index] = np.median(points[members], axis=0)
        fused_colors[output_index] = np.median(rgb[members], axis=0)
        fused_scales[output_index] = np.median(gaussian_scales[members], axis=0)
        fused_opacities[output_index] = np.max(gaussian_opacities[members])
        fused_actors[output_index] = actors[members[0]]
        fused_geoms[output_index] = geoms[members[0]]
        fused_from_base[output_index] = bool(np.any(from_base[members]))

        member_normals = normals[members]
        reference_candidates = members[from_base[members]]
        reference_index = reference_candidates[0] if len(reference_candidates) else members[0]
        reference = normals[reference_index]
        aligned = member_normals.copy()
        aligned[np.einsum("ij,j->i", aligned, reference) < 0.0] *= -1.0
        fused_normal = np.mean(aligned, axis=0)
        norm = float(np.linalg.norm(fused_normal))
        fused_normals[output_index] = reference if norm < 1e-12 else fused_normal / norm

    report = {
        "schema": "fedsplat.actor_local_voxel_fusion.v1",
        "method": "median_position_color_scale_max_opacity_aligned_mean_normal",
        "key": "actor_id+geom_id+floor(local_xyz/voxel_size)",
        "voxel_size_m": float(voxel_size_m),
        "input_observations": int(count),
        "fused_gaussians": int(fused_count),
        "multi_observation_voxels": int(np.count_nonzero(observation_counts > 1)),
        "base_supported_voxels": int(np.count_nonzero(fused_from_base)),
        "auxiliary_only_voxels": int(np.count_nonzero(~fused_from_base)),
        "reduction_fraction": float(1.0 - fused_count / count),
        "observations_per_voxel": {
            "minimum": int(np.min(observation_counts)),
            "median": float(np.median(observation_counts)),
            "p95": float(np.percentile(observation_counts, 95)),
            "maximum": int(np.max(observation_counts)),
        },
    }
    return (
        fused_points.astype(np.float32),
        fused_normals.astype(np.float32),
        fused_colors.astype(np.float32),
        fused_actors,
        fused_geoms,
        fused_scales.astype(np.float32),
        fused_opacities.astype(np.float32),
        fused_from_base,
        report,
    )


def compose_component_gs_policy_observation(
    execution_observation: dict[str, Any],
    rendered_images: np.ndarray,
    *,
    pixel_key: str = POLICY_PIXEL_KEY,
) -> dict[str, Any]:
    pixels = execution_observation.get("pixels")
    if not isinstance(pixels, dict) or pixel_key not in pixels:
        raise ValueError(f"Execution observation has no pixels.{pixel_key}")
    reference = np.asarray(pixels[pixel_key])
    rendered = np.asarray(rendered_images, dtype=np.uint8)
    if reference.shape != rendered.shape:
        raise ValueError(
            f"Component-GS image shape differs for {pixel_key}: "
            f"{rendered.shape} != {reference.shape}"
        )
    policy_pixels = dict(pixels)
    policy_pixels[pixel_key] = np.ascontiguousarray(rendered)
    result = dict(execution_observation)
    result["pixels"] = policy_pixels
    return result


@dataclass
class _Capture:
    rgb: np.ndarray
    policy_rgb: np.ndarray
    depth_m: np.ndarray
    actor_ids: np.ndarray
    geom_ids: np.ndarray
    intrinsic: np.ndarray
    world_to_camera: np.ndarray


def _capture_rgbd(sim: Any, camera_name: str, width: int, height: int) -> _Capture:
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
        get_real_depth_map,
    )

    policy_rgb, normalized_depth = sim.render(
        camera_name=camera_name,
        width=int(width),
        height=int(height),
        depth=True,
    )
    context = getattr(sim, "_render_context_offscreen", None)
    sitegroup = getattr(getattr(context, "vopt", None), "sitegroup", None)
    original_sitegroup = None if sitegroup is None else np.asarray(sitegroup).copy()
    try:
        if sitegroup is not None:
            sitegroup[:] = 0
        segmentation = sim.render(
            camera_name=camera_name,
            width=int(width),
            height=int(height),
            segmentation=True,
        )
    finally:
        if sitegroup is not None:
            sitegroup[:] = original_sitegroup
    camera_to_world = get_camera_extrinsic_matrix(sim, camera_name)
    real_depth = np.asarray(
        get_real_depth_map(sim, np.asarray(normalized_depth)), dtype=np.float32
    )
    segmentation_camera = policy_image_to_camera_coordinates(segmentation)
    return _Capture(
        rgb=policy_image_to_camera_coordinates(policy_rgb).astype(np.uint8),
        policy_rgb=np.ascontiguousarray(policy_rgb, dtype=np.uint8),
        depth_m=policy_image_to_camera_coordinates(real_depth).astype(np.float32),
        actor_ids=actor_ids_from_segmentation(
            sim.model, segmentation_camera
        ).astype(np.int64),
        geom_ids=geom_ids_from_segmentation(
            sim.model, segmentation_camera
        ).astype(np.int64),
        intrinsic=np.asarray(
            get_camera_intrinsic_matrix(sim, camera_name, height, width),
            dtype=np.float64,
        ),
        world_to_camera=np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64)),
    )


def _capture_points(capture: _Capture, max_depth_m: float) -> tuple[np.ndarray, ...]:
    points, colors, actor_ids, audit = unproject_rgbd(
        capture.rgb,
        capture.depth_m,
        capture.actor_ids,
        capture.intrinsic,
        capture.world_to_camera,
        stride=1,
        min_depth_m=0.01,
        max_depth_m=max_depth_m,
    )
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float32)]
    camera_points = (capture.world_to_camera @ homogeneous.T).T
    height, width = capture.depth_m.shape
    rows, columns = np.mgrid[0:height, 0:width]
    depth = np.asarray(capture.depth_m, dtype=np.float64)
    fx, fy = float(capture.intrinsic[0, 0]), float(capture.intrinsic[1, 1])
    cx, cy = float(capture.intrinsic[0, 2]), float(capture.intrinsic[1, 2])
    camera_grid = np.stack(
        [
            (columns - cx) * depth / fx,
            (rows - cy) * depth / fy,
            depth,
            np.ones_like(depth),
        ],
        axis=-1,
    )
    camera_to_world = np.linalg.inv(capture.world_to_camera)
    world_grid = np.einsum("ij,hwj->hwi", camera_to_world, camera_grid)[..., :3]
    tangent_x = np.gradient(world_grid, axis=1)
    tangent_y = np.gradient(world_grid, axis=0)
    world_normals = np.cross(tangent_x, tangent_y)
    world_normals /= np.maximum(
        np.linalg.norm(world_normals, axis=-1, keepdims=True), 1e-12
    )
    camera_position = camera_to_world[:3, 3]
    view_to_camera = camera_position[None, None, :] - world_grid
    flip = np.sum(world_normals * view_to_camera, axis=-1) < 0.0
    world_normals[flip] *= -1.0
    valid = (
        np.isfinite(depth)
        & (depth >= 0.01)
        & (depth <= float(max_depth_m))
    )
    normals = world_normals[valid].astype(np.float32)
    geom_ids = capture.geom_ids[valid].astype(np.int64)
    if len(normals) != len(points):
        raise RuntimeError("RGB-D normal and point counts differ")
    audit["normal_estimation"] = {
        "method": "central_difference_world_surfel",
        "finite_fraction": float(np.isfinite(normals).all(axis=1).mean()),
        "median_norm": float(np.median(np.linalg.norm(normals, axis=1))),
    }
    return (
        points,
        colors,
        actor_ids,
        geom_ids,
        camera_points[:, 2].astype(np.float32),
        normals,
        audit,
    )


def _capture_component_points(
    capture: _Capture,
    max_depth_m: float,
    sim: Any,
    dynamic_bodies: np.ndarray,
    support_margin_m: float | None,
) -> tuple[np.ndarray, ...]:
    captured = list(_capture_points(capture, max_depth_m))
    audit = captured[-1]
    if support_margin_m is None:
        audit["dynamic_geom_support_filter"] = {
            "enabled": False,
            "schema": "fedsplat.dynamic_geom_support_filter.v1",
        }
        return tuple(captured)
    keep, support_audit = dynamic_geom_support_mask(
        captured[0],
        captured[2],
        captured[3],
        sim.model,
        sim.data,
        dynamic_bodies,
        margin_m=float(support_margin_m),
    )
    for index in range(6):
        captured[index] = captured[index][keep]
    audit["dynamic_geom_support_filter"] = {
        "enabled": True,
        **support_audit,
        "retained_point_count": int(np.count_nonzero(keep)),
    }
    return tuple(captured)


class ComponentGaussianScene:
    """One persistent Gaussian scene bound to one vector-environment member."""

    def __init__(
        self,
        *,
        local_means: np.ndarray,
        local_normals: np.ndarray,
        colors: np.ndarray,
        material_albedos: np.ndarray,
        relight_mask: np.ndarray,
        scales: np.ndarray,
        opacities: np.ndarray,
        actor_ids: np.ndarray,
        intrinsic: np.ndarray,
        world_to_camera: np.ndarray,
        reference_rgb: np.ndarray,
        initialization_report: dict[str, Any],
        initial_body_positions: np.ndarray,
        initial_body_rotations: np.ndarray,
        backface_culling: bool,
        relighting_blend: float,
        relighting_intensity_scale: float,
        relighting_gamma: float,
        device: str,
        background_rgb: tuple[float, float, float],
    ) -> None:
        import torch

        self.device = str(device)
        self.width = int(reference_rgb.shape[1])
        self.height = int(reference_rgb.shape[0])
        self.reference_rgb = np.ascontiguousarray(reference_rgb, dtype=np.uint8)
        self.initialization_report = initialization_report
        self.initial_body_positions = np.ascontiguousarray(
            initial_body_positions, dtype=np.float64
        )
        self.initial_body_rotations = np.ascontiguousarray(
            initial_body_rotations, dtype=np.float64
        )
        self.local_means_np = np.ascontiguousarray(local_means, dtype=np.float32)
        self.local_normals_np = np.ascontiguousarray(local_normals, dtype=np.float32)
        self.colors_np = np.ascontiguousarray(colors, dtype=np.float32)
        self.material_albedos_np = np.ascontiguousarray(
            material_albedos, dtype=np.float32
        )
        self.relight_mask_np = np.ascontiguousarray(relight_mask, dtype=bool)
        self.scales_np = np.ascontiguousarray(scales, dtype=np.float32)
        self.opacities_np = np.ascontiguousarray(opacities, dtype=np.float32)
        self.actor_ids_np = np.ascontiguousarray(actor_ids, dtype=np.int64)
        self.local_means = torch.as_tensor(
            self.local_means_np, dtype=torch.float32, device=self.device
        )
        self.local_normals = torch.as_tensor(
            self.local_normals_np, dtype=torch.float32, device=self.device
        )
        self.colors = torch.as_tensor(
            self.colors_np, dtype=torch.float32, device=self.device
        )
        self.material_albedos = torch.as_tensor(
            self.material_albedos_np, dtype=torch.float32, device=self.device
        )
        self.relight_mask = torch.as_tensor(
            self.relight_mask_np, dtype=torch.bool, device=self.device
        )
        self.backface_culling = bool(backface_culling)
        self.relighting_blend = float(relighting_blend)
        self.relighting_intensity_scale = float(relighting_intensity_scale)
        self.relighting_gamma = float(relighting_gamma)
        self.scales = torch.as_tensor(
            self.scales_np, dtype=torch.float32, device=self.device
        )
        self.opacities = torch.as_tensor(
            self.opacities_np, dtype=torch.float32, device=self.device
        )
        self.actor_ids = torch.as_tensor(
            self.actor_ids_np, dtype=torch.long, device=self.device
        )
        self.viewmat = torch.as_tensor(
            world_to_camera, dtype=torch.float32, device=self.device
        )[None]
        self.camera_position = torch.as_tensor(
            np.linalg.inv(world_to_camera)[:3, 3],
            dtype=torch.float32,
            device=self.device,
        )
        self.intrinsic = torch.as_tensor(
            intrinsic, dtype=torch.float32, device=self.device
        )[None]
        self.background = torch.as_tensor(
            background_rgb, dtype=torch.float32, device=self.device
        )[None]
        self.render_latencies_s: list[float] = []
        self.render_count = 0

    @classmethod
    def build(cls, sim: Any, config: dict[str, Any], environment_index: int) -> "ComponentGaussianScene":
        camera_name = str(config.get("camera_name", "agentview"))
        width = int(config["width"])
        height = int(config["height"])
        max_depth_m = float(config.get("max_depth_m", 5.0))
        max_gaussians = int(config.get("max_gaussians", 140_000))
        base_scale_factor = float(config.get("base_scale_factor", 0.75))
        base_opacity = float(config.get("base_opacity", 0.88))
        normal_scale_ratio = float(config.get("surfel_normal_scale_ratio", 0.15))
        auxiliary_scale_factor = float(config.get("auxiliary_scale_factor", 0.65))
        auxiliary_opacity = float(config.get("auxiliary_opacity", 0.88))
        dynamic_auxiliary_scale_factor = float(
            config.get("dynamic_auxiliary_scale_factor", 0.55)
        )
        dynamic_auxiliary_opacity = float(
            config.get("dynamic_auxiliary_opacity", 0.72)
        )
        dynamic_geom_support_margin = config.get(
            "dynamic_geom_support_margin_m"
        )
        if dynamic_geom_support_margin is not None:
            dynamic_geom_support_margin = float(dynamic_geom_support_margin)
        static_voxel_size = float(
            config.get("auxiliary_novel_voxel_size_m", 0.0025)
        )
        dynamic_voxel_size = float(
            config.get("dynamic_auxiliary_novel_voxel_size_m", 0.0015)
        )
        occlusion_margin = float(config.get("auxiliary_occlusion_margin_m", 0.008))
        dynamic_visibility_mode = str(
            config.get("auxiliary_dynamic_visibility_mode", "all_visible")
        )
        dynamic_voxel_fusion = bool(
            config.get("dynamic_actor_local_voxel_fusion", False)
        )
        static_disocclusion_completion = bool(
            config.get("static_disocclusion_completion", False)
        )
        calibration_offsets = list(
            config.get("articulation_calibration_qpos_offsets", [])
        )
        calibration_body_patterns = list(
            config.get("articulation_calibration_body_patterns", [])
        )
        calibration_joint_patterns = list(
            config.get("articulation_calibration_joint_patterns", [])
        )
        material_albedo_scale = float(
            config.get("articulation_material_albedo_scale", 0.85)
        )
        material_albedo_blend = float(
            config.get("articulation_material_albedo_blend", 1.0)
        )
        backface_culling = bool(config.get("articulation_backface_culling", True))
        relighting_intensity_scale = float(
            config.get("articulation_relighting_intensity_scale", 1.0)
        )
        relighting_gamma = float(
            config.get("articulation_relighting_gamma", 2.2)
        )

        body_positions = np.asarray(sim.data.body_xpos, dtype=np.float64).copy()
        body_rotations = np.asarray(sim.data.body_xmat, dtype=np.float64).reshape(-1, 3, 3).copy()
        dynamic = dynamic_body_mask(sim.model)
        nominal = _capture_rgbd(sim, camera_name, width, height)
        (
            points,
            colors,
            actor_ids,
            geom_ids,
            depths,
            world_normals,
            unprojection_audit,
        ) = _capture_component_points(
            nominal,
            max_depth_m,
            sim,
            dynamic,
            dynamic_geom_support_margin,
        )
        local = body_local_coordinates(points, actor_ids, body_positions, body_rotations)
        local_normals = body_local_directions(
            world_normals, actor_ids, body_rotations
        )
        body_distances = np.linalg.norm(points - body_positions[actor_ids], axis=1)
        dynamic_distances = body_distances[dynamic[actor_ids]]
        binding_gate = {
            "maximum_dynamic_body_distance_median_m": float(
                config.get("maximum_dynamic_body_distance_median_m", 0.2)
            ),
            "maximum_dynamic_body_distance_p95_m": float(
                config.get("maximum_dynamic_body_distance_p95_m", 0.5)
            ),
        }
        body_binding = {
            "dynamic_point_count": int(len(dynamic_distances)),
            "dynamic_body_distance_median_m": float(np.median(dynamic_distances)),
            "dynamic_body_distance_p95_m": float(np.percentile(dynamic_distances, 95)),
            "gate": binding_gate,
        }
        body_binding["passed"] = bool(
            len(dynamic_distances)
            and body_binding["dynamic_body_distance_median_m"]
            <= binding_gate["maximum_dynamic_body_distance_median_m"]
            and body_binding["dynamic_body_distance_p95_m"]
            <= binding_gate["maximum_dynamic_body_distance_p95_m"]
        )
        if body_binding["passed"] is not True:
            raise RuntimeError("Component-GS body-binding geometry gate failed")
        base_count = len(points)
        focal = float(nominal.intrinsic[0, 0])
        scales = surfel_scales(
            depths, focal, base_scale_factor, normal_scale_ratio
        )
        opacities = np.full(base_count, base_opacity, dtype=np.float32)

        _, seen_static, seen_dynamic = select_novel_voxels_by_motion_class(
            local,
            actor_ids,
            dynamic,
            static_voxel_size_m=static_voxel_size,
            dynamic_voxel_size_m=dynamic_voxel_size,
        )
        auxiliary_local: list[np.ndarray] = []
        auxiliary_local_normals: list[np.ndarray] = []
        auxiliary_colors: list[np.ndarray] = []
        auxiliary_scales: list[np.ndarray] = []
        auxiliary_opacities: list[np.ndarray] = []
        auxiliary_actor_ids: list[np.ndarray] = []
        auxiliary_geom_ids: list[np.ndarray] = []
        auxiliary_reports = []
        background_completion_report = {
            "enabled": static_disocclusion_completion,
            "method": "temporarily_move_dynamic_geoms_out_of_view",
        }
        if static_disocclusion_completion:
            geom_body_ids = np.asarray(sim.model.geom_bodyid, dtype=np.int64)
            dynamic_geom_ids = np.flatnonzero(dynamic[geom_body_ids])
            if not len(dynamic_geom_ids):
                raise RuntimeError("Static disocclusion completion found no dynamic geoms")
            original_geom_positions = np.asarray(
                sim.model.geom_pos[dynamic_geom_ids], dtype=np.float64
            ).copy()
            original_geom_rgba = np.asarray(
                sim.model.geom_rgba[dynamic_geom_ids], dtype=np.float64
            ).copy()
            original_geom_groups = np.asarray(
                sim.model.geom_group[dynamic_geom_ids], dtype=np.int32
            ).copy()
            render_context = getattr(sim, "_render_context_offscreen", None)
            rendered_geom_groups = getattr(
                getattr(render_context, "vopt", None), "geomgroup", None
            )
            original_rendered_geom_groups = (
                None
                if rendered_geom_groups is None
                else np.asarray(rendered_geom_groups).copy()
            )
            try:
                sim.model.geom_pos[dynamic_geom_ids] = np.asarray(
                    [1000.0, 1000.0, 1000.0], dtype=np.float64
                )
                sim.model.geom_rgba[dynamic_geom_ids, 3] = 0.0
                sim.model.geom_group[dynamic_geom_ids] = 5
                if rendered_geom_groups is not None:
                    rendered_geom_groups[5] = 0
                sim.forward()
                background_capture = _capture_rgbd(
                    sim, camera_name, width, height
                )
            finally:
                sim.model.geom_pos[dynamic_geom_ids] = original_geom_positions
                sim.model.geom_rgba[dynamic_geom_ids] = original_geom_rgba
                sim.model.geom_group[dynamic_geom_ids] = original_geom_groups
                if original_rendered_geom_groups is not None:
                    rendered_geom_groups[:] = original_rendered_geom_groups
                sim.forward()
            (
                background_points,
                background_colors,
                background_ids,
                background_geom_ids,
                background_depths,
                background_world_normals,
                background_audit,
            ) = _capture_component_points(
                background_capture,
                max_depth_m,
                sim,
                dynamic,
                dynamic_geom_support_margin,
            )
            dynamic_background_points = int(
                np.count_nonzero(dynamic[background_ids])
            )
            if dynamic_background_points:
                raise RuntimeError(
                    "Static disocclusion capture still contains "
                    f"{dynamic_background_points} dynamic-body points"
                )
            background_local = body_local_coordinates(
                background_points,
                background_ids,
                body_positions,
                body_rotations,
            )
            background_local_normals = body_local_directions(
                background_world_normals,
                background_ids,
                body_rotations,
            )
            novel, seen_static, seen_dynamic = select_novel_voxels_by_motion_class(
                background_local,
                background_ids,
                dynamic,
                static_voxel_size_m=static_voxel_size,
                dynamic_voxel_size_m=dynamic_voxel_size,
                seen_static=seen_static,
                seen_dynamic=seen_dynamic,
            )
            background_local = background_local[novel]
            background_local_normals = background_local_normals[novel]
            background_colors = background_colors[novel]
            background_ids = background_ids[novel]
            background_geom_ids = background_geom_ids[novel]
            background_depths = background_depths[novel]
            auxiliary_local.append(background_local.astype(np.float32))
            auxiliary_local_normals.append(
                background_local_normals.astype(np.float32)
            )
            auxiliary_colors.append(background_colors.astype(np.float32))
            auxiliary_actor_ids.append(background_ids.astype(np.int64))
            auxiliary_geom_ids.append(background_geom_ids.astype(np.int64))
            auxiliary_scales.append(
                surfel_scales(
                    background_depths,
                    float(background_capture.intrinsic[0, 0]),
                    auxiliary_scale_factor,
                    normal_scale_ratio,
                )
            )
            auxiliary_opacities.append(
                np.full(
                    len(background_local),
                    auxiliary_opacity,
                    dtype=np.float32,
                )
            )
            background_completion_report.update(
                {
                    "hidden_dynamic_geom_count": int(len(dynamic_geom_ids)),
                    "captured_point_count": int(len(background_points)),
                    "captured_dynamic_point_count": dynamic_background_points,
                    "novel_static_gaussians": int(len(background_local)),
                    "model_geometry_restored": bool(
                        np.array_equal(
                            np.asarray(sim.model.geom_pos[dynamic_geom_ids]),
                            original_geom_positions,
                        )
                        and np.array_equal(
                            np.asarray(sim.model.geom_rgba[dynamic_geom_ids]),
                            original_geom_rgba,
                        )
                        and np.array_equal(
                            np.asarray(sim.model.geom_group[dynamic_geom_ids]),
                            original_geom_groups,
                        )
                        and (
                            original_rendered_geom_groups is None
                            or np.array_equal(
                                np.asarray(rendered_geom_groups),
                                original_rendered_geom_groups,
                            )
                        )
                    ),
                    "unprojection": background_audit,
                }
            )
        camera_id = int(sim.model.camera_name2id(camera_name))
        camera_position = np.asarray(sim.model.cam_pos[camera_id], dtype=np.float64).copy()
        camera_quaternion = np.asarray(sim.model.cam_quat[camera_id], dtype=np.float64).copy()
        try:
            for view_index, view in enumerate(config.get("auxiliary_views", [])):
                position_delta = np.asarray(
                    view.get("position_delta_m", [0.0, 0.0, 0.0]), dtype=np.float64
                )
                rotation_delta = _quaternion_from_euler_xyz_deg(
                    view.get("rotation_delta_deg", [0.0, 0.0, 0.0])
                )
                sim.model.cam_pos[camera_id] = camera_position + position_delta
                sim.model.cam_quat[camera_id] = _quaternion_multiply_wxyz(
                    camera_quaternion, rotation_delta
                )
                sim.forward()
                capture = _capture_rgbd(sim, camera_name, width, height)
                (
                    aux_points,
                    aux_colors,
                    aux_ids,
                    aux_geom_ids,
                    aux_depths,
                    aux_world_normals,
                    aux_audit,
                ) = _capture_component_points(
                    capture,
                    max_depth_m,
                    sim,
                    dynamic,
                    dynamic_geom_support_margin,
                )
                keep_occluded, occlusion_audit = select_component_auxiliary_points(
                    aux_points,
                    aux_ids,
                    dynamic,
                    nominal.depth_m,
                    nominal.intrinsic,
                    nominal.world_to_camera,
                    occlusion_margin_m=occlusion_margin,
                    dynamic_visibility_mode=dynamic_visibility_mode,
                )
                aux_points = aux_points[keep_occluded]
                aux_colors = aux_colors[keep_occluded]
                aux_ids = aux_ids[keep_occluded]
                aux_geom_ids = aux_geom_ids[keep_occluded]
                aux_depths = aux_depths[keep_occluded]
                aux_world_normals = aux_world_normals[keep_occluded]
                aux_local = body_local_coordinates(
                    aux_points, aux_ids, body_positions, body_rotations
                )
                aux_local_normals = body_local_directions(
                    aux_world_normals, aux_ids, body_rotations
                )
                novel, seen_static, seen_dynamic = select_novel_voxels_by_motion_class(
                    aux_local,
                    aux_ids,
                    dynamic,
                    static_voxel_size_m=static_voxel_size,
                    dynamic_voxel_size_m=dynamic_voxel_size,
                    seen_static=seen_static,
                    seen_dynamic=seen_dynamic,
                )
                novel_before_fusion = novel.copy()
                if dynamic_voxel_fusion:
                    novel |= dynamic[aux_ids]
                aux_local = aux_local[novel]
                aux_colors = aux_colors[novel]
                aux_ids = aux_ids[novel]
                aux_geom_ids = aux_geom_ids[novel]
                aux_depths = aux_depths[novel]
                aux_local_normals = aux_local_normals[novel]
                auxiliary_local.append(aux_local.astype(np.float32))
                auxiliary_local_normals.append(
                    aux_local_normals.astype(np.float32)
                )
                auxiliary_colors.append(aux_colors.astype(np.float32))
                auxiliary_actor_ids.append(aux_ids.astype(np.int64))
                auxiliary_geom_ids.append(aux_geom_ids.astype(np.int64))
                is_dynamic_auxiliary = dynamic[aux_ids]
                scale_factors = np.where(
                    is_dynamic_auxiliary,
                    dynamic_auxiliary_scale_factor,
                    auxiliary_scale_factor,
                ).astype(np.float32)
                tangent_scales = (
                    aux_depths
                    / float(capture.intrinsic[0, 0])
                    * scale_factors
                )
                auxiliary_scales.append(
                    np.stack(
                        [
                            tangent_scales,
                            tangent_scales,
                            tangent_scales * normal_scale_ratio,
                        ],
                        axis=1,
                    ).astype(np.float32)
                )
                auxiliary_opacities.append(
                    np.where(
                        is_dynamic_auxiliary,
                        dynamic_auxiliary_opacity,
                        auxiliary_opacity,
                    ).astype(np.float32)
                )
                auxiliary_reports.append(
                    {
                        "view_index": view_index,
                        "occlusion_selection": occlusion_audit,
                        "novel_gaussians": int(len(aux_local)),
                        "strictly_novel_gaussians": int(
                            np.count_nonzero(novel_before_fusion)
                        ),
                        "retained_dynamic_reobservations": int(
                            np.count_nonzero(novel & ~novel_before_fusion)
                        ),
                        "novel_dynamic_gaussians": int(
                            np.count_nonzero(is_dynamic_auxiliary)
                        ),
                        "novel_static_gaussians": int(
                            np.count_nonzero(~is_dynamic_auxiliary)
                        ),
                        "unprojection": aux_audit,
                    }
                )
        finally:
            sim.model.cam_pos[camera_id] = camera_position
            sim.model.cam_quat[camera_id] = camera_quaternion
            sim.forward()

        calibration_reports = []
        if calibration_offsets:
            calibration_bodies = body_name_mask(
                sim.model, calibration_body_patterns
            )
            calibration_joints = matching_scalar_joint_qpos(
                sim.model, calibration_joint_patterns
            )
            if not np.any(calibration_bodies):
                raise ValueError("Articulation calibration matched no bodies")
            if not calibration_joints:
                raise ValueError("Articulation calibration matched no joints")
            if any(len(offsets) != len(calibration_joints) for offsets in calibration_offsets):
                raise ValueError(
                    "Every articulation calibration offset must match the joint count"
                )
            original_qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
            original_qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()
            try:
                for pose_index, offsets in enumerate(calibration_offsets):
                    calibrated_qpos = original_qpos.copy()
                    applied_offsets = []
                    for joint, requested_offset in zip(
                        calibration_joints, offsets, strict=True
                    ):
                        address = joint["qpos_address"]
                        target = original_qpos[address] + float(requested_offset)
                        if joint["limited"]:
                            lower, upper = joint["range"]
                            target = float(np.clip(target, lower + 1e-4, upper - 1e-4))
                        calibrated_qpos[address] = target
                        applied_offsets.append(float(target - original_qpos[address]))
                    sim.data.qpos[:] = calibrated_qpos
                    sim.data.qvel[:] = 0.0
                    sim.forward()
                    capture = _capture_rgbd(sim, camera_name, width, height)
                    (
                        pose_points,
                        pose_colors,
                        pose_ids,
                        pose_geom_ids,
                        pose_depths,
                        pose_world_normals,
                        pose_audit,
                    ) = _capture_component_points(
                        capture,
                        max_depth_m,
                        sim,
                        dynamic,
                        dynamic_geom_support_margin,
                    )
                    calibration_selection = calibration_bodies[pose_ids]
                    pose_points = pose_points[calibration_selection]
                    pose_colors = pose_colors[calibration_selection]
                    pose_ids = pose_ids[calibration_selection]
                    pose_geom_ids = pose_geom_ids[calibration_selection]
                    pose_depths = pose_depths[calibration_selection]
                    pose_world_normals = pose_world_normals[calibration_selection]
                    current_positions = np.asarray(
                        sim.data.body_xpos, dtype=np.float64
                    ).copy()
                    current_rotations = np.asarray(
                        sim.data.body_xmat, dtype=np.float64
                    ).reshape(-1, 3, 3).copy()
                    pose_local = body_local_coordinates(
                        pose_points,
                        pose_ids,
                        current_positions,
                        current_rotations,
                    )
                    pose_local_normals = body_local_directions(
                        pose_world_normals,
                        pose_ids,
                        current_rotations,
                    )
                    novel, seen_static, seen_dynamic = (
                        select_novel_voxels_by_motion_class(
                            pose_local,
                            pose_ids,
                            dynamic,
                            static_voxel_size_m=static_voxel_size,
                            dynamic_voxel_size_m=dynamic_voxel_size,
                            seen_static=seen_static,
                            seen_dynamic=seen_dynamic,
                        )
                    )
                    novel_before_fusion = novel.copy()
                    if dynamic_voxel_fusion:
                        novel |= dynamic[pose_ids]
                    pose_local = pose_local[novel]
                    pose_colors = pose_colors[novel]
                    pose_ids = pose_ids[novel]
                    pose_geom_ids = pose_geom_ids[novel]
                    pose_depths = pose_depths[novel]
                    pose_local_normals = pose_local_normals[novel]
                    auxiliary_local.append(pose_local.astype(np.float32))
                    auxiliary_local_normals.append(
                        pose_local_normals.astype(np.float32)
                    )
                    auxiliary_colors.append(pose_colors.astype(np.float32))
                    auxiliary_actor_ids.append(pose_ids.astype(np.int64))
                    auxiliary_geom_ids.append(pose_geom_ids.astype(np.int64))
                    auxiliary_scales.append(
                        surfel_scales(
                            pose_depths,
                            float(capture.intrinsic[0, 0]),
                            dynamic_auxiliary_scale_factor,
                            normal_scale_ratio,
                        )
                    )
                    auxiliary_opacities.append(
                        np.full(
                            len(pose_local),
                            dynamic_auxiliary_opacity,
                            dtype=np.float32,
                        )
                    )
                    calibration_reports.append(
                        {
                            "pose_index": pose_index,
                            "requested_qpos_offsets": [
                                float(value) for value in offsets
                            ],
                            "applied_qpos_offsets": applied_offsets,
                            "selected_body_point_count": int(
                                np.count_nonzero(calibration_selection)
                            ),
                            "novel_gaussians": int(len(pose_local)),
                            "strictly_novel_gaussians": int(
                                np.count_nonzero(novel_before_fusion)
                            ),
                            "retained_dynamic_reobservations": int(
                                np.count_nonzero(novel & ~novel_before_fusion)
                            ),
                            "unprojection": pose_audit,
                        }
                    )
            finally:
                sim.data.qpos[:] = original_qpos
                sim.data.qvel[:] = original_qvel
                sim.forward()

        raw_base_count = base_count
        if auxiliary_local:
            local = np.concatenate([local, *auxiliary_local], axis=0)
            local_normals = np.concatenate(
                [local_normals, *auxiliary_local_normals], axis=0
            )
            colors = np.concatenate([colors, *auxiliary_colors], axis=0)
            actor_ids = np.concatenate([actor_ids, *auxiliary_actor_ids], axis=0)
            geom_ids = np.concatenate([geom_ids, *auxiliary_geom_ids], axis=0)
            scales = np.concatenate([scales, *auxiliary_scales], axis=0)
            opacities = np.concatenate([opacities, *auxiliary_opacities], axis=0)

        dynamic_fusion_report = {
            "enabled": False,
            "schema": "fedsplat.actor_local_voxel_fusion.v1",
        }
        if dynamic_voxel_fusion:
            source_is_base = np.arange(len(local), dtype=np.int64) < raw_base_count
            is_dynamic = dynamic[actor_ids]
            (
                fused_local,
                fused_local_normals,
                fused_colors,
                fused_actor_ids,
                fused_geom_ids,
                fused_scales,
                fused_opacities,
                fused_from_base,
                dynamic_fusion_report,
            ) = fuse_actor_local_gaussian_voxels(
                local[is_dynamic],
                local_normals[is_dynamic],
                colors[is_dynamic],
                actor_ids[is_dynamic],
                geom_ids[is_dynamic],
                scales[is_dynamic],
                opacities[is_dynamic],
                source_is_base[is_dynamic],
                voxel_size_m=dynamic_voxel_size,
            )
            dynamic_fusion_report = {
                "enabled": True,
                **dynamic_fusion_report,
                "raw_base_source_points": int(raw_base_count),
            }
            static_base = (~is_dynamic) & source_is_base
            static_auxiliary = (~is_dynamic) & ~source_is_base
            fused_base = fused_from_base
            fused_auxiliary = ~fused_from_base

            local = np.concatenate(
                [
                    local[static_base],
                    fused_local[fused_base],
                    fused_local[fused_auxiliary],
                    local[static_auxiliary],
                ],
                axis=0,
            )
            local_normals = np.concatenate(
                [
                    local_normals[static_base],
                    fused_local_normals[fused_base],
                    fused_local_normals[fused_auxiliary],
                    local_normals[static_auxiliary],
                ],
                axis=0,
            )
            colors = np.concatenate(
                [
                    colors[static_base],
                    fused_colors[fused_base],
                    fused_colors[fused_auxiliary],
                    colors[static_auxiliary],
                ],
                axis=0,
            )
            actor_ids = np.concatenate(
                [
                    actor_ids[static_base],
                    fused_actor_ids[fused_base],
                    fused_actor_ids[fused_auxiliary],
                    actor_ids[static_auxiliary],
                ]
            )
            geom_ids = np.concatenate(
                [
                    geom_ids[static_base],
                    fused_geom_ids[fused_base],
                    fused_geom_ids[fused_auxiliary],
                    geom_ids[static_auxiliary],
                ]
            )
            scales = np.concatenate(
                [
                    scales[static_base],
                    fused_scales[fused_base],
                    fused_scales[fused_auxiliary],
                    scales[static_auxiliary],
                ],
                axis=0,
            )
            opacities = np.concatenate(
                [
                    opacities[static_base],
                    fused_opacities[fused_base],
                    fused_opacities[fused_auxiliary],
                    opacities[static_auxiliary],
                ]
            )
            base_count = int(np.count_nonzero(static_base) + np.count_nonzero(fused_base))

        material_albedos = colors.copy()
        relight_mask = np.zeros(len(colors), dtype=bool)
        articulated_color_count = 0
        if calibration_offsets:
            articulated_color_mask = calibration_bodies[actor_ids] & (geom_ids >= 0)
            articulated_color_count = int(np.count_nonzero(articulated_color_mask))
            material_albedos[articulated_color_mask] = geom_material_albedo(
                sim.model,
                geom_ids[articulated_color_mask],
                colors[articulated_color_mask],
                scale=material_albedo_scale,
                blend=1.0,
            )
            relight_mask[articulated_color_mask] = True

        prebudget_count = len(local)
        prebudget_dynamic_count = int(np.count_nonzero(dynamic[actor_ids]))
        budget_report = {
            "policy": "preserve_base_then_dynamic_auxiliary_then_static_auxiliary",
            "maximum_gaussians": max_gaussians,
            "prebudget_gaussians": int(prebudget_count),
            "prebudget_dynamic_gaussians": prebudget_dynamic_count,
        }
        if len(local) > max_gaussians:
            rng = np.random.default_rng(int(config.get("seed", 0)) + environment_index)
            auxiliary_indices = np.arange(base_count, len(local), dtype=np.int64)
            dynamic_auxiliary = auxiliary_indices[dynamic[actor_ids[auxiliary_indices]]]
            static_auxiliary = auxiliary_indices[~dynamic[actor_ids[auxiliary_indices]]]
            remaining = max(0, max_gaussians - base_count)
            if len(dynamic_auxiliary) > remaining:
                dynamic_auxiliary = np.sort(
                    rng.choice(dynamic_auxiliary, remaining, replace=False)
                )
                static_auxiliary = np.empty(0, dtype=np.int64)
            else:
                remaining -= len(dynamic_auxiliary)
                if len(static_auxiliary) > remaining:
                    static_auxiliary = np.sort(
                        rng.choice(static_auxiliary, remaining, replace=False)
                    )
            indices = np.r_[
                np.arange(base_count), dynamic_auxiliary, static_auxiliary
            ]
            local = local[indices]
            local_normals = local_normals[indices]
            colors = colors[indices]
            material_albedos = material_albedos[indices]
            relight_mask = relight_mask[indices]
            actor_ids = actor_ids[indices]
            geom_ids = geom_ids[indices]
            scales = scales[indices]
            opacities = opacities[indices]
        budget_report.update(
            {
                "postbudget_gaussians": int(len(local)),
                "postbudget_dynamic_gaussians": int(
                    np.count_nonzero(dynamic[actor_ids])
                ),
                "dropped_gaussians": int(prebudget_count - len(local)),
            }
        )

        unique_actor_ids, actor_counts = np.unique(actor_ids, return_counts=True)
        body_components = []
        for body_id, count in zip(unique_actor_ids, actor_counts, strict=True):
            body_name = sim.model.body_id2name(int(body_id))
            body_components.append(
                {
                    "body_id": int(body_id),
                    "body_name": body_name,
                    "dynamic": bool(dynamic[body_id]),
                    "gaussian_count": int(count),
                }
            )

        report = {
            "schema": "fedsplat.libero_component_gaussian_scene.v1",
            "environment_index": int(environment_index),
            "camera_name": camera_name,
            "width": width,
            "height": height,
            "base_gaussians": base_count,
            "auxiliary_gaussians": int(len(local) - base_count),
            "base_dynamic_gaussians": int(
                np.count_nonzero(dynamic[actor_ids[:base_count]])
            ),
            "auxiliary_dynamic_gaussians": int(
                np.count_nonzero(dynamic[actor_ids[base_count:]])
            ),
            "auxiliary_static_gaussians": int(
                np.count_nonzero(~dynamic[actor_ids[base_count:]])
            ),
            "gaussian_count": int(len(local)),
            "body_count": int(len(body_positions)),
            "dynamic_body_count": int(dynamic.sum()),
            "dynamic_body_ids": np.flatnonzero(dynamic).astype(int).tolist(),
            "simulator_rgbd_capture_count": (
                1
                + int(static_disocclusion_completion)
                + len(auxiliary_reports)
                + len(calibration_reports)
            ),
            "simulator_rgb_per_policy_step": False,
            "persistent_scene": True,
            "actor_local_components": True,
            "image_orientation": {
                "unprojection_input": "vertical_flip_of_mujoco_policy_rgb_to_camera_coordinates",
                "policy_output": "vertical_flip_of_gsplat_camera_output",
            },
            "body_binding": body_binding,
            "component_completion": {
                "dynamic_visibility_mode": dynamic_visibility_mode,
                "static_voxel_size_m": static_voxel_size,
                "dynamic_voxel_size_m": dynamic_voxel_size,
                "dynamic_auxiliary_scale_factor": dynamic_auxiliary_scale_factor,
                "dynamic_auxiliary_opacity": dynamic_auxiliary_opacity,
                "dynamic_geom_support_margin_m": dynamic_geom_support_margin,
                "dynamic_actor_local_voxel_fusion": dynamic_fusion_report,
                "static_disocclusion_completion": background_completion_report,
                "surfel_covariance": {
                    "orientation_source": "rgbd_world_normal_bound_in_body_local_frame",
                    "normal_scale_ratio": normal_scale_ratio,
                    "orientation_updated_from_body_pose_per_render": True,
                },
                "segmentation_contract": {
                    "site_groups_disabled": True,
                    "transparent_geoms_excluded": True,
                    "minimum_geom_alpha": 0.05,
                },
                "articulation_calibration": {
                    "body_patterns": calibration_body_patterns,
                    "joint_patterns": calibration_joint_patterns,
                    "joints": (
                        calibration_joints if calibration_offsets else []
                    ),
                    "pose_count": len(calibration_reports),
                    "material_albedo_scale": material_albedo_scale,
                    "material_albedo_blend": material_albedo_blend,
                    "backface_culling": backface_culling,
                    "relighting_intensity_scale": relighting_intensity_scale,
                    "relighting_gamma": relighting_gamma,
                    "material_colored_gaussians_prebudget": articulated_color_count,
                    "poses": calibration_reports,
                },
                "budget": budget_report,
            },
            "body_components": body_components,
            "unprojection": unprojection_audit,
            "auxiliary_views": auxiliary_reports,
            "gaussian_tensor_sha256": _sha256_arrays(
                local,
                local_normals,
                colors,
                material_albedos,
                relight_mask,
                scales,
                opacities,
                actor_ids,
                geom_ids,
            ),
        }
        return cls(
            local_means=local,
            local_normals=local_normals,
            colors=colors,
            material_albedos=material_albedos,
            relight_mask=relight_mask,
            scales=scales,
            opacities=opacities,
            actor_ids=actor_ids,
            intrinsic=nominal.intrinsic,
            world_to_camera=nominal.world_to_camera,
            reference_rgb=nominal.policy_rgb,
            initialization_report=report,
            initial_body_positions=body_positions,
            initial_body_rotations=body_rotations,
            backface_culling=backface_culling,
            relighting_blend=material_albedo_blend,
            relighting_intensity_scale=relighting_intensity_scale,
            relighting_gamma=relighting_gamma,
            device=str(config.get("device", "cuda")),
            background_rgb=tuple(float(value) for value in config.get("background_rgb", [0.0, 0.0, 0.0])),
        )

    def render(self, sim: Any) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from gsplat import rasterization

        started = time.perf_counter()
        body_positions = torch.as_tensor(
            np.asarray(sim.data.body_xpos), dtype=torch.float32, device=self.device
        )
        body_rotations = torch.as_tensor(
            np.asarray(sim.data.body_xmat).reshape(-1, 3, 3),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            selected_rotations = body_rotations[self.actor_ids]
            means = torch.bmm(
                selected_rotations, self.local_means[..., None]
            )[..., 0] + body_positions[self.actor_ids]
            world_normals = torch.bmm(
                selected_rotations, self.local_normals[..., None]
            )[..., 0]
            world_normals = torch.nn.functional.normalize(world_normals, dim=1)
            quaternions = torch.stack(
                [
                    1.0 + world_normals[:, 2],
                    -world_normals[:, 1],
                    world_normals[:, 0],
                    torch.zeros_like(world_normals[:, 0]),
                ],
                dim=1,
            )
            opposite = world_normals[:, 2] < -0.999999
            quaternions[opposite] = torch.tensor(
                [0.0, 1.0, 0.0, 0.0],
                dtype=torch.float32,
                device=self.device,
            )
            quaternions = torch.nn.functional.normalize(quaternions, dim=1)
            render_opacities = self.opacities
            if self.backface_culling and bool(torch.any(self.relight_mask)):
                view_directions = torch.nn.functional.normalize(
                    self.camera_position[None] - means, dim=1
                )
                front_facing = (
                    torch.sum(world_normals * view_directions, dim=1) > 0.0
                )
                render_opacities = self.opacities.clone()
                render_opacities[self.relight_mask] *= front_facing[
                    self.relight_mask
                ].to(render_opacities.dtype)
            render_colors = self.colors
            if self.relighting_blend > 0.0 and bool(torch.any(self.relight_mask)):
                import mujoco

                mask = self.relight_mask
                relit_means = means[mask]
                relit_normals = world_normals[mask]
                irradiance = torch.zeros_like(self.material_albedos[mask])
                light_positions = np.asarray(sim.data.light_xpos, dtype=np.float32)
                light_directions = np.asarray(sim.data.light_xdir, dtype=np.float32)
                for light_index in range(int(sim.model.nlight)):
                    if not bool(sim.model.light_active[light_index]):
                        continue
                    ambient = torch.as_tensor(
                        np.asarray(sim.model.light_ambient[light_index]),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    diffuse = torch.as_tensor(
                        np.asarray(sim.model.light_diffuse[light_index]),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    light_type = int(sim.model.light_type[light_index])
                    attenuation = torch.ones(
                        len(relit_means), dtype=torch.float32, device=self.device
                    )
                    spot = torch.ones_like(attenuation)
                    if light_type == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL):
                        direction = -torch.as_tensor(
                            light_directions[light_index],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        direction = torch.nn.functional.normalize(direction, dim=0)
                        to_light = direction[None].expand_as(relit_means)
                    elif light_type in {
                        int(mujoco.mjtLightType.mjLIGHT_POINT),
                        int(mujoco.mjtLightType.mjLIGHT_SPOT),
                    }:
                        delta = torch.as_tensor(
                            light_positions[light_index],
                            dtype=torch.float32,
                            device=self.device,
                        )[None] - relit_means
                        distance = torch.linalg.norm(delta, dim=1).clamp_min(1e-6)
                        to_light = delta / distance[:, None]
                        coefficients = torch.as_tensor(
                            np.asarray(sim.model.light_attenuation[light_index]),
                            dtype=torch.float32,
                            device=self.device,
                        )
                        attenuation = 1.0 / (
                            coefficients[0]
                            + coefficients[1] * distance
                            + coefficients[2] * distance.square()
                        ).clamp_min(1e-6)
                        attenuation = torch.where(
                            distance <= float(sim.model.light_range[light_index]),
                            attenuation,
                            torch.zeros_like(attenuation),
                        )
                        if light_type == int(mujoco.mjtLightType.mjLIGHT_SPOT):
                            from_light = -to_light
                            spot_direction = torch.as_tensor(
                                light_directions[light_index],
                                dtype=torch.float32,
                                device=self.device,
                            )
                            spot_direction = torch.nn.functional.normalize(
                                spot_direction, dim=0
                            )
                            cosine = torch.sum(
                                from_light * spot_direction[None], dim=1
                            ).clamp(0.0, 1.0)
                            cutoff = math.cos(
                                math.radians(float(sim.model.light_cutoff[light_index]))
                            )
                            spot = torch.where(
                                cosine >= cutoff,
                                cosine.pow(
                                    float(sim.model.light_exponent[light_index])
                                ),
                                torch.zeros_like(cosine),
                            )
                    else:
                        continue
                    lambert = torch.sum(relit_normals * to_light, dim=1).clamp_min(0.0)
                    irradiance += ambient[None] + (
                        diffuse[None]
                        * lambert[:, None]
                        * attenuation[:, None]
                        * spot[:, None]
                    )
                lit_linear = (
                    self.material_albedos[mask]
                    * irradiance
                    * self.relighting_intensity_scale
                ).clamp(0.0, 1.0)
                lit_srgb = lit_linear.pow(1.0 / self.relighting_gamma)
                render_colors = self.colors.clone()
                render_colors[mask] = (
                    (1.0 - self.relighting_blend) * self.colors[mask]
                    + self.relighting_blend * lit_srgb
                )
            rendered, alpha, _ = rasterization(
                means,
                quaternions,
                self.scales,
                render_opacities,
                render_colors,
                self.viewmat,
                self.intrinsic,
                self.width,
                self.height,
                backgrounds=self.background,
                packed=False,
                render_mode="RGB",
                rasterize_mode="antialiased",
            )
            rgb = camera_image_to_policy_coordinates(
                rendered[0].clamp(0, 1).mul(255).byte().cpu().numpy()
            )
            alpha_np = camera_image_to_policy_coordinates(
                alpha[0, ..., 0].detach().cpu().numpy()
            )
        self.render_latencies_s.append(time.perf_counter() - started)
        self.render_count += 1
        return rgb, alpha_np

    def graphdeco_cloud(self, sim: Any) -> GaussianCloud:
        positions = np.asarray(sim.data.body_xpos, dtype=np.float64)
        rotations = np.asarray(sim.data.body_xmat, dtype=np.float64).reshape(-1, 3, 3)
        means = world_coordinates_from_body_local(
            self.local_means_np,
            self.actor_ids_np,
            positions,
            rotations,
        ).astype(np.float32)
        world_normals = np.einsum(
            "nij,nj->ni",
            rotations[self.actor_ids_np],
            self.local_normals_np,
        )
        return GaussianCloud(
            means=means,
            colors=self.colors_np.copy(),
            scales=self.scales_np.copy(),
            quaternions=quaternions_from_z_axis(world_normals),
            opacities=self.opacities_np.copy(),
            actor_ids=self.actor_ids_np.copy(),
        )

    def pose_delta_report(self, sim: Any) -> dict[str, Any]:
        positions = np.asarray(sim.data.body_xpos, dtype=np.float64)
        rotations = np.asarray(sim.data.body_xmat, dtype=np.float64).reshape(-1, 3, 3)
        dynamic_ids = np.asarray(
            self.initialization_report["dynamic_body_ids"], dtype=np.int64
        )
        translations = np.linalg.norm(
            positions[dynamic_ids] - self.initial_body_positions[dynamic_ids], axis=1
        )
        relative = np.einsum(
            "nij,njk->nik",
            rotations[dynamic_ids],
            self.initial_body_rotations[dynamic_ids].transpose(0, 2, 1),
        )
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        angles_deg = np.rad2deg(np.arccos(cosine))
        return {
            "maximum_dynamic_translation_m": float(np.max(translations, initial=0.0)),
            "mean_dynamic_translation_m": float(np.mean(translations)) if len(translations) else 0.0,
            "maximum_dynamic_rotation_deg": float(np.max(angles_deg, initial=0.0)),
            "mean_dynamic_rotation_deg": float(np.mean(angles_deg)) if len(angles_deg) else 0.0,
        }


class LiberoComponentGSProvider:
    def __init__(
        self,
        scenes: list[ComponentGaussianScene],
        config: dict[str, Any],
        *,
        out_dir: Path | None = None,
        sims: list[Any] | None = None,
    ) -> None:
        self.scenes = scenes
        self.config = dict(config)
        self.startup_quality: dict[str, Any] | None = None
        self.reference_diagnostics: list[dict[str, Any]] = []
        if bool(config.get("export_first_scene_ply", False)):
            if out_dir is None or not sims:
                raise ValueError("PLY export requires an output directory and simulator")
            from .robotwin_3dgs import write_graphdeco_ply

            path = Path(out_dir) / "component_gs_scene_env00.ply"
            write_graphdeco_ply(scenes[0].graphdeco_cloud(sims[0]), path, sh_degree=0)
            scenes[0].initialization_report["exported_ply"] = {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

    @classmethod
    def build(
        cls,
        sims: list[Any],
        config: dict[str, Any],
        *,
        width: int,
        height: int,
        seed: int,
        out_dir: Path | None = None,
    ) -> "LiberoComponentGSProvider":
        build_config = dict(config)
        build_config.update({"width": width, "height": height, "seed": seed})
        scenes = [
            ComponentGaussianScene.build(sim, build_config, index)
            for index, sim in enumerate(sims)
        ]
        return cls(scenes, build_config, out_dir=out_dir, sims=sims)

    def render_policy_observation(
        self,
        execution_observation: dict[str, Any],
        sims: list[Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if len(sims) != len(self.scenes):
            raise ValueError("Component-GS scene and simulator counts differ")
        rendered = []
        alphas = []
        for scene, sim in zip(self.scenes, sims, strict=True):
            rgb, alpha = scene.render(sim)
            rendered.append(rgb)
            alphas.append(alpha)
        observation = compose_component_gs_policy_observation(
            execution_observation,
            np.stack(rendered),
            pixel_key=POLICY_PIXEL_KEY,
        )
        startup = None
        if self.startup_quality is None:
            gate = self.config.get("startup_quality_gate", {})
            rows = []
            for index, (scene, rgb, alpha) in enumerate(
                zip(self.scenes, rendered, alphas, strict=True)
            ):
                reference = scene.reference_rgb
                row = {
                    "environment_index": index,
                    "psnr_db": psnr(reference, rgb),
                    "ssim": ssim(reference, rgb),
                    "mean_absolute_uint8": float(
                        np.abs(reference.astype(np.int16) - rgb.astype(np.int16)).mean()
                    ),
                    "mean_alpha": float(np.mean(alpha)),
                    "reference_rgb_sha256": _sha256_arrays(reference),
                    "rendered_rgb_sha256": _sha256_arrays(rgb),
                    "renderer_output_differs_from_simulator_rgb": bool(
                        not np.array_equal(reference, rgb)
                    ),
                }
                row["passed"] = bool(
                    row["psnr_db"] >= float(gate.get("minimum_psnr_db", 20.0))
                    and row["ssim"] >= float(gate.get("minimum_ssim", 0.75))
                    and row["mean_alpha"] >= float(gate.get("minimum_mean_alpha", 0.85))
                    and row["mean_absolute_uint8"]
                    <= float(gate.get("maximum_mean_absolute_uint8", 16.0))
                    and row["renderer_output_differs_from_simulator_rgb"]
                )
                rows.append(row)
            startup = {
                "schema": "fedsplat.component_gs_startup_quality.v1",
                "quality_gate": gate,
                "environments": rows,
                "passed": bool(rows) and all(row["passed"] for row in rows),
            }
            self.startup_quality = startup
        return observation, startup

    def record_diagnostic_reference(
        self,
        policy_observation: dict[str, Any],
        sims: list[Any],
        *,
        step: int,
        out_dir: Path,
    ) -> dict[str, Any] | None:
        """Compare against simulator RGB without routing the reference to policy."""
        requested = set(int(value) for value in self.config.get("diagnostic_reference_steps", []))
        if int(step) not in requested:
            return None
        from PIL import Image

        rendered = np.asarray(policy_observation["pixels"][POLICY_PIXEL_KEY], dtype=np.uint8)
        camera_name = str(self.config.get("camera_name", "agentview"))
        rows = []
        for index, (sim, prediction, scene) in enumerate(
            zip(sims, rendered, self.scenes, strict=True)
        ):
            reference = np.asarray(
                sim.render(
                    camera_name=camera_name,
                    width=int(prediction.shape[1]),
                    height=int(prediction.shape[0]),
                    segmentation=False,
                ),
                dtype=np.uint8,
            )
            segmentation = sim.render(
                camera_name=camera_name,
                width=int(prediction.shape[1]),
                height=int(prediction.shape[0]),
                segmentation=True,
            )
            current_actor_ids = actor_ids_from_segmentation(sim.model, segmentation)
            current_dynamic_mask = dynamic_body_mask(sim.model)[current_actor_ids]
            articulation_joints = matching_scalar_joint_qpos(
                sim.model,
                self.config.get("articulation_calibration_joint_patterns", []),
            )
            reference_path = Path(out_dir) / (
                f"component_gs_reference__step-{int(step):04d}__env-{index:02d}__simulator.png"
            )
            prediction_path = Path(out_dir) / (
                f"component_gs_reference__step-{int(step):04d}__env-{index:02d}__gaussian.png"
            )
            if bool(self.config.get("export_diagnostic_reference_png", True)):
                Image.fromarray(reference).save(reference_path)
                Image.fromarray(prediction).save(prediction_path)
            row = {
                "environment_index": index,
                "psnr_db": psnr(reference, prediction),
                "ssim": ssim(reference, prediction),
                "mean_absolute_uint8": float(
                    np.abs(reference.astype(np.int16) - prediction.astype(np.int16)).mean()
                ),
                "simulator_reference_sha256": _sha256_arrays(reference),
                "gaussian_prediction_sha256": _sha256_arrays(prediction),
                "reference_used_for_policy_input": False,
                "task_success_or_reward_used": False,
                "region_metrics": {
                    "dynamic_bodies": _masked_image_metrics(
                        reference, prediction, current_dynamic_mask
                    ),
                    "static_bodies": _masked_image_metrics(
                        reference, prediction, ~current_dynamic_mask
                    ),
                },
                "dynamic_pose_delta": scene.pose_delta_report(sim),
                "dynamic_body_pixel_count": int(
                    np.count_nonzero(current_dynamic_mask)
                ),
                "articulation_joint_qpos": [
                    {
                        "joint_id": int(joint["joint_id"]),
                        "joint_name": str(joint["joint_name"]),
                        "qpos_address": int(joint["qpos_address"]),
                        "qpos": float(sim.data.qpos[joint["qpos_address"]]),
                    }
                    for joint in articulation_joints
                ],
            }
            if reference_path.is_file():
                row["simulator_reference_png"] = reference_path.name
                row["gaussian_prediction_png"] = prediction_path.name
            rows.append(row)
        block = {
            "schema": "fedsplat.component_gs_sparse_reference_audit.v1",
            "step": int(step),
            "camera_name": camera_name,
            "diagnostic_only": True,
            "reference_used_for_policy_input": False,
            "task_success_or_reward_used": False,
            "environments": rows,
        }
        self.reference_diagnostics.append(block)
        return block

    def report(self) -> dict[str, Any]:
        latencies = np.asarray(
            [value for scene in self.scenes for value in scene.render_latencies_s],
            dtype=np.float64,
        )
        reference_rows = [
            row
            for block in self.reference_diagnostics
            for row in block["environments"]
        ]
        reference_summary = {
            "requested_steps": [
                int(value) for value in self.config.get("diagnostic_reference_steps", [])
            ],
            "completed_steps": [block["step"] for block in self.reference_diagnostics],
            "simulator_rgb_capture_count": len(reference_rows),
            "reference_used_for_policy_input": False,
            "task_success_or_reward_used": False,
            "mean_psnr_db": (
                float(np.mean([row["psnr_db"] for row in reference_rows]))
                if reference_rows
                else None
            ),
            "minimum_psnr_db": (
                float(min(row["psnr_db"] for row in reference_rows))
                if reference_rows
                else None
            ),
            "mean_ssim": (
                float(np.mean([row["ssim"] for row in reference_rows]))
                if reference_rows
                else None
            ),
            "minimum_ssim": (
                float(min(row["ssim"] for row in reference_rows))
                if reference_rows
                else None
            ),
            "records": self.reference_diagnostics,
        }
        return {
            "schema": "fedsplat.libero_component_gs_provider.v1",
            "mode": MODE,
            "evidence_role": EVIDENCE_ROLE,
            "component_gaussian_renderer_used": True,
            "renderer_backend": "gsplat.cuda.rasterization",
            "persistent_gaussian_scene": True,
            "actor_local_rigid_components": True,
            "pose_source": "simulator_hidden_body_pose_oracle",
            "deployment_eligible": False,
            "simulator_rgb_per_policy_step": False,
            "canonicalized_pixel_keys": [POLICY_PIXEL_KEY],
            "retained_execution_pixel_keys": ["image2"],
            "scene_count": len(self.scenes),
            "total_gaussians": int(
                sum(scene.initialization_report["gaussian_count"] for scene in self.scenes)
            ),
            "total_simulator_rgbd_captures": int(
                sum(
                    scene.initialization_report["simulator_rgbd_capture_count"]
                    for scene in self.scenes
                )
            ),
            "diagnostic_simulator_rgb_captures": len(reference_rows),
            "render_calls": int(sum(scene.render_count for scene in self.scenes)),
            "render_latency_ms": {
                "mean": float(latencies.mean() * 1000.0) if len(latencies) else None,
                "p50": float(np.percentile(latencies, 50) * 1000.0) if len(latencies) else None,
                "p95": float(np.percentile(latencies, 95) * 1000.0) if len(latencies) else None,
            },
            "startup_quality": self.startup_quality,
            "sparse_reference_audit": reference_summary,
            "scenes": [scene.initialization_report for scene in self.scenes],
        }
