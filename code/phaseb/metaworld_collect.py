"""Collect Meta-World demonstrations into Phase B JSONL schema."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class TaskSpec:
    alias: str
    env_name: str
    policy_class_name: str
    instruction: str
    foreground_objects: List[str]
    background_objects: List[str]


TASK_SPECS: Dict[str, TaskSpec] = {
    "door-lock": TaskSpec(
        alias="door-lock",
        env_name="door-lock-v3",
        policy_class_name="SawyerDoorLockV3Policy",
        instruction="lock the door",
        foreground_objects=["door", "lock", "handle"],
        background_objects=["table", "cabinet"],
    ),
    "close-drawer": TaskSpec(
        alias="close-drawer",
        env_name="drawer-close-v3",
        policy_class_name="SawyerDrawerCloseV3Policy",
        instruction="close the drawer",
        foreground_objects=["drawer", "handle"],
        background_objects=["table", "cabinet"],
    ),
    "sweep-into": TaskSpec(
        alias="sweep-into",
        env_name="sweep-into-v3",
        policy_class_name="SawyerSweepIntoV3Policy",
        instruction="sweep object into hole",
        foreground_objects=["object", "sweep-tool", "hole"],
        background_objects=["table", "cabinet"],
    ),
    "open-window": TaskSpec(
        alias="open-window",
        env_name="window-open-v3",
        policy_class_name="SawyerWindowOpenV3Policy",
        instruction="open the window",
        foreground_objects=["window", "handle"],
        background_objects=["table", "cabinet"],
    ),
}

_PAPER_CAMERA_VIEWS: Tuple[str, ...] = ("corner", "top", "shoulder")
# Camera configs are expressed in Gymnasium MujocoRenderer default_cam_config format.
_PAPER_CAMERA_CONFIGS: Dict[str, Dict[str, object]] = {
    # Corner view: slightly elevated, diagonally toward the table.
    "corner": {"azimuth": 45.0, "elevation": -30.0, "distance": 2.2, "lookat": None},
    # Top view: near-overhead.
    "top": {"azimuth": 90.0, "elevation": -89.0, "distance": 1.8, "lookat": None},
    # Shoulder view: over-the-shoulder angle.
    "shoulder": {"azimuth": 135.0, "elevation": -20.0, "distance": 2.0, "lookat": None},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_tasks(raw: str) -> List[str]:
    pieces = [piece.strip() for piece in raw.split(",")]
    return [piece for piece in pieces if piece]

def _parse_views(raw: str) -> List[str]:
    pieces = [piece.strip() for piece in (raw or "").split(",")]
    return [piece for piece in pieces if piece]

_PROJ_CACHE: Dict[tuple[int, int, int, int, int, int], object] = {}


def _projection_matrix(
    grid_h: int,
    grid_w: int,
    row: int,
    col: int,
    out_dim: int,
    in_dim: int,
) -> object:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError("numpy is required for image tokenization in Meta-World collection.") from exc
    key = (int(grid_h), int(grid_w), int(row), int(col), int(out_dim), int(in_dim))
    cached = _PROJ_CACHE.get(key)
    if cached is not None:
        return cached
    seed = (
        1337
        + 1000003 * (int(row) + 1)
        + 9176 * (int(col) + 1)
        + 6113 * int(grid_h)
        + 2711 * int(grid_w)
        + 19 * int(out_dim)
        + 23 * int(in_dim)
    ) % (2**32)
    rng = np.random.default_rng(int(seed))
    scale = 1.0 / math.sqrt(max(int(in_dim), 1))
    mat = rng.standard_normal(size=(int(out_dim), int(in_dim)), dtype=np.float32) * np.float32(scale)
    _PROJ_CACHE[key] = mat
    return mat


def _avg_pool_patch_rgb(patch: object, out_h: int = 4, out_w: int = 4) -> object:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError("numpy is required for image tokenization in Meta-World collection.") from exc
    if not isinstance(patch, np.ndarray):
        raise TypeError("Expected patch to be a numpy.ndarray.")
    safe_h = max(int(out_h), 1)
    safe_w = max(int(out_w), 1)
    pooled_rows = []
    for row_chunk in np.array_split(patch, safe_h, axis=0):
        pooled_cols = []
        for col_chunk in np.array_split(row_chunk, safe_w, axis=1):
            if col_chunk.size <= 0:
                pooled_cols.append(np.zeros((3,), dtype=np.float32))
            else:
                pooled_cols.append(col_chunk.mean(axis=(0, 1)).astype(np.float32))
        pooled_rows.append(np.stack(pooled_cols, axis=0))
    return np.stack(pooled_rows, axis=0)  # (out_h, out_w, 3)


def _tokenize_image(image: object, grid_h: int, grid_w: int, token_dim: int = 8) -> List[List[float]]:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for image tokenization in Meta-World collection.") from exc
    resolved_token_dim = int(token_dim)
    if resolved_token_dim <= 0:
        raise ValueError("token_dim must be > 0.")
    if not isinstance(image, np.ndarray):
        raise TypeError("Expected rendered image to be a numpy.ndarray.")
    height, width, _ = image.shape
    patch_h = max(height // max(grid_h, 1), 1)
    patch_w = max(width // max(grid_w, 1), 1)
    normalized = image.astype(np.float32) / 255.0

    tokens: List[List[float]] = []
    for row in range(grid_h):
        for col in range(grid_w):
            start_h = row * patch_h
            end_h = min((row + 1) * patch_h, height)
            start_w = col * patch_w
            end_w = min((col + 1) * patch_w, width)
            patch = normalized[start_h:end_h, start_w:end_w, :]
            if patch.size <= 0:
                tokens.append([0.0 for _ in range(resolved_token_dim)])
                continue

            rgb_mean = patch.mean(axis=(0, 1))
            rgb_std = patch.std(axis=(0, 1))
            luminance = 0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]
            base = [
                float(rgb_mean[0]),
                float(rgb_mean[1]),
                float(rgb_mean[2]),
                float(rgb_std[0]),
                float(rgb_std[1]),
                float(rgb_std[2]),
                float(luminance.mean()),
                float(luminance.std()),
            ]
            if resolved_token_dim <= len(base):
                tokens.append(base[:resolved_token_dim])
                continue

            pooled = _avg_pool_patch_rgb(patch, out_h=4, out_w=4).reshape(-1)
            proj_dim = resolved_token_dim - len(base)
            mat = _projection_matrix(
                grid_h=int(grid_h),
                grid_w=int(grid_w),
                row=int(row),
                col=int(col),
                out_dim=int(proj_dim),
                in_dim=int(pooled.shape[0]),
            )
            projected = (mat @ pooled.astype(np.float32)).astype(np.float32)
            token = base + [float(value) for value in projected.tolist()]
            tokens.append(token)
    return tokens


def _as_uint8_image(image: object) -> object:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("numpy is required for image processing in Meta-World collection.") from exc
    if not isinstance(image, np.ndarray):
        raise TypeError("Expected rendered image to be a numpy.ndarray.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected rendered image shape (H, W, 3).")
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return clipped


def _save_png(image_uint8: object, path: Path) -> None:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for saving PNG screenshots.") from exc
    _ensure_parent(path)
    Image.fromarray(image_uint8, mode="RGB").save(path)


def _save_gif(frames_uint8: Sequence[object], path: Path, fps: int) -> None:
    if not frames_uint8:
        return
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for saving GIF videos.") from exc
    safe_fps = max(int(fps), 1)
    duration_ms = int(round(1000.0 / float(safe_fps)))
    images = [Image.fromarray(frame, mode="RGB") for frame in frames_uint8]
    _ensure_parent(path)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    _ensure_parent(path)
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _require_metaworld_imports() -> tuple[object, object]:
    # Meta-World rollouts/collection often run headless on compute nodes. When
    # DISPLAY is not available, default Mujoco to EGL so offscreen rendering can
    # create a valid OpenGL context.
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    try:
        from metaworld import MT1
        import metaworld.policies as policies
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError(
            "Meta-World imports failed. Install metaworld/mujoco in the active Python environment."
        ) from exc
    return MT1, policies


def _apply_camera_view(env: object, view: str) -> None:
    if not view:
        return
    renderer = getattr(env, "mujoco_renderer", None)
    if renderer is None:
        return
    config = _PAPER_CAMERA_CONFIGS.get(view, None)
    if config is None:
        return

    viewer_map = getattr(renderer, "_viewers", None)
    viewer = viewer_map.get("rgb_array") if isinstance(viewer_map, dict) else None
    lookat = None
    if viewer is not None:
        cam = getattr(viewer, "cam", None)
        if cam is not None and hasattr(cam, "lookat"):
            try:
                lookat = tuple(float(value) for value in cam.lookat)
            except Exception:
                lookat = None

    resolved = dict(config)
    if resolved.get("lookat", None) is None and lookat is not None:
        resolved["lookat"] = lookat
    renderer.default_cam_config = resolved
    renderer._set_cam_config()


def _save_rgb_npz(frames: object, path: Path, views: Sequence[str]) -> None:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError("numpy is required for saving rgb npz files.") from exc
    if not isinstance(frames, np.ndarray):
        raise TypeError("frames must be a numpy.ndarray.")
    _ensure_parent(path)
    np.savez_compressed(path, frames=frames, views=np.array(list(views)))


def collect_metaworld_demonstrations(
    output_root: str,
    tasks: List[str],
    episodes_per_task: int = 40,
    max_steps_per_episode: int = 100,
    seed: int = 42,
    render_width: int = 256,
    render_height: int = 256,
    token_grid_h: int = 4,
    token_grid_w: int = 6,
    token_dim: int = 8,
    camera_views: List[str] | None = None,
    save_rgb_npz: bool = False,
    media_root: str = "",
    media_max_episodes: int = 0,
    media_max_frames: int = 120,
    media_fps: int = 10,
    media_save_frames: bool = False,
    media_save_gif: bool = False,
) -> Dict[str, object]:
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be > 0.")
    if max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be > 0.")
    if render_width <= 0 or render_height <= 0:
        raise ValueError("render size must be > 0.")
    if token_grid_h <= 0 or token_grid_w <= 0:
        raise ValueError("token grid size must be > 0.")
    if token_dim <= 0:
        raise ValueError("token_dim must be > 0.")
    resolved_camera_views = list(camera_views or [])
    if resolved_camera_views:
        unknown_views = [view for view in resolved_camera_views if view not in _PAPER_CAMERA_CONFIGS]
        if unknown_views:
            raise ValueError(f"Unknown camera view(s): {unknown_views}. Supported: {sorted(_PAPER_CAMERA_CONFIGS)}")
    if media_max_episodes < 0:
        raise ValueError("media_max_episodes must be >= 0.")
    if media_max_frames <= 0:
        raise ValueError("media_max_frames must be > 0.")
    if media_fps <= 0:
        raise ValueError("media_fps must be > 0.")
    if not tasks:
        raise ValueError("At least one task is required.")
    unknown = [task for task in tasks if task not in TASK_SPECS]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Supported: {sorted(TASK_SPECS.keys())}")

    MT1, policies = _require_metaworld_imports()
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    media_path = Path(media_root) if media_root else None
    media_enabled = bool(media_path) and (bool(media_save_frames) or bool(media_save_gif)) and media_max_episodes > 0
    if media_enabled and media_path is not None:
        media_path.mkdir(parents=True, exist_ok=True)

    random_state = random.Random(seed)
    report: Dict[str, object] = {
        "time_utc": _utc_now(),
        "output_root": str(output_path),
        "tasks": list(tasks),
        "episodes_per_task": int(episodes_per_task),
        "max_steps_per_episode": int(max_steps_per_episode),
        "seed": int(seed),
        "render_width": int(render_width),
        "render_height": int(render_height),
        "camera_views": list(resolved_camera_views),
        "save_rgb_npz": bool(save_rgb_npz),
        "token_grid": [int(token_grid_h), int(token_grid_w)],
        "token_dim": int(token_dim),
        "media": {
            "enabled": bool(media_enabled),
            "media_root": str(media_path) if media_path is not None else "",
            "max_episodes_per_task": int(media_max_episodes),
            "max_frames_per_episode": int(media_max_frames),
            "fps": int(media_fps),
            "save_frames": bool(media_save_frames),
            "save_gif": bool(media_save_gif),
        },
        "task_stats": {},
    }

    for task_index, task_alias in enumerate(tasks):
        spec = TASK_SPECS[task_alias]
        mt1 = MT1(spec.env_name, seed=seed + task_index * 1009)
        env_factory = mt1.train_classes[spec.env_name]
        policy_class = getattr(policies, spec.policy_class_name)
        policy = policy_class()

        rows: List[Dict[str, object]] = []
        success_episodes = 0
        step_counts: List[int] = []
        media_saved_episodes = 0
        media_saved_frames = 0
        media_saved_gifs: List[str] = []
        media_sample_frames: List[str] = []

        # Treat episodes_per_task as the number of *successful* expert demonstrations to retain.
        # Meta-World expert policies can occasionally fail under randomized initializations.
        desired_success = int(episodes_per_task)
        max_attempts = max(desired_success * 3, desired_success)
        attempt = 0
        while success_episodes < desired_success and attempt < max_attempts:
            episode = attempt
            capture_episode_media = bool(media_enabled) and success_episodes < int(media_max_episodes)
            episode_media_frames: List[object] = []
            frame_index = 0
            env = env_factory(
                render_mode="rgb_array",
                width=int(render_width),
                height=int(render_height),
            )
            try:
                task_obj = mt1.train_tasks[random_state.randrange(len(mt1.train_tasks))]
                env.set_task(task_obj)
                obs, _ = env.reset()
                # Prime offscreen renderer so subsequent camera config updates take effect.
                try:
                    _ = env.render()
                except Exception:
                    pass
                episode_success = 0.0
                local_steps = 0
                trajectory_id = f"{task_alias}_episode_{success_episodes:04d}"
                episode_rows: List[Dict[str, object]] = []
                episode_rgb_steps: List[object] = []
                rgb_views = resolved_camera_views
                if not rgb_views:
                    rgb_views = ["default"]

                for step in range(max_steps_per_episode):
                    view_tokens: List[List[float]] = []
                    step_rgb_frames: List[object] = []
                    media_frame: object | None = None

                    for view in rgb_views:
                        if view != "default":
                            _apply_camera_view(env, view=view)
                        image = env.render()
                        if image is None:
                            media_frame = None
                            break
                        image_uint8 = _as_uint8_image(image)
                        step_rgb_frames.append(image_uint8)
                        view_tokens.extend(
                            _tokenize_image(
                                image=image_uint8,
                                grid_h=int(token_grid_h),
                                grid_w=int(token_grid_w),
                                token_dim=int(token_dim),
                            )
                        )
                        if media_frame is None:
                            media_frame = image_uint8

                    if media_frame is None:
                        break

                    if bool(save_rgb_npz):
                        try:
                            import numpy as np
                        except Exception as exc:  # pragma: no cover - depends on external env
                            raise RuntimeError("numpy is required for saving rgb npz files.") from exc
                        episode_rgb_steps.append(np.stack(step_rgb_frames, axis=0))

                    if capture_episode_media and media_path is not None and frame_index < int(media_max_frames):
                        episode_dir = media_path / task_alias / f"episode_{episode:04d}"
                        if media_save_frames:
                            frame_path = episode_dir / f"frame_{frame_index:04d}.png"
                            _save_png(image_uint8=media_frame, path=frame_path)
                            media_saved_frames += 1
                            if frame_index == 0:
                                media_sample_frames.append(str(frame_path))
                        if media_save_gif:
                            episode_media_frames.append(media_frame.copy())
                        frame_index += 1
                    action_raw = policy.get_action(obs)
                    # Meta-World environments clip actions to [-1, 1]. Store the clipped action so
                    # behavior cloning labels match the executed control signal.
                    action = [
                        max(-1.0, min(1.0, float(action_raw[idx]))) if idx < len(action_raw) else 0.0
                        for idx in range(4)
                    ]
                    next_obs, _, terminated, truncated, info = env.step(action)
                    success = float(info.get("success", 0.0)) if isinstance(info, dict) else 0.0
                    episode_success = max(episode_success, success)
                    local_steps += 1
                    record = {
                        "instruction": spec.instruction,
                        "proprio": [float(value) for value in obs],
                        "action": [float(value) for value in action[:4]],
                        "success": float(success),
                        "trajectory_id": trajectory_id,
                        "step_in_trajectory": int(step),
                        "foreground_objects": list(spec.foreground_objects),
                        "background_objects": list(spec.background_objects),
                        "image_tokens": view_tokens,
                        "camera_views": list(resolved_camera_views) if resolved_camera_views else [],
                    }
                    if bool(save_rgb_npz):
                        record["rgb_npz_path"] = f"rgb/{trajectory_id}.npz"
                        record["rgb_step"] = int(step)
                        record["rgb_views"] = list(rgb_views)
                    episode_rows.append(record)
                    obs = next_obs
                    # Keep demonstrations short and aligned with evaluation horizons when the task is solved.
                    if success >= 1.0:
                        break
                    if bool(terminated) or bool(truncated):
                        break
                if episode_success >= 1.0:
                    rows.extend(episode_rows)
                    if bool(save_rgb_npz) and episode_rgb_steps:
                        try:
                            import numpy as np
                        except Exception as exc:  # pragma: no cover - depends on external env
                            raise RuntimeError("numpy is required for saving rgb npz files.") from exc
                        frames = np.stack([np.asarray(step) for step in episode_rgb_steps], axis=0)
                        _save_rgb_npz(
                            frames=frames,
                            path=(output_path / task_alias / "rgb" / f"{trajectory_id}.npz"),
                            views=list(rgb_views),
                        )
                    if capture_episode_media and media_save_gif and media_path is not None and episode_media_frames:
                        gif_path = media_path / task_alias / f"episode_{success_episodes:04d}.gif"
                        _save_gif(frames_uint8=episode_media_frames, path=gif_path, fps=int(media_fps))
                        media_saved_gifs.append(str(gif_path))
                    if capture_episode_media:
                        media_saved_episodes += 1
                    success_episodes += 1
                    step_counts.append(local_steps)
            finally:
                env.close()
            attempt += 1

        task_root = output_path / task_alias
        raw_path = task_root / "raw.jsonl"
        _write_jsonl(raw_path, rows)
        mean_steps = sum(step_counts) / float(max(len(step_counts), 1))
        task_report = {
            "env_name": spec.env_name,
            "policy_class_name": spec.policy_class_name,
            "episodes": int(desired_success),
            "attempts": int(attempt),
            "rows_written": int(len(rows)),
            "success_episodes": int(success_episodes),
            "episode_success_rate": float(success_episodes / float(max(desired_success, 1))),
            "mean_steps_per_episode": float(mean_steps),
            "raw_path": str(raw_path),
            "media_saved_episodes": int(media_saved_episodes),
            "media_saved_frames": int(media_saved_frames),
            "media_saved_gif_count": int(len(media_saved_gifs)),
            "media_sample_frames": list(media_sample_frames[:3]),
            "media_sample_gifs": list(media_saved_gifs[:3]),
        }
        report["task_stats"][task_alias] = task_report  # type: ignore[index]

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect Meta-World demonstrations for Phase B.")
    parser.add_argument("--output-root", default="data/metaworld_real_raw")
    parser.add_argument("--tasks", default="door-lock,close-drawer,sweep-into,open-window")
    parser.add_argument("--episodes-per-task", type=int, default=40)
    parser.add_argument("--max-steps-per-episode", type=int, default=100)
    parser.add_argument("--render-width", type=int, default=256)
    parser.add_argument("--render-height", type=int, default=256)
    parser.add_argument(
        "--camera-views",
        default="",
        help="Comma-separated camera views to render (e.g., 'corner,top,shoulder'). Default: single view.",
    )
    parser.add_argument(
        "--paper-cameras",
        action="store_true",
        help="Shortcut for paper-style 3-view RGB (corner/top/shoulder) at 128x128 unless overridden.",
    )
    parser.add_argument("--token-grid-h", type=int, default=4)
    parser.add_argument("--token-grid-w", type=int, default=6)
    parser.add_argument(
        "--token-dim",
        type=int,
        default=8,
        help="Per-patch token vector width stored under `image_tokens`.",
    )
    parser.add_argument("--media-root", default="")
    parser.add_argument("--media-max-episodes", type=int, default=0)
    parser.add_argument("--media-max-frames", type=int, default=120)
    parser.add_argument("--media-fps", type=int, default=10)
    parser.add_argument("--media-save-frames", action="store_true")
    parser.add_argument("--media-save-gif", action="store_true")
    parser.add_argument(
        "--save-rgb-npz",
        action="store_true",
        help="Save per-trajectory multi-view RGB frames to `output_root/<task>/rgb/<trajectory_id>.npz`.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-json", default="runs/phase_b/phaseb_metaworld_collect_report.json")
    parser.add_argument("--report-markdown", default="project/phaseb_metaworld_collect_report.md")
    args = parser.parse_args()

    camera_views = _parse_views(args.camera_views)
    render_width = args.render_width
    render_height = args.render_height
    if bool(args.paper_cameras):
        if not camera_views:
            camera_views = list(_PAPER_CAMERA_VIEWS)
        if int(render_width) == 256 and int(render_height) == 256:
            render_width = 128
            render_height = 128

    report = collect_metaworld_demonstrations(
        output_root=args.output_root,
        tasks=_parse_tasks(args.tasks),
        episodes_per_task=args.episodes_per_task,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        render_width=render_width,
        render_height=render_height,
        token_grid_h=args.token_grid_h,
        token_grid_w=args.token_grid_w,
        token_dim=args.token_dim,
        camera_views=camera_views,
        save_rgb_npz=bool(args.save_rgb_npz),
        media_root=args.media_root,
        media_max_episodes=args.media_max_episodes,
        media_max_frames=args.media_max_frames,
        media_fps=args.media_fps,
        media_save_frames=args.media_save_frames,
        media_save_gif=args.media_save_gif,
    )

    report_json_path = Path(args.report_json)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_markdown_path = Path(args.report_markdown)
    report_markdown_path.parent.mkdir(parents=True, exist_ok=True)

    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    media_summary = report.get("media", {}) if isinstance(report.get("media", {}), dict) else {}
    markdown_lines = [
        "# Phase B Meta-World Collection Report",
        "",
        f"- Time (UTC): `{report['time_utc']}`",
        f"- Output root: `{report['output_root']}`",
        f"- Tasks: `{', '.join(report['tasks'])}`",
        f"- Episodes per task: `{report['episodes_per_task']}`",
        f"- Max steps per episode: `{report['max_steps_per_episode']}`",
        f"- Media enabled: `{media_summary.get('enabled', False)}`",
        f"- Media root: `{media_summary.get('media_root', '')}`",
        "",
        "| Task | Env | Episodes | Rows | Episode Success Rate | Mean Steps | Media Episodes | GIF Count | Raw Path |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    task_stats = report["task_stats"]  # type: ignore[index]
    for task in sorted(task_stats.keys()):
        row = task_stats[task]
        markdown_lines.append(
            "| "
            + f"{task} | "
            + f"{row['env_name']} | "
            + f"{row['episodes']} | "
            + f"{row['rows_written']} | "
            + f"{row['episode_success_rate'] * 100.0:.2f}% | "
            + f"{row['mean_steps_per_episode']:.2f} | "
            + f"{row.get('media_saved_episodes', 0)} | "
            + f"{row.get('media_saved_gif_count', 0)} | "
            + f"{row['raw_path']} |"
        )
    markdown_lines.append("")
    report_markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    report["report_json"] = str(report_json_path)
    report["report_markdown"] = str(report_markdown_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
