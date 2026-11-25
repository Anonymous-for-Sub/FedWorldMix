"""Meta-World rollout evaluator for HPT-backed Phase B checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import random
import ssl
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

from .deps import require_numpy, require_torch
from .hpt.configuration_hpt import HPTConfig
from .hpt.modeling_hpt import HPTPolicy
from .metaworld_collect import TASK_SPECS, _PAPER_CAMERA_CONFIGS


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _maybe_fix_ssl_cert_env() -> None:
    """Work around relocated Python envs with broken OpenSSL default CA paths."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    cafile = None
    try:
        cafile = ssl.get_default_verify_paths().openssl_cafile
    except Exception:
        cafile = None
    if cafile and Path(cafile).exists():
        return
    try:
        import certifi
    except Exception:
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()


def _as_uint8_image(image: object) -> object:
    require_numpy("HPT rollout image conversion")
    import numpy as np

    if not isinstance(image, np.ndarray):
        raise TypeError("Expected rendered image to be a numpy.ndarray.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected rendered image shape (H, W, 3).")
    if image.dtype == np.uint8:
        return image
    clipped = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return clipped


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_png(image_uint8: object, path: Path) -> None:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError("Pillow is required for saving PNG screenshots.") from exc
    _ensure_parent(path)
    Image.fromarray(image_uint8, mode="RGB").save(path)


def _save_gif(frames_uint8: Sequence[object], path: Path, fps: int) -> None:
    if not frames_uint8:
        return
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on external env
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


def _require_metaworld_imports() -> tuple[object, object]:
    if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"
    try:
        from metaworld import MT1
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError("Meta-World imports failed. Install metaworld/mujoco in the active Python env.") from exc
    return MT1, None


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


def _clamp_action(values: Sequence[float], action_dim: int = 4) -> List[float]:
    out: List[float] = []
    for idx in range(int(action_dim)):
        v = float(values[idx]) if idx < len(values) else 0.0
        out.append(max(-1.0, min(1.0, v)))
    return out


def _build_policy_from_checkpoint(
    checkpoint: Dict[str, object],
    state_dict: Dict[str, object],
    device: str,
) -> HPTPolicy:
    require_torch("HPT rollout policy build")
    import torch

    cfg_payload = checkpoint.get("hpt_config", None)
    if not isinstance(cfg_payload, dict):
        raise RuntimeError("Checkpoint missing hpt_config.")
    cfg_payload = dict(cfg_payload)
    # Avoid re-downloading trunk during rollout; we load full state_dict next.
    cfg_payload["load_pretrained"] = ""
    if isinstance(cfg_payload.get("modalities", None), list):
        cfg_payload["modalities"] = tuple(cfg_payload["modalities"])
    hpt_cfg = HPTConfig(**cfg_payload)
    policy = HPTPolicy(config=hpt_cfg)
    policy.load_state_dict(state_dict)
    policy.eval()
    resolved_device = torch.device(device) if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy.to(resolved_device)
    return policy


def evaluate_metaworld_rollouts_hpt(
    checkpoint_path: str,
    tasks: List[str],
    *,
    seed: int = 0,
    trials: int = 15,
    repeats: int = 5,
    max_steps_per_episode: int = 250,
    render_width: int = 128,
    render_height: int = 128,
    camera_views: Sequence[str] = ("corner", "top", "shoulder"),
    torch_device: str = "auto",
    media_root: str = "",
    media_max_episodes: int = 1,
    media_max_frames: int = 120,
    media_fps: int = 10,
    media_save_frames: bool = False,
    media_save_gif: bool = False,
) -> Dict[str, object]:
    require_torch("HPT rollout")
    import torch

    _maybe_fix_ssl_cert_env()
    MT1, _ = _require_metaworld_imports()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unexpected checkpoint format (expected dict).")
    cfg_payload = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
    task_embed_dim = 0
    task_embed_scale = 1.0
    text_embed_dim = 0
    text_embed_scale = 1.0
    text_embed_mode = "state"
    iosp_embed_dim = 0
    iosp_embed_scale = 1.0
    iosp_mode = "proxy"
    task_text_embeddings: Dict[str, List[float]] = {}
    task_iosp_embeddings: Dict[str, List[float]] = {}
    try:
        model_cfg = cfg_payload.get("model", {}) if isinstance(cfg_payload.get("model", {}), dict) else {}
        task_embed_dim = int(model_cfg.get("hpt_task_embed_dim", 0) or 0)
        task_embed_scale = float(model_cfg.get("hpt_task_embed_scale", 1.0) or 1.0)
        text_embed_dim = int(model_cfg.get("hpt_text_embed_dim", 0) or 0)
        text_embed_scale = float(model_cfg.get("hpt_text_embed_scale", 1.0) or 1.0)
        text_embed_mode = str(model_cfg.get("hpt_text_embed_mode", "state") or "state").strip().lower()
        iosp_embed_dim = int(model_cfg.get("hpt_iosp_embed_dim", model_cfg.get("hpt_text_embed_dim", 0)) or 0)
        iosp_embed_scale = float(model_cfg.get("hpt_iosp_embed_scale", 1.0) or 1.0)
        iosp_mode = str(model_cfg.get("hpt_iosp_mode", "proxy") or "proxy").strip().lower()
    except Exception:
        task_embed_dim = 0
        task_embed_scale = 1.0
        text_embed_dim = 0
        text_embed_scale = 1.0
        text_embed_mode = "state"
        iosp_embed_dim = 0
        iosp_embed_scale = 1.0
        iosp_mode = "proxy"

    raw_text_embeds = ckpt.get("task_text_embeddings", {})
    if isinstance(raw_text_embeds, dict):
        for key, value in raw_text_embeds.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, list):
                task_text_embeddings[key] = [float(v) for v in value if isinstance(v, (int, float))]

    raw_iosp_embeds = ckpt.get("task_iosp_embeddings", {})
    if isinstance(raw_iosp_embeds, dict):
        for key, value in raw_iosp_embeds.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, list):
                task_iosp_embeddings[key] = [float(v) for v in value if isinstance(v, (int, float))]

    client_to_task = ckpt.get("client_to_task", {})
    if not isinstance(client_to_task, dict):
        client_to_task = {}
    client_state_dicts = ckpt.get("client_state_dicts", {})
    if not isinstance(client_state_dicts, dict):
        raise RuntimeError("Checkpoint missing client_state_dicts.")

    def resolve_device(raw: str) -> str:
        requested = (raw or "auto").strip().lower()
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return raw

    device = resolve_device(torch_device)
    policies: Dict[str, HPTPolicy] = {}
    for client_id, sd in client_state_dicts.items():
        if not isinstance(client_id, str) or not isinstance(sd, dict):
            continue
        policies[client_id] = _build_policy_from_checkpoint(ckpt, sd, device=device)

    media_path = Path(media_root) if media_root else None
    media_enabled = bool(media_path) and (bool(media_save_frames) or bool(media_save_gif)) and int(media_max_episodes) > 0
    if media_enabled and media_path is not None:
        media_path.mkdir(parents=True, exist_ok=True)

    report: Dict[str, object] = {
        "time_utc": _utc_now(),
        "backbone": "hpt",
        "checkpoint_path": str(checkpoint_path),
        "tasks": list(tasks),
        "seed": int(seed),
        "trials": int(trials),
        "repeats": int(repeats),
        "max_steps_per_episode": int(max_steps_per_episode),
        "render_width": int(render_width),
        "render_height": int(render_height),
        "camera_views": list(camera_views),
        "torch_device": str(device),
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
        print(f"[rollout][hpt] task={task_alias} env={spec.env_name} start")
        # Pick the corresponding client policy when available; otherwise fall back to the first policy.
        client_id = ""
        for cid, tname in client_to_task.items():
            if str(tname) == task_alias and cid in policies:
                client_id = str(cid)
                break
        if not client_id and policies:
            client_id = next(iter(policies.keys()))
        if not client_id:
            raise RuntimeError("No policies available in checkpoint.")
        policy = policies[client_id]

        mt1 = MT1(spec.env_name, seed=seed + task_index * 1009)
        env_factory = mt1.train_classes[spec.env_name]

        repeat_success_rates: List[float] = []
        step_counts: List[int] = []
        media_saved_episodes = 0
        media_saved_frames = 0
        media_saved_gifs: List[str] = []
        media_sample_gifs: List[str] = []

        for repeat_index in range(int(repeats)):
            rng = random.Random(int(seed) + task_index * 1009 + repeat_index * 4099)
            success_episodes = 0
            for trial_index in range(int(trials)):
                capture_episode_media = bool(media_enabled) and media_saved_episodes < int(media_max_episodes)
                episode_media_frames: List[object] = []
                frame_index = 0
                env = env_factory(render_mode="rgb_array", width=int(render_width), height=int(render_height))
                try:
                    task_obj = mt1.train_tasks[rng.randrange(len(mt1.train_tasks))]
                    env.set_task(task_obj)
                    obs, _ = env.reset()
                    # prime viewer
                    try:
                        _ = env.render()
                    except Exception:
                        pass
                    policy.reset()
                    episode_success = 0.0
                    local_steps = 0

                    for step in range(int(max_steps_per_episode)):
                        # Render multi-view observations.
                        images: Dict[str, object] = {}
                        media_frame = None
                        for view in camera_views:
                            _apply_camera_view(env, view=str(view))
                            image = env.render()
                            if image is None:
                                media_frame = None
                                break
                            image_uint8 = _as_uint8_image(image)
                            images[str(view)] = image_uint8
                            if media_frame is None:
                                media_frame = image_uint8
                        if media_frame is None:
                            break

                        require_numpy("HPT rollout input pack")
                        import numpy as np

                        batch: Dict[str, object] = {}
                        for view in camera_views:
                            frame = images[str(view)]
                            frame_arr = np.asarray(frame)
                            if not frame_arr.flags["C_CONTIGUOUS"]:
                                frame_arr = np.ascontiguousarray(frame_arr)
                            tensor = (
                                torch.from_numpy(frame_arr.copy())
                                .to(dtype=torch.float32)
                                .permute(2, 0, 1)
                                / 255.0
                            )
                            batch[f"observation.images.{view}"] = tensor.unsqueeze(0).to(device=device)
                        batch["observation.state"] = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                        iosp_vec = task_iosp_embeddings.get(str(task_alias), [])
                        if iosp_vec:
                            resolved_iosp_dim = int(iosp_embed_dim or 0)
                            if str(iosp_mode) == "clip":
                                resolved_iosp_dim = 512
                            if resolved_iosp_dim > 0:
                                expected = 3 * int(resolved_iosp_dim)
                                vec = [float(v) for v in iosp_vec]
                                if len(vec) < expected:
                                    vec = vec + [0.0] * (expected - len(vec))
                                elif len(vec) > expected:
                                    vec = vec[:expected]
                                vec = [float(v) * float(iosp_embed_scale) for v in vec]
                            else:
                                vec = [float(v) * float(iosp_embed_scale) for v in iosp_vec]
                            batch["observation.iosp"] = torch.tensor(
                                vec,
                                dtype=torch.float32,
                                device=device,
                            ).unsqueeze(0)
                        if task_embed_dim > 0:
                            from .task_embed import task_embedding

                            embed = torch.tensor(
                                task_embedding(task_alias, task_embed_dim, scale=task_embed_scale),
                                dtype=torch.float32,
                                device=device,
                            ).unsqueeze(0)
                            batch["observation.state"] = torch.cat([batch["observation.state"], embed], dim=1)
                        if text_embed_dim > 0:
                            raw = task_text_embeddings.get(str(task_alias), [])
                            vec = [float(v) for v in raw]
                            if len(vec) < int(text_embed_dim):
                                vec = vec + [0.0] * (int(text_embed_dim) - len(vec))
                            elif len(vec) > int(text_embed_dim):
                                vec = vec[: int(text_embed_dim)]
                            embed = torch.tensor(
                                [float(v) * float(text_embed_scale) for v in vec],
                                dtype=torch.float32,
                                device=device,
                            ).unsqueeze(0)
                            if str(text_embed_mode) == "language":
                                batch["observation.language"] = embed
                            else:
                                batch["observation.state"] = torch.cat([batch["observation.state"], embed], dim=1)

                        action_tensor = policy.select_action(batch)
                        action_raw = action_tensor.detach().cpu().view(-1).tolist()
                        action = _clamp_action(action_raw, action_dim=4)

                        if capture_episode_media and media_path is not None and frame_index < int(media_max_frames):
                            episode_dir = media_path / task_alias / f"episode_{media_saved_episodes:04d}"
                            if media_save_frames:
                                frame_path = episode_dir / f"frame_{frame_index:04d}.png"
                                _save_png(image_uint8=media_frame, path=frame_path)
                                media_saved_frames += 1
                            if media_save_gif:
                                episode_media_frames.append(media_frame.copy())
                            frame_index += 1

                        next_obs, _, terminated, truncated, info = env.step(action)
                        success = float(info.get("success", 0.0)) if isinstance(info, dict) else 0.0
                        episode_success = max(episode_success, success)
                        local_steps += 1
                        obs = next_obs
                        if success >= 1.0:
                            break
                        if bool(terminated) or bool(truncated):
                            break

                    repeat_success_rates.append(float(episode_success))
                    step_counts.append(int(local_steps))
                    if episode_success >= 1.0:
                        success_episodes += 1
                    if capture_episode_media and media_save_gif and media_path is not None and episode_media_frames:
                        gif_path = media_path / task_alias / f"episode_{media_saved_episodes:04d}.gif"
                        _save_gif(frames_uint8=episode_media_frames, path=gif_path, fps=int(media_fps))
                        media_saved_gifs.append(str(gif_path))
                        if not media_sample_gifs:
                            media_sample_gifs.append(str(gif_path))
                    if capture_episode_media:
                        media_saved_episodes += 1
                finally:
                    env.close()

        import math

        mean_success = sum(repeat_success_rates) / float(max(len(repeat_success_rates), 1))
        variance = (
            sum((rate - mean_success) ** 2 for rate in repeat_success_rates) / float(max(len(repeat_success_rates), 1))
        )
        std_success = math.sqrt(variance)
        mean_steps = sum(step_counts) / float(max(len(step_counts), 1))

        report["task_stats"][task_alias] = {
            "env_name": spec.env_name,
            "client_id": client_id,
            "episodes": int(len(repeat_success_rates)),
            "mean_success_rate": float(mean_success),
            "std_success_rate": float(std_success),
            "mean_steps_per_episode": float(mean_steps),
            "media_saved_episodes": int(media_saved_episodes),
            "media_saved_frames": int(media_saved_frames),
            "media_saved_gif_count": int(len(media_saved_gifs)),
            "media_sample_gifs": list(media_sample_gifs[:3]),
        }
        print(f"[rollout][hpt] task={task_alias} done mean_success={float(mean_success) * 100.0:.2f}%")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Meta-World rollouts from HPT Phase B checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", default="door-lock,close-drawer,sweep-into,open-window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-steps-per-episode", type=int, default=250)
    parser.add_argument("--render-width", type=int, default=128)
    parser.add_argument("--render-height", type=int, default=128)
    parser.add_argument("--camera-views", default="corner,top,shoulder")
    parser.add_argument("--torch-device", default="auto")
    parser.add_argument("--media-root", default="")
    parser.add_argument("--media-max-episodes", type=int, default=1)
    parser.add_argument("--media-max-frames", type=int, default=120)
    parser.add_argument("--media-fps", type=int, default=10)
    parser.add_argument("--media-save-frames", action="store_true")
    parser.add_argument("--media-save-gif", action="store_true")
    parser.add_argument("--report-json", default="runs/phase_b/phaseb_metaworld_rollout_hpt.json")
    parser.add_argument("--report-markdown", default="project/phaseb_metaworld_rollout_hpt.md")
    args = parser.parse_args()

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    views = [v.strip() for v in str(args.camera_views).split(",") if v.strip()]

    report = evaluate_metaworld_rollouts_hpt(
        checkpoint_path=str(args.checkpoint),
        tasks=tasks,
        seed=int(args.seed),
        trials=int(args.trials),
        repeats=int(args.repeats),
        max_steps_per_episode=int(args.max_steps_per_episode),
        render_width=int(args.render_width),
        render_height=int(args.render_height),
        camera_views=views,
        torch_device=str(args.torch_device),
        media_root=str(args.media_root),
        media_max_episodes=int(args.media_max_episodes),
        media_max_frames=int(args.media_max_frames),
        media_fps=int(args.media_fps),
        media_save_frames=bool(args.media_save_frames),
        media_save_gif=bool(args.media_save_gif),
    )

    report_json_path = Path(args.report_json)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_markdown_path = Path(args.report_markdown)
    report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    markdown_lines = [
        "# Phase B Meta-World Rollout Report (HPT)",
        "",
        f"- Time (UTC): `{report['time_utc']}`",
        f"- Checkpoint: `{report['checkpoint_path']}`",
        f"- Tasks: `{', '.join(report['tasks'])}`",
        f"- Eval seed: `{report['seed']}`",
        f"- Trials × repeats: `{report['trials']} × {report['repeats']}`",
        f"- Max steps/episode: `{report['max_steps_per_episode']}`",
        "",
        "| Task | Env | Client | Mean Success | Std | Episodes | Mean Steps | GIF Count | Sample GIF |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    task_stats = report["task_stats"]  # type: ignore[index]
    for task in sorted(task_stats.keys()):
        row = task_stats[task]
        sample_gifs = row.get("media_sample_gifs", []) if isinstance(row, dict) else []
        sample_gif = sample_gifs[0] if isinstance(sample_gifs, list) and sample_gifs else ""
        markdown_lines.append(
            "| "
            + f"{task} | "
            + f"{row['env_name']} | "
            + f"{row['client_id']} | "
            + f"{row['mean_success_rate'] * 100.0:.2f}% | "
            + f"{row['std_success_rate'] * 100.0:.2f}% | "
            + f"{row['episodes']} | "
            + f"{row['mean_steps_per_episode']:.2f} | "
            + f"{row.get('media_saved_gif_count', 0)} | "
            + f"{sample_gif} |"
        )
    markdown_lines.append("")
    report_markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    report["report_json"] = str(report_json_path)
    report["report_markdown"] = str(report_markdown_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
