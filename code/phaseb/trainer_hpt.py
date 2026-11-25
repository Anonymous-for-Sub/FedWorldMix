"""Phase B trainer for the paper-aligned HPT backbone path.

This module is intentionally scoped to:
- 3-view RGB (corner/top/shoulder) datasets collected into rgb npz files
- HPT trunk pretrained weights (HF: `liruiw/hpt-<size>`)
- A lightweight regression head for single-step BC

FedVLA-specific DGMoE + EDA (expert-driven aggregation) is wired separately; this
trainer currently focuses on getting the HPT backbone + RGB data loop running.
"""

from __future__ import annotations

import json
import os
import random
import signal
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import RunConfig
from .data import build_dataset_layout, build_task_clients, missing_paths
from .deps import require_torch
from .eda import compute_layer_weights
from .hpt.configuration_hpt import HPTConfig
from .hpt.modeling_hpt import HPTPolicy
from .hpt_data import IOSPPermNPZCache, PhaseBHPTDataset, RGBNPZCache, TrajectoryBatchSampler


class ExpertStatsAccumulator:
    """Accumulate DGMoE expert activation counts from the HPT trunk during training."""

    def __init__(self) -> None:
        self.counts = None
        self.token_count = 0

    def update_from_policy(self, policy: HPTPolicy) -> None:
        require_torch("HPT expert stats accumulation")
        import torch

        trunk = getattr(getattr(policy, "model", None), "trunk", None)
        if trunk is None:
            return
        transformer = None
        if isinstance(trunk, dict):
            transformer = trunk.get("trunk", None)
        else:
            try:
                transformer = trunk["trunk"]
            except Exception:
                transformer = None
        if transformer is None:
            return
        latest = getattr(transformer, "latest_expert_counts", None)
        if latest is None:
            return
        token_count = int(getattr(transformer, "latest_token_count", 0) or 0)
        latest = latest.detach().to(device="cpu", dtype=torch.float32)
        if self.counts is None:
            self.counts = latest.clone()
        else:
            self.counts = self.counts + latest
        self.token_count += token_count

    def export_matrix(self) -> list[list[float]]:
        if self.counts is None:
            return []
        return self.counts.tolist()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = Path(str(path) + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _maybe_fix_ssl_cert_env() -> None:
    """Work around relocated Python envs with broken OpenSSL default CA paths."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import ssl

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


def _resolve_torch_device(requested: str) -> "object":
    require_torch("HPT trainer device resolution")
    import torch

    raw = (requested or "auto").strip()
    normalized = raw.lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Torch backend requested CUDA device, but CUDA is not available. "
                "Use --torch-device cpu or install CUDA-enabled torch."
            )
        device = torch.device(raw)
        device_index = 0 if device.index is None else int(device.index)
        device_count = int(torch.cuda.device_count())
        if device_index < 0 or device_index >= device_count:
            raise RuntimeError(
                f"Requested CUDA device index {device_index} is out of range "
                f"(available count: {device_count})."
            )
        return torch.device(f"cuda:{device_index}")
    raise RuntimeError(f"Unsupported torch device '{requested}'. Use one of: auto, cpu, cuda, cuda:<index>.")


def _load_hpt_trunk_spec(model_size: str) -> dict[str, object]:
    """Load trunk spec (embed_dim/blocks/heads) from the HF `config.json` when present."""
    candidate_dir = Path(str(model_size))
    if candidate_dir.exists() and candidate_dir.is_dir():
        config_path = candidate_dir / "config.json"
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            return {
                "embed_dim": int(payload.get("embed_dim", 0)),
                "num_blocks": int(payload.get("num_blocks", 0)),
                "num_heads": int(payload.get("num_heads", 0)),
                "drop_path": float(payload.get("drop_path", 0.0)),
                "weight_init_style": str(payload.get("weight_init_style", "pytorch")),
            }
    local_override = os.environ.get("HPT_LOCAL_DIR", "").strip()
    if local_override:
        config_path = Path(local_override) / "config.json"
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            return {
                "embed_dim": int(payload.get("embed_dim", 0)),
                "num_blocks": int(payload.get("num_blocks", 0)),
                "num_heads": int(payload.get("num_heads", 0)),
                "drop_path": float(payload.get("drop_path", 0.0)),
                "weight_init_style": str(payload.get("weight_init_style", "pytorch")),
            }

    from huggingface_hub import hf_hub_download

    repo_id = f"liruiw/hpt-{model_size}"
    try:
        config_path = hf_hub_download(repo_id, "config.json")
    except Exception as exc:  # pragma: no cover - depends on HF availability
        raise RuntimeError(
            f"Unable to fetch {repo_id}/config.json. "
            "Phase B currently supports HPT sizes with config.json (e.g., base, large)."
        ) from exc
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return {
        "embed_dim": int(payload.get("embed_dim", 0)),
        "num_blocks": int(payload.get("num_blocks", 0)),
        "num_heads": int(payload.get("num_heads", 0)),
        "drop_path": float(payload.get("drop_path", 0.0)),
        "weight_init_style": str(payload.get("weight_init_style", "pytorch")),
    }


def build_hpt_config(
    config: RunConfig,
    *,
    action_dim: int,
    proprio_dim: int,
    image_hw: int = 128,
) -> HPTConfig:
    model_size = str(config.model.hpt_pretrained)
    trunk = _load_hpt_trunk_spec(model_size=model_size)
    embed_dim = int(trunk["embed_dim"])
    num_blocks = int(trunk["num_blocks"])
    num_heads = int(trunk["num_heads"])
    if embed_dim <= 0 or num_blocks <= 0 or num_heads <= 0:
        raise RuntimeError(f"Invalid trunk spec from HPT config.json: {trunk}")

    views = list(config.model.hpt_camera_views)
    if not views:
        raise ValueError("config.model.hpt_camera_views must be non-empty for HPT training.")

    input_shapes = {f"observation.images.{view}": [3, int(image_hw), int(image_hw)] for view in views}
    modalities: tuple = ("image", "state") if bool(config.model.hpt_use_state) else ("image",)
    if bool(config.model.hpt_use_state):
        input_shapes["observation.state"] = [int(proprio_dim)]
    text_embed_model = str(getattr(config.model, "hpt_text_embed_model", "") or "").strip()
    text_embed_dim = int(getattr(config.model, "hpt_text_embed_dim", 0) or 0)
    text_embed_mode = str(getattr(config.model, "hpt_text_embed_mode", "state") or "state").strip().lower()
    if text_embed_model and text_embed_dim > 0 and text_embed_mode == "language":
        input_shapes["observation.language"] = [int(text_embed_dim)]
        if "language" not in modalities:
            modalities = tuple(list(modalities) + ["language"])
    iosp_mode = str(getattr(config.model, "hpt_iosp_mode", "proxy") or "proxy").strip().lower()
    iosp_embed_model = str(getattr(config.model, "hpt_iosp_embed_model", "") or "").strip()
    iosp_embed_dim = int(getattr(config.model, "hpt_iosp_embed_dim", 0) or 0)
    if not iosp_embed_model:
        iosp_embed_model = text_embed_model
    if iosp_embed_dim <= 0:
        iosp_embed_dim = 512 if str(iosp_mode) == "clip" else int(text_embed_dim)
    iosp_enabled = bool(config.train.use_iosp) and bool(iosp_embed_model) and int(iosp_embed_dim) > 0
    if iosp_enabled:
        input_shapes["observation.iosp"] = [int(3 * int(iosp_embed_dim))]
    iosp_clip_model = str(getattr(config.model, "hpt_iosp_clip_model", "") or "").strip()
    if not iosp_clip_model:
        iosp_clip_model = iosp_embed_model or text_embed_model
    iosp_clip_image_size = int(getattr(config.model, "hpt_iosp_clip_image_size", 224) or 224)
    iosp_clip_strategy = str(getattr(config.model, "hpt_iosp_clip_strategy", "direct") or "direct").strip().lower()
    iosp_clip_remaining_max = int(getattr(config.model, "hpt_iosp_clip_remaining_max", 0) or 0)
    iosp_group_moe_enabled = bool(getattr(config.model, "hpt_iosp_group_moe", False))

    # Disable normalization by default to avoid dataset_stats requirements.
    return HPTConfig(
        input_shapes=input_shapes,
        output_shapes={"action": [int(action_dim)]},
        input_normalization_modes={},
        output_normalization_modes={},
        domain_name="metaworld",
        head_architecture=str(config.model.hpt_head_architecture),
        head_tanh_end=True,
        action_chunk_size=1,
        n_action_steps=1,
        n_obs_steps=1,
        embed_dim=int(embed_dim),
        num_blocks=int(num_blocks),
        num_heads=int(num_heads),
        token_postprocessing="mean",
        weight_init_style=str(trunk["weight_init_style"]),
        drop_path=float(trunk["drop_path"]),
        no_trunk=False,
        load_pretrained=str(model_size),
        modalities=modalities,
        state_input_dim=int(proprio_dim),
        language_input_dim=int(text_embed_dim) if ("observation.language" in input_shapes) else 768,
        head_action_dim=int(action_dim),
        head_input_dim=int(embed_dim),
        dgmoe_enabled=bool(config.model.hpt_use_dgmoe),
        dgmoe_impl=str(getattr(config.model, "hpt_dgmoe_impl", "adapter") or "adapter").strip().lower(),
        dgmoe_num_experts=int(config.model.hpt_dgmoe_num_experts),
        dgmoe_expert_hidden_dim=int(config.model.hpt_dgmoe_expert_hidden_dim),
        dgmoe_lambda=float(config.model.hpt_dgmoe_lambda),
        dgmoe_temperature=float(config.model.hpt_dgmoe_temperature),
        dgmoe_renormalize_gates=bool(config.model.hpt_dgmoe_renormalize_gates),
        dgmoe_gate_residual=bool(config.model.hpt_dgmoe_gate_residual),
        dgmoe_force_active=bool(getattr(config.model, "hpt_dgmoe_force_active", True)),
        iosp_enabled=bool(iosp_enabled),
        iosp_mode=str(iosp_mode),
        iosp_top_k=int(getattr(config.model, "top_k_tokens_per_group", 8) or 8),
        iosp_embed_dim=int(iosp_embed_dim),
        iosp_clip_model_name=str(iosp_clip_model),
        iosp_clip_image_size=int(iosp_clip_image_size),
        iosp_clip_strategy=str(iosp_clip_strategy),
        iosp_clip_remaining_max=int(iosp_clip_remaining_max),
        iosp_group_moe_enabled=bool(iosp_group_moe_enabled),
        iosp_group_moe_num_experts=int(getattr(config.model, "hpt_iosp_group_moe_num_experts", 4) or 4),
        iosp_group_moe_hidden_dim=int(getattr(config.model, "hpt_iosp_group_moe_hidden_dim", 128) or 128),
        iosp_group_moe_lambda=float(getattr(config.model, "hpt_iosp_group_moe_lambda", 0.5) or 0.5),
        iosp_group_moe_temperature=float(getattr(config.model, "hpt_iosp_group_moe_temperature", 1.0) or 1.0),
        iosp_group_moe_renormalize_gates=bool(getattr(config.model, "hpt_iosp_group_moe_renormalize_gates", False)),
        iosp_group_moe_gate_residual=bool(getattr(config.model, "hpt_iosp_group_moe_gate_residual", True)),
    )


def _set_requires_grad(module: "object", requires_grad: bool) -> None:
    for param in getattr(module, "parameters")():
        param.requires_grad = bool(requires_grad)


def _configure_hpt_trainable(policy: HPTPolicy, config: RunConfig) -> None:
    """Apply paper-style freezing rules for HPT + optional DGMoE."""
    # Encoders (vision stem) freezing.
    if bool(config.model.hpt_freeze_encoders) and hasattr(policy.model, "encoders"):
        _set_requires_grad(policy.model.encoders, False)

    if not hasattr(policy.model, "trunk"):
        return

    if bool(config.model.hpt_use_dgmoe):
        # Paper intent: keep attention/FFN frozen; train DGMoE experts + gates.
        _set_requires_grad(policy.model.trunk, False)

        trunk = policy.model.trunk
        transformer = trunk.get("trunk", None) if hasattr(trunk, "get") else None
        blocks = getattr(transformer, "blocks", None) if transformer is not None else None
        if blocks is None:
            return
        for blk in blocks:
            if not bool(config.model.hpt_freeze_trunk_attention) and hasattr(blk, "attn"):
                _set_requires_grad(blk.attn, True)
            if not bool(config.model.hpt_freeze_trunk_mlp) and hasattr(blk, "mlp"):
                _set_requires_grad(blk.mlp, True)
            dgmoe = getattr(blk, "dgmoe", None)
            if dgmoe is not None:
                _set_requires_grad(dgmoe, True)
        return

    # Default path: freeze full trunk if requested.
    if bool(config.model.hpt_freeze_trunk):
        _set_requires_grad(policy.model.trunk, False)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _move_batch_to_device(batch: Dict[str, object], device: "object") -> Dict[str, object]:
    require_torch("HPT batch move")
    import torch

    out: Dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                out[key] = value.to(device=device, dtype=torch.float32)
            else:
                out[key] = value.to(device=device)
        else:
            out[key] = value
    return out


def _trainable_parameters(policy: HPTPolicy) -> list[tuple[str, "object"]]:
    return [(name, param) for name, param in policy.named_parameters() if bool(getattr(param, "requires_grad", False))]


def _all_finite_tensors(tensors: list["object"]) -> bool:
    require_torch("HPT finite tensor checks")
    import torch

    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        if not bool(torch.isfinite(tensor).all().item()):
            return False
    return True


def _optimizer_step_with_guards(
    policy: HPTPolicy,
    *,
    grad_clip_norm: float,
    fail_on_nonfinite: bool,
    context: str,
) -> None:
    require_torch("HPT guarded optimizer step")
    import torch

    named_params = _trainable_parameters(policy)
    grads = [param.grad for _, param in named_params if getattr(param, "grad", None) is not None]
    if not _all_finite_tensors(grads):
        policy.zero_grad(set_to_none=True)
        if fail_on_nonfinite:
            raise FloatingPointError(f"{context}: non-finite gradients")
        return
    if float(grad_clip_norm) > 0.0 and grads:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [param for _, param in named_params],
            max_norm=float(grad_clip_norm),
            error_if_nonfinite=False,
        )
        grad_norm_value = float(grad_norm.detach().cpu().item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        if not (grad_norm_value == grad_norm_value) or grad_norm_value == float("inf"):
            policy.zero_grad(set_to_none=True)
            if fail_on_nonfinite:
                raise FloatingPointError(f"{context}: invalid gradient norm {grad_norm_value}")
            return
    policy.optimizer.step()  # type: ignore[attr-defined]
    params = [param for _, param in named_params]
    if not _all_finite_tensors(params):
        policy.zero_grad(set_to_none=True)
        if fail_on_nonfinite:
            raise FloatingPointError(f"{context}: non-finite parameters after optimizer step")
        return


def _train_epoch(
    policy: HPTPolicy,
    loader: "object",
    device: "object",
    *,
    expert_stats: ExpertStatsAccumulator | None = None,
    grad_clip_norm: float = 0.0,
    fail_on_nonfinite: bool = False,
) -> float:
    require_torch("HPT train epoch")
    import torch

    policy.train()
    losses: List[float] = []
    for batch_index, batch in enumerate(loader):
        policy.zero_grad(set_to_none=True)
        batch = _move_batch_to_device(batch, device=device)
        out = policy(batch)
        loss = out.get("loss", None)
        if loss is None:
            raise RuntimeError("HPTPolicy forward did not return a loss.")
        if not bool(torch.isfinite(loss).all().item()):
            policy.zero_grad(set_to_none=True)
            if fail_on_nonfinite:
                raise FloatingPointError(f"batch {int(batch_index)}: non-finite loss")
            continue
        if expert_stats is not None:
            expert_stats.update_from_policy(policy)
        loss.backward()
        _optimizer_step_with_guards(
            policy,
            grad_clip_norm=float(grad_clip_norm),
            fail_on_nonfinite=bool(fail_on_nonfinite),
            context=f"batch {int(batch_index)}",
        )
        losses.append(float(loss.detach().cpu().item()))
    return _mean(losses)


def _train_epoch_multitask(
    policy: HPTPolicy,
    loaders: Dict[str, "object"],
    device: "object",
    *,
    seed: int,
    grad_clip_norm: float = 0.0,
    fail_on_nonfinite: bool = False,
) -> float:
    require_torch("HPT train epoch multitask")
    import torch

    order = list(loaders.keys())
    random.Random(int(seed)).shuffle(order)
    losses: List[float] = []
    for task in order:
        policy.train()
        for batch_index, batch in enumerate(loaders[task]):
            policy.zero_grad(set_to_none=True)
            batch = _move_batch_to_device(batch, device=device)
            out = policy(batch)
            loss = out.get("loss", None)
            if loss is None:
                raise RuntimeError("HPTPolicy forward did not return a loss.")
            if not bool(torch.isfinite(loss).all().item()):
                policy.zero_grad(set_to_none=True)
                if fail_on_nonfinite:
                    raise FloatingPointError(f"task {task} batch {int(batch_index)}: non-finite loss")
                continue
            loss.backward()
            _optimizer_step_with_guards(
                policy,
                grad_clip_norm=float(grad_clip_norm),
                fail_on_nonfinite=bool(fail_on_nonfinite),
                context=f"task {task} batch {int(batch_index)}",
            )
            losses.append(float(loss.detach().cpu().item()))
    return _mean(losses)


def _eval_epoch(policy: HPTPolicy, loader: "object", device: "object") -> float:
    require_torch("HPT eval epoch")
    import torch

    policy.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device=device)
            out = policy(batch)
            loss = out.get("loss", None)
            if loss is None:
                raise RuntimeError("HPTPolicy forward did not return a loss.")
            losses.append(float(loss.detach().cpu().item()))
    return _mean(losses)


def _eval_epoch_multitask(policy: HPTPolicy, loaders: Dict[str, "object"], device: "object") -> float:
    require_torch("HPT eval epoch")
    import torch

    losses: List[float] = []
    policy.eval()
    with torch.no_grad():
        for task in sorted(loaders.keys()):
            for batch in loaders[task]:
                batch = _move_batch_to_device(batch, device=device)
                out = policy(batch)
                loss = out.get("loss", None)
                if loss is None:
                    raise RuntimeError("HPTPolicy forward did not return a loss.")
                losses.append(float(loss.detach().cpu().item()))
    return _mean(losses)


def _average_state_dict(state_dicts: List[Dict[str, "object"]]) -> Dict[str, "object"]:
    require_torch("HPT FedAvg")
    import torch

    if not state_dicts:
        return {}
    keys = list(state_dicts[0].keys())
    out: Dict[str, object] = {}
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            continue
        stacked = torch.stack([t.to(dtype=torch.float32) for t in tensors], dim=0)
        out[key] = stacked.mean(dim=0)
    return out


def _blend_state_dict(
    local_state: Dict[str, "object"],
    global_state: Dict[str, "object"],
    *,
    alpha: float,
) -> Dict[str, "object"]:
    """Blend local parameters with global parameters: alpha*local + (1-alpha)*global."""
    require_torch("HPT trunk blending")
    import torch

    a = float(alpha)
    if a <= 0.0:
        return dict(global_state)
    if a >= 1.0:
        return dict(local_state)
    merged: Dict[str, object] = dict(local_state)
    for key, global_val in global_state.items():
        local_val = local_state.get(key, None)
        if isinstance(global_val, torch.Tensor) and isinstance(local_val, torch.Tensor):
            merged[key] = a * local_val.to(dtype=torch.float32) + (1.0 - a) * global_val.to(dtype=torch.float32)
        else:
            merged[key] = global_val
    return merged


def _parse_block_index_from_key(key: str) -> int | None:
    marker = ".blocks."
    idx = key.find(marker)
    if idx < 0:
        return None
    rest = key[idx + len(marker) :]
    piece = rest.split(".", 1)[0]
    if not piece.isdigit():
        return None
    return int(piece)


def _eda_aggregate_state_dict(
    client_to_state: Dict[str, Dict[str, "object"]],
    *,
    client_to_expert_stats: Dict[str, list[list[float]]],
    average_state: Dict[str, "object"],
    eda_debug: bool,
) -> tuple[Dict[str, "object"], Dict[str, list[float]]]:
    """Apply EDA aggregation to DGMoE parameters, falling back to FedAvg elsewhere."""
    require_torch("HPT EDA aggregation")
    import torch

    if not client_to_expert_stats:
        return dict(average_state), {}
    matrices = list(client_to_expert_stats.values())
    if not matrices or any((not mat) for mat in matrices):
        return dict(average_state), {}
    num_layers = len(matrices[0])
    if any(len(mat) != num_layers for mat in matrices):
        return dict(average_state), {}

    weights = compute_layer_weights(client_to_expert_stats)
    if not weights:
        return dict(average_state), {}

    any_weights = next(iter(weights.values()), None)
    if not isinstance(any_weights, list) or len(any_weights) != num_layers:
        return dict(average_state), {}
    out: Dict[str, object] = dict(average_state)
    for key, value in list(out.items()):
        if ".dgmoe." not in str(key):
            continue
        layer_index = _parse_block_index_from_key(str(key))
        if layer_index is None or layer_index < 0 or layer_index >= num_layers:
            continue
        acc = None
        for cid, state in client_to_state.items():
            coeffs = weights.get(cid, None)
            if coeffs is None or layer_index >= len(coeffs):
                continue
            coeff = float(coeffs[layer_index])
            tensor = state.get(key, None)
            if not isinstance(tensor, torch.Tensor):
                continue
            if acc is None:
                acc = tensor.to(dtype=torch.float32) * coeff
            else:
                acc = acc + tensor.to(dtype=torch.float32) * coeff
        if acc is not None:
            out[key] = acc

    if bool(eda_debug):
        # Keep weights as plain lists for JSON logs.
        return out, {cid: list(ws) for cid, ws in weights.items()}
    return out, {}


def build_hpt_summary_markdown(report: Dict[str, object]) -> str:
    lines = [
        "# Phase B HPT Summary",
        "",
        f"- Time (UTC): `{report.get('time_utc', '')}`",
        f"- Backbone: `{report.get('backbone', '')}`",
        f"- Experiment mode: `{report.get('experiment_mode', '')}`",
        f"- HPT pretrained: `{report.get('hpt_pretrained', '')}`",
        f"- Dataset root: `{report.get('dataset_root', '')}`",
        f"- Checkpoint: `{report.get('checkpoint_path', '')}`",
        "",
    ]
    history = report.get("history", [])
    if isinstance(history, list) and history:
        tail = history[-1]
        if isinstance(tail, dict):
            lines.append("## Last Round")
            for key, value in tail.items():
                if key == "round":
                    continue
                lines.append(f"- {key}: `{value}`")
            lines.append("")
    return "\n".join(lines)


def run_phaseb_hpt(config: RunConfig, dataset_root: str) -> Dict[str, object]:
    require_torch("Phase B HPT trainer")
    import torch

    _maybe_fix_ssl_cert_env()
    output_file = str(getattr(config, "output_file", "") or "").strip()
    progress_every_rounds = int(getattr(getattr(config, "train", object()), "progress_every_rounds", 0) or 0)
    checkpoint_every_rounds = int(getattr(getattr(config, "train", object()), "checkpoint_every_rounds", 0) or 0)

    stop_requested: Dict[str, str] = {"signal": ""}
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)

    def _handle_stop(signum: int, frame: object) -> None:  # noqa: ARG001
        stop_requested["signal"] = str(signum)
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _handle_stop)
        signal.signal(signal.SIGINT, _handle_stop)
    except Exception:
        prev_sigterm = None
        prev_sigint = None

    device = _resolve_torch_device(config.train.torch_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    random.seed(int(config.train.seed))
    torch.manual_seed(int(config.train.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.train.seed))

    client_to_task = build_task_clients(
        tasks=config.task_split.tasks,
        clients_per_task=config.task_split.clients_per_task,
    )
    layout = build_dataset_layout(dataset_root, client_to_task)
    missing = missing_paths(layout)
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing[:3]} (and {max(len(missing) - 3, 0)} more).")

    # Infer action/proprio dims from the dataset (first non-empty record).
    observed_action_dim = 0
    observed_proprio_dim = 0
    for task in config.task_split.tasks:
        train_path = Path(dataset_root) / task / "train.jsonl"
        if not train_path.exists():
            continue
        for line in train_path.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            action = payload.get("action", [])
            proprio = payload.get("proprio", [])
            if isinstance(action, list):
                observed_action_dim = max(observed_action_dim, len([v for v in action if isinstance(v, (int, float))]))
            if isinstance(proprio, list):
                observed_proprio_dim = max(observed_proprio_dim, len([v for v in proprio if isinstance(v, (int, float))]))
            break
        if observed_action_dim and observed_proprio_dim:
            break

    action_dim = int(config.model.action_dim) if int(config.model.action_dim) > 0 else int(observed_action_dim)
    if action_dim <= 0:
        raise RuntimeError("Unable to resolve action_dim from config or dataset.")
    proprio_dim = int(observed_proprio_dim) if observed_proprio_dim > 0 else 39
    task_embed_dim = int(getattr(config.model, "hpt_task_embed_dim", 0) or 0)
    task_embed_scale = float(getattr(config.model, "hpt_task_embed_scale", 1.0) or 1.0)
    text_embed_model = str(getattr(config.model, "hpt_text_embed_model", "") or "").strip()
    text_embed_dim = int(getattr(config.model, "hpt_text_embed_dim", 0) or 0)
    text_embed_scale = float(getattr(config.model, "hpt_text_embed_scale", 1.0) or 1.0)
    text_embed_mode = str(getattr(config.model, "hpt_text_embed_mode", "state") or "state").strip().lower()
    iosp_mode = str(getattr(config.model, "hpt_iosp_mode", "proxy") or "proxy").strip().lower()
    iosp_embed_model = str(getattr(config.model, "hpt_iosp_embed_model", "") or "").strip()
    iosp_embed_dim = int(getattr(config.model, "hpt_iosp_embed_dim", 0) or 0)
    iosp_embed_scale = float(getattr(config.model, "hpt_iosp_embed_scale", 1.0) or 1.0)
    if not iosp_embed_model:
        iosp_embed_model = text_embed_model

    task_text_embeddings: Dict[str, List[float]] = {}
    if text_embed_model and text_embed_dim > 0:
        from .metaworld_collect import TASK_SPECS
        from .text_embed import build_task_instruction_embeddings

        task_to_instruction: Dict[str, str] = {}
        for task in config.task_split.tasks:
            spec = TASK_SPECS.get(str(task), None)
            instruction = spec.instruction if spec is not None else str(task)
            task_to_instruction[str(task)] = str(instruction)
        task_text_embeddings = build_task_instruction_embeddings(
            task_to_instruction=task_to_instruction,
            model_name=text_embed_model,
            out_dim=int(text_embed_dim),
            device="cpu",
            normalize=True,
        )

    task_iosp_embeddings: Dict[str, List[float]] = {}
    if bool(config.train.use_iosp) and iosp_embed_model and int(iosp_embed_dim) > 0:
        from .metaworld_collect import TASK_SPECS
        if str(iosp_mode) == "clip":
            from .text_embed import build_task_object_group_embeddings_clip as build_task_object_group_embeddings
            iosp_out_dim = 512
        else:
            from .text_embed import build_task_object_group_embeddings
            iosp_out_dim = int(iosp_embed_dim)

        task_to_groups: Dict[str, Dict[str, str]] = {}
        for task in config.task_split.tasks:
            spec = TASK_SPECS.get(str(task), None)
            if spec is None:
                task_to_groups[str(task)] = {"to": str(task), "so": "", "bo": ""}
                continue

            instruction = str(spec.instruction or "").lower()
            foreground = [str(obj) for obj in (spec.foreground_objects or []) if str(obj)]
            background = [str(obj) for obj in (spec.background_objects or []) if str(obj)]

            target_objects = [obj for obj in foreground if obj.lower() in instruction]
            if not target_objects and foreground:
                target_objects = [foreground[0]]
            surrounding_objects = [obj for obj in foreground if obj not in target_objects]
            task_to_groups[str(task)] = {
                "to": " ".join(target_objects).strip(),
                "so": " ".join(surrounding_objects).strip(),
                "bo": " ".join(background).strip(),
            }

        task_iosp_embeddings = build_task_object_group_embeddings(
            task_to_groups=task_to_groups,
            model_name=iosp_embed_model,
            out_dim=int(iosp_out_dim),
            device="cpu",
            normalize=True,
        )

    hpt_cfg = build_hpt_config(
        config,
        action_dim=action_dim,
        proprio_dim=proprio_dim + max(task_embed_dim, 0) + (max(text_embed_dim, 0) if text_embed_mode != "language" else 0),
        image_hw=128,
    )

    cache = RGBNPZCache(max_items=32)
    perm_cache = IOSPPermNPZCache(max_items=64)
    iosp_perm_suffix = str(getattr(config.model, "hpt_iosp_perm_suffix", "") or "").strip()

    def make_loader(task: str, split: str, *, shuffle: bool) -> "object":
        text_vec = task_text_embeddings.get(str(task), [])
        iosp_vec = task_iosp_embeddings.get(str(task), [])
        if int(text_embed_dim) > 0:
            if len(text_vec) < int(text_embed_dim):
                text_vec = list(text_vec) + [0.0] * (int(text_embed_dim) - len(text_vec))
            elif len(text_vec) > int(text_embed_dim):
                text_vec = list(text_vec)[: int(text_embed_dim)]
        resolved_iosp_dim = int(iosp_embed_dim)
        if str(iosp_mode) == "clip":
            resolved_iosp_dim = 512
        if resolved_iosp_dim > 0:
            expected_iosp = int(resolved_iosp_dim) * 3
            if len(iosp_vec) < expected_iosp:
                iosp_vec = list(iosp_vec) + [0.0] * (expected_iosp - len(iosp_vec))
            elif len(iosp_vec) > expected_iosp:
                iosp_vec = list(iosp_vec)[:expected_iosp]
        dataset = PhaseBHPTDataset(
            dataset_root=dataset_root,
            task=task,
            split=split,
            camera_views=list(config.model.hpt_camera_views),
            max_samples=int(config.train.max_samples_per_client),
            cache=cache,
            perm_cache=perm_cache,
            task_embed_dim=task_embed_dim,
            task_embed_scale=task_embed_scale,
            text_embed=[float(v) * float(text_embed_scale) for v in text_vec],
            text_embed_mode=text_embed_mode,
            iosp_embed=[float(v) * float(iosp_embed_scale) for v in iosp_vec] if bool(config.train.use_iosp) else None,
            iosp_perm_suffix=iosp_perm_suffix,
        )
        if len(dataset) <= 0:
            raise RuntimeError(f"Empty HPT dataset for task={task} split={split}.")
        batch_sampler = TrajectoryBatchSampler(
            dataset.indices_by_rgb_npz_path,
            batch_size=int(config.train.batch_size),
            shuffle=bool(shuffle),
            seed=int(config.train.seed),
        )
        # Avoid CUDA + DataLoader multiprocessing interactions (forked workers can retain CUDA context
        # and cause severe GPU memory bloat). HPT runs are typically GPU-bound anyway.
        num_workers = 0
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )

    experiment_mode = str(config.train.experiment_mode)
    rounds = int(config.train.rounds)
    local_epochs = int(config.train.local_epochs)

    history: List[Dict[str, object]] = []
    checkpoint_path = ""
    interrupted = False
    interrupted_signal: str = ""
    last_completed_round = 0
    client_state_dicts: Dict[str, Dict[str, object]] = {}

    def _atomic_torch_save(payload: Dict[str, object], path: str) -> None:
        checkpoint_target = Path(path)
        checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(str(checkpoint_target) + ".tmp")
        torch.save(payload, str(tmp_path))
        tmp_path.replace(checkpoint_target)

    def _build_checkpoint_payload() -> Dict[str, object]:
        payload: Dict[str, object] = {
            "time_utc": _utc_now(),
            "backbone": "hpt",
            "experiment_mode": experiment_mode,
            "config": asdict(config),
            "hpt_config": asdict(hpt_cfg),
            "client_to_task": dict(client_to_task),
            "client_state_dicts": client_state_dicts,
        }
        if task_text_embeddings:
            payload["task_text_embeddings"] = dict(task_text_embeddings)
        if task_iosp_embeddings:
            payload["task_iosp_embeddings"] = dict(task_iosp_embeddings)
        return payload

    def _emit_progress() -> None:
        if not output_file:
            return
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary_path = out.with_suffix(".summary.md")
        report: Dict[str, object] = {
            "time_utc": _utc_now(),
            "backbone": "hpt",
            "experiment_mode": experiment_mode,
            "dataset_root": dataset_root,
            "missing_dataset_paths": missing,
            "client_to_task": dict(client_to_task),
            "hpt_pretrained": str(config.model.hpt_pretrained),
            "history": history,
            "checkpoint_path": checkpoint_path,
            "interrupted": bool(interrupted),
            "last_completed_round": int(last_completed_round),
            "stop_signal": str(interrupted_signal),
        }
        _atomic_write_json(out, report)
        _atomic_write_text(summary_path, build_hpt_summary_markdown(report))

    def _maybe_progress(round_index: int, *, force: bool = False) -> None:
        every = int(progress_every_rounds)
        if not force and (every <= 0 or round_index <= 0 or (round_index % every) != 0):
            return
        _emit_progress()

    def _maybe_checkpoint(round_index: int, *, force: bool = False) -> None:
        nonlocal checkpoint_path
        if not bool(getattr(config, "save_checkpoint", False)):
            return
        if not client_state_dicts:
            return
        every = int(checkpoint_every_rounds)
        if not force and (every <= 0 or round_index <= 0 or (round_index % every) != 0):
            return
        checkpoint_path = str(getattr(config, "checkpoint_output", "")).strip()
        if not checkpoint_path:
            checkpoint_path = str(Path(config.output_dir) / "phaseb_hpt_checkpoint.pt")
        _atomic_torch_save(_build_checkpoint_payload(), checkpoint_path)

    try:
        if experiment_mode == "centralized":
            policy = HPTPolicy(config=hpt_cfg)
            policy.to(device)
            _configure_hpt_trainable(policy, config)
            policy.optimizer = torch.optim.Adam(
                [p for p in policy.parameters() if p.requires_grad],
                lr=float(config.train.learning_rate),
            )
            train_loaders = {task: make_loader(task, "train", shuffle=True) for task in config.task_split.tasks}
            val_loaders = {task: make_loader(task, "val", shuffle=False) for task in config.task_split.tasks}
            for rnd in range(rounds):
                train_losses = []
                for _ in range(local_epochs):
                    train_losses.append(
                        _train_epoch_multitask(
                            policy,
                            train_loaders,
                            device=device,
                            seed=int(config.train.seed) + int(rnd) * 1009,
                            grad_clip_norm=float(getattr(config.train, "grad_clip_norm", 0.0) or 0.0),
                            fail_on_nonfinite=bool(getattr(config.train, "skip_nonfinite_updates", False)),
                        )
                    )
                val_loss = _eval_epoch_multitask(policy, val_loaders, device=device)
                if rounds >= 10 and ((rnd + 1) % 5 == 0 or rnd == 0 or rnd + 1 == rounds):
                    print(
                        f"[centralized][round {rnd + 1}/{rounds}] "
                        f"train_loss={float(_mean(train_losses)):.4f} val_loss={float(val_loss):.4f}"
                    )
                history.append(
                    {
                        "round": int(rnd + 1),
                        "train_loss": float(_mean(train_losses)),
                        "val_loss": float(val_loss),
                    }
                )
                last_completed_round = int(rnd + 1)
                _maybe_progress(last_completed_round)
            client_state_dicts = {cid: policy.state_dict() for cid in client_to_task.keys()}
            _maybe_checkpoint(last_completed_round, force=True)
        else:
            def _optimizer_state_to_cpu(state: dict) -> dict:
                out = dict(state)
                raw_state = out.get("state", {})
                if isinstance(raw_state, dict):
                    for slot in raw_state.values():
                        if not isinstance(slot, dict):
                            continue
                        for key, value in list(slot.items()):
                            if isinstance(value, torch.Tensor):
                                slot[key] = value.detach().cpu()
                return out

            def _optimizer_state_to_device(optimizer: "object", device: "object") -> None:
                raw_state = getattr(optimizer, "state", {})
                if not isinstance(raw_state, dict):
                    return
                for slot in raw_state.values():
                    if not isinstance(slot, dict):
                        continue
                    for key, value in list(slot.items()):
                        if isinstance(value, torch.Tensor):
                            slot[key] = value.to(device=device)

            # Bootstrap a single pretrained HPT state dict, then train clients sequentially to avoid
            # holding multiple HPT(+CLIP IOSP) replicas on GPU at once.
            bootstrap_policy = HPTPolicy(config=hpt_cfg)
            bootstrap_state = {k: v.detach().cpu() for k, v in bootstrap_policy.state_dict().items()}
            del bootstrap_policy
            if device.type == "cuda":
                torch.cuda.empty_cache()

            runtime_payload = asdict(hpt_cfg)
            runtime_payload["load_pretrained"] = ""
            if isinstance(runtime_payload.get("modalities", None), list):
                runtime_payload["modalities"] = tuple(runtime_payload["modalities"])
            runtime_hpt_cfg = HPTConfig(**runtime_payload)

            train_loaders: Dict[str, object] = {}
            val_loaders: Dict[str, object] = {}
            for client_id, task in client_to_task.items():
                train_loaders[client_id] = make_loader(task, "train", shuffle=True)
                val_loaders[client_id] = make_loader(task, "val", shuffle=False)

            client_state_dicts = {cid: dict(bootstrap_state) for cid in client_to_task.keys()}
            client_opt_state: Dict[str, dict | None] = {cid: None for cid in client_to_task.keys()}

            global_trunk: Dict[str, object] | None = None
            for rnd in range(rounds):
                trunk_states: List[Dict[str, object]] = []
                trunk_state_by_client: Dict[str, Dict[str, object]] = {}
                client_train_loss: Dict[str, float] = {}
                client_val_loss: Dict[str, float] = {}
                client_expert_stats: Dict[str, list[list[float]]] = {}

                for cid in client_to_task.keys():
                    policy = HPTPolicy(config=runtime_hpt_cfg)
                    policy.load_state_dict(client_state_dicts[cid])
                    policy.to(device)
                    _configure_hpt_trainable(policy, config)
                    policy.optimizer = torch.optim.Adam(
                        [p for p in policy.parameters() if p.requires_grad],
                        lr=float(config.train.learning_rate),
                    )
                    if client_opt_state[cid] is not None:
                        policy.optimizer.load_state_dict(client_opt_state[cid])  # type: ignore[arg-type]
                        _optimizer_state_to_device(policy.optimizer, device=device)

                    if global_trunk is not None:
                        alpha = float(getattr(config.train, "trunk_merge_alpha", 0.0) or 0.0)
                        if alpha > 0.0:
                            local_state = {k: v.detach().cpu() for k, v in policy.model.trunk.state_dict().items()}
                            blended = _blend_state_dict(local_state, global_trunk, alpha=alpha)
                            policy.model.trunk.load_state_dict(blended)  # type: ignore[arg-type]
                        else:
                            policy.model.trunk.load_state_dict(global_trunk)  # type: ignore[arg-type]

                    stats_acc = ExpertStatsAccumulator()
                    epoch_losses = []
                    local_update_ok = True
                    try:
                        for _ in range(local_epochs):
                            epoch_losses.append(
                                _train_epoch(
                                    policy,
                                    train_loaders[cid],
                                    device=device,
                                    expert_stats=stats_acc,
                                    grad_clip_norm=float(getattr(config.train, "grad_clip_norm", 0.0) or 0.0),
                                    fail_on_nonfinite=bool(getattr(config.train, "skip_nonfinite_updates", False)),
                                )
                            )
                    except FloatingPointError as exc:
                        local_update_ok = False
                        print(
                            f"[{experiment_mode}][round {rnd + 1}/{rounds}][{cid}] "
                            f"skipping unstable local update: {exc}"
                        )
                        policy.load_state_dict(client_state_dicts[cid])
                        policy.to(device)
                    client_train_loss[cid] = float(_mean(epoch_losses))
                    client_val_loss[cid] = float(_eval_epoch(policy, val_loaders[cid], device=device))
                    client_expert_stats[cid] = stats_acc.export_matrix()

                    trunk_state = {k: v.detach().cpu() for k, v in policy.model.trunk.state_dict().items()}
                    trunk_state_by_client[cid] = trunk_state
                    trunk_states.append(trunk_state)

                    client_state_dicts[cid] = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
                    if local_update_ok:
                        client_opt_state[cid] = _optimizer_state_to_cpu(policy.optimizer.state_dict())

                    del policy
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                avg_trunk = _average_state_dict(trunk_states)
                eda_layer_weights: Dict[str, list[float]] = {}
                if experiment_mode == "fedvla":
                    global_trunk, eda_layer_weights = _eda_aggregate_state_dict(
                        trunk_state_by_client,
                        client_to_expert_stats=client_expert_stats,
                        average_state=avg_trunk,
                        eda_debug=bool(config.train.eda_debug),
                    )
                else:
                    global_trunk = avg_trunk

                if rounds >= 10 and ((rnd + 1) % 5 == 0 or rnd == 0 or rnd + 1 == rounds):
                    mean_train = _mean(list(client_train_loss.values()))
                    mean_val = _mean(list(client_val_loss.values()))
                    print(
                        f"[{experiment_mode}][round {rnd + 1}/{rounds}] "
                        f"mean_train_loss={float(mean_train):.4f} mean_val_loss={float(mean_val):.4f}"
                    )

                row: Dict[str, object] = {
                    "round": int(rnd + 1),
                    "client_train_loss": dict(client_train_loss),
                    "client_val_loss": dict(client_val_loss),
                }
                if bool(config.train.eda_debug):
                    row["client_expert_stats"] = dict(client_expert_stats)
                    if eda_layer_weights:
                        row["eda_layer_weights"] = dict(eda_layer_weights)
                history.append(row)

                last_completed_round = int(rnd + 1)
                _maybe_checkpoint(last_completed_round)
                _maybe_progress(last_completed_round)

            _maybe_checkpoint(last_completed_round, force=True)
    except KeyboardInterrupt:
        interrupted = True
        interrupted_signal = str(stop_requested.get("signal", "") or "")
        if history:
            tail = history[-1]
            if isinstance(tail, dict) and isinstance(tail.get("round", None), int):
                last_completed_round = int(tail["round"])
            else:
                last_completed_round = int(len(history))
        _maybe_checkpoint(last_completed_round, force=True)
        _maybe_progress(last_completed_round, force=True)
    finally:
        try:
            if prev_sigterm is not None:
                signal.signal(signal.SIGTERM, prev_sigterm)  # type: ignore[arg-type]
            if prev_sigint is not None:
                signal.signal(signal.SIGINT, prev_sigint)  # type: ignore[arg-type]
        except Exception:
            pass

    if not interrupted:
        _maybe_progress(last_completed_round, force=True)

    return {
        "time_utc": _utc_now(),
        "backbone": "hpt",
        "experiment_mode": experiment_mode,
        "dataset_root": dataset_root,
        "missing_dataset_paths": missing,
        "client_to_task": dict(client_to_task),
        "hpt_pretrained": str(config.model.hpt_pretrained),
        "history": history,
        "checkpoint_path": checkpoint_path,
        "interrupted": bool(interrupted),
        "last_completed_round": int(last_completed_round),
        "stop_signal": str(interrupted_signal),
    }
