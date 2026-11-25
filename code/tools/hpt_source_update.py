"""Protected heterogeneous-source updates for the isolated HPT experiment.

This module intentionally uses recorded multi-camera RGB as its structured
source. It does not create, consume, or claim 3D Gaussian Splatting data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


IMAGE_PREFIX = "observation.images."


@dataclass(frozen=True)
class SourceUpdateConfig:
    method: str
    lambda_2d: float = 0.75
    lambda_structured: float = 0.25
    norm_cap: float = 0.50
    eps: float = 1.0e-12


def _image_keys(batch: Mapping[str, object]) -> List[str]:
    return sorted(key for key in batch if str(key).startswith(IMAGE_PREFIX))


def photometric_twin(batch: Mapping[str, object]) -> Dict[str, object]:
    """Apply a fixed, reproducible photometric shift to every camera stream."""
    import torch

    out = dict(batch)
    channel_scales = (1.10, 0.92, 1.04)
    for view_index, key in enumerate(_image_keys(batch)):
        image = batch[key]
        if not isinstance(image, torch.Tensor):
            continue
        scale = torch.tensor(channel_scales, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
        brightness = 0.92 + 0.04 * float(view_index)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        shifted = (image - mean) * 1.12 + mean
        out[key] = (shifted * scale * brightness + 0.025).clamp(0.0, 1.0)
    return out


def structured_multiview_twin(batch: Mapping[str, object], batch_index: int) -> Dict[str, object]:
    """Create a camera-availability twin from the recorded three-view sample.

    Exactly one camera stream is masked while all remaining streams retain
    their original camera identity, state, and action label.
    """
    import torch

    out = dict(batch)
    keys = _image_keys(batch)
    if not keys:
        return out
    masked_key = keys[int(batch_index) % len(keys)]
    image = batch[masked_key]
    if isinstance(image, torch.Tensor):
        out[masked_key] = torch.zeros_like(image)
    return out


def sample_validity(batch: Mapping[str, object]) -> Tuple[float, int, int]:
    """Return sample-level validity acceptance ratio and counts."""
    import torch

    action = batch.get("action")
    keys = _image_keys(batch)
    if not isinstance(action, torch.Tensor) or action.ndim == 0:
        return 0.0, 0, 0
    batch_size = int(action.shape[0])
    accepted = 0
    for sample_index in range(batch_size):
        valid = bool(torch.isfinite(action[sample_index]).all().item())
        for key in keys:
            image = batch[key]
            if not isinstance(image, torch.Tensor) or sample_index >= int(image.shape[0]):
                valid = False
                break
            item = image[sample_index]
            valid = valid and bool(torch.isfinite(item).all().item()) and int(item.numel()) > 0
        accepted += int(valid)
    return float(accepted) / float(max(batch_size, 1)), int(accepted), int(batch_size)


def is_correction_parameter(name: str) -> bool:
    """Restrict structured correction to visual/connector trainable parameters."""
    return any(
        token in name
        for token in (
            "model.encoders.image",
            "model.stems.metaworld_image",
            "model.trunk",
            "model.heads.metaworld",
            "model.iosp",
        )
    )


def _dot(left: Sequence[object], right: Sequence[object]) -> "object":
    import torch

    terms = [torch.sum(a * b) for a, b in zip(left, right) if a is not None and b is not None]
    if not terms:
        return torch.tensor(0.0)
    return torch.stack(terms).sum()


def _norm_sq(grads: Sequence[object]) -> "object":
    return _dot(grads, grads)


def compose_protected_gradients(
    names: Sequence[str],
    base_grads: Sequence[object],
    correction_grads: Sequence[object],
    *,
    correction_weight: float,
    quality: float,
    norm_cap: float,
    eps: float = 1.0e-12,
) -> Tuple[List[object], Dict[str, float | bool | int]]:
    """Project and cap the correction in the declared parameter subspace."""
    import torch

    if not (len(names) == len(base_grads) == len(correction_grads)):
        raise ValueError("Gradient/name lengths do not match.")

    selected = [is_correction_parameter(name) for name in names]
    base_selected = [grad if keep else None for grad, keep in zip(base_grads, selected)]
    corr_selected = [grad if keep else None for grad, keep in zip(correction_grads, selected)]
    dot_before = _dot(corr_selected, base_selected)
    base_norm_sq = _norm_sq(base_selected)
    corr_norm_sq = _norm_sq(corr_selected)
    base_norm = torch.sqrt(torch.clamp(base_norm_sq, min=0.0))
    corr_norm = torch.sqrt(torch.clamp(corr_norm_sq, min=0.0))

    projected = bool(float(dot_before.detach().cpu().item()) < 0.0)
    projection_coeff = dot_before / (base_norm_sq + float(eps)) if projected else dot_before * 0.0
    corrected: List[object] = []
    for base_grad, corr_grad, keep in zip(base_grads, correction_grads, selected):
        if not keep or corr_grad is None:
            corrected.append(None)
        elif projected and base_grad is not None:
            corrected.append(corr_grad - projection_coeff * base_grad)
        else:
            corrected.append(corr_grad)

    corrected_norm = torch.sqrt(torch.clamp(_norm_sq(corrected), min=0.0))
    cap_limit = float(norm_cap) * base_norm
    cap_scale_tensor = torch.minimum(
        torch.ones_like(corrected_norm),
        cap_limit / (corrected_norm + float(eps)),
    )
    cap_scale = float(cap_scale_tensor.detach().cpu().item())
    corrected = [grad * cap_scale_tensor if grad is not None else None for grad in corrected]
    dot_after = _dot(corrected, base_selected)

    combined: List[object] = []
    effective_weight = float(correction_weight) * float(quality)
    for base_grad, corr_grad in zip(base_grads, corrected):
        if base_grad is None and corr_grad is None:
            combined.append(None)
        elif base_grad is None:
            combined.append(corr_grad * effective_weight)
        elif corr_grad is None:
            combined.append(base_grad)
        else:
            combined.append(base_grad + corr_grad * effective_weight)

    denominator = float(base_norm.detach().cpu().item()) * float(corr_norm.detach().cpu().item()) + float(eps)
    return combined, {
        "correction_parameter_count": int(sum(selected)),
        "projection_applied": projected,
        "dot_before": float(dot_before.detach().cpu().item()),
        "dot_after": float(dot_after.detach().cpu().item()),
        "cosine_before": float(dot_before.detach().cpu().item()) / denominator,
        "base_grad_norm": float(base_norm.detach().cpu().item()),
        "correction_grad_norm": float(corr_norm.detach().cpu().item()),
        "corrected_grad_norm": float(torch.sqrt(torch.clamp(_norm_sq(corrected), min=0.0)).detach().cpu().item()),
        "cap_scale": cap_scale,
        "effective_correction_weight": effective_weight,
    }


class HPTSourceEpoch:
    """Callable replacement for ``trainer_hpt._train_epoch``."""

    def __init__(self, config: SourceUpdateConfig, diagnostics_path: Path) -> None:
        if config.method not in {"naive_joint_hpt", "fedworldmix_hpt"}:
            raise ValueError(f"Unsupported source method: {config.method}")
        self.config = config
        self.diagnostics_path = diagnostics_path
        self.epoch_index = 0
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _loss(policy: object, batch: Mapping[str, object]) -> "object":
        out = policy(batch)
        loss = out.get("loss", None)
        if loss is None:
            raise RuntimeError("HPTPolicy forward did not return a loss.")
        return loss

    @staticmethod
    def _grad(loss: object, params: Sequence[object]) -> Tuple[object, ...]:
        import torch

        return torch.autograd.grad(loss, params, allow_unused=True)

    @staticmethod
    def _weighted_sum(
        left: Sequence[object], right: Sequence[object], right_weight: float
    ) -> List[object]:
        out: List[object] = []
        for a, b in zip(left, right):
            if a is None and b is None:
                out.append(None)
            elif a is None:
                out.append(b * float(right_weight))
            elif b is None:
                out.append(a)
            else:
                out.append(a + b * float(right_weight))
        return out

    def _append(self, payload: Mapping[str, object]) -> None:
        with self.diagnostics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")

    def __call__(
        self,
        policy: object,
        loader: object,
        device: object,
        *,
        expert_stats: object | None = None,
        grad_clip_norm: float = 0.0,
        fail_on_nonfinite: bool = False,
    ) -> float:
        import torch
        from phaseb import trainer_hpt

        policy.train()
        named_params = trainer_hpt._trainable_parameters(policy)
        names = [name for name, _ in named_params]
        params = [param for _, param in named_params]
        losses: List[float] = []
        epoch = self.epoch_index
        self.epoch_index += 1

        for batch_index, raw_batch in enumerate(loader):
            policy.zero_grad(set_to_none=True)
            real_batch = trainer_hpt._move_batch_to_device(raw_batch, device=device)
            two_d_batch = photometric_twin(real_batch)
            structured_batch = structured_multiview_twin(real_batch, batch_index=batch_index)
            quality, accepted, total = sample_validity(structured_batch)

            real_loss = self._loss(policy, real_batch)
            if expert_stats is not None:
                expert_stats.update_from_policy(policy)
            real_grads = self._grad(real_loss, params)
            two_d_loss = self._loss(policy, two_d_batch)
            two_d_grads = self._grad(two_d_loss, params)
            structured_loss = self._loss(policy, structured_batch)
            structured_grads = self._grad(structured_loss, params)
            base_grads = self._weighted_sum(real_grads, two_d_grads, self.config.lambda_2d)

            diagnostics: Dict[str, object] = {}
            if self.config.method == "fedworldmix_hpt":
                final_grads, diagnostics = compose_protected_gradients(
                    names,
                    base_grads,
                    structured_grads,
                    correction_weight=self.config.lambda_structured,
                    quality=quality,
                    norm_cap=self.config.norm_cap,
                    eps=self.config.eps,
                )
            else:
                final_grads = self._weighted_sum(
                    base_grads,
                    structured_grads,
                    self.config.lambda_structured * quality,
                )
                base_norm = math.sqrt(
                    sum(float(torch.sum(g * g).detach().cpu().item()) for g in base_grads if g is not None)
                )
                corr_norm = math.sqrt(
                    sum(float(torch.sum(g * g).detach().cpu().item()) for g in structured_grads if g is not None)
                )
                diagnostics = {
                    "projection_applied": False,
                    "base_grad_norm": base_norm,
                    "correction_grad_norm": corr_norm,
                    "cap_scale": 1.0,
                    "effective_correction_weight": self.config.lambda_structured * quality,
                }

            for param, grad in zip(params, final_grads):
                param.grad = None if grad is None else grad.detach()
            trainer_hpt._optimizer_step_with_guards(
                policy,
                grad_clip_norm=float(grad_clip_norm),
                fail_on_nonfinite=bool(fail_on_nonfinite),
                context=f"source epoch {epoch} batch {batch_index}",
            )

            combined_loss = (
                float(real_loss.detach().cpu().item())
                + self.config.lambda_2d * float(two_d_loss.detach().cpu().item())
                + self.config.lambda_structured * quality * float(structured_loss.detach().cpu().item())
            )
            losses.append(combined_loss)
            self._append(
                {
                    "epoch": int(epoch),
                    "batch": int(batch_index),
                    "method": self.config.method,
                    "real_loss": float(real_loss.detach().cpu().item()),
                    "two_d_loss": float(two_d_loss.detach().cpu().item()),
                    "structured_multiview_loss": float(structured_loss.detach().cpu().item()),
                    "combined_loss": combined_loss,
                    "quality_acceptance": quality,
                    "accepted_samples": accepted,
                    "source_samples": total,
                    **diagnostics,
                }
            )
        return sum(losses) / float(max(len(losses), 1))

