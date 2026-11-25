"""Text embedding utilities for Phase B.

This module provides a lightweight, optional instruction-conditioning path for
HPT-based baselines by appending a fixed embedding vector to `observation.state`.

Design goals:
- Deterministic per-instruction vectors (given the same model + text).
- Avoid requiring the text encoder at rollout time by allowing embeddings to be
  stored inside checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, Iterable, List, Sequence


def _maybe_fix_ssl_cert_env() -> None:
    import os
    from pathlib import Path

    if os.environ.get("SSL_CERT_FILE"):
        return
    cafile = None
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


def _l2_normalize(vec: Sequence[float], eps: float = 1e-12) -> List[float]:
    import math

    denom = math.sqrt(sum(float(v) * float(v) for v in vec))
    denom = denom if denom > float(eps) else float(eps)
    return [float(v) / denom for v in vec]


def _project(vec: Sequence[float], out_dim: int, *, seed: int) -> List[float]:
    """Deterministic random projection to a target dimension (if needed)."""
    out_dim = int(out_dim)
    if out_dim <= 0:
        return []
    in_dim = int(len(vec))
    if in_dim == out_dim:
        return [float(v) for v in vec]
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on external env
        raise RuntimeError("numpy is required for deterministic projection.") from exc
    rng = np.random.default_rng(int(seed) % (2**32))
    scale = 1.0 / max(float(in_dim) ** 0.5, 1.0)
    mat = rng.standard_normal(size=(out_dim, in_dim), dtype=np.float32) * np.float32(scale)
    out = mat @ np.asarray(list(vec), dtype=np.float32)
    return [float(v) for v in out.tolist()]


@dataclass
class CLIPTextEmbedder:
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"
    normalize: bool = True

    def embed(self, texts: Iterable[str]) -> Dict[str, List[float]]:
        """Return a dict mapping text -> embedding list."""
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return {}
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on external env
            raise RuntimeError("torch is required for CLIP text embeddings.") from exc
        try:
            from transformers import CLIPTextModel, CLIPTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on external env
            raise RuntimeError("transformers is required for CLIP text embeddings.") from exc

        _maybe_fix_ssl_cert_env()
        tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        model = CLIPTextModel.from_pretrained(self.model_name)
        model.eval()
        model.to(self.device)

        encoded = tokenizer(
            texts_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        pooled = outputs.pooler_output  # (B, hidden)
        pooled = pooled.detach().to("cpu")
        result: Dict[str, List[float]] = {}
        for idx, text in enumerate(texts_list):
            vec = pooled[idx].view(-1).tolist()
            vec = [float(v) for v in vec]
            if self.normalize:
                vec = _l2_normalize(vec)
            result[text] = vec
        return result


@dataclass
class CLIPTextEmbedderProjected:
    """CLIP text embeddings in the joint CLIP projection space (paper-aligned)."""

    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"
    normalize: bool = True

    def embed(self, texts: Iterable[str]) -> Dict[str, List[float]]:
        texts_list = [str(t) for t in texts]
        if not texts_list:
            return {}
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on external env
            raise RuntimeError("torch is required for CLIP text embeddings.") from exc
        try:
            from transformers import CLIPTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on external env
            raise RuntimeError("transformers is required for CLIP text embeddings.") from exc

        _maybe_fix_ssl_cert_env()
        tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        model = None
        # Prefer a text-only projection model to avoid loading vision weights.
        try:
            from transformers import CLIPTextModelWithProjection  # type: ignore

            model = CLIPTextModelWithProjection.from_pretrained(self.model_name)
        except Exception:
            from transformers import CLIPModel  # type: ignore

            model = CLIPModel.from_pretrained(self.model_name)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.to(self.device)

        encoded = tokenizer(
            texts_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            if hasattr(model, "get_text_features"):
                feats = model.get_text_features(**encoded)
            else:
                outputs = model(**encoded)
                feats = getattr(outputs, "text_embeds", None)
                if feats is None:
                    feats = getattr(outputs, "pooler_output", None)
        if not isinstance(feats, torch.Tensor):
            raise RuntimeError(
                "CLIP projected text feature extraction returned an unexpected type "
                f"({type(feats)}; missing tensor output)."
            )
        feats = feats.detach().to("cpu")
        result: Dict[str, List[float]] = {}
        for idx, text in enumerate(texts_list):
            vec = feats[idx].view(-1).tolist()
            vec = [float(v) for v in vec]
            if self.normalize:
                vec = _l2_normalize(vec)
            result[text] = vec
        return result


def build_task_instruction_embeddings(
    *,
    task_to_instruction: Dict[str, str],
    model_name: str,
    out_dim: int,
    device: str = "cpu",
    normalize: bool = True,
) -> Dict[str, List[float]]:
    """Compute instruction embeddings per task alias."""
    embedder = CLIPTextEmbedder(model_name=str(model_name), device=str(device), normalize=bool(normalize))
    instructions = [str(v) for v in task_to_instruction.values()]
    instruction_to_vec = embedder.embed(instructions)

    projected: Dict[str, List[float]] = {}
    digest = hashlib.sha256((str(model_name) + "::proj_v1").encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
    for task, instruction in task_to_instruction.items():
        vec = instruction_to_vec.get(str(instruction), [])
        if not vec:
            projected[str(task)] = []
            continue
        vec2 = _project(vec, int(out_dim), seed=int(seed))
        projected[str(task)] = [float(v) for v in vec2]
    return projected


def build_task_object_group_embeddings(
    *,
    task_to_groups: Dict[str, Dict[str, str]],
    model_name: str,
    out_dim: int,
    device: str = "cpu",
    normalize: bool = True,
) -> Dict[str, List[float]]:
    """Compute per-task IOSP object-group embeddings (TO/SO/BO) and concatenate them.

    Returns a dict mapping task -> [TO_embed, SO_embed, BO_embed] flattened (length = 3*out_dim).
    """
    embedder = CLIPTextEmbedder(model_name=str(model_name), device=str(device), normalize=bool(normalize))
    phrases: List[str] = []
    for groups in task_to_groups.values():
        for key in ("to", "so", "bo"):
            phrase = str(groups.get(key, "") or "").strip()
            if phrase:
                phrases.append(phrase)
    phrase_to_vec = embedder.embed(sorted(set(phrases)))

    digest = hashlib.sha256((str(model_name) + "::proj_v1").encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)

    projected: Dict[str, List[float]] = {}
    for task, groups in task_to_groups.items():
        flat: List[float] = []
        for key in ("to", "so", "bo"):
            phrase = str(groups.get(key, "") or "").strip()
            vec = phrase_to_vec.get(phrase, []) if phrase else []
            if vec:
                vec = _project(vec, int(out_dim), seed=int(seed))
            else:
                vec = []
            if int(out_dim) > 0:
                if len(vec) < int(out_dim):
                    vec = list(vec) + [0.0] * (int(out_dim) - len(vec))
                elif len(vec) > int(out_dim):
                    vec = list(vec)[: int(out_dim)]
            flat.extend([float(v) for v in vec])
        projected[str(task)] = flat
    return projected


def build_task_object_group_embeddings_clip(
    *,
    task_to_groups: Dict[str, Dict[str, str]],
    model_name: str,
    out_dim: int,
    device: str = "cpu",
    normalize: bool = True,
) -> Dict[str, List[float]]:
    """IOSP object-group embeddings using CLIP-projected text features (no random projection)."""
    if int(out_dim) != 512:
        raise ValueError("build_task_object_group_embeddings_clip requires out_dim=512 (CLIP projection dim).")
    embedder = CLIPTextEmbedderProjected(model_name=str(model_name), device=str(device), normalize=bool(normalize))
    phrases: List[str] = []
    for groups in task_to_groups.values():
        for key in ("to", "so", "bo"):
            phrase = str(groups.get(key, "") or "").strip()
            if phrase:
                phrases.append(phrase)
    phrase_to_vec = embedder.embed(sorted(set(phrases)))

    projected: Dict[str, List[float]] = {}
    for task, groups in task_to_groups.items():
        flat: List[float] = []
        for key in ("to", "so", "bo"):
            phrase = str(groups.get(key, "") or "").strip()
            vec = phrase_to_vec.get(phrase, []) if phrase else []
            if not vec:
                vec = [0.0] * int(out_dim)
            elif len(vec) != int(out_dim):
                vec = list(vec)[: int(out_dim)] + [0.0] * max(0, int(out_dim) - len(vec))
            flat.extend([float(v) for v in vec])
        projected[str(task)] = flat
    return projected
