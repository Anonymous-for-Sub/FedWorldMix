"""Dataset utilities for the HPT (RGB) Phase B path."""

from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .deps import require_numpy, require_torch


@dataclass(frozen=True)
class HPTSampleIndex:
    task: str
    rgb_npz_path: str
    rgb_step: int
    proprio: List[float]
    action: List[float]


class RGBNPZCache:
    """Tiny LRU cache for decoded trajectory RGB arrays."""

    def __init__(self, max_items: int = 8) -> None:
        self.max_items = int(max_items)
        self._cache: "OrderedDict[str, Tuple[object, List[str]]]" = OrderedDict()

    def get(self, path: Path) -> Tuple[object, List[str]]:
        require_numpy("RGB npz cache")
        import numpy as np

        key = str(path)
        cached = self._cache.get(key, None)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        if not path.exists():
            raise FileNotFoundError(f"RGB npz not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            frames = data["frames"]
            views_raw = data.get("views", None)
            if views_raw is None:
                views = []
            else:
                views = [str(v) for v in views_raw.tolist()]
        self._cache[key] = (frames, views)
        self._cache.move_to_end(key)
        while len(self._cache) > max(self.max_items, 0):
            self._cache.popitem(last=False)
        return frames, views


class IOSPPermNPZCache:
    """Tiny LRU cache for IOSP permutation arrays stored next to RGB trajectories."""

    def __init__(self, max_items: int = 64) -> None:
        self.max_items = int(max_items)
        self._cache: "OrderedDict[str, object]" = OrderedDict()

    def get(self, path: Path) -> "object":
        require_numpy("IOSP perm npz cache")
        import numpy as np

        key = str(path)
        cached = self._cache.get(key, None)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        if not path.exists():
            raise FileNotFoundError(f"IOSP perm npz not found: {path}")
        with np.load(path, allow_pickle=False) as data:
            perm = data["perm"]
        self._cache[key] = perm
        self._cache.move_to_end(key)
        while len(self._cache) > max(self.max_items, 0):
            self._cache.popitem(last=False)
        return perm


def _read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def build_hpt_indices(
    dataset_root: str,
    task: str,
    split: str,
    *,
    max_samples: int = 0,
) -> List[HPTSampleIndex]:
    root = Path(dataset_root) / task / f"{split}.jsonl"
    rows = _read_jsonl(root, limit=int(max_samples))
    indices: List[HPTSampleIndex] = []
    for payload in rows:
        rgb_npz_path = payload.get("rgb_npz_path", None)
        rgb_step = payload.get("rgb_step", None)
        if not isinstance(rgb_npz_path, str) or not rgb_npz_path:
            continue
        if not isinstance(rgb_step, int):
            if isinstance(rgb_step, float):
                rgb_step = int(rgb_step)
            else:
                continue
        proprio = payload.get("proprio", None)
        action = payload.get("action", None)
        if not isinstance(proprio, list) or not isinstance(action, list):
            continue
        indices.append(
            HPTSampleIndex(
                task=str(task),
                rgb_npz_path=str(rgb_npz_path),
                rgb_step=int(rgb_step),
                proprio=[float(v) for v in proprio if isinstance(v, (int, float))],
                action=[float(v) for v in action if isinstance(v, (int, float))],
            )
        )
    return indices


class PhaseBHPTDataset:
    """Torch Dataset for Phase B HPT path.

    Returns a dict compatible with the HPTPolicy forward/select_action interface.
    """

    def __init__(
        self,
        dataset_root: str,
        task: str,
        split: str,
        *,
        camera_views: Sequence[str],
        max_samples: int = 0,
        cache: RGBNPZCache | None = None,
        perm_cache: IOSPPermNPZCache | None = None,
        task_embed_dim: int = 0,
        task_embed_scale: float = 1.0,
        text_embed: Sequence[float] | None = None,
        text_embed_mode: str = "state",
        iosp_embed: Sequence[float] | None = None,
        iosp_perm_suffix: str = "",
    ) -> None:
        require_torch("HPT dataset")
        self.dataset_root = str(dataset_root)
        self.task = str(task)
        self.split = str(split)
        self.camera_views = [str(v) for v in camera_views if str(v)]
        if not self.camera_views:
            raise ValueError("camera_views must be non-empty for HPT datasets.")
        self.indices = build_hpt_indices(
            dataset_root=self.dataset_root,
            task=self.task,
            split=self.split,
            max_samples=int(max_samples),
        )
        self.cache = cache or RGBNPZCache(max_items=8)
        self.perm_cache = perm_cache or IOSPPermNPZCache(max_items=64)
        self.task_embed_dim = int(task_embed_dim)
        self.task_embed_scale = float(task_embed_scale)
        self.text_embed = [float(v) for v in (text_embed or [])]
        self.text_embed_mode = str(text_embed_mode or "state").strip().lower()
        self.iosp_embed = [float(v) for v in (iosp_embed or [])]
        self.iosp_perm_suffix = str(iosp_perm_suffix or "").strip()
        grouped: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.indices):
            key = str(item.rgb_npz_path)
            grouped.setdefault(key, []).append(int(idx))
        self.indices_by_rgb_npz_path = grouped

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int) -> Dict[str, object]:
        require_torch("HPT dataset")
        require_numpy("HPT dataset")
        import numpy as np
        import torch

        item = self.indices[int(index)]
        npz_path = Path(self.dataset_root) / item.task / item.rgb_npz_path
        frames, views = self.cache.get(npz_path)
        if not isinstance(frames, np.ndarray):
            raise TypeError("RGB npz frames must be a numpy array.")
        if frames.ndim != 5:
            raise ValueError(f"Expected frames shape (T,V,H,W,3), got {frames.shape}.")
        step = int(item.rgb_step)
        if step < 0 or step >= int(frames.shape[0]):
            raise IndexError(f"rgb_step {step} is out of range for {npz_path} with T={int(frames.shape[0])}.")
        step_frames = frames[step]  # (V,H,W,3)
        if step_frames.ndim != 4 or int(step_frames.shape[-1]) != 3:
            raise ValueError(f"Expected step frames shape (V,H,W,3), got {step_frames.shape}.")

        view_to_index = {name: idx for idx, name in enumerate(views)}
        sample: Dict[str, object] = {}
        for view_name in self.camera_views:
            if view_name not in view_to_index:
                raise KeyError(f"View '{view_name}' not found in {npz_path} (available: {views})")
            frame = step_frames[int(view_to_index[view_name])]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            image = torch.from_numpy(frame).to(dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3,H,W)
            sample[f"observation.images.{view_name}"] = image  # (3,H,W)

        sample["observation.state"] = torch.tensor(item.proprio, dtype=torch.float32)
        if self.task_embed_dim > 0:
            from .task_embed import task_embedding

            embed = torch.tensor(
                task_embedding(self.task, self.task_embed_dim, scale=self.task_embed_scale),
                dtype=torch.float32,
            )
            sample["observation.state"] = torch.cat([sample["observation.state"], embed], dim=0)
        if self.text_embed:
            embed = torch.tensor(self.text_embed, dtype=torch.float32)
            if self.text_embed_mode == "language":
                sample["observation.language"] = embed
            else:
                sample["observation.state"] = torch.cat([sample["observation.state"], embed], dim=0)
        if self.iosp_embed:
            sample["observation.iosp"] = torch.tensor(self.iosp_embed, dtype=torch.float32)
        if self.iosp_perm_suffix:
            perm_path = npz_path.with_suffix(self.iosp_perm_suffix)
            perm = self.perm_cache.get(perm_path)
            if not isinstance(perm, np.ndarray):
                raise TypeError(f"IOSP perm must be a numpy array, got {type(perm)!r} for {perm_path}.")
            if perm.ndim != 2:
                raise ValueError(f"Expected IOSP perm shape (T,L), got {perm.shape} for {perm_path}.")
            if step < 0 or step >= int(perm.shape[0]):
                raise IndexError(f"rgb_step {step} out of range for IOSP perm {perm_path} with T={int(perm.shape[0])}.")
            sample["observation.iosp_perm"] = torch.from_numpy(perm[step]).to(dtype=torch.int64)
        sample["action"] = torch.tensor(item.action, dtype=torch.float32)
        return sample


class TrajectoryBatchSampler:
    """Batch sampler that groups indices by trajectory (rgb_npz_path).

    Without this, each batch tends to load many distinct compressed `.npz` files,
    which is extremely slow and leaves the GPU under-utilized.
    """

    def __init__(
        self,
        indices_by_rgb_npz_path: Dict[str, List[int]],
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        shuffle_within_trajectory: bool = True,
    ) -> None:
        self.indices_by_rgb_npz_path = {str(k): list(v) for k, v in indices_by_rgb_npz_path.items()}
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.shuffle_within_trajectory = bool(shuffle_within_trajectory)

    def __iter__(self):
        rng = random.Random(self.seed)
        keys = list(self.indices_by_rgb_npz_path.keys())
        if self.shuffle:
            rng.shuffle(keys)
        for key in keys:
            indices = list(self.indices_by_rgb_npz_path.get(key, []))
            if self.shuffle and self.shuffle_within_trajectory:
                rng.shuffle(indices)
            for start in range(0, len(indices), max(self.batch_size, 1)):
                batch = indices[start : start + max(self.batch_size, 1)]
                if batch:
                    yield batch

    def __len__(self) -> int:
        total = 0
        bsz = max(self.batch_size, 1)
        for indices in self.indices_by_rgb_npz_path.values():
            total += int(math.ceil(len(indices) / float(bsz)))
        return int(total)
