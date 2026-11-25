"""Deterministic task embedding utilities (local, lightweight).

Used to provide a stable task-conditioning signal when a single shared head is
trained across multiple tasks (e.g., centralized baselines).
"""

from __future__ import annotations

import hashlib
import random
from typing import List


def task_embedding(task: str, dim: int, *, scale: float = 1.0) -> List[float]:
    """Return a deterministic pseudo-random embedding for a task alias."""
    task = str(task)
    dim = int(dim)
    if dim <= 0:
        return []
    digest = hashlib.sha256(task.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = random.Random(seed)
    vec = [rng.gauss(0.0, 1.0) * float(scale) for _ in range(dim)]
    return vec

