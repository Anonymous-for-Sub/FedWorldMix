"""Optional dependency helpers for Phase B."""

from __future__ import annotations

import importlib.util


def has_numpy() -> bool:
    return importlib.util.find_spec("numpy") is not None


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def require_numpy(feature_name: str) -> None:
    if has_numpy():
        return
    raise RuntimeError(
        f"{feature_name} requires numpy, but numpy is not installed. "
        "Install dependencies from requirements-phaseb.txt first."
    )


def require_torch(feature_name: str) -> None:
    if has_torch():
        return
    raise RuntimeError(
        f"{feature_name} requires torch, but torch is not installed. "
        "Install dependencies from requirements-phaseb.txt first."
    )
