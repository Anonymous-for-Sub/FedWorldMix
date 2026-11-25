"""Compatibility helpers for external LeRobot runtimes.

The project intentionally executes heavy VLA code in prepared external
environments.  Some source checkouts expose optional policy packages from
``lerobot.policies.__init__`` that are incompatible with an otherwise usable
runtime.  Importing the namespace package directly keeps optional policies
lazy while preserving normal submodule imports.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def bootstrap_lerobot_policies() -> None:
    """Install a lazy ``lerobot.policies`` namespace when none is loaded."""
    if "lerobot.policies" in sys.modules:
        return

    import lerobot

    policies_path = Path(lerobot.__file__).resolve().parent / "policies"
    if not policies_path.is_dir():
        raise FileNotFoundError(f"LeRobot policies package was not found: {policies_path}")
    package = types.ModuleType("lerobot.policies")
    package.__path__ = [str(policies_path)]
    package.__package__ = "lerobot.policies"
    package.__file__ = str(policies_path / "__init__.py")
    sys.modules["lerobot.policies"] = package


def bootstrap_smolvla_processors() -> None:
    """Register checkpoint-referenced SmolVLA processor steps explicitly."""
    bootstrap_lerobot_policies()
    import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
