"""Backward-compatible imports for older FedSplat scripts."""

from __future__ import annotations

from .generator_adapter import GaussianClientGenerator, load_client_config


RoboSplatAdapter = GaussianClientGenerator
