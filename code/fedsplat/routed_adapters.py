"""Per-environment static LoRA-bank application for image-routed rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .lora_updates import ClientAdapterUpdate, load_client_update


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RoutedAdapterController:
    def __init__(self, route_names: list[str], environment_count: int) -> None:
        if not route_names or route_names[0] != "base":
            raise ValueError("Routed adapter bank must place base first")
        if environment_count <= 0:
            raise ValueError("Routed adapter controller requires environments")
        self.route_names = tuple(route_names)
        self.route_to_index = {name: index for index, name in enumerate(route_names)}
        self.route_indices = np.zeros(environment_count, dtype=np.int64)

    def set_routes(self, routes: list[str]) -> None:
        if len(routes) != len(self.route_indices):
            raise ValueError(
                f"Expected {len(self.route_indices)} routes, observed {len(routes)}"
            )
        unknown = sorted(set(routes) - set(self.route_to_index))
        if unknown:
            raise ValueError(f"Router selected unsupported adapter routes: {unknown}")
        self.route_indices = np.asarray(
            [self.route_to_index[route] for route in routes], dtype=np.int64
        )

    def expanded_indices(self, batch_size: int) -> np.ndarray:
        environments = len(self.route_indices)
        if batch_size % environments:
            raise ValueError(
                "Routed adapter cannot map the leading tensor dimension to "
                f"{environments} environments: batch={batch_size}"
            )
        return np.repeat(self.route_indices, batch_size // environments)


def _resolved_routes(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    route_records = manifest.get("routes")
    if not isinstance(route_records, dict) or "base" not in route_records:
        raise ValueError("Adapter-bank manifest must define a base route")
    routes = {}
    for name, record in route_records.items():
        path = Path(record["adapter"])
        routes[name] = path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
        if not (routes[name] / "adapter_update.json").is_file():
            raise FileNotFoundError(f"Incomplete adapter route {name}: {routes[name]}")
    return manifest, routes


def apply_routed_adapter_bank(
    policy,
    manifest_path: Path,
    *,
    environment_count: int,
) -> tuple[RoutedAdapterController, list[str], dict[str, Any]]:
    """Wrap policy linears once and select full adapters per environment."""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    manifest, route_paths = _resolved_routes(manifest_path)
    route_names = ["base", *sorted(name for name in route_paths if name != "base")]
    updates: dict[str, ClientAdapterUpdate] = {
        name: load_client_update(route_paths[name]) for name in route_names
    }
    reference = updates["base"]
    reference_layers = {layer.layer_name: layer for layer in reference.layers}
    for route_name, update in updates.items():
        layers = {layer.layer_name: layer for layer in update.layers}
        if set(layers) != set(reference_layers):
            raise ValueError(f"Route {route_name} has a different adapter layer set")
        for layer_name, reference_layer in reference_layers.items():
            layer = layers[layer_name]
            if layer.a.shape != reference_layer.a.shape or layer.b.shape != reference_layer.b.shape:
                raise ValueError(
                    f"Route {route_name} has incompatible low-rank shape for {layer_name}"
                )

    controller = RoutedAdapterController(route_names, environment_count)

    class RoutedStaticAdapterLinear(nn.Module):
        def __init__(self, base: nn.Linear, layer_name: str) -> None:
            super().__init__()
            self.base = base
            a = np.stack(
                [
                    next(
                        layer.a
                        for layer in updates[route_name].layers
                        if layer.layer_name == layer_name
                    )
                    for route_name in route_names
                ]
            )
            b = np.stack(
                [
                    next(
                        layer.b
                        for layer in updates[route_name].layers
                        if layer.layer_name == layer_name
                    )
                    for route_name in route_names
                ]
            )
            self.register_buffer(
                "route_a", torch.as_tensor(a, dtype=torch.float32, device=base.weight.device)
            )
            self.register_buffer(
                "route_b", torch.as_tensor(b, dtype=torch.float32, device=base.weight.device)
            )

        @property
        def weight(self):
            return self.base.weight

        @property
        def bias(self):
            return self.base.bias

        @property
        def in_features(self) -> int:
            return self.base.in_features

        @property
        def out_features(self) -> int:
            return self.base.out_features

        def forward(self, inputs):
            base_output = self.base(inputs)
            route_indices = controller.expanded_indices(int(inputs.shape[0]))
            update = torch.zeros_like(base_output, dtype=torch.float32)
            for route_index in np.unique(route_indices):
                positions = np.flatnonzero(route_indices == route_index)
                tensor_positions = torch.as_tensor(
                    positions, dtype=torch.long, device=inputs.device
                )
                selected = inputs.index_select(0, tensor_positions)
                selected_update = functional.linear(
                    functional.linear(
                        selected.float(), self.route_a[int(route_index)]
                    ),
                    self.route_b[int(route_index)],
                )
                update.index_copy_(0, tensor_positions, selected_update)
            return base_output + update.to(dtype=base_output.dtype)

    applied = []
    for layer in reference.layers:
        parent_name, separator, attribute = layer.layer_name.rpartition(".")
        parent = policy if not separator else policy.get_submodule(parent_name)
        base = getattr(parent, attribute)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Adapter target is not torch.nn.Linear: {layer.layer_name}")
        setattr(parent, attribute, RoutedStaticAdapterLinear(base, layer.layer_name))
        applied.append(layer.layer_name)
    audit = {
        "schema": "fedsplat.routed_adapter_bank_runtime.v1",
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256(Path(manifest_path).resolve()),
        "route_order": route_names,
        "route_adapter_paths": {name: str(route_paths[name]) for name in route_names},
        "route_adapter_fingerprints": {
            name: manifest["routes"][name].get("adapter_fingerprint")
            for name in route_names
        },
        "environment_count": environment_count,
        "layers": applied,
    }
    return controller, applied, audit
