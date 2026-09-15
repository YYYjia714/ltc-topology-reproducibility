from __future__ import annotations

from typing import Any

import torch

try:
    import torch_directml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch_directml = None


def default_device_arg() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch_directml is not None:
        return "dml"
    return "cpu"


def resolve_runtime_device(device_name: str) -> tuple[Any, str]:
    normalized = device_name.lower()
    if normalized == "auto":
        return resolve_runtime_device(default_device_arg())
    if normalized == "cpu":
        return torch.device("cpu"), "cpu"
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but CUDA is not available.")
        return torch.device(device_name), "cuda"
    if normalized == "dml" or normalized.startswith("dml:"):
        if torch_directml is None:
            raise RuntimeError(
                "DirectML device requested, but torch-directml is not installed."
            )
        if normalized == "dml":
            return torch_directml.device(), "dml"
        _, _, index_text = normalized.partition(":")
        if not index_text.isdigit():
            raise RuntimeError(f"Invalid DirectML device specifier: {device_name}")
        return torch_directml.device(int(index_text)), "dml"
    return torch.device(device_name), torch.device(device_name).type


def should_pin_memory(device_backend: str) -> bool:
    return device_backend == "cuda"
