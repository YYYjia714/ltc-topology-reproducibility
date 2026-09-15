from __future__ import annotations

import time
from typing import Callable

import torch


def mpjpe(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.norm(predictions - targets, dim=-1).mean()


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def measure_inference_time(
    fn: Callable[[torch.Tensor], torch.Tensor],
    sample: torch.Tensor,
    warmup: int = 10,
    steps: int = 50,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn(sample)
        if sample.is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            fn(sample)
        if sample.is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) / steps
