from __future__ import annotations

import torch


def add_gaussian_noise(inputs: torch.Tensor, std: float, generator: torch.Generator | None = None) -> torch.Tensor:
    if std <= 0:
        return inputs
    noise = torch.randn(
        inputs.shape,
        generator=generator,
        device=inputs.device,
        dtype=inputs.dtype,
    ) * std
    return inputs + noise


def apply_frame_dropout(
    inputs: torch.Tensor,
    drop_prob: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if drop_prob <= 0:
        return inputs
    outputs = inputs.clone()
    batch, history = outputs.shape[:2]
    mask = torch.rand(
        batch,
        history,
        generator=generator,
        device=outputs.device,
        dtype=torch.float32,
    ) < drop_prob
    mask[:, 0] = False
    for t in range(1, history):
        dropped = mask[:, t]
        if dropped.any():
            outputs[dropped, t] = outputs[dropped, t - 1]
    return outputs


def temporal_warp(inputs: torch.Tensor, factor: float) -> torch.Tensor:
    if factor == 1.0:
        return inputs
    batch, history, joints, dims = inputs.shape
    source = torch.arange(history, device=inputs.device, dtype=inputs.dtype)
    center = (history - 1) / 2.0
    warped = (source - center) / factor + center
    warped = warped.clamp(0, history - 1)
    left = warped.floor().long()
    right = warped.ceil().long()
    alpha = (warped - left.to(inputs.dtype)).view(1, history, 1, 1)
    left_vals = inputs[:, left]
    right_vals = inputs[:, right]
    return left_vals * (1.0 - alpha) + right_vals * alpha


def apply_perturbation(
    inputs: torch.Tensor,
    perturbation: str,
    level: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if perturbation == "clean":
        return inputs
    if perturbation == "noise":
        return add_gaussian_noise(inputs, level, generator=generator)
    if perturbation == "dropout":
        return apply_frame_dropout(inputs, level, generator=generator)
    if perturbation == "speed":
        return temporal_warp(inputs, level)
    raise ValueError(f"Unsupported perturbation: {perturbation}")
