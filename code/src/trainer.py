from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import torch
from torch.utils.data import DataLoader

from .metrics import count_parameters, measure_inference_time, mpjpe


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    num_items = 0
    total_batches = len(loader)
    for batch_index, batch in enumerate(loader, start=1):
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(inputs)
        loss = mpjpe(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        num_items += batch_size
        if progress_callback is not None:
            progress_callback(
                batch_index,
                total_batches,
                running_loss / max(num_items, 1),
            )
    return running_loss / max(num_items, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    running_loss = 0.0
    num_items = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        predictions = model(inputs)
        loss = mpjpe(predictions, targets)
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        num_items += batch_size
    return running_loss / max(num_items, 1)


@torch.no_grad()
def summarize_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    first_batch = next(iter(loader))
    sample = first_batch["inputs"][:1].to(device)
    model.eval()
    inference_seconds = measure_inference_time(model, sample)
    return {
        "parameters": count_parameters(model),
        "inference_seconds": inference_seconds,
        "samples_per_second": 1.0 / inference_seconds if inference_seconds > 0 else 0.0,
    }


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        output_dir / "best.pt",
    )
    (output_dir / "best_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
