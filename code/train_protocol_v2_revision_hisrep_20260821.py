from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.progress import ProgressTracker
from train_hisrep_cmu_clean import (
    COMMON_JOINTS,
    Config,
    CmuHisRepWindowDataset,
    _hisrep_future_from_output,
    make_model,
    read_json,
)


MM_PER_METER = 1000.0
OBSERVATION_FRAMES = 25
HISREP_INTERNAL_INPUT_FRAMES = 50


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def set_deterministic_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def predict_block_mm(model: torch.nn.Module, current_mm: torch.Tensor, cfg: Config) -> torch.Tensor:
    batch_size = current_mm.shape[0]
    if current_mm.shape[1] != OBSERVATION_FRAMES or cfg.input_n != HISREP_INTERNAL_INPUT_FRAMES:
        raise RuntimeError("HisRep observation adapter requires 25 observed frames and internal input_n=50")
    repeated_last = current_mm[:, -1:].expand(-1, cfg.input_n - OBSERVATION_FRAMES, -1, -1)
    src = torch.cat([current_mm, repeated_last], dim=1).reshape(batch_size, cfg.input_n, cfg.in_features)
    output = model(src, output_n=cfg.model_output_n, input_n=cfg.input_n, itera=cfg.itera)
    return _hisrep_future_from_output(output, cfg, batch_size)


def forward_hisrep_observation_adapter(
    model: torch.nn.Module,
    batch_m: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_m.shape[1] != OBSERVATION_FRAMES + cfg.output_n:
        raise RuntimeError(f"Expected {OBSERVATION_FRAMES + cfg.output_n} frames, got {batch_m.shape[1]}")
    observed_mm = batch_m[:, :OBSERVATION_FRAMES].float() * MM_PER_METER
    target_mm = batch_m[:, OBSERVATION_FRAMES : OBSERVATION_FRAMES + cfg.output_n].float() * MM_PER_METER
    repeated_last = observed_mm[:, -1:].expand(-1, cfg.input_n - OBSERVATION_FRAMES, -1, -1)
    source_mm = torch.cat([observed_mm, repeated_last], dim=1)
    batch_size = int(batch_m.shape[0])
    output = model(
        source_mm.reshape(batch_size, cfg.input_n, cfg.in_features),
        output_n=cfg.model_output_n,
        input_n=cfg.input_n,
        itera=cfg.itera,
    )
    prediction_mm = _hisrep_future_from_output(output, cfg, batch_size)
    mpjpe = torch.linalg.norm(prediction_mm - target_mm, dim=-1).mean()
    if cfg.model_output_n == cfg.output_n and cfg.itera == 1:
        prediction_all = output[:, :, 0].reshape(
            batch_size, cfg.kernel_size + cfg.output_n, len(COMMON_JOINTS), 3
        )
        supervised = torch.cat([observed_mm[:, -cfg.kernel_size :], target_mm], dim=1)
        loss = torch.linalg.norm(prediction_all - supervised, dim=-1).mean()
    else:
        loss = mpjpe
    return loss, mpjpe


def recursive_rollout_mm(
    model: torch.nn.Module,
    inputs_m: torch.Tensor,
    cfg: Config,
    horizon: int = 75,
) -> torch.Tensor:
    current_mm = inputs_m * MM_PER_METER
    chunks = []
    generated = 0
    while generated < horizon:
        block_mm = predict_block_mm(model, current_mm, cfg)
        needed = min(cfg.output_n, horizon - generated)
        chunks.append(block_mm[:, :needed])
        current_mm = torch.cat([current_mm[:, needed:], block_mm[:, :needed]], dim=1)
        generated += needed
    return torch.cat(chunks, dim=1)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    tracker: ProgressTracker | None,
    epoch: int,
    global_batch: int,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss = 0.0
    total_mpjpe = 0.0
    count = 0
    for batch_index, batch in enumerate(loader, 1):
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, mpjpe = forward_hisrep_observation_adapter(model, batch, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
        optimizer.step()
        batch_size = batch.shape[0]
        count += batch_size
        global_batch += 1
        total_loss += float(loss.detach().cpu()) * batch_size
        total_mpjpe += float(mpjpe.detach().cpu()) * batch_size
        if tracker and (batch_index % cfg.progress_every == 0 or batch_index == len(loader)):
            tracker.update(
                processed=global_batch,
                success=global_batch,
                failed=0,
                current_item=f"epoch {epoch}/{cfg.epochs} batch {batch_index}/{len(loader)}",
                extra={
                    "epoch": epoch,
                    "epochs_total": cfg.epochs,
                    "batch": batch_index,
                    "batches_per_epoch": len(loader),
                    "train_physical_mpjpe_1_25_mm": round(total_mpjpe / count, 6),
                },
            )
    return {"loss_mm": total_loss / count, "physical_mpjpe_1_25_mm": total_mpjpe / count}, global_batch


@torch.no_grad()
def evaluate_recursive_75(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    block_totals = [0.0, 0.0, 0.0]
    total = 0.0
    count = 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        inputs = batch[:, :OBSERVATION_FRAMES]
        targets_mm = batch[:, OBSERVATION_FRAMES : OBSERVATION_FRAMES + 75] * MM_PER_METER
        predictions_mm = recursive_rollout_mm(model, inputs, cfg, 75)
        errors = torch.linalg.norm(predictions_mm - targets_mm, dim=-1)
        batch_size = batch.shape[0]
        count += batch_size
        for index in range(3):
            block_totals[index] += float(errors[:, index * 25 : (index + 1) * 25].mean().cpu()) * batch_size
        total += float(errors.mean().cpu()) * batch_size
    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return {
        "physical_mpjpe_1_25_mm": block_totals[0] / count,
        "physical_mpjpe_26_50_mm": block_totals[1] / count,
        "physical_mpjpe_51_75_mm": block_totals[2] / count,
        "physical_mpjpe_1_75_mm": total / count,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    lr: float,
    best_value: float,
    best_epoch: int,
    history: list[dict],
    config: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "lr": lr,
            "best_epoch": best_epoch,
            "best_val_physical_mpjpe_1_75_mm": best_value,
            "model_state": model.state_dict(),
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "config": config,
            "architecture_signature": {
                "model": "hisrep",
                "input_n": 25,
                "output_n": 25,
                "d_model": config["hisrep_config"]["d_model"],
                "num_stage": config["hisrep_config"]["num_stage"],
                "dct_n": config["hisrep_config"]["dct_n"],
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol v2 common18 formal revision HisRepItself trainer.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--progress-path", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--frozen-inventory-sha256", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_deterministic_seed(args.seed)
    device = torch.device(args.device)
    metadata = read_json(args.metadata)
    expected = {
        "history": 25,
        "future": 25,
        "rollout_horizon": 75,
        "stride": args.stride,
        "representation": "common18",
        "num_joints": 18,
        "coordinate_source": "joints_root_relative",
        "center_on_first_frame": False,
        "split_unit": "subject",
    }
    mismatches = {key: {"expected": value, "actual": metadata.get(key)} for key, value in expected.items() if metadata.get(key) != value}
    if str(metadata.get("split_manifest_sha256", "")).lower() != args.expected_manifest_sha256.lower():
        mismatches["split_manifest_sha256"] = {
            "expected": args.expected_manifest_sha256.lower(),
            "actual": str(metadata.get("split_manifest_sha256", "")).lower(),
        }
    if mismatches:
        raise RuntimeError(f"Protocol metadata mismatch: {mismatches}")

    cfg = Config(
        dataset=args.dataset_name,
        input_n=HISREP_INTERNAL_INPUT_FRAMES,
        output_n=25,
        model_output_n=25,
        itera=1,
        dct_n=20,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        stride=args.stride,
        seed=args.seed,
        num_workers=args.num_workers,
        progress_every=20,
        disable_tqdm=True,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    train_set = CmuHisRepWindowDataset(
        metadata,
        cfg.dataset,
        "train",
        OBSERVATION_FRAMES,
        cfg.output_n,
        cfg.stride,
        args.max_train_windows or None,
        cfg.seed,
    )
    val_set = CmuHisRepWindowDataset(
        metadata,
        cfg.dataset,
        "val",
        OBSERVATION_FRAMES,
        75,
        cfg.stride,
        args.max_val_windows or None,
        cfg.seed + 1,
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = make_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_gamma = cfg.lr_decay_factor ** (1.0 / cfg.epochs)
    lr_now = cfg.lr
    config = {
        "dataset_name": args.dataset_name,
        "representation": "common18",
        "hisrep_config": asdict(cfg),
        "observation_frames": OBSERVATION_FRAMES,
        "observation_adapter": "repeat_last_observed_pose_from_25_to_internal_50_frames",
        "uses_additional_real_history": False,
        "checkpoint_selection": "validation_recursive_75_frame_physical_mpjpe_mm",
        "manifest_sha256": args.expected_manifest_sha256.lower(),
        "metadata_sha256": sha256_file(args.metadata),
        "trainer_sha256": sha256_file(Path(__file__)),
        "frozen_inventory_sha256": args.frozen_inventory_sha256.lower(),
        "deterministic_algorithms": True,
    }
    atomic_write_json(args.run_dir / "config.json", config)
    tracker = ProgressTracker(args.progress_path)
    tracker.start(
        stage="formal_revision_hisrep_training",
        total=cfg.epochs * len(train_loader),
        extra={
            "dataset": args.dataset_name,
            "seed": args.seed,
            "checkpoint_selection": config["checkpoint_selection"],
            "output_dir": str(args.run_dir),
        },
    )
    history: list[dict] = []
    best_value = math.inf
    best_epoch = 0
    global_batch = 0
    for epoch in range(1, cfg.epochs + 1):
        lr_now *= lr_gamma
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        train_metrics, global_batch = train_epoch(
            model,
            train_loader,
            cfg,
            device,
            optimizer,
            tracker,
            epoch,
            global_batch,
        )
        val_metrics = evaluate_recursive_75(model, val_loader, cfg, device)
        record = {
            "epoch": epoch,
            "lr": lr_now,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        atomic_write_json(args.run_dir / "train_history.json", history)
        current_value = val_metrics["physical_mpjpe_1_75_mm"]
        if current_value < best_value:
            best_value = current_value
            best_epoch = epoch
            save_checkpoint(args.run_dir / "best.pt", model, optimizer, epoch, lr_now, best_value, best_epoch, history, config)
        save_checkpoint(args.run_dir / "last.pt", model, optimizer, epoch, lr_now, best_value, best_epoch, history, config)
        tracker.update(
            processed=epoch * len(train_loader),
            success=epoch * len(train_loader),
            failed=0,
            current_item=f"epoch {epoch}/{cfg.epochs} complete",
            extra={
                "epoch": epoch,
                "epochs_total": cfg.epochs,
                "train_physical_mpjpe_1_25_mm": round(train_metrics["physical_mpjpe_1_25_mm"], 6),
                "val_physical_mpjpe_1_75_mm": round(current_value, 6),
                "best_epoch": best_epoch,
                "best_val_physical_mpjpe_1_75_mm": round(best_value, 6),
            },
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)

    best_path = args.run_dir / "best.pt"
    if best_epoch == 0 or not best_path.is_file():
        raise RuntimeError("HisRepItself training produced no selected checkpoint")
    result = {
        "status": "completed",
        "dataset": args.dataset_name,
        "representation": "common18",
        "method_name": "HisRepItself",
        "seed": args.seed,
        "epochs": cfg.epochs,
        "best_epoch": best_epoch,
        "best_val_physical_mpjpe_1_75_mm": best_value,
        "checkpoint_selection": config["checkpoint_selection"],
        "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "history_file": str(args.run_dir / "train_history.json"),
        "history_sha256": sha256_file(args.run_dir / "train_history.json"),
        "config": config,
    }
    atomic_write_json(args.run_dir / "results.json", result)
    tracker.finish(
        "completed",
        extra={
            "best_epoch": best_epoch,
            "best_val_physical_mpjpe_1_75_mm": round(best_value, 6),
            "results_file": str(args.run_dir / "results.json"),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
