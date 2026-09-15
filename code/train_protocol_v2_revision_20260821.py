from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.metrics import count_parameters
from src.models import MotionForecastModel
from src.progress import ProgressTracker
from train_protocol_v2_supplementary_rollout_20260820 import (
    RolloutWindowDataset,
    read_json,
    recursive_rollout,
    validate_protocol_v2_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MM_PER_METER = 1000.0


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def denormalize(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return values * std + mean


def mpjpe(values: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(values - targets, dim=-1).mean()


def bone_error(values: torch.Tensor, targets: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return values.new_tensor(0.0)
    value_bones = values[:, :, edges[:, 0], :] - values[:, :, edges[:, 1], :]
    target_bones = targets[:, :, edges[:, 0], :] - targets[:, :, edges[:, 1], :]
    return torch.abs(
        torch.linalg.norm(value_bones, dim=-1) - torch.linalg.norm(target_bones, dim=-1)
    ).mean()


def anchor_error(
    values: torch.Tensor,
    targets: torch.Tensor,
    anchor_joint_index: int,
) -> torch.Tensor:
    return torch.linalg.norm(
        values[:, :, anchor_joint_index, :] - targets[:, :, anchor_joint_index, :],
        dim=-1,
    ).mean()


def velocity_error(
    input_values: torch.Tensor,
    values: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    value_full = torch.cat([input_values[:, -1:], values], dim=1)
    target_full = torch.cat([input_values[:, -1:], targets], dim=1)
    value_velocity = value_full[:, 1:] - value_full[:, :-1]
    target_velocity = target_full[:, 1:] - target_full[:, :-1]
    return torch.linalg.norm(value_velocity - target_velocity, dim=-1).mean()


def physical_rollout_loss(
    inputs_normalized: torch.Tensor,
    predictions_normalized: torch.Tensor,
    targets_normalized: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    edges: torch.Tensor,
    anchor_joint_index: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    inputs_m = denormalize(inputs_normalized, mean, std)
    predictions_m = denormalize(predictions_normalized, mean, std)
    targets_m = denormalize(targets_normalized, mean, std)
    loss_1_25 = mpjpe(predictions_m[:, :25], targets_m[:, :25])
    loss_26_50 = mpjpe(predictions_m[:, 25:50], targets_m[:, 25:50])
    loss_51_75 = mpjpe(predictions_m[:, 50:75], targets_m[:, 50:75])
    loss_bone = bone_error(predictions_m, targets_m, edges)
    loss_anchor = anchor_error(predictions_m, targets_m, anchor_joint_index)
    loss_velocity = velocity_error(inputs_m, predictions_m, targets_m)
    total = (
        loss_1_25
        + args.lambda_26_50 * loss_26_50
        + args.lambda_51_75 * loss_51_75
        + args.lambda_bone * loss_bone
        + args.lambda_anchor * loss_anchor
        + args.lambda_velocity * loss_velocity
    )
    normalized = mpjpe(predictions_normalized, targets_normalized)
    return total, {
        "loss_total_m": float(total.detach().cpu()),
        "normalized_mpjpe_1_75": float(normalized.detach().cpu()),
        "physical_mpjpe_1_25_mm": float(loss_1_25.detach().cpu()) * MM_PER_METER,
        "physical_mpjpe_26_50_mm": float(loss_26_50.detach().cpu()) * MM_PER_METER,
        "physical_mpjpe_51_75_mm": float(loss_51_75.detach().cpu()) * MM_PER_METER,
        "physical_mpjpe_1_75_mm": float(mpjpe(predictions_m, targets_m).detach().cpu())
        * MM_PER_METER,
        "physical_bone_1_75_mm": float(loss_bone.detach().cpu()) * MM_PER_METER,
        "physical_anchor_1_75_mm": float(loss_anchor.detach().cpu()) * MM_PER_METER,
        "physical_velocity_1_75_mm": float(loss_velocity.detach().cpu()) * MM_PER_METER,
    }


def physical_short_loss(
    predictions_normalized: torch.Tensor,
    targets_normalized: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    predictions_m = denormalize(predictions_normalized, mean, std)
    targets_m = denormalize(targets_normalized, mean, std)
    loss = mpjpe(predictions_m, targets_m)
    normalized = mpjpe(predictions_normalized, targets_normalized)
    return loss, {
        "loss_total_m": float(loss.detach().cpu()),
        "normalized_mpjpe_1_25": float(normalized.detach().cpu()),
        "physical_mpjpe_1_25_mm": float(loss.detach().cpu()) * MM_PER_METER,
    }


def architecture_signature(args: argparse.Namespace) -> dict[str, object]:
    if args.model == "ltc":
        return {
            "model": "ltc",
            "adjacency_mode": "not_applicable",
            "use_anchor_guidance": False,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        }
    return {
        "model": "ltc_topology",
        "adjacency_mode": args.adjacency_mode,
        "use_anchor_guidance": not args.disable_anchor_guidance,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }


def checkpoint_config(args: argparse.Namespace, metadata: dict) -> dict[str, object]:
    return {
        "dataset_name": args.dataset_name,
        "representation": "common18",
        "training_objective": args.training_objective,
        "architecture": architecture_signature(args),
        "history": args.history,
        "future_step": args.future_step,
        "validation_horizon": args.validation_horizon,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_26_50": args.lambda_26_50,
        "lambda_51_75": args.lambda_51_75,
        "lambda_bone": args.lambda_bone,
        "lambda_anchor": args.lambda_anchor,
        "lambda_velocity": args.lambda_velocity,
        "loss_coordinate_space": "denormalized_meter",
        "checkpoint_selection": "validation_recursive_75_frame_physical_mpjpe_mm",
        "seed": args.seed,
        "manifest_sha256": str(metadata["split_manifest_sha256"]).lower(),
        "metadata_sha256": sha256_file(args.metadata),
        "normalization_sha256": sha256_file(args.data_root / "normalization_stats.npz"),
        "trainer_sha256": sha256_file(Path(__file__)),
        "frozen_inventory_sha256": args.frozen_inventory_sha256.lower(),
        "deterministic_algorithms": True,
    }


def make_model(
    args: argparse.Namespace,
    joints: int,
    skeleton_edges: list[list[int]],
    anchor_joint_index: int,
) -> MotionForecastModel:
    if args.model == "ltc":
        effective_edges: list[list[int]] = []
        use_anchor = False
    else:
        effective_edges = skeleton_edges if args.adjacency_mode == "skeleton" else []
        use_anchor = not args.disable_anchor_guidance
    return MotionForecastModel(
        model_type=args.model,
        joints=joints,
        history=args.history,
        future=args.future_step,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_topology_encoder=True,
        use_topology_decoder=True,
        use_root_guidance=use_anchor,
        skeleton_edges=effective_edges,
        guidance_joint_index=anchor_joint_index,
    )


@torch.no_grad()
def evaluate_physical_75(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    edges: torch.Tensor,
    anchor_joint_index: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        predictions = recursive_rollout(
            model,
            inputs,
            horizon=args.validation_horizon,
            step=args.future_step,
        )
        _, metrics = physical_rollout_loss(
            inputs,
            predictions,
            targets,
            mean,
            std,
            edges,
            anchor_joint_index,
            args,
        )
        batch_size = inputs.shape[0]
        count += batch_size
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return {key: value / count for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    best_value: float,
    best_epoch: int,
    config: dict[str, object],
    warm_start_sha256: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_physical_mpjpe_1_75_mm": best_value,
            "config": config,
            "architecture_signature": config["architecture"],
            "warm_start_sha256": warm_start_sha256,
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
    parser = argparse.ArgumentParser(description="Protocol v2 common18 formal revision LTC trainer.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--training-objective", choices=["no_rollout", "rollout", "continued_no_rollout"], required=True)
    parser.add_argument("--model", choices=["ltc", "ltc_topology"], default="ltc_topology")
    parser.add_argument("--adjacency-mode", choices=["skeleton", "identity"], default="skeleton")
    parser.add_argument("--disable-anchor-guidance", action="store_true")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--future-step", type=int, default=25)
    parser.add_argument("--validation-horizon", type=int, default=75)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-26-50", type=float, default=0.7)
    parser.add_argument("--lambda-51-75", type=float, default=0.5)
    parser.add_argument("--lambda-bone", type=float, default=0.1)
    parser.add_argument("--lambda-anchor", type=float, default=0.1)
    parser.add_argument("--lambda-velocity", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--frozen-inventory-sha256", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warm_start and args.resume_from:
        raise ValueError("--warm-start and --resume-from are mutually exclusive")
    if args.training_objective in {"rollout", "continued_no_rollout"} and not args.warm_start and not args.resume_from:
        raise ValueError(f"{args.training_objective} requires --warm-start or --resume-from")
    if args.model == "ltc" and (args.adjacency_mode != "skeleton" or not args.disable_anchor_guidance):
        raise ValueError("Original LTC requires canonical adjacency=skeleton and --disable-anchor-guidance flags")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_deterministic_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(args.metadata)
    validate_protocol_v2_metadata(metadata, argparse.Namespace(
        history=args.history,
        future_step=args.future_step,
        horizon=args.validation_horizon,
        stride=args.stride,
        expected_manifest_sha256=args.expected_manifest_sha256,
    ))
    if metadata.get("representation") != "common18" or int(metadata.get("num_joints", 0)) != 18:
        raise RuntimeError("Formal revision experiments require common18 metadata")
    anchor_joint_index = int(metadata["guidance_joint_index"])
    anchor_joint_name = str(metadata["joint_names"][anchor_joint_index])
    if anchor_joint_name != "spine2":
        raise RuntimeError(f"Expected common18 spine2 anchor, got {anchor_joint_name}")
    skeleton_edges = [[int(src), int(dst)] for src, dst in metadata["skeleton_edges"]]

    stats_path = args.data_root / "normalization_stats.npz"
    train_horizon = args.validation_horizon if args.training_objective == "rollout" else args.future_step
    train_set = RolloutWindowDataset(
        args.dataset_name,
        args.metadata,
        "train",
        args.history,
        train_horizon,
        args.stride,
        stats_path,
        args.max_train_windows,
        args.seed,
    )
    val_set = RolloutWindowDataset(
        args.dataset_name,
        args.metadata,
        "val",
        args.history,
        args.validation_horizon,
        args.stride,
        stats_path,
        args.max_val_windows,
        args.seed + 1,
    )
    if not train_set or not val_set:
        raise RuntimeError(f"Empty data: train={len(train_set)}, val={len(val_set)}")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    mean = torch.from_numpy(train_set.mean).to(device)
    std = torch.from_numpy(train_set.std).to(device)
    edges = torch.tensor(skeleton_edges, dtype=torch.long, device=device)
    model = make_model(args, int(metadata["num_joints"]), skeleton_edges, anchor_joint_index).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = checkpoint_config(args, metadata)
    history: list[dict] = []
    best_value = float("inf")
    best_epoch = 0
    start_epoch = 1
    warm_start_sha256 = sha256_file(args.warm_start) if args.warm_start else None

    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("config") != config:
            raise RuntimeError("Resume checkpoint configuration does not match the frozen run configuration")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint.get("history", []))
        best_value = float(checkpoint["best_val_physical_mpjpe_1_75_mm"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1
    elif args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture_signature") != config["architecture"]:
            raise RuntimeError(
                f"Warm-start architecture mismatch: {checkpoint.get('architecture_signature')} != {config['architecture']}"
            )
        model.load_state_dict(checkpoint["model_state"], strict=True)

    tracker = ProgressTracker(args.progress_file) if args.progress_file else None
    total_batches = args.epochs * len(train_loader)
    if tracker:
        tracker.start(
            stage="formal_revision_ltc_training",
            total=total_batches,
            extra={
                "dataset": args.dataset_name,
                "training_objective": args.training_objective,
                "architecture": config["architecture"],
                "anchor_joint_name": anchor_joint_name,
                "checkpoint_selection": config["checkpoint_selection"],
                "output_dir": str(args.output_dir),
                "seed": args.seed,
            },
        )

    global_batch = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        seen = 0
        for batch_index, batch in enumerate(train_loader, 1):
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if args.training_objective == "rollout":
                predictions = recursive_rollout(model, inputs, args.validation_horizon, args.future_step)
                loss, metrics = physical_rollout_loss(
                    inputs,
                    predictions,
                    targets,
                    mean,
                    std,
                    edges,
                    anchor_joint_index,
                    args,
                )
            else:
                predictions = model(inputs)
                loss, metrics = physical_short_loss(predictions, targets, mean, std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = inputs.shape[0]
            seen += batch_size
            global_batch += 1
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size
            if tracker and (batch_index % 20 == 0 or batch_index == len(train_loader)):
                tracker.update(
                    processed=global_batch,
                    success=global_batch,
                    failed=0,
                    current_item=f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)}",
                    extra={
                        "epoch": epoch,
                        "epochs_total": args.epochs,
                        "batch": batch_index,
                        "batches_per_epoch": len(train_loader),
                        "train_physical_mpjpe_mm": round(
                            totals["physical_mpjpe_1_75_mm" if args.training_objective == "rollout" else "physical_mpjpe_1_25_mm"]
                            / seen,
                            6,
                        ),
                        "best_epoch": best_epoch or None,
                        "best_val_physical_mpjpe_1_75_mm": None if best_epoch == 0 else round(best_value, 6),
                    },
                )
        train_metrics = {key: value / seen for key, value in totals.items()}
        val_metrics = evaluate_physical_75(
            model,
            val_loader,
            device,
            mean,
            std,
            edges,
            anchor_joint_index,
            args,
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        atomic_write_json(args.output_dir / "train_history.json", history)
        current_value = float(val_metrics["physical_mpjpe_1_75_mm"])
        if current_value < best_value:
            best_value = current_value
            best_epoch = epoch
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                history,
                best_value,
                best_epoch,
                config,
                warm_start_sha256,
            )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            history,
            best_value,
            best_epoch,
            config,
            warm_start_sha256,
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if best_epoch == 0:
        raise RuntimeError("Training produced no selected checkpoint")
    best_path = args.output_dir / "best.pt"
    result = {
        "status": "completed",
        "dataset": args.dataset_name,
        "representation": "common18",
        "method_name": (
            "Original LTC"
            if args.model == "ltc"
            else f"LTC-Topology ({args.adjacency_mode} adjacency)"
            + (" w/o spine2 anchor guidance" if args.disable_anchor_guidance else "")
        ),
        "training_objective": args.training_objective,
        "architecture": config["architecture"],
        "anchor_joint_index": anchor_joint_index,
        "anchor_joint_name": anchor_joint_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_physical_mpjpe_1_75_mm": best_value,
        "checkpoint_selection": config["checkpoint_selection"],
        "loss_coordinate_space": config["loss_coordinate_space"],
        "parameters": count_parameters(model),
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "warm_start_sha256": warm_start_sha256,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "history_file": str(args.output_dir / "train_history.json"),
        "history_sha256": sha256_file(args.output_dir / "train_history.json"),
        "config": config,
    }
    atomic_write_json(args.output_dir / "results.json", result)
    if tracker:
        tracker.finish(
            status="completed",
            extra={
                "best_epoch": best_epoch,
                "best_val_physical_mpjpe_1_75_mm": round(best_value, 6),
                "results_file": str(args.output_dir / "results.json"),
            },
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
