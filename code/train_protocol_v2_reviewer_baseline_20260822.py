from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.metrics import count_parameters
from src.progress import ProgressTracker
from src.reviewer_baselines import (
    BASELINE_SPECS,
    HumanMACCommon18,
    build_reviewer_baseline,
)
from train_protocol_v2_revision_20260821 import (
    atomic_write_json,
    evaluate_physical_75,
    physical_short_loss,
    set_deterministic_seed,
    sha256_file,
)
from train_protocol_v2_supplementary_rollout_20260820 import (
    RolloutWindowDataset,
    read_json,
    validate_protocol_v2_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled common18 baselines required by the reviewer."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--baseline", choices=sorted(BASELINE_SPECS), required=True)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--frozen-inventory-sha256", default="")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def checkpoint_config(
    args: argparse.Namespace, metadata: dict, source_path: Path
) -> dict[str, object]:
    spec = BASELINE_SPECS[args.baseline]
    return {
        "dataset_name": args.dataset_name,
        "representation": "common18",
        "method": args.baseline,
        "method_name": spec.name,
        "method_family": spec.family,
        "implementation_status": spec.implementation_status,
        "history": args.history,
        "future_step": args.future_step,
        "validation_horizon": args.validation_horizon,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "loss_coordinate_space": (
            "normalized_noise_mse" if args.baseline == "humanmac" else "denormalized_meter_mpjpe"
        ),
        "checkpoint_selection": "validation_recursive_75_frame_physical_mpjpe_mm",
        "diffusion_primary_metric_sampling": (
            "deterministic_zero_initialization" if args.baseline == "humanmac" else None
        ),
        "seed": args.seed,
        "manifest_sha256": str(metadata["split_manifest_sha256"]).lower(),
        "metadata_sha256": sha256_file(args.metadata),
        "normalization_sha256": sha256_file(args.data_root / "normalization_stats.npz"),
        "trainer_sha256": sha256_file(Path(__file__)),
        "model_source_sha256": sha256_file(source_path),
        "frozen_inventory_sha256": args.frozen_inventory_sha256.lower(),
        "deterministic_algorithms": True,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    best_value: float,
    best_epoch: int,
    config: dict[str, object],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_physical_mpjpe_1_75_mm": best_value,
            "config": config,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_deterministic_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(args.metadata)
    validate_protocol_v2_metadata(
        metadata,
        argparse.Namespace(
            history=args.history,
            future_step=args.future_step,
            horizon=args.validation_horizon,
            stride=args.stride,
            expected_manifest_sha256=args.expected_manifest_sha256,
        ),
    )
    if metadata.get("representation") != "common18" or int(
        metadata.get("num_joints", 0)
    ) != 18:
        raise RuntimeError("Reviewer baselines require the frozen common18 representation")
    if str(metadata["joint_names"][int(metadata["guidance_joint_index"])]) != "spine2":
        raise RuntimeError("Frozen common18 anchor must be spine2")

    stats_path = args.data_root / "normalization_stats.npz"
    train_set = RolloutWindowDataset(
        args.dataset_name,
        args.metadata,
        "train",
        args.history,
        args.future_step,
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
    skeleton_edges = [
        [int(source), int(target)] for source, target in metadata["skeleton_edges"]
    ]
    edges = torch.tensor(skeleton_edges, dtype=torch.long, device=device)
    anchor_index = int(metadata["guidance_joint_index"])
    model = build_reviewer_baseline(
        args.baseline,
        int(metadata["num_joints"]),
        args.history,
        args.future_step,
        skeleton_edges,
        args.hidden_size,
        args.num_layers,
        args.dropout,
    ).to(device)
    source_path = Path(__file__).resolve().parent / "src" / "reviewer_baselines.py"
    config = checkpoint_config(args, metadata, source_path)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history: list[dict] = []
    best_value = float("inf")
    best_epoch = 0
    start_epoch = 1
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if checkpoint.get("config") != config:
            raise RuntimeError("Resume checkpoint configuration changed")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint.get("history", []))
        best_value = float(checkpoint["best_val_physical_mpjpe_1_75_mm"])
        best_epoch = int(checkpoint["best_epoch"])
        start_epoch = int(checkpoint["epoch"]) + 1

    metric_args = argparse.Namespace(
        validation_horizon=args.validation_horizon,
        future_step=args.future_step,
        lambda_26_50=0.0,
        lambda_51_75=0.0,
        lambda_bone=0.0,
        lambda_anchor=0.0,
        lambda_velocity=0.0,
    )
    tracker = ProgressTracker(args.progress_file) if args.progress_file else None
    total_batches = args.epochs * len(train_loader)
    if tracker:
        tracker.start(
            stage="formal_reviewer_baseline_training",
            total=total_batches,
            extra={
                "dataset": args.dataset_name,
                "baseline": args.baseline,
                "family": BASELINE_SPECS[args.baseline].family,
                "checkpoint_selection": config["checkpoint_selection"],
                "output_dir": str(args.output_dir),
                "seed": args.seed,
            },
        )

    global_batch = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_physical = 0.0
        seen = 0
        for batch_index, batch in enumerate(train_loader, 1):
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(model, HumanMACCommon18):
                loss = model.training_loss(inputs, targets)
                with torch.no_grad():
                    predictions = model(inputs)
                    _, physical = physical_short_loss(predictions, targets, mean, std)
            else:
                predictions = model(inputs)
                loss, physical = physical_short_loss(predictions, targets, mean, std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_size = inputs.shape[0]
            seen += batch_size
            global_batch += 1
            total_loss += float(loss.detach().cpu()) * batch_size
            total_physical += physical["physical_mpjpe_1_25_mm"] * batch_size
            if tracker and (batch_index % 20 == 0 or batch_index == len(train_loader)):
                tracker.update(
                    processed=global_batch,
                    success=global_batch,
                    failed=0,
                    current_item=(
                        f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)}"
                    ),
                    extra={
                        "epoch": epoch,
                        "epochs_total": args.epochs,
                        "batch": batch_index,
                        "batches_per_epoch": len(train_loader),
                        "train_physical_mpjpe_mm": round(total_physical / seen, 6),
                        "best_epoch": best_epoch or None,
                        "best_val_physical_mpjpe_1_75_mm": (
                            None if best_epoch == 0 else round(best_value, 6)
                        ),
                    },
                )
        val_metrics = evaluate_physical_75(
            model,
            val_loader,
            device,
            mean,
            std,
            edges,
            anchor_index,
            metric_args,
        )
        record = {
            "epoch": epoch,
            "train_objective": total_loss / seen,
            "train_physical_mpjpe_1_25_mm": total_physical / seen,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        atomic_write_json(args.output_dir / "train_history.json", history)
        current = float(val_metrics["physical_mpjpe_1_75_mm"])
        if current < best_value:
            best_value = current
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
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)

    if best_epoch == 0:
        raise RuntimeError("Training produced no selected checkpoint")
    best_path = args.output_dir / "best.pt"
    result = {
        "status": "completed",
        "dataset": args.dataset_name,
        "representation": "common18",
        "method": args.baseline,
        "method_name": BASELINE_SPECS[args.baseline].name,
        "method_family": BASELINE_SPECS[args.baseline].family,
        "implementation_status": BASELINE_SPECS[args.baseline].implementation_status,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_physical_mpjpe_1_75_mm": best_value,
        "checkpoint_selection": config["checkpoint_selection"],
        "parameters": count_parameters(model),
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
