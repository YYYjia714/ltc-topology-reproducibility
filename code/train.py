from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import ForecastingDataset
from src.models import MotionForecastModel
from src.progress import ProgressTracker
from src.runtime import default_device_arg, resolve_runtime_device, should_pin_memory
from src.trainer import evaluate, save_checkpoint, summarize_model, train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train motion forecasting baselines.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=["ltc", "lstm", "gru", "ltc_topology"], default="ltc")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--stats-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-manifest-sha256", default="")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--device", default=default_device_arg())
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--progress-update-interval", type=int, default=50)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--disable-topology-encoder", action="store_true")
    parser.add_argument("--disable-topology-decoder", action="store_true")
    parser.add_argument("--disable-root-guidance", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: ForecastingDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def configure_runtime(args: argparse.Namespace, device_backend: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    elif device_backend == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def load_existing_history(output_dir: Path) -> list[dict]:
    history_path = output_dir / "train_history.json"
    if not history_path.exists():
        return []
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def save_last_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict],
    best_val: float,
    epochs_without_improvement: int,
    current_metrics: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_val_mpjpe": best_val,
            "epochs_without_improvement": epochs_without_improvement,
            "current_metrics": current_metrics,
        },
        output_dir / "last.pt",
    )


def main() -> None:
    args = parse_args()
    device, device_backend = resolve_runtime_device(args.device)
    configure_runtime(args, device_backend)
    set_seed(args.seed)
    pin_memory = should_pin_memory(device_backend)
    metadata_path = args.metadata or (args.data_root / "metadata.json")
    metadata = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("protocol_version", "")).startswith("2."):
            expected = {
                "history": 25,
                "future": 25,
                "stride": 5,
                "coordinate_source": "joints_root_relative",
                "center_on_first_frame": False,
                "split_unit": "subject",
            }
            mismatches = {
                key: {"expected": value, "actual": metadata.get(key)}
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            expected_hash = args.expected_manifest_sha256.strip().lower()
            actual_hash = str(metadata.get("split_manifest_sha256", "")).lower()
            if expected_hash and actual_hash != expected_hash:
                mismatches["split_manifest_sha256"] = {
                    "expected": expected_hash,
                    "actual": actual_hash,
                }
            if mismatches:
                raise RuntimeError(f"Protocol v2 metadata mismatch: {mismatches}")

    train_set = ForecastingDataset(
        args.data_root / "train.npz",
        normalize=args.normalize,
        stats_file=args.stats_file,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_set = ForecastingDataset(
        args.data_root / "val.npz",
        normalize=args.normalize,
        stats_file=args.stats_file,
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
    )
    test_set = ForecastingDataset(
        args.data_root / "test.npz",
        normalize=args.normalize,
        stats_file=args.stats_file,
        max_samples=args.max_test_samples,
        seed=args.seed + 2,
    )

    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers, pin_memory)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers, pin_memory)
    test_loader = make_loader(test_set, args.batch_size, False, args.num_workers, pin_memory)

    skeleton_edges = None if metadata is None else metadata.get("skeleton_edges")
    guidance_joint_index = 0 if metadata is None else int(metadata.get("guidance_joint_index", 0))
    model = MotionForecastModel(
        model_type=args.model,
        joints=train_set.info.joints,
        history=train_set.info.history,
        future=train_set.info.future,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_topology_encoder=not args.disable_topology_encoder,
        use_topology_decoder=not args.disable_topology_decoder,
        use_root_guidance=not args.disable_root_guidance,
        skeleton_edges=skeleton_edges,
        guidance_joint_index=guidance_joint_index,
    ).to(device)

    optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if device_backend == "dml":
        # DirectML currently falls back to CPU for foreach AdamW updates.
        optimizer_kwargs["foreach"] = False
    optimizer_cls = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    print(f"Using device backend: {device_backend}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_val = float("inf")
    history = []
    epochs_without_improvement = 0
    if args.resume_from is not None:
        resume_path = args.resume_from.expanduser()
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("best_val_mpjpe", best_val))
        history = checkpoint.get("history", []) or load_existing_history(args.output_dir)
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(
            f"Resuming from {resume_path} at epoch {start_epoch:03d} with best val {best_val:.6f}",
            flush=True,
        )
    tracker = None
    train_batches_per_epoch = len(train_loader)
    total_train_steps = args.epochs * train_batches_per_epoch
    if args.progress_file is not None:
        tracker = ProgressTracker(args.progress_file)
        tracker.start(
            stage=f"train_{args.model}",
            total=total_train_steps,
            extra={
                "output_dir": str(args.output_dir),
                "data_root": str(args.data_root),
                "device_backend": device_backend,
                "model": args.model,
                "use_topology_encoder": not args.disable_topology_encoder,
                "use_topology_decoder": not args.disable_topology_decoder,
                "use_root_guidance": not args.disable_root_guidance,
                "best_val_mpjpe": None,
                "best_epoch": None,
                "epochs_total": args.epochs,
                "batches_per_epoch": train_batches_per_epoch,
                "resume_from": None if args.resume_from is None else str(args.resume_from),
                "resume_epoch": start_epoch,
                "metadata": str(metadata_path) if metadata_path.exists() else None,
                "manifest_sha256": None if metadata is None else metadata.get("split_manifest_sha256"),
            },
        )

    for epoch in range(start_epoch, args.epochs + 1):
        def on_train_batch(batch_index: int, total_batches: int, running_train_loss: float) -> None:
            if tracker is None:
                return
            if batch_index != total_batches and batch_index % max(1, args.progress_update_interval) != 0:
                return
            global_step = (epoch - 1) * train_batches_per_epoch + batch_index
            tracker.update(
                processed=global_step,
                success=global_step,
                failed=0,
                current_item=f"epoch_{epoch:03d} batch_{batch_index:04d}/{total_batches:04d}",
                extra={
                    "current_epoch": epoch,
                    "current_batch": batch_index,
                    "epochs_total": args.epochs,
                    "batches_per_epoch": total_batches,
                    "train_mpjpe": round(running_train_loss, 6),
                    "val_mpjpe": None,
                    "best_val_mpjpe": None if best_val == float('inf') else round(best_val, 6),
                    "best_epoch": checkpoint_epoch_from_history(history, best_val),
                    "epochs_without_improvement": epochs_without_improvement,
                },
            )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            progress_callback=on_train_batch,
        )
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        val_loss = evaluate(model, val_loader, device) if should_eval else None
        record = {"epoch": epoch, "train_mpjpe": train_loss, "val_mpjpe": val_loss}
        history.append(record)
        if val_loss is None:
            print(f"Epoch {epoch:03d} | train MPJPE={train_loss:.6f} | val MPJPE=skipped", flush=True)
            save_last_checkpoint(
                args.output_dir,
                model,
                optimizer,
                epoch,
                history,
                best_val,
                epochs_without_improvement,
                {
                    "epoch": epoch,
                    "train_mpjpe": train_loss,
                    "val_mpjpe": None,
                },
            )
            if tracker is not None:
                global_step = epoch * train_batches_per_epoch
                tracker.update(
                    processed=global_step,
                    success=global_step,
                    failed=0,
                    current_item=f"epoch_{epoch:03d} completed",
                    extra={
                        "current_epoch": epoch,
                        "current_batch": train_batches_per_epoch,
                        "epochs_total": args.epochs,
                        "batches_per_epoch": train_batches_per_epoch,
                        "train_mpjpe": round(train_loss, 6),
                        "val_mpjpe": None,
                        "best_val_mpjpe": None if best_val == float("inf") else round(best_val, 6),
                        "best_epoch": checkpoint_epoch_from_history(history, best_val),
                    },
                )
            continue

        print(
            f"Epoch {epoch:03d} | train MPJPE={train_loss:.6f} | val MPJPE={val_loss:.6f}",
            flush=True,
        )
        if val_loss < (best_val - args.min_delta):
            best_val = val_loss
            epochs_without_improvement = 0
            metrics = {
                "best_epoch": epoch,
                "train_mpjpe": train_loss,
                "val_mpjpe": val_loss,
            }
            save_checkpoint(args.output_dir, model, optimizer, epoch, metrics)
        else:
            epochs_without_improvement += 1
        save_last_checkpoint(
            args.output_dir,
            model,
            optimizer,
            epoch,
            history,
            best_val,
            epochs_without_improvement,
            {
                "epoch": epoch,
                "train_mpjpe": train_loss,
                "val_mpjpe": val_loss,
            },
        )
        if val_loss is not None:
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                if tracker is not None:
                    global_step = epoch * train_batches_per_epoch
                    tracker.update(
                        processed=global_step,
                        success=global_step,
                        failed=0,
                        current_item=f"epoch_{epoch:03d} early_stop",
                        extra={
                            "current_epoch": epoch,
                            "current_batch": train_batches_per_epoch,
                            "epochs_total": args.epochs,
                            "batches_per_epoch": train_batches_per_epoch,
                            "train_mpjpe": round(train_loss, 6),
                            "val_mpjpe": round(val_loss, 6),
                            "best_val_mpjpe": None if best_val == float("inf") else round(best_val, 6),
                            "best_epoch": checkpoint_epoch_from_history(history, best_val),
                            "epochs_without_improvement": epochs_without_improvement,
                        },
                    )
                print(
                    f"Early stopping at epoch {epoch:03d} after {epochs_without_improvement} validations without improvement.",
                    flush=True,
                )
                break
        if tracker is not None:
            global_step = epoch * train_batches_per_epoch
            tracker.update(
                processed=global_step,
                success=global_step,
                failed=0,
                current_item=f"epoch_{epoch:03d} completed",
                extra={
                    "current_epoch": epoch,
                    "current_batch": train_batches_per_epoch,
                    "epochs_total": args.epochs,
                    "batches_per_epoch": train_batches_per_epoch,
                    "train_mpjpe": round(train_loss, 6),
                    "val_mpjpe": round(val_loss, 6),
                    "best_val_mpjpe": None if best_val == float("inf") else round(best_val, 6),
                    "best_epoch": checkpoint_epoch_from_history(history, best_val),
                    "epochs_without_improvement": epochs_without_improvement,
                },
            )

    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    test_mpjpe = evaluate(model, test_loader, device)
    summary = summarize_model(model, test_loader, device)
    result = {
        "model": args.model,
        "history": train_set.info.history,
        "future": train_set.info.future,
        "joints": train_set.info.joints,
        "normalized": args.normalize,
        "optimizer": args.optimizer,
        "use_topology_encoder": not args.disable_topology_encoder,
        "use_topology_decoder": not args.disable_topology_decoder,
        "use_root_guidance": not args.disable_root_guidance,
        "guidance_joint_index": guidance_joint_index,
        "skeleton_edges": skeleton_edges,
        "metadata": str(metadata_path) if metadata_path.exists() else None,
        "manifest_sha256": None if metadata is None else metadata.get("split_manifest_sha256"),
        "best_epoch": checkpoint["epoch"],
        "best_epoch_label": f"{checkpoint['epoch']}*" if checkpoint["epoch"] == args.epochs else str(checkpoint["epoch"]),
        "best_at_final_epoch": checkpoint["epoch"] == args.epochs,
        "best_val_mpjpe": checkpoint["metrics"]["val_mpjpe"],
        "test_mpjpe": test_mpjpe,
        "parameters": summary["parameters"],
        "inference_seconds": summary["inference_seconds"],
        "samples_per_second": summary["samples_per_second"],
    }
    (args.output_dir / "train_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if tracker is not None:
        tracker.finish(
            status="completed",
            extra={
                "best_epoch": checkpoint["epoch"],
                "best_val_mpjpe": round(checkpoint["metrics"]["val_mpjpe"], 6),
                "test_mpjpe": round(test_mpjpe, 6),
                "results_file": str(args.output_dir / "results.json"),
                "history_file": str(args.output_dir / "train_history.json"),
            },
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def checkpoint_epoch_from_history(history: list[dict], best_val: float) -> int | None:
    if best_val == float("inf"):
        return None
    for item in reversed(history):
        if item["val_mpjpe"] == best_val:
            return int(item["epoch"])
    return None


if __name__ == "__main__":
    main()
