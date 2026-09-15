from __future__ import annotations

import argparse
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.metrics import count_parameters, measure_inference_time
from src.models import MotionForecastModel
from src.progress import ProgressTracker
from src.topology import get_edges


PROJECT_ROOT = Path(__file__).resolve().parent
JOINT_ROOT = PROJECT_ROOT / "data/interim/joints"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol_v2_metadata(metadata: dict, args: argparse.Namespace) -> None:
    if not str(metadata.get("protocol_version", "")).startswith("2."):
        return
    expected = {
        "history": args.history,
        "future": args.future_step,
        "rollout_horizon": args.horizon,
        "stride": args.stride,
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


def resolve_sequence_path(dataset_name: str, sequence: str) -> Path:
    candidates = [
        JOINT_ROOT / sequence,
        JOINT_ROOT / dataset_name / sequence,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Sequence not found: {sequence}")


@lru_cache(maxsize=512)
def load_joints(
    path_text: str,
    center_on_first_frame: bool = False,
    joint_indices: tuple[int, ...] = tuple(range(25)),
) -> np.ndarray:
    with np.load(path_text, allow_pickle=True) as data:
        key = "joints_root_relative" if "joints_root_relative" in data else "joints"
        joints = np.asarray(data[key], dtype=np.float32)
        if center_on_first_frame:
            joints = joints - joints[0:1]
        return joints[:, joint_indices, :]


class RolloutWindowDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        metadata_path: Path,
        split: str,
        history: int,
        horizon: int,
        stride: int,
        stats_file: Path,
        max_windows: int = 0,
        seed: int = 42,
    ) -> None:
        self.dataset_name = dataset_name
        self.history = history
        self.horizon = horizon
        self.stride = stride
        stats = np.load(stats_file, allow_pickle=True)
        self.mean = stats["mean"].astype(np.float32)[0]
        self.std = stats["std"].astype(np.float32)[0]

        metadata = read_json(metadata_path)
        self.center_on_first_frame = bool(metadata.get("center_on_first_frame", False))
        self.joint_indices = tuple(int(index) for index in metadata["joint_indices_smplx25"])
        records: list[tuple[str, int]] = []
        for item in metadata["sequences"]:
            if item["split"] != split:
                continue
            sequence_path = resolve_sequence_path(dataset_name, item["sequence"])
            frames = int(item.get("frames", 0))
            if frames <= 0:
                frames = load_joints(
                    str(sequence_path),
                    self.center_on_first_frame,
                    self.joint_indices,
                ).shape[0]
            limit = frames - history - horizon + 1
            if limit <= 0:
                continue
            for start in range(0, limit, stride):
                records.append((str(sequence_path), start))

        if max_windows and len(records) > max_windows:
            rng = random.Random(seed)
            rng.shuffle(records)
            records = records[:max_windows]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        sequence_path, start = self.records[index]
        joints = load_joints(sequence_path, self.center_on_first_frame, self.joint_indices)
        clip = joints[start : start + self.history + self.horizon]
        inputs = (clip[: self.history] - self.mean) / self.std
        targets = (clip[self.history :] - self.mean) / self.std
        return {
            "inputs": torch.from_numpy(inputs.astype(np.float32)),
            "targets": torch.from_numpy(targets.astype(np.float32)),
        }


def recursive_rollout(model: torch.nn.Module, inputs: torch.Tensor, horizon: int, step: int) -> torch.Tensor:
    current = inputs
    chunks = []
    generated = 0
    while generated < horizon:
        pred = model(current)
        needed = min(step, horizon - generated)
        chunks.append(pred[:, :needed])
        generated += needed
        current = torch.cat([current[:, needed:], pred[:, :needed]], dim=1)
    return torch.cat(chunks, dim=1)


def mpjpe(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(predictions - targets, dim=-1).mean()


def bone_length_loss(predictions: torch.Tensor, targets: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return predictions.new_tensor(0.0)
    pred_bones = predictions[:, :, edges[:, 0], :] - predictions[:, :, edges[:, 1], :]
    tgt_bones = targets[:, :, edges[:, 0], :] - targets[:, :, edges[:, 1], :]
    return torch.abs(torch.linalg.norm(pred_bones, dim=-1) - torch.linalg.norm(tgt_bones, dim=-1)).mean()


def root_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    guidance_joint_index: int = 0,
) -> torch.Tensor:
    return torch.linalg.norm(
        predictions[:, :, guidance_joint_index, :] - targets[:, :, guidance_joint_index, :],
        dim=-1,
    ).mean()


def velocity_loss(inputs: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_full = torch.cat([inputs[:, -1:, :, :], predictions], dim=1)
    tgt_full = torch.cat([inputs[:, -1:, :, :], targets], dim=1)
    pred_velocity = pred_full[:, 1:] - pred_full[:, :-1]
    tgt_velocity = tgt_full[:, 1:] - tgt_full[:, :-1]
    return torch.linalg.norm(pred_velocity - tgt_velocity, dim=-1).mean()


def rollout_loss(
    inputs: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    edges: torch.Tensor,
    lambda_26_50: float,
    lambda_51_75: float,
    lambda_bone: float,
    lambda_root: float,
    lambda_velocity: float,
    guidance_joint_index: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_1_25 = mpjpe(predictions[:, :25], targets[:, :25])
    loss_26_50 = mpjpe(predictions[:, 25:50], targets[:, 25:50])
    loss_51_75 = mpjpe(predictions[:, 50:75], targets[:, 50:75])
    loss_bone = bone_length_loss(predictions, targets, edges)
    loss_root = root_loss(predictions, targets, guidance_joint_index=guidance_joint_index)
    loss_velocity = velocity_loss(inputs, predictions, targets)
    total = (
        loss_1_25
        + lambda_26_50 * loss_26_50
        + lambda_51_75 * loss_51_75
        + lambda_bone * loss_bone
        + lambda_root * loss_root
        + lambda_velocity * loss_velocity
    )
    metrics = {
        "loss_total": float(total.detach().cpu()),
        "mpjpe_1_25": float(loss_1_25.detach().cpu()),
        "mpjpe_26_50": float(loss_26_50.detach().cpu()),
        "mpjpe_51_75": float(loss_51_75.detach().cpu()),
        "mpjpe_1_75": float(mpjpe(predictions, targets).detach().cpu()),
        "bone": float(loss_bone.detach().cpu()),
        "root": float(loss_root.detach().cpu()),
        "velocity": float(loss_velocity.detach().cpu()),
    }
    return total, metrics


def make_model(
    args: argparse.Namespace,
    joints: int,
    history: int,
    future: int,
    skeleton_edges: list[list[int]],
    guidance_joint_index: int,
) -> MotionForecastModel:
    return MotionForecastModel(
        model_type="ltc_topology",
        joints=joints,
        history=history,
        future=future,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_topology_encoder=True,
        use_topology_decoder=True,
        use_root_guidance=True,
        skeleton_edges=skeleton_edges,
        guidance_joint_index=guidance_joint_index,
    )


@torch.no_grad()
def evaluate_rollout(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    horizon: int,
    step: int,
    edges: torch.Tensor,
    guidance_joint_index: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        predictions = recursive_rollout(model, inputs, horizon=horizon, step=step)
        _, metrics = rollout_loss(
            inputs,
            predictions,
            targets,
            edges,
            args.lambda_26_50,
            args.lambda_51_75,
            args.lambda_bone,
            args.lambda_root,
            args.lambda_velocity,
            guidance_joint_index,
        )
        batch_size = inputs.size(0)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        count += batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(output_dir: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight rollout consistency training for LTC-Topology.")
    parser.add_argument("--dataset-name", default="CMU")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/windows_cmu")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/windows_cmu/metadata_cmu.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument("--progress-file", type=Path, default=None)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--future-step", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=75)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-train-windows", type=int, default=8192)
    parser.add_argument("--max-val-windows", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lambda-26-50", type=float, default=0.7)
    parser.add_argument("--lambda-51-75", type=float, default=0.5)
    parser.add_argument("--lambda-bone", type=float, default=0.1)
    parser.add_argument("--lambda-root", type=float, default=0.1)
    parser.add_argument("--lambda-velocity", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-manifest-sha256", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_file = args.data_root / "normalization_stats.npz"
    train_set = RolloutWindowDataset(
        dataset_name=args.dataset_name,
        metadata_path=args.metadata,
        split="train",
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        stats_file=stats_file,
        max_windows=args.max_train_windows,
        seed=args.seed,
    )
    val_set = RolloutWindowDataset(
        dataset_name=args.dataset_name,
        metadata_path=args.metadata,
        split="val",
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        stats_file=stats_file,
        max_windows=args.max_val_windows,
        seed=args.seed + 1,
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    metadata = read_json(args.metadata)
    validate_protocol_v2_metadata(metadata, args)
    joints = int(metadata["num_joints"])
    skeleton_edges = [[int(src), int(dst)] for src, dst in metadata["skeleton_edges"]]
    guidance_joint_index = int(metadata.get("guidance_joint_index", 0))
    model = make_model(
        args,
        joints=joints,
        history=args.history,
        future=args.future_step,
        skeleton_edges=skeleton_edges,
        guidance_joint_index=guidance_joint_index,
    ).to(device)
    if args.warm_start is not None:
        checkpoint = torch.load(args.warm_start, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    edges = torch.tensor(skeleton_edges, dtype=torch.long, device=device)
    tracker = ProgressTracker(args.progress_file) if args.progress_file is not None else None
    total_steps = args.epochs * len(train_loader)
    if tracker is not None:
        tracker.start(
            stage="train_ltc_topology_rollout_consistency",
            total=total_steps,
            extra={
                "dataset": args.dataset_name,
                "output_dir": str(args.output_dir),
                "warm_start": None if args.warm_start is None else str(args.warm_start),
                "train_windows": len(train_set),
                "val_windows": len(val_set),
                "horizon": args.horizon,
                "lambda_26_50": args.lambda_26_50,
                "lambda_51_75": args.lambda_51_75,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "pin_memory": bool(args.pin_memory and device.type == "cuda"),
            },
        )

    history: list[dict] = []
    best_val = float("inf")
    best_epoch = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: dict[str, float] = {}
        seen = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = recursive_rollout(model, inputs, horizon=args.horizon, step=args.future_step)
            loss, metrics = rollout_loss(
                inputs,
                predictions,
                targets,
                edges,
                args.lambda_26_50,
                args.lambda_51_75,
                args.lambda_bone,
                args.lambda_root,
                args.lambda_velocity,
                guidance_joint_index,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = inputs.size(0)
            seen += batch_size
            global_step += 1
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value * batch_size
            if tracker is not None and (batch_index == len(train_loader) or batch_index % 20 == 0):
                train_mpjpe = running["mpjpe_1_75"] / max(seen, 1)
                tracker.update(
                    processed=global_step,
                    success=global_step,
                    failed=0,
                    current_item=f"epoch {epoch}/{args.epochs} batch {batch_index}/{len(train_loader)}",
                    extra={
                        "epoch": epoch,
                        "epochs_total": args.epochs,
                        "batch": batch_index,
                        "batches_per_epoch": len(train_loader),
                        "train_mpjpe_1_75": round(train_mpjpe, 6),
                        "best_val_mpjpe_1_75": None if best_epoch == 0 else round(best_val, 6),
                        "best_epoch": best_epoch,
                    },
                )

        train_metrics = {key: value / max(seen, 1) for key, value in running.items()}
        val_metrics = evaluate_rollout(
            model,
            val_loader,
            device,
            args.horizon,
            args.future_step,
            edges,
            guidance_joint_index,
            args,
        )
        record = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        (args.output_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)

        if val_metrics["mpjpe_1_75"] < best_val:
            best_val = val_metrics["mpjpe_1_75"]
            best_epoch = epoch
            save_checkpoint(args.output_dir, model, optimizer, epoch, {"val_mpjpe_1_75": best_val, **val_metrics})

    last_batch = next(iter(val_loader))["inputs"][:1].to(device)
    inference_seconds = measure_inference_time(model, last_batch)
    result = {
        "dataset": args.dataset_name,
        "model": "ltc_topology",
        "method_name": "Full LTC-Topology with rollout consistency training",
        "epochs_total": args.epochs,
        "best_epoch": best_epoch,
        "best_epoch_label": f"{best_epoch}*" if best_epoch == args.epochs else str(best_epoch),
        "best_at_final_epoch": best_epoch == args.epochs,
        "best_val_mpjpe": best_val,
        "best_val_mpjpe_1_75": best_val,
        "parameters": count_parameters(model),
        "inference_seconds": inference_seconds,
        "samples_per_second": 1.0 / inference_seconds if inference_seconds > 0 else 0.0,
        "optimizer": "adam",
        "normalized": True,
        "warm_start": None if args.warm_start is None else str(args.warm_start),
        "rollout_consistency": {
            "horizon": args.horizon,
            "future_step": args.future_step,
            "lambda_26_50": args.lambda_26_50,
            "lambda_51_75": args.lambda_51_75,
            "lambda_bone": args.lambda_bone,
            "lambda_root": args.lambda_root,
            "lambda_velocity": args.lambda_velocity,
            "train_windows": len(train_set),
            "val_windows": len(val_set),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": bool(args.pin_memory and device.type == "cuda"),
            "guidance_joint_index": guidance_joint_index,
            "skeleton_edges": skeleton_edges,
        },
    }
    (args.output_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if tracker is not None:
        tracker.finish(
            status="completed",
            extra={
                "best_epoch": best_epoch,
                "best_val_mpjpe_1_75": round(best_val, 6),
                "results_file": str(args.output_dir / "results.json"),
                "history_file": str(args.output_dir / "train_history.json"),
            },
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
