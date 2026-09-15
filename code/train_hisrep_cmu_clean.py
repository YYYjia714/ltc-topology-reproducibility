from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
HISREP_ROOT = PROJECT_ROOT / "sota_models" / "HisRepItself"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "windows_cmu" / "metadata_cmu.json"
JOINT_ROOT = PROJECT_ROOT / "data" / "interim" / "joints"
PROGRESS_ROOT = PROJECT_ROOT / "runs" / "progress"
RUNS_ROOT = PROJECT_ROOT / "runs"
COMMON_JOINTS = list(range(4, 22))
MM_PER_METER = 1000.0

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HISREP_ROOT))
from src.progress import ProgressTracker  # noqa: E402
from model import AttModel  # type: ignore  # noqa: E402


@dataclass
class Config:
    dataset: str = "CMU"
    model: str = "HisRepItself"
    protocol: str = "clean retrain, SMPL-X common 18 joints"
    input_n: int = 50
    output_n: int = 25
    model_output_n: int = 25
    itera: int = 1
    kernel_size: int = 10
    in_features: int = 54
    d_model: int = 256
    num_stage: int = 12
    dct_n: int = 30
    batch_size: int = 128
    test_batch_size: int = 256
    epochs: int = 50
    lr: float = 5e-4
    lr_decay_factor: float = 0.1
    max_norm: float = 10000.0
    stride: int = 5
    seed: int = 42
    num_workers: int = 0
    progress_every: int = 50
    disable_tqdm: bool = False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cpu_byte_rng_state(state: Any, name: str) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
        raise TypeError(f"{name} RNG state must be a torch.Tensor, got {type(state).__name__}")
    if state.dtype != torch.uint8:
        raise TypeError(f"{name} RNG state must have dtype torch.uint8, got {state.dtype}")
    normalized = state.detach().to(device="cpu").contiguous()
    if normalized.ndim != 1:
        raise ValueError(f"{name} RNG state must be one-dimensional, got shape {tuple(normalized.shape)}")
    return normalized


def capture_rng_state() -> dict[str, Any]:
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [
            _cpu_byte_rng_state(state, f"CUDA device {index}")
            for index, state in enumerate(torch.cuda.get_rng_state_all())
        ]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": _cpu_byte_rng_state(torch.get_rng_state(), "Torch CPU"),
        "cuda": cuda_states,
    }


def restore_rng_state(rng_state: dict[str, Any], device: torch.device) -> None:
    required = {"python", "numpy", "torch"}
    missing = sorted(required.difference(rng_state))
    if missing:
        raise KeyError(f"Checkpoint RNG state is missing: {missing}")

    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(_cpu_byte_rng_state(rng_state["torch"], "Torch CPU"))

    cuda_states = rng_state.get("cuda")
    if device.type != "cuda" or cuda_states is None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Checkpoint contains CUDA RNG state, but CUDA is unavailable")
    expected_devices = torch.cuda.device_count()
    if len(cuda_states) != expected_devices:
        raise RuntimeError(
            f"CUDA RNG state count mismatch: checkpoint={len(cuda_states)}, available={expected_devices}"
        )
    normalized_cuda_states = [
        _cpu_byte_rng_state(state, f"CUDA device {index}")
        for index, state in enumerate(cuda_states)
    ]
    torch.cuda.set_rng_state_all(normalized_cuda_states)


def resolve_sequence_path(sequence: str, dataset_name: str) -> Path | None:
    candidates = [JOINT_ROOT / sequence, JOINT_ROOT / dataset_name / sequence]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=128)
def load_sequence(path: Path, joint_indices: tuple[int, ...] = tuple(COMMON_JOINTS)) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        key = "joints_root_relative" if "joints_root_relative" in data else "joints"
        joints = np.asarray(data[key], dtype=np.float32)
    return joints[:, joint_indices, :]


class CmuHisRepWindowDataset(Dataset):
    def __init__(
        self,
        metadata: dict[str, Any],
        dataset_name: str,
        split: str,
        input_n: int,
        output_n: int,
        stride: int,
        max_windows: int | None = None,
        seed: int = 42,
    ) -> None:
        self.split = split
        self.dataset_name = dataset_name
        self.input_n = input_n
        self.output_n = output_n
        self.total_n = input_n + output_n
        self.stride = stride
        self.joint_indices = tuple(int(index) for index in metadata.get("joint_indices_smplx25", COMMON_JOINTS))
        if len(self.joint_indices) != len(COMMON_JOINTS):
            raise RuntimeError(f"HisRepItself requires 18 joints, got {self.joint_indices}")
        self.sequence_paths: list[Path] = []
        self.sequence_names: list[str] = []
        self.index: list[tuple[int, int]] = []
        self.skipped: list[dict[str, str]] = []

        for item in metadata.get("sequences", []):
            if item.get("split") != split:
                continue
            seq_name = str(item.get("sequence", ""))
            seq_path = resolve_sequence_path(seq_name, dataset_name)
            if seq_path is None:
                self.skipped.append({"sequence": seq_name, "reason": "missing"})
                continue
            frames = int(item.get("frames", 0))
            if frames <= 0:
                frames = load_sequence(seq_path, self.joint_indices).shape[0]
            if frames < self.total_n:
                self.skipped.append({"sequence": seq_name, "reason": "too_short"})
                continue
            seq_idx = len(self.sequence_paths)
            self.sequence_paths.append(seq_path)
            self.sequence_names.append(seq_name)
            for start in range(0, frames - self.total_n + 1, stride):
                self.index.append((seq_idx, start))

        if max_windows is not None and max_windows > 0 and len(self.index) > max_windows:
            rng = random.Random(seed)
            self.index = rng.sample(self.index, max_windows)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq_idx, start = self.index[idx]
        sequence = load_sequence(self.sequence_paths[seq_idx], self.joint_indices)
        arr = sequence[start : start + self.total_n]
        return torch.from_numpy(arr)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(cfg: Config, device: torch.device) -> nn.Module:
    model = AttModel.AttModel(
        in_features=cfg.in_features,
        kernel_size=cfg.kernel_size,
        d_model=cfg.d_model,
        num_stage=cfg.num_stage,
        dct_n=cfg.dct_n,
    )
    return model.to(device)


def lr_decay(optimizer: torch.optim.Optimizer, lr_now: float, gamma: float) -> float:
    lr_now = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_now
    return lr_now


def _hisrep_future_from_output(out_all: torch.Tensor, cfg: Config, batch_size: int) -> torch.Tensor:
    if cfg.model_output_n == cfg.output_n and cfg.itera == 1:
        pred_all = out_all[:, :, 0].reshape(batch_size, cfg.kernel_size + cfg.output_n, len(COMMON_JOINTS), 3)
        return pred_all[:, cfg.kernel_size :]
    pred_flat = (
        out_all[:, cfg.kernel_size :]
        .transpose(1, 2)
        .reshape(batch_size, cfg.model_output_n * cfg.itera, len(COMMON_JOINTS), 3)
    )
    return pred_flat[:, : cfg.output_n]


def forward_hisrep(model: nn.Module, batch_m: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = batch_m.shape[0]
    batch_mm = batch_m.float() * MM_PER_METER
    src = batch_mm.reshape(batch_size, cfg.input_n + cfg.output_n, cfg.in_features)
    out_all = model(src, output_n=cfg.model_output_n, input_n=cfg.input_n, itera=cfg.itera)
    pred_future = _hisrep_future_from_output(out_all, cfg, batch_size)
    target_future = batch_mm[:, cfg.input_n : cfg.input_n + cfg.output_n]
    mpjpe = torch.mean(torch.linalg.norm(pred_future - target_future, dim=3))
    if cfg.model_output_n == cfg.output_n and cfg.itera == 1:
        target_sup = batch_mm[:, -cfg.output_n - cfg.kernel_size :]
        pred_all = out_all[:, :, 0].reshape(batch_size, cfg.kernel_size + cfg.output_n, len(COMMON_JOINTS), 3)
        loss = torch.mean(torch.linalg.norm(pred_all - target_sup, dim=3))
    else:
        loss = mpjpe
    return loss, mpjpe


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    progress: ProgressTracker | None = None,
    progress_base: int = 0,
    epoch: int = 0,
    phase: str = "train",
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_mpjpe = 0.0
    total_count = 0
    iterator = tqdm(
        loader,
        desc=f"{phase} epoch {epoch}",
        unit="batch",
        leave=False,
        disable=cfg.disable_tqdm,
        mininterval=5.0,
    )
    for batch_idx, batch in enumerate(iterator, start=1):
        batch = batch.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss, mpjpe = forward_hisrep(model, batch, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=cfg.max_norm)
            optimizer.step()
        else:
            with torch.no_grad():
                loss, mpjpe = forward_hisrep(model, batch, cfg)
        batch_size = int(batch.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_mpjpe += float(mpjpe.detach().cpu()) * batch_size
        total_count += batch_size
        if not cfg.disable_tqdm:
            iterator.set_postfix(loss=f"{total_loss / max(total_count, 1):.3f}", mpjpe=f"{total_mpjpe / max(total_count, 1):.3f}")
        should_update_progress = batch_idx == len(loader) or batch_idx % max(cfg.progress_every, 1) == 0
        if progress is not None and is_train and should_update_progress:
            processed = progress_base + batch_idx
            progress.update(
                processed=processed,
                success=processed,
                failed=0,
                current_item=f"epoch {epoch}/{cfg.epochs}, train batch {batch_idx}/{len(loader)}",
                extra={
                    "epoch": epoch,
                    "batch": batch_idx,
                    "batches_per_epoch": len(loader),
                    "train_loss_running": round(total_loss / max(total_count, 1), 6),
                    "train_mpjpe_running": round(total_mpjpe / max(total_count, 1), 6),
                },
            )
    return {
        "loss": total_loss / max(total_count, 1),
        "mpjpe": total_mpjpe / max(total_count, 1),
    }


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, lr: float, best_val: float, cfg: Config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "lr": lr,
            "best_val_mpjpe": best_val,
            "model_state": model.state_dict(),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "common_joints": COMMON_JOINTS,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    cfg = Config(
        dataset=args.dataset_name,
        input_n=args.input_n,
        output_n=args.output_n,
        model_output_n=args.model_output_n,
        itera=args.itera,
        dct_n=args.dct_n,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        lr=args.lr,
        stride=args.stride,
        seed=args.seed,
        num_workers=args.num_workers,
        progress_every=args.progress_every,
        disable_tqdm=args.disable_tqdm,
    )
    seed_everything(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    metadata_path = Path(args.metadata)
    metadata = read_json(metadata_path)
    if str(metadata.get("protocol_version", "")).startswith("2."):
        expected = {
            "history": cfg.input_n,
            "future": cfg.output_n,
            "stride": cfg.stride,
            "representation": "common18",
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
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress_path)
    progress = ProgressTracker(progress_path)

    print(f"device={device}")
    print(f"loading {cfg.dataset} train/val windows...")
    train_dataset = CmuHisRepWindowDataset(metadata, cfg.dataset, "train", cfg.input_n, cfg.output_n, cfg.stride, args.max_train_windows, cfg.seed)
    val_dataset = CmuHisRepWindowDataset(metadata, cfg.dataset, "val", cfg.input_n, cfg.output_n, cfg.stride, args.max_val_windows, cfg.seed)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"empty dataset: train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model = make_model(cfg, device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    lr_now = cfg.lr
    lr_gamma = cfg.lr_decay_factor ** (1.0 / cfg.epochs)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"train_windows={len(train_dataset)}, val_windows={len(val_dataset)}, params={num_params/1e6:.3f}M")

    history: list[dict[str, Any]] = []
    best_val = math.inf
    best_epoch = 0
    start_epoch = 1
    resume_epoch = 0
    resume_path: Path | None = Path(args.resume_from).resolve() if args.resume_from else None
    resume_rng_restored = False
    elapsed_before_resume = 0.0
    prior_progress_history = ""

    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        compatibility_keys = (
            "dataset",
            "input_n",
            "output_n",
            "model_output_n",
            "itera",
            "dct_n",
            "batch_size",
            "test_batch_size",
            "stride",
            "seed",
        )
        mismatches = {
            key: {"checkpoint": checkpoint_config.get(key), "current": getattr(cfg, key)}
            for key in compatibility_keys
            if checkpoint_config.get(key) != getattr(cfg, key)
        }
        if mismatches:
            raise RuntimeError(f"Resume checkpoint configuration mismatch: {mismatches}")

        resume_epoch = int(checkpoint.get("epoch", 0))
        if resume_epoch < 1 or resume_epoch >= cfg.epochs:
            raise RuntimeError(f"Invalid resume epoch {resume_epoch} for target epoch count {cfg.epochs}")
        model.load_state_dict(checkpoint.get("model_state", checkpoint.get("state_dict")), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_now = float(checkpoint["lr"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now
        best_val = float(checkpoint["best_val_mpjpe"])
        start_epoch = resume_epoch + 1

        history_path = run_dir / "train_history.json"
        if not history_path.is_file():
            raise FileNotFoundError(f"Training history required for resume: {history_path}")
        loaded_history = read_json(history_path)
        if not isinstance(loaded_history, list) or len(loaded_history) != resume_epoch:
            raise RuntimeError(
                f"Resume history length mismatch: checkpoint epoch={resume_epoch}, history rows={len(loaded_history)}"
            )
        if int(loaded_history[-1].get("epoch", -1)) != resume_epoch:
            raise RuntimeError("Resume history does not end at the checkpoint epoch")
        history = loaded_history
        best_epoch = int(history[-1]["best_epoch"])

        if progress_path.is_file():
            prior_progress = read_json(progress_path)
            elapsed_before_resume = float(prior_progress.get("elapsed_seconds", 0.0))
        if progress.history_path.is_file():
            prior_progress_history = progress.history_path.read_text(encoding="utf-8")

        rng_state = checkpoint.get("rng_state")
        if rng_state:
            restore_rng_state(rng_state, device)
            resume_rng_restored = True
        print(
            f"resuming from {resume_path}: completed_epoch={resume_epoch}, "
            f"next_epoch={start_epoch}, lr={lr_now:.12g}, best_val={best_val:.6f}, "
            f"rng_restored={resume_rng_restored}"
        )

    total_train_batches = cfg.epochs * len(train_loader)
    dataset_slug = cfg.dataset.lower().replace(" ", "_")
    stage_name = f"train_hisrep_{dataset_slug}_clean_{cfg.input_n}in{cfg.output_n}out_cuda"
    model_name = f"HisRepItself-{cfg.dataset}-clean-retrain-{cfg.input_n}in{cfg.output_n}out"
    progress.start(
        stage=stage_name,
        total=total_train_batches,
        extra={
            "dataset": cfg.dataset,
            "model": model_name,
            "device": str(device),
            "run_dir": str(run_dir),
            "input_n": cfg.input_n,
            "output_n": cfg.output_n,
            "model_output_n": cfg.model_output_n,
            "itera": cfg.itera,
            "dct_n": cfg.dct_n,
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
            "epochs_total": cfg.epochs,
            "best_epoch": None,
            "best_val_mpjpe": None,
            "resume_from": str(resume_path) if resume_path is not None else None,
            "resume_epoch": resume_epoch,
            "resume_rng_restored": resume_rng_restored,
        },
    )
    if resume_path is not None:
        if prior_progress_history:
            progress.history_path.write_text(prior_progress_history, encoding="utf-8")
        progress.state["started_at"] = (
            datetime.now() - timedelta(seconds=max(elapsed_before_resume, 0.0))
        ).isoformat(timespec="seconds")
        completed_batches = resume_epoch * len(train_loader)
        progress.update(
            processed=completed_batches,
            success=completed_batches,
            failed=0,
            current_item=f"resumed after epoch {resume_epoch}/{cfg.epochs}",
            extra={
                "epoch": resume_epoch,
                "epochs_completed": resume_epoch,
                "best_epoch": best_epoch,
                "best_val_mpjpe": round(best_val, 6),
                "lr": lr_now,
                "resume_from": str(resume_path),
                "resume_rng_restored": resume_rng_restored,
            },
        )
    write_json(
        run_dir / "config.json",
        {
            "config": asdict(cfg),
            "common_joints": COMMON_JOINTS,
            "metadata": {"path": str(metadata_path)},
            "resume": {
                "checkpoint": str(resume_path) if resume_path is not None else None,
                "completed_epoch": resume_epoch,
                "rng_restored": resume_rng_restored,
            },
        },
    )

    started = time.time()
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            lr_now = lr_decay(optimizer, lr_now, lr_gamma)
            progress_base = (epoch - 1) * len(train_loader)
            train_metrics = run_epoch(model, train_loader, cfg, device, optimizer, progress, progress_base, epoch, "train")
            val_metrics = run_epoch(model, val_loader, cfg, device, None, None, 0, epoch, "val")
            is_best = val_metrics["mpjpe"] < best_val
            if is_best:
                best_val = val_metrics["mpjpe"]
                best_epoch = epoch
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, lr_now, best_val, cfg)
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, lr_now, best_val, cfg)
            row = {
                "epoch": epoch,
                "lr": lr_now,
                "train_loss": train_metrics["loss"],
                "train_mpjpe": train_metrics["mpjpe"],
                "val_loss": val_metrics["loss"],
                "val_mpjpe": val_metrics["mpjpe"],
                "best_epoch": best_epoch,
                "best_val_mpjpe": best_val,
                "is_best": is_best,
            }
            history.append(row)
            write_json(run_dir / "train_history.json", history)
            write_json(
                run_dir / "results.json",
                {
                    "status": "running",
                    "dataset": cfg.dataset,
                    "model": model_name,
                    "input_n": cfg.input_n,
                    "output_n": cfg.output_n,
                    "best_epoch": best_epoch,
                    "best_val_mpjpe": best_val,
                    "epochs_completed": epoch,
                    "epochs_total": cfg.epochs,
                    "parameters": num_params,
                    "run_dir": str(run_dir),
                },
            )
            progress.update(
                processed=epoch * len(train_loader),
                success=epoch * len(train_loader),
                failed=0,
                current_item=f"epoch {epoch}/{cfg.epochs} complete, val MPJPE {val_metrics['mpjpe']:.3f} mm, best epoch {best_epoch}",
                extra={
                    "epoch": epoch,
                    "epochs_completed": epoch,
                    "train_mpjpe": round(train_metrics["mpjpe"], 6),
                    "val_mpjpe": round(val_metrics["mpjpe"], 6),
                    "best_epoch": best_epoch,
                    "best_val_mpjpe": round(best_val, 6),
                    "lr": lr_now,
                },
            )
        elapsed = elapsed_before_resume + (time.time() - started)
        write_json(
            run_dir / "results.json",
            {
                "status": "completed",
                "dataset": cfg.dataset,
                "model": model_name,
                "input_n": cfg.input_n,
                "output_n": cfg.output_n,
                "best_epoch": best_epoch,
                "best_val_mpjpe": best_val,
                "epochs_completed": cfg.epochs,
                "epochs_total": cfg.epochs,
                "parameters": num_params,
                "elapsed_seconds": elapsed,
                "run_dir": str(run_dir),
                "best_checkpoint": str(run_dir / "best.pt"),
                "resume_from": str(resume_path) if resume_path is not None else None,
                "resumed_after_epoch": resume_epoch,
                "resume_rng_restored": resume_rng_restored,
            },
        )
        progress.finish(
            "completed",
            extra={
                "best_epoch": best_epoch,
                "best_val_mpjpe": round(best_val, 6),
                "elapsed_seconds_total": round(elapsed, 2),
            },
        )
    except Exception as exc:
        progress.finish("failed", extra={"last_error": repr(exc)})
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean retrain HisRepItself on CMU root-relative common-18 joints.")
    parser.add_argument("--dataset-name", type=str, default="CMU")
    parser.add_argument("--metadata", type=str, default=str(METADATA_PATH))
    parser.add_argument("--input-n", type=int, default=50)
    parser.add_argument("--output-n", type=int, default=25)
    parser.add_argument("--model-output-n", type=int, default=25)
    parser.add_argument("--itera", type=int, default=1)
    parser.add_argument("--dct-n", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-manifest-sha256", default="")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=str(RUNS_ROOT / "hisrep_cmu_clean_50in25out_18j_cuda"))
    parser.add_argument("--progress-path", type=str, default=str(PROGRESS_ROOT / "train_hisrep_cmu_clean_50in25out_cuda.json"))
    parser.add_argument("--resume-from", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
