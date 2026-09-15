from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_protocol_v2_supplementary_rollout_20260820 as rollout_train  # noqa: E402
from scripts.evaluate_reviewer_minimum33_20260823 import (  # noqa: E402
    build_entries,
    hisrep_rollout,
    load_baseline_model,
    load_hisrep_model,
    load_ltc_model,
    read_json,
)


METHODS = {
    "Full LTC-Topology",
    "HisRepItself",
    "Controlled LSTM",
    "MSR-GCN common18 adaptation",
    "ST-Transformer common18 adaptation",
    "siMLPe common18 adaptation",
    "HumanMAC common18 adaptation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled common18 efficiency benchmark")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--throughput-batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_predictor(entry: dict, metadata: dict, mean: torch.Tensor, std: torch.Tensor, device: torch.device):
    if entry["kind"] == "hisrep":
        model, config = load_hisrep_model(entry, device)

        def predict(inputs_n: torch.Tensor) -> torch.Tensor:
            return hisrep_rollout(model, inputs_n * std + mean, config)

    elif entry["kind"] == "baseline":
        model = load_baseline_model(entry, metadata, device)

        def predict(inputs_n: torch.Tensor) -> torch.Tensor:
            return rollout_train.recursive_rollout(model, inputs_n, horizon=75, step=25)

    else:
        model = load_ltc_model(entry, metadata, device)

        def predict(inputs_n: torch.Tensor) -> torch.Tensor:
            return rollout_train.recursive_rollout(model, inputs_n, horizon=75, step=25)

    return model, predict


@torch.inference_mode()
def measure(predict, inputs: torch.Tensor, warmup: int, repeats: int, device: torch.device) -> tuple[float, float]:
    for _ in range(warmup):
        predict(inputs)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        predict(inputs)
    synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else float("nan")
    )
    return elapsed / repeats, peak_mb


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    plan = read_json(args.plan)
    entries = [
        entry
        for entry in build_entries(plan)
        if int(entry["seed"]) == 42 and entry["method"] in METHODS
    ]
    core_methods = {"Full LTC-Topology", "HisRepItself", "Controlled LSTM"}
    recent_methods = METHODS - core_methods
    core_entries = [entry for entry in entries if entry["method"] in core_methods]
    recent_entries = [entry for entry in entries if entry["method"] in recent_methods]
    if (
        len(core_entries) != 9
        or len(recent_entries) != 4
        or any(entry["dataset"] != "cmu" for entry in recent_entries)
    ):
        raise RuntimeError(
            "Expected three core methods on all datasets and four recent methods on representative CMU; "
            f"found {len(core_entries)} core and {len(recent_entries)} recent entries"
        )
    rows = []
    for dataset_name in ("cmu", "kit", "bmlmovi"):
        data_root = Path(plan["data_root"]) / dataset_name / "common18"
        metadata = read_json(data_root / "metadata.json")
        dataset = rollout_train.RolloutWindowDataset(
            dataset_name=str(metadata["dataset"]),
            metadata_path=data_root / "metadata.json",
            split="test",
            history=25,
            horizon=75,
            stride=5,
            stats_file=data_root / "normalization_stats.npz",
            max_windows=max(1, args.throughput_batch_size),
            seed=20260822,
        )
        samples = torch.stack([dataset[index]["inputs"] for index in range(min(len(dataset), args.throughput_batch_size))]).to(device)
        stats_payload = np.load(data_root / "normalization_stats.npz")
        mean = torch.from_numpy(stats_payload["mean"].astype(np.float32)[0]).to(device)
        std = torch.from_numpy(stats_payload["std"].astype(np.float32)[0]).to(device)
        for entry in [item for item in entries if item["dataset"] == dataset_name]:
            model, predict = make_predictor(entry, metadata, mean, std, device)
            repeats = min(args.repeats, 5) if entry["method"] == "HumanMAC common18 adaptation" else args.repeats
            latency_s, peak_single = measure(predict, samples[:1], args.warmup, repeats, device)
            batch_s, peak_batch = measure(predict, samples, args.warmup, repeats, device)
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": entry["method"],
                    "seed": 42,
                    "horizon_frames": 75,
                    "batch_size": int(samples.shape[0]),
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "latency_ms_per_sequence": latency_s * 1000.0,
                    "throughput_sequences_per_second": float(samples.shape[0]) / batch_s,
                    "peak_memory_single_mb": peak_single,
                    "peak_memory_batch_mb": peak_batch,
                    "device": str(device),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "common18_efficiency.csv", rows)
    payload = {
        "status": "completed",
        "plan": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "controlled_protocol": "common18, 25 observed frames, recursive 75-frame prediction, same device",
        "rows": rows,
    }
    (args.output_dir / "common18_efficiency.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
