from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from train_hisrep_cmu_clean import capture_rng_state, restore_rng_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HisRep RNG checkpoint recovery.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    original_state = capture_rng_state()
    cpu_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restore_rng_state(cpu_checkpoint["rng_state"], torch.device("cpu"))
    cpu_sample_a = (random.random(), np.random.random(), torch.rand(8))
    restore_rng_state(cpu_checkpoint["rng_state"], torch.device("cpu"))
    cpu_sample_b = (random.random(), np.random.random(), torch.rand(8))
    if cpu_sample_a[:2] != cpu_sample_b[:2] or not torch.equal(cpu_sample_a[2], cpu_sample_b[2]):
        raise AssertionError("CPU RNG restoration is not deterministic")
    corrupt_state = dict(cpu_checkpoint["rng_state"])
    corrupt_state["torch"] = corrupt_state["torch"].float()
    try:
        restore_rng_state(corrupt_state, torch.device("cpu"))
    except TypeError:
        pass
    else:
        raise AssertionError("A non-uint8 RNG state was accepted")

    report: dict[str, object] = {
        "status": "passed",
        "checkpoint": str(checkpoint_path),
        "cpu_restore": "passed",
        "cpu_determinism": "passed",
        "invalid_dtype_rejected": "passed",
        "cuda_available": torch.cuda.is_available(),
        "cuda_map_location_restore": "not_run",
        "cuda_determinism": "not_run",
    }
    try:
        if torch.cuda.is_available():
            cuda_checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            loaded_states = cuda_checkpoint["rng_state"].get("cuda")
            if loaded_states is None or not all(state.is_cuda for state in loaded_states):
                raise AssertionError("CUDA map_location did not reproduce GPU-resident RNG states")
            restore_rng_state(cuda_checkpoint["rng_state"], torch.device("cuda"))
            cuda_sample_a = torch.rand(8, device="cuda")
            restore_rng_state(cuda_checkpoint["rng_state"], torch.device("cuda"))
            cuda_sample_b = torch.rand(8, device="cuda")
            if not torch.equal(cuda_sample_a, cuda_sample_b):
                raise AssertionError("CUDA RNG restoration is not deterministic")
            report["cuda_map_location_restore"] = "passed"
            report["cuda_determinism"] = "passed"
    finally:
        restore_rng_state(
            original_state,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
