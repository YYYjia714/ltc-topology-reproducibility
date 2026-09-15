from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology import JOINT_NAMES_25, SKELETON_EDGES_25


SPLITS = ("train", "val", "test")
COMMON_JOINTS = list(range(4, 22))
COMMON_EDGES = sorted(
    {
        (left - 4, right - 4)
        for left, right in SKELETON_EDGES_25
        if left in COMMON_JOINTS and right in COMMON_JOINTS
    }
    | {(0, 2), (1, 2)}
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen Protocol v2 windows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("CMU", "KIT", "BMLmovi"), required=True)
    parser.add_argument("--representation", choices=("native25", "common18"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--future", type=int, default=25)
    parser.add_argument("--rollout-horizon", type=int, default=75)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path, dataset: str) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["dataset"] == dataset]
    if not rows:
        raise RuntimeError(f"No {dataset} rows in {path}")
    sequence_ids = [row["sequence_id"] for row in rows]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise RuntimeError(f"Duplicate sequence IDs in {dataset} manifest")
    subject_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["split"] not in SPLITS:
            raise RuntimeError(f"Invalid split: {row}")
        subject_splits[row["subject_id"]].add(row["split"])
    overlap = [subject for subject, splits in subject_splits.items() if len(splits) > 1]
    if overlap:
        raise RuntimeError(f"Subject overlap in frozen manifest: {overlap[:5]}")
    return sorted(rows, key=lambda row: row["sequence_id"])


def resolve_sequence_path(dataset: str, sequence: str) -> Path:
    root = PROJECT_ROOT / "data/interim/joints"
    candidates = [root / sequence, root / dataset / sequence]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(sequence)


def count_windows(frames: int, history: int, future: int, stride: int) -> int:
    total = history + future
    return 0 if frames < total else ((frames - total) // stride) + 1


def load_root_relative(path: Path, joint_indices: list[int]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "joints_root_relative" not in payload:
            raise KeyError(f"joints_root_relative missing: {path}")
        joints = np.asarray(payload["joints_root_relative"], dtype=np.float32)
        root_index = int(np.asarray(payload.get("root_index", 0)).reshape(-1)[0])
    if root_index != 0:
        raise RuntimeError(f"Expected root index 0, got {root_index}: {path}")
    return joints[:, joint_indices, :]


def create_writer(
    output_root: Path,
    split: str,
    count: int,
    history: int,
    future: int,
    joints: int,
    sequence_width: int,
    subject_width: int,
) -> dict:
    return {
        "inputs": open_memmap(
            output_root / f"{split}_inputs.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, history, joints, 3),
        ),
        "targets": open_memmap(
            output_root / f"{split}_targets.npy",
            mode="w+",
            dtype=np.float32,
            shape=(count, future, joints, 3),
        ),
        "sequence_ids": open_memmap(
            output_root / f"{split}_sequence_ids.npy",
            mode="w+",
            dtype=f"<U{sequence_width}",
            shape=(count,),
        ),
        "subject_ids": open_memmap(
            output_root / f"{split}_subject_ids.npy",
            mode="w+",
            dtype=f"<U{subject_width}",
            shape=(count,),
        ),
        "window_starts": open_memmap(
            output_root / f"{split}_window_starts.npy",
            mode="w+",
            dtype=np.int32,
            shape=(count,),
        ),
    }


def compute_stats(inputs_path: Path) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.load(inputs_path, mmap_mode="r")
    sums = np.zeros((1, 1, inputs.shape[2], 3), dtype=np.float64)
    sum_squares = np.zeros_like(sums)
    frames = 0
    for start in range(0, inputs.shape[0], 256):
        chunk = np.asarray(inputs[start : start + 256], dtype=np.float64)
        sums += chunk.sum(axis=(0, 1), keepdims=True)
        sum_squares += np.square(chunk).sum(axis=(0, 1), keepdims=True)
        frames += chunk.shape[0] * chunk.shape[1]
    mean = (sums / frames).astype(np.float32)
    variance = sum_squares / frames - np.square(mean.astype(np.float64))
    std = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"Output is not empty: {args.output_root}")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest, args.dataset)
    joint_indices = list(range(25)) if args.representation == "native25" else COMMON_JOINTS
    joint_names = [JOINT_NAMES_25[index] for index in joint_indices]
    skeleton_edges = SKELETON_EDGES_25 if args.representation == "native25" else COMMON_EDGES
    guidance_joint_index = 0 if args.representation == "native25" else 2
    guidance_joint_name = joint_names[guidance_joint_index]
    manifest_hash = sha256_file(args.manifest)

    counts = {split: 0 for split in SPLITS}
    rollout_counts = {split: 0 for split in SPLITS}
    sequence_stats = []
    max_sequence_width = max(len(row["sequence_id"]) for row in rows)
    max_subject_width = max(len(row["subject_id"]) for row in rows)
    print(f"Counting {args.dataset} {args.representation} sequences...", flush=True)
    for index, row in enumerate(rows, start=1):
        path = resolve_sequence_path(args.dataset, row["sequence_id"])
        with np.load(path, allow_pickle=False) as payload:
            frames = int(payload["joints_root_relative"].shape[0])
        windows = count_windows(frames, args.history, args.future, args.stride)
        rollout_windows = count_windows(frames, args.history, args.rollout_horizon, args.stride)
        counts[row["split"]] += windows
        rollout_counts[row["split"]] += rollout_windows
        sequence_stats.append(
            {
                "dataset": args.dataset,
                "subject_id": row["subject_id"],
                "sequence": row["sequence_id"],
                "split": row["split"],
                "frames": frames,
                "num_windows": windows,
                "num_rollout_windows": rollout_windows,
                "source_path": str(path.resolve()),
            }
        )
        if index % 250 == 0 or index == len(rows):
            print(f"  counted {index}/{len(rows)}", flush=True)

    writers = {
        split: create_writer(
            args.output_root,
            split,
            counts[split],
            args.history,
            args.future,
            len(joint_indices),
            max_sequence_width,
            max_subject_width,
        )
        for split in SPLITS
    }
    offsets = {split: 0 for split in SPLITS}

    print(f"Writing {args.dataset} {args.representation} windows...", flush=True)
    for index, (row, record) in enumerate(zip(rows, sequence_stats), start=1):
        split = row["split"]
        joints = load_root_relative(Path(record["source_path"]), joint_indices)
        starts = np.arange(
            0,
            joints.shape[0] - args.history - args.future + 1,
            args.stride,
            dtype=np.int32,
        )
        count = len(starts)
        if count:
            inputs = np.stack([joints[start : start + args.history] for start in starts])
            targets = np.stack(
                [joints[start + args.history : start + args.history + args.future] for start in starts]
            )
            offset = offsets[split]
            end = offset + count
            writer = writers[split]
            writer["inputs"][offset:end] = inputs
            writer["targets"][offset:end] = targets
            writer["sequence_ids"][offset:end] = row["sequence_id"]
            writer["subject_ids"][offset:end] = row["subject_id"]
            writer["window_starts"][offset:end] = starts
            offsets[split] = end
        if index % 100 == 0 or index == len(rows):
            print(
                f"  wrote {index}/{len(rows)}; train/val/test={offsets['train']}/{offsets['val']}/{offsets['test']}",
                flush=True,
            )

    for writer in writers.values():
        for array in writer.values():
            array.flush()
    if offsets != counts:
        raise RuntimeError(f"Written counts differ: expected={counts}, written={offsets}")

    mean, std = compute_stats(args.output_root / "train_inputs.npy")
    np.savez_compressed(args.output_root / "normalization_stats.npz", mean=mean, std=std)

    split_subjects = {
        split: sorted({row["subject_id"] for row in rows if row["split"] == split})
        for split in SPLITS
    }
    metadata = {
        "protocol_name": "Protocol v2 subject-grouped root-relative pose",
        "protocol_version": "2.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "representation": args.representation,
        "history": args.history,
        "future": args.future,
        "rollout_horizon": args.rollout_horizon,
        "stride": args.stride,
        "num_joints": len(joint_indices),
        "joint_dims": 3,
        "joint_indices_smplx25": joint_indices,
        "joint_names": joint_names,
        "skeleton_edges": [list(edge) for edge in skeleton_edges],
        "root_index_native25": 0,
        "root_joint_present": args.representation == "native25",
        "guidance_joint_index": guidance_joint_index,
        "guidance_joint_name": guidance_joint_name,
        "guidance_semantics": "root-centered latent anchor",
        "coordinate_source": "joints_root_relative",
        "use_root_relative": True,
        "center_on_first_frame": False,
        "stored_unit": "meter",
        "normalization": "per-channel mean/std computed from train_inputs only",
        "split_unit": "subject",
        "split_manifest": str(args.manifest.resolve()),
        "split_manifest_sha256": manifest_hash,
        "split_subjects": split_subjects,
        "splits": {
            split: {
                "num_samples": counts[split],
                "num_rollout_samples": rollout_counts[split],
                "num_sequences": sum(1 for row in rows if row["split"] == split),
                "num_subjects": len(split_subjects[split]),
                "files": {
                    "inputs": f"{split}_inputs.npy",
                    "targets": f"{split}_targets.npy",
                    "sequence_ids": f"{split}_sequence_ids.npy",
                    "subject_ids": f"{split}_subject_ids.npy",
                    "window_starts": f"{split}_window_starts.npy",
                },
            }
            for split in SPLITS
        },
        "sequences": sequence_stats,
    }
    metadata_path = args.output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_root / "BUILD_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "metadata": str(metadata_path),
                "manifest_sha256": manifest_hash,
                "counts": counts,
                "rollout_counts": rollout_counts,
                "normalization_stats": str(args.output_root / "normalization_stats.npz"),
                "pid": os.getpid(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_root": str(args.output_root), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
