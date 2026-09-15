from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

from src.progress import ProgressTracker
from src.topology import get_edges, get_joint_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build forecasting windows from extracted AMASS joints."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--history", type=int, required=True)
    parser.add_argument("--future", type=int, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-frames", type=int, default=0)
    parser.add_argument("--use-root-relative", action="store_true")
    parser.add_argument("--center-on-first-frame", action="store_true")
    parser.add_argument("--save-normalization", action="store_true")
    parser.add_argument("--include-subsets", nargs="*", default=None)
    parser.add_argument("--metadata-name", default="metadata.json")
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("runs/progress/window_progress.json"),
    )
    parser.add_argument("--ext", default=".npz")
    return parser.parse_args()


def iter_joint_files(root: Path, ext: str) -> List[Path]:
    return sorted(
        path
        for path in root.rglob(f"*{ext}")
        if path.is_file() and path.name != "manifest.json"
    )


def filter_subset_files(
    files: List[Path], root: Path, include_subsets: List[str] | None
) -> List[Path]:
    if not include_subsets:
        return files
    allowed = {item.lower() for item in include_subsets}
    selected = []
    for path in files:
        try:
            first_part = path.relative_to(root).parts[0].lower()
        except Exception:
            continue
        if first_part in allowed:
            selected.append(path)
    return selected


def deterministic_split(sequence_key: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.md5(sequence_key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def build_windows(
    joints: np.ndarray, history: int, future: int, stride: int
) -> Tuple[np.ndarray, np.ndarray]:
    total = history + future
    if joints.shape[0] < total:
        return (
            np.empty((0, history, joints.shape[1], joints.shape[2]), dtype=np.float32),
            np.empty((0, future, joints.shape[1], joints.shape[2]), dtype=np.float32),
        )

    inputs = []
    targets = []
    for start in range(0, joints.shape[0] - total + 1, stride):
        clip = joints[start : start + total]
        inputs.append(clip[:history])
        targets.append(clip[history:])
    return np.stack(inputs).astype(np.float32), np.stack(targets).astype(np.float32)


def count_windows(num_frames: int, history: int, future: int, stride: int) -> int:
    total = history + future
    if num_frames < total:
        return 0
    return ((num_frames - total) // stride) + 1


def load_sequence(
    file_path: Path,
    input_root: Path,
    use_root_relative: bool,
    center_on_first_frame: bool,
    min_frames: int,
) -> Dict[str, object]:
    sequence_key = str(file_path.relative_to(input_root)).replace("\\", "/")
    with np.load(file_path, allow_pickle=True) as data:
        joint_key = (
            "joints_root_relative"
            if use_root_relative and "joints_root_relative" in data
            else "joints"
        )
        joints = np.asarray(data[joint_key], dtype=np.float32)
        if min_frames and joints.shape[0] < min_frames:
            return {"sequence_key": sequence_key, "skip_note": "skipped_short_sequence"}
        if center_on_first_frame:
            joints = joints - joints[0:1]
        return {
            "sequence_key": sequence_key,
            "joints": joints,
            "joint_key": joint_key,
            "fps": float(np.asarray(data.get("fps", 30.0)).reshape(-1)[0]),
            "root_index": int(np.asarray(data.get("root_index", 0)).reshape(-1)[0]),
        }


def split_paths(output_root: Path, split_name: str) -> Dict[str, Path]:
    return {
        "inputs": output_root / f"{split_name}_inputs.npy",
        "targets": output_root / f"{split_name}_targets.npy",
        "sequence_ids": output_root / f"{split_name}_sequence_ids.npy",
    }


def compute_normalization_stats(inputs_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    train_inputs = np.load(inputs_path, mmap_mode="r")
    if train_inputs.shape[0] == 0:
        raise SystemExit("Train split is empty, cannot compute normalization stats.")

    sum_array = np.zeros((1, 1, train_inputs.shape[2], train_inputs.shape[3]), dtype=np.float64)
    sum_sq_array = np.zeros_like(sum_array)
    total_frames = 0
    chunk_size = 256

    for start in range(0, train_inputs.shape[0], chunk_size):
        chunk = np.asarray(train_inputs[start : start + chunk_size], dtype=np.float64)
        sum_array += chunk.sum(axis=(0, 1), keepdims=True)
        sum_sq_array += np.square(chunk).sum(axis=(0, 1), keepdims=True)
        total_frames += chunk.shape[0] * chunk.shape[1]

    mean = (sum_array / total_frames).astype(np.float32)
    variance = (sum_sq_array / total_frames) - np.square(mean.astype(np.float64))
    std = np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    files = filter_subset_files(
        iter_joint_files(args.input_root, args.ext), args.input_root, args.include_subsets
    )
    if not files:
        raise SystemExit(f"No joint files found under {args.input_root}")

    tracker = ProgressTracker(args.progress_file)
    tracker.start(
        stage="build_windows",
        total=len(files),
        extra={
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "history": args.history,
            "future": args.future,
            "include_subsets": args.include_subsets or [],
            "storage": "streaming_npy",
        },
    )

    split_counts = {"train": 0, "val": 0, "test": 0}
    split_sequence_sets = {"train": set(), "val": set(), "test": set()}
    sequence_stats = []
    first_joint_shape = None
    detected_root_index = 0
    max_sequence_id_length = 1
    success_count = 0
    failed_count = 0

    count_bar = tqdm(files, desc="Counting windows", unit="seq")
    for index, file_path in enumerate(count_bar, start=1):
        try:
            record = load_sequence(
                file_path=file_path,
                input_root=args.input_root,
                use_root_relative=args.use_root_relative,
                center_on_first_frame=args.center_on_first_frame,
                min_frames=args.min_frames,
            )
            sequence_key = str(record["sequence_key"])
            if "skip_note" in record:
                success_count += 1
                tracker.update(
                    processed=index,
                    success=success_count,
                    failed=failed_count,
                    current_item=sequence_key,
                    extra={"phase": "counting", "last_note": str(record["skip_note"])},
                )
                count_bar.set_postfix(tracker.tqdm_postfix())
                continue

            joints = np.asarray(record["joints"], dtype=np.float32)
            if first_joint_shape is None:
                first_joint_shape = joints.shape[1:]
            detected_root_index = int(record["root_index"])
            max_sequence_id_length = max(max_sequence_id_length, len(sequence_key))

            num_windows = count_windows(
                num_frames=int(joints.shape[0]),
                history=args.history,
                future=args.future,
                stride=args.stride,
            )
            if num_windows == 0:
                success_count += 1
                tracker.update(
                    processed=index,
                    success=success_count,
                    failed=failed_count,
                    current_item=sequence_key,
                    extra={"phase": "counting", "last_note": "no_windows_created"},
                )
                continue

            split_name = deterministic_split(sequence_key, args.train_ratio, args.val_ratio)
            split_counts[split_name] += num_windows
            split_sequence_sets[split_name].add(sequence_key)
            sequence_stats.append(
                {
                    "sequence": sequence_key,
                    "split": split_name,
                    "frames": int(joints.shape[0]),
                    "num_windows": int(num_windows),
                    "fps": float(record["fps"]),
                    "joint_key": str(record["joint_key"]),
                }
            )
            success_count += 1
            tracker.update(
                processed=index,
                success=success_count,
                failed=failed_count,
                current_item=sequence_key,
                extra={
                    "phase": "counting",
                    "last_num_windows": int(num_windows),
                    "estimated_splits": split_counts,
                },
            )
            count_bar.set_postfix(tracker.tqdm_postfix())
        except Exception as exc:
            failed_count += 1
            tracker.update(
                processed=index,
                success=success_count,
                failed=failed_count,
                current_item=str(file_path.relative_to(args.input_root)).replace("\\", "/"),
                last_error=str(exc),
                extra={"phase": "counting"},
            )
            count_bar.set_postfix(tracker.tqdm_postfix())

    if first_joint_shape is None:
        raise SystemExit("No valid joint files were processed.")

    split_writers = {}
    split_offsets = {"train": 0, "val": 0, "test": 0}
    sequence_dtype = f"<U{max_sequence_id_length}"
    for split_name, count in split_counts.items():
        paths = split_paths(args.output_root, split_name)
        split_writers[split_name] = {
            "paths": paths,
            "inputs": open_memmap(
                paths["inputs"],
                mode="w+",
                dtype=np.float32,
                shape=(count, args.history, first_joint_shape[0], first_joint_shape[1]),
            ),
            "targets": open_memmap(
                paths["targets"],
                mode="w+",
                dtype=np.float32,
                shape=(count, args.future, first_joint_shape[0], first_joint_shape[1]),
            ),
            "sequence_ids": open_memmap(
                paths["sequence_ids"],
                mode="w+",
                dtype=sequence_dtype,
                shape=(count,),
            ),
        }

    tracker.start(
        stage="build_windows_write",
        total=len(files),
        extra={
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "history": args.history,
            "future": args.future,
            "include_subsets": args.include_subsets or [],
            "storage": "streaming_npy",
            "estimated_splits": split_counts,
        },
    )
    success_count = 0
    failed_count = 0

    write_bar = tqdm(files, desc="Writing windows", unit="seq")
    for index, file_path in enumerate(write_bar, start=1):
        try:
            record = load_sequence(
                file_path=file_path,
                input_root=args.input_root,
                use_root_relative=args.use_root_relative,
                center_on_first_frame=args.center_on_first_frame,
                min_frames=args.min_frames,
            )
            sequence_key = str(record["sequence_key"])
            if "skip_note" in record:
                success_count += 1
                tracker.update(
                    processed=index,
                    success=success_count,
                    failed=failed_count,
                    current_item=sequence_key,
                    extra={"phase": "writing", "last_note": str(record["skip_note"])},
                )
                write_bar.set_postfix(tracker.tqdm_postfix())
                continue

            joints = np.asarray(record["joints"], dtype=np.float32)
            inputs, targets = build_windows(joints, args.history, args.future, args.stride)
            if inputs.shape[0] == 0:
                success_count += 1
                tracker.update(
                    processed=index,
                    success=success_count,
                    failed=failed_count,
                    current_item=sequence_key,
                    extra={"phase": "writing", "last_note": "no_windows_created"},
                )
                write_bar.set_postfix(tracker.tqdm_postfix())
                continue

            split_name = deterministic_split(sequence_key, args.train_ratio, args.val_ratio)
            offset = split_offsets[split_name]
            count = inputs.shape[0]
            writer = split_writers[split_name]
            writer["inputs"][offset : offset + count] = inputs
            writer["targets"][offset : offset + count] = targets
            writer["sequence_ids"][offset : offset + count] = np.asarray(
                [sequence_key] * count, dtype=sequence_dtype
            )
            split_offsets[split_name] += count

            success_count += 1
            tracker.update(
                processed=index,
                success=success_count,
                failed=failed_count,
                current_item=sequence_key,
                extra={
                    "phase": "writing",
                    "last_num_windows": int(count),
                    "written_splits": split_offsets,
                },
            )
            write_bar.set_postfix(tracker.tqdm_postfix())
        except Exception as exc:
            failed_count += 1
            tracker.update(
                processed=index,
                success=success_count,
                failed=failed_count,
                current_item=str(file_path.relative_to(args.input_root)).replace("\\", "/"),
                last_error=str(exc),
                extra={"phase": "writing"},
            )
            write_bar.set_postfix(tracker.tqdm_postfix())

    for split_name, writer in split_writers.items():
        writer["inputs"].flush()
        writer["targets"].flush()
        writer["sequence_ids"].flush()

    if args.save_normalization:
        mean, std = compute_normalization_stats(
            split_writers["train"]["paths"]["inputs"]
        )
        np.savez_compressed(args.output_root / "normalization_stats.npz", mean=mean, std=std)

    metadata = {
        "history": args.history,
        "future": args.future,
        "stride": args.stride,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "num_joints": int(first_joint_shape[0]),
        "joint_dims": int(first_joint_shape[1]),
        "root_index": detected_root_index,
        "joint_names": get_joint_names(int(first_joint_shape[0])),
        "skeleton_edges": get_edges(int(first_joint_shape[0])),
        "use_root_relative": args.use_root_relative,
        "center_on_first_frame": args.center_on_first_frame,
        "save_normalization": args.save_normalization,
        "include_subsets": args.include_subsets or [],
        "storage_format": "npy",
        "split_files": {
            split_name: {
                "inputs": writer["paths"]["inputs"].name,
                "targets": writer["paths"]["targets"].name,
                "sequence_ids": writer["paths"]["sequence_ids"].name,
            }
            for split_name, writer in split_writers.items()
        },
        "splits": {
            split_name: {
                "num_samples": int(split_counts[split_name]),
                "num_sequences": int(len(split_sequence_sets[split_name])),
            }
            for split_name in split_counts
        },
        "sequences": sequence_stats,
    }
    metadata_path = args.output_root / args.metadata_name
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tracker.finish(
        status="completed",
        extra={
            "metadata_path": str(metadata_path),
            "splits": metadata["splits"],
            "success": success_count,
            "failed": failed_count,
            "storage": "streaming_npy",
        },
    )
    print(f"Saved processed windows to {args.output_root}")


if __name__ == "__main__":
    main()
