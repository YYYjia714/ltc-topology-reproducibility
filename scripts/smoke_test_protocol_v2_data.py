from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")
COMMON_JOINTS = list(range(4, 22))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Protocol v2 data only.")
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, default=None)
    parser.add_argument("--samples-per-split", type=int, default=12)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def connected(num_nodes: int, edges: list[list[int]]) -> bool:
    graph = [[] for _ in range(num_nodes)]
    for left, right in edges:
        graph[left].append(right)
        graph[right].append(left)
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == num_nodes


def recompute_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.load(path, mmap_mode="r")
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
    std = np.maximum(np.sqrt(np.maximum(variance, 1e-12)).astype(np.float32), 1e-6)
    return mean, std


def resolve_source(sequence_record: dict) -> Path:
    path = Path(sequence_record["source_path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def sample_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, num=count, dtype=int).tolist()))


def check_representation(dataset: str, representation_root: Path, manifest_hash: str, samples: int) -> tuple[list[dict], dict]:
    metadata = read_json(representation_root / "metadata.json")
    checks = []

    def record(name: str, passed: bool, details: str) -> None:
        checks.append({"dataset": dataset, "representation": metadata["representation"], "check": name, "passed": bool(passed), "details": details})

    record("build_complete", (representation_root / "BUILD_COMPLETE.json").exists(), "BUILD_COMPLETE.json exists")
    record("manifest_hash", metadata["split_manifest_sha256"] == manifest_hash, metadata["split_manifest_sha256"])
    record("root_relative", metadata["use_root_relative"] is True, str(metadata["use_root_relative"]))
    record("no_first_frame_centering", metadata["center_on_first_frame"] is False, str(metadata["center_on_first_frame"]))
    record("stored_unit_meter", metadata["stored_unit"] == "meter", metadata["stored_unit"])
    record("subject_split", metadata["split_unit"] == "subject", metadata["split_unit"])
    record("history_future_stride", (metadata["history"], metadata["future"], metadata["stride"]) == (25, 25, 5), f"{metadata['history']}/{metadata['future']}/{metadata['stride']}")
    record("graph_connected", connected(metadata["num_joints"], metadata["skeleton_edges"]), f"edges={len(metadata['skeleton_edges'])}")

    subject_sets = {split: set(metadata["split_subjects"][split]) for split in SPLITS}
    overlap = (subject_sets["train"] & subject_sets["val"]) | (subject_sets["train"] & subject_sets["test"]) | (subject_sets["val"] & subject_sets["test"])
    record("zero_subject_overlap", not overlap, f"overlap={sorted(overlap)}")

    sequence_sets = {
        split: {item["sequence"] for item in metadata["sequences"] if item["split"] == split}
        for split in SPLITS
    }
    sequence_overlap = (sequence_sets["train"] & sequence_sets["val"]) | (sequence_sets["train"] & sequence_sets["test"]) | (sequence_sets["val"] & sequence_sets["test"])
    record("zero_sequence_overlap", not sequence_overlap, f"overlap_count={len(sequence_overlap)}")

    sequence_lookup = {item["sequence"]: item for item in metadata["sequences"]}
    joint_indices = metadata["joint_indices_smplx25"]
    max_raw_diff = 0.0
    max_root = 0.0
    for split in SPLITS:
        inputs = np.load(representation_root / f"{split}_inputs.npy", mmap_mode="r")
        targets = np.load(representation_root / f"{split}_targets.npy", mmap_mode="r")
        sequence_ids = np.load(representation_root / f"{split}_sequence_ids.npy", mmap_mode="r")
        subject_ids = np.load(representation_root / f"{split}_subject_ids.npy", mmap_mode="r")
        starts = np.load(representation_root / f"{split}_window_starts.npy", mmap_mode="r")
        expected = metadata["splits"][split]["num_samples"]
        shape_ok = inputs.shape[0] == targets.shape[0] == sequence_ids.shape[0] == subject_ids.shape[0] == starts.shape[0] == expected
        record(f"{split}_array_counts", shape_ok, f"expected={expected}, actual={inputs.shape[0]}")
        for index in sample_indices(expected, samples):
            sequence = str(sequence_ids[index])
            start = int(starts[index])
            source = resolve_source(sequence_lookup[sequence])
            with np.load(source, allow_pickle=False) as payload:
                raw = np.asarray(payload["joints_root_relative"], dtype=np.float32)[:, joint_indices, :]
            expected_input = raw[start : start + 25]
            expected_target = raw[start + 25 : start + 50]
            max_raw_diff = max(max_raw_diff, float(np.max(np.abs(np.asarray(inputs[index]) - expected_input))))
            max_raw_diff = max(max_raw_diff, float(np.max(np.abs(np.asarray(targets[index]) - expected_target))))
            if 0 in joint_indices:
                native_root_index = joint_indices.index(0)
                max_root = max(max_root, float(np.max(np.abs(expected_input[:, native_root_index, :]))))
    record("sampled_windows_match_raw_source", max_raw_diff == 0.0, f"max_abs_diff_m={max_raw_diff}")
    if 0 in joint_indices:
        record("root_is_zero", max_root == 0.0, f"max_abs_root_m={max_root}")

    saved = np.load(representation_root / "normalization_stats.npz", allow_pickle=False)
    recomputed_mean, recomputed_std = recompute_stats(representation_root / "train_inputs.npy")
    mean_diff = float(np.max(np.abs(saved["mean"] - recomputed_mean)))
    std_diff = float(np.max(np.abs(saved["std"] - recomputed_std)))
    record("train_stats_recomputed", mean_diff <= 1e-7 and std_diff <= 1e-7, f"mean_diff={mean_diff}, std_diff={std_diff}")
    summary = {
        "dataset": dataset,
        "representation": metadata["representation"],
        "num_joints": metadata["num_joints"],
        "manifest_hash": metadata["split_manifest_sha256"],
        "all_checks_pass": all(item["passed"] for item in checks),
        "max_sampled_raw_diff_m": max_raw_diff,
        "normalization_mean_diff": mean_diff,
        "normalization_std_diff": std_diff,
    }
    return checks, summary


def cross_representation_check(dataset: str, native_root: Path, common_root: Path, samples: int) -> tuple[list[dict], dict]:
    checks = []
    max_input_diff = 0.0
    max_target_diff = 0.0
    ids_equal = True
    starts_equal = True
    subjects_equal = True
    for split in SPLITS:
        native_inputs = np.load(native_root / f"{split}_inputs.npy", mmap_mode="r")
        native_targets = np.load(native_root / f"{split}_targets.npy", mmap_mode="r")
        common_inputs = np.load(common_root / f"{split}_inputs.npy", mmap_mode="r")
        common_targets = np.load(common_root / f"{split}_targets.npy", mmap_mode="r")
        native_ids = np.load(native_root / f"{split}_sequence_ids.npy", mmap_mode="r")
        common_ids = np.load(common_root / f"{split}_sequence_ids.npy", mmap_mode="r")
        native_starts = np.load(native_root / f"{split}_window_starts.npy", mmap_mode="r")
        common_starts = np.load(common_root / f"{split}_window_starts.npy", mmap_mode="r")
        native_subjects = np.load(native_root / f"{split}_subject_ids.npy", mmap_mode="r")
        common_subjects = np.load(common_root / f"{split}_subject_ids.npy", mmap_mode="r")
        ids_equal = ids_equal and np.array_equal(native_ids, common_ids)
        starts_equal = starts_equal and np.array_equal(native_starts, common_starts)
        subjects_equal = subjects_equal and np.array_equal(native_subjects, common_subjects)
        for index in sample_indices(len(native_inputs), samples):
            max_input_diff = max(max_input_diff, float(np.max(np.abs(np.asarray(native_inputs[index])[:, COMMON_JOINTS, :] - np.asarray(common_inputs[index])))))
            max_target_diff = max(max_target_diff, float(np.max(np.abs(np.asarray(native_targets[index])[:, COMMON_JOINTS, :] - np.asarray(common_targets[index])))))
    native_stats = np.load(native_root / "normalization_stats.npz", allow_pickle=False)
    common_stats = np.load(common_root / "normalization_stats.npz", allow_pickle=False)
    mean_diff = float(np.max(np.abs(native_stats["mean"][:, :, COMMON_JOINTS, :] - common_stats["mean"])))
    std_diff = float(np.max(np.abs(native_stats["std"][:, :, COMMON_JOINTS, :] - common_stats["std"])))
    values = {
        "sequence_ids_equal": ids_equal,
        "subject_ids_equal": subjects_equal,
        "window_starts_equal": starts_equal,
        "max_input_subset_diff_m": max_input_diff,
        "max_target_subset_diff_m": max_target_diff,
        "max_mean_subset_diff": mean_diff,
        "max_std_subset_diff": std_diff,
    }
    for name, value in values.items():
        passed = value is True if isinstance(value, bool) else float(value) <= 1e-7
        checks.append({"dataset": dataset, "representation": "native25_vs_common18", "check": name, "passed": passed, "details": str(value)})
    return checks, {"dataset": dataset, **values, "all_checks_pass": all(item["passed"] for item in checks)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256_file(args.frozen_manifest)
    checks = []
    summaries = []
    cross_summaries = []
    for dataset_slug, dataset_name in (("cmu", "CMU"), ("kit", "KIT"), ("bmlmovi", "BMLmovi")):
        native_root = args.protocol_root / dataset_slug / "native25"
        common_root = args.protocol_root / dataset_slug / "common18"
        for root in (native_root, common_root):
            new_checks, summary = check_representation(dataset_name, root, manifest_hash, args.samples_per_split)
            checks.extend(new_checks)
            summaries.append(summary)
        new_checks, cross_summary = cross_representation_check(dataset_name, native_root, common_root, args.samples_per_split)
        checks.extend(new_checks)
        cross_summaries.append(cross_summary)

    failed = [item for item in checks if not item["passed"]]
    payload = {
        "smoke_test_id": "protocol_v2_data_smoke_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_root": str(args.protocol_root.resolve()),
        "frozen_manifest": str(args.frozen_manifest.resolve()),
        "frozen_manifest_sha256": manifest_hash,
        "status": "PASS" if not failed else "FAIL",
        "training_started": False,
        "representation_summaries": summaries,
        "cross_representation_summaries": cross_summaries,
        "checks": checks,
        "failed_checks": failed,
    }
    (args.output_dir / "protocol_v2_data_smoke_test.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Protocol v2 Data Smoke Test",
        "",
        f"Status: **{payload['status']}**",
        "",
        "No model training was started.",
        "",
        "| Dataset | Representation | Joints | All checks pass | Raw max diff (m) | Mean diff | Std diff |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['dataset']} | {summary['representation']} | {summary['num_joints']} | {summary['all_checks_pass']} | "
            f"{summary['max_sampled_raw_diff_m']:.3g} | {summary['normalization_mean_diff']:.3g} | {summary['normalization_std_diff']:.3g} |"
        )
    lines.extend(["", "## Cross-Representation Checks", ""])
    for summary in cross_summaries:
        lines.append(
            f"- {summary['dataset']}: pass={summary['all_checks_pass']}, input_diff={summary['max_input_subset_diff_m']:.3g} m, target_diff={summary['max_target_subset_diff_m']:.3g} m."
        )
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        for item in failed:
            lines.append(f"- {item['dataset']} / {item['representation']} / {item['check']}: {item['details']}")
    (args.output_dir / "SMOKE_TEST_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.mirror_dir is not None:
        args.mirror_dir.mkdir(parents=True, exist_ok=True)
        for path in args.output_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, args.mirror_dir / path.name)
    print(json.dumps({"status": payload["status"], "failed_checks": len(failed), "training_started": False}, indent=2))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
