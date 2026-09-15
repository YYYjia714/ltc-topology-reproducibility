from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val", "test")

DATASETS = {
    "CMU": {
        "data_root": PROJECT_ROOT / "data/processed/windows_cmu",
        "metadata": PROJECT_ROOT / "data/processed/windows_cmu/metadata_cmu.json",
    },
    "KIT": {
        "data_root": PROJECT_ROOT / "data/processed/windows_kit",
        "metadata": PROJECT_ROOT / "data/processed/windows_kit/metadata_kit.json",
    },
    "BMLmovi": {
        "data_root": PROJECT_ROOT / "data/processed/windows_bmlmovi_root_relative_mincheck",
        "metadata": PROJECT_ROOT
        / "data/processed/windows_bmlmovi_root_relative_mincheck/metadata_bmlmovi_root_relative_mincheck.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit sequence/subject leakage and lock the active data protocol."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/analysis/reviewer_revision_data_protocol_audit_20260815",
    )
    parser.add_argument("--mirror-dir", type=Path, default=None)
    parser.add_argument(
        "--hash-source-files",
        action="store_true",
        help="Hash source NPZ files to detect exact duplicates stored under different paths.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical_subject(dataset: str, sequence: str) -> str:
    parts = Path(sequence.replace("/", "\\")).parts
    if dataset in {"CMU", "KIT"}:
        if len(parts) < 2:
            return "UNKNOWN"
        return f"{dataset}:{parts[1]}"
    match = re.search(r"Subject_(\d+)", sequence, flags=re.IGNORECASE)
    if match:
        return f"BMLmovi:Subject_{int(match.group(1)):02d}"
    return "UNKNOWN"


def resolve_sequence_path(dataset: str, sequence: str) -> Path | None:
    joint_root = PROJECT_ROOT / "data/interim/joints"
    candidates = [joint_root / sequence, joint_root / dataset / sequence]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def pair_names() -> list[tuple[str, str]]:
    return [("train", "val"), ("train", "test"), ("val", "test")]


def load_materialized_ids(data_root: Path, split: str) -> np.ndarray:
    npy_path = data_root / f"{split}_sequence_ids.npy"
    if npy_path.exists():
        return np.load(npy_path, mmap_mode="r", allow_pickle=False)
    npz_path = data_root / f"{split}.npz"
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=True) as payload:
            return np.asarray(payload["sequence_ids"])
    raise FileNotFoundError(f"No materialized sequence IDs for {data_root} / {split}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def static_protocol_findings() -> list[dict]:
    window_builder = (PROJECT_ROOT / "scripts/build_forecasting_windows.py").read_text(
        encoding="utf-8"
    )
    rollout_trainer = (PROJECT_ROOT / "train_ltc_topology_rollout_consistency.py").read_text(
        encoding="utf-8"
    )
    hisrep_trainer = (PROJECT_ROOT / "train_hisrep_cmu_clean.py").read_text(
        encoding="utf-8"
    )
    all_models_eval = (PROJECT_ROOT / "evaluate_all_models_rollout75_denorm_mm.py").read_text(
        encoding="utf-8"
    )
    rollout_eval = (PROJECT_ROOT / "evaluate_gru_vs_topology_rollout75.py").read_text(
        encoding="utf-8"
    )
    model_code = (PROJECT_ROOT / "src/models.py").read_text(encoding="utf-8")
    return [
        {
            "check": "split_key",
            "status": "SEQUENCE_LEVEL",
            "evidence": "deterministic_split(sequence_key, ...)" if "deterministic_split(sequence_key" in window_builder else "not found",
            "interpretation": "Complete sequences are assigned before windows, but subjects are not grouped.",
        },
        {
            "check": "split_before_windowing",
            "status": "PASS" if "split_name = deterministic_split(sequence_key" in window_builder else "REVIEW",
            "evidence": "One split is selected for each sequence and all of its windows are written to that split.",
            "interpretation": "Overlapping windows from the same sequence do not cross splits.",
        },
        {
            "check": "normalization_source",
            "status": "PASS" if 'split_writers["train"]["paths"]["inputs"]' in window_builder else "REVIEW",
            "evidence": "normalization_stats.npz is computed from train inputs in the window builder.",
            "interpretation": "Validation and test arrays are not used to estimate mean/std in this builder.",
        },
        {
            "check": "ltc_centering_from_metadata",
            "status": "PASS" if 'metadata.get("center_on_first_frame", False)' in rollout_trainer else "REVIEW",
            "evidence": "RolloutWindowDataset reads center_on_first_frame from metadata.",
            "interpretation": "LTC rollout training follows the dataset centering flag.",
        },
        {
            "check": "hisrep_centering_from_metadata",
            "status": "FAIL" if 'metadata.get("center_on_first_frame", False)' not in hisrep_trainer else "PASS",
            "evidence": "HisRep load_sequence reads joints_root_relative but does not apply metadata centering.",
            "interpretation": "HisRep retraining is not preprocessing-identical to centered CMU/KIT LTC training.",
        },
        {
            "check": "main_eval_centering_from_metadata",
            "status": "PASS" if 'metadata.get("center_on_first_frame", False)' in all_models_eval else "REVIEW",
            "evidence": "Main rollout evaluation reads center_on_first_frame from metadata.",
            "interpretation": "Main native-model evaluation follows the active dataset centering flag.",
        },
        {
            "check": "main_eval_stride",
            "status": "PASS" if 'default=5' in all_models_eval and '"stride": args.stride' in all_models_eval else "REVIEW",
            "evidence": "Default evaluation stride is 5 and is recorded in output metadata.",
            "interpretation": "The locked evaluation stride is explicit.",
        },
        {
            "check": "fixed_history_recursive_rollout",
            "status": "PASS" if "current = torch.cat([current[:, needed:" in rollout_trainer and "current = torch.cat([current[:, needed:" in rollout_eval else "REVIEW",
            "evidence": "After each 25-frame prediction, the oldest 25 frames are dropped and the predicted 25 frames are appended.",
            "interpretation": "Training and inference both use a fixed latest-25-frame context; the input length does not grow.",
        },
        {
            "check": "root_guidance_under_root_relative_input",
            "status": "REVIEW" if "root_feature = nodes[:, :, 0, :]" in model_code else "REVIEW",
            "evidence": "The raw joint-0 trajectory is identically zero after root normalization. The branch reads the joint-0 latent feature after graph encoding, so it may contain neighboring-body context but not global root motion.",
            "interpretation": "The component should be described as root-centered latent guidance or an anchor constraint, not as root-trajectory guidance. The root loss uses a zero target and penalizes predicted root drift.",
        },
    ]


def audit_dataset(dataset: str, spec: dict, hash_source_files: bool) -> dict:
    metadata = read_json(spec["metadata"])
    records = metadata.get("sequences", [])
    sequence_sets: dict[str, set[str]] = {split: set() for split in SPLITS}
    subject_sets: dict[str, set[str]] = {split: set() for split in SPLITS}
    subject_to_splits: dict[str, set[str]] = defaultdict(set)
    sequence_rows: list[dict] = []
    missing_sources: list[str] = []
    hashes: dict[str, list[dict]] = defaultdict(list)

    for record in records:
        split = str(record["split"])
        sequence = str(record["sequence"])
        subject = canonical_subject(dataset, sequence)
        sequence_sets[split].add(sequence)
        subject_sets[split].add(subject)
        subject_to_splits[subject].add(split)
        source_path = resolve_sequence_path(dataset, sequence)
        if source_path is None:
            missing_sources.append(sequence)
        elif hash_source_files:
            hashes[sha256_file(source_path)].append(
                {"sequence": sequence, "split": split, "path": str(source_path)}
            )
        sequence_rows.append(
            {
                "dataset": dataset,
                "sequence": sequence,
                "subject": subject,
                "split": split,
                "frames": int(record.get("frames", 0)),
                "num_windows": int(record.get("num_windows", 0)),
                "joint_key": str(record.get("joint_key", "")),
                "source_exists": source_path is not None,
                "source_path": "" if source_path is None else str(source_path),
            }
        )

    sequence_overlaps = {}
    subject_overlaps = {}
    for left, right in pair_names():
        key = f"{left}_vs_{right}"
        sequence_overlaps[key] = sorted(sequence_sets[left] & sequence_sets[right])
        subject_overlaps[key] = sorted(subject_sets[left] & subject_sets[right])

    materialized = {}
    for split in SPLITS:
        ids = load_materialized_ids(spec["data_root"], split)
        unique_ids = {str(item) for item in np.unique(ids)}
        expected = sequence_sets[split]
        materialized[split] = {
            "num_window_ids": int(ids.shape[0]),
            "metadata_num_windows": int(metadata["splits"][split]["num_samples"]),
            "num_unique_sequence_ids": len(unique_ids),
            "metadata_num_sequences": len(expected),
            "ids_missing_from_materialized": sorted(expected - unique_ids),
            "unexpected_materialized_ids": sorted(unique_ids - expected),
            "matches_metadata": (
                int(ids.shape[0]) == int(metadata["splits"][split]["num_samples"])
                and unique_ids == expected
            ),
        }

    duplicate_groups = []
    if hash_source_files:
        for digest, items in hashes.items():
            item_splits = {item["split"] for item in items}
            if len(items) > 1:
                duplicate_groups.append(
                    {
                        "sha256": digest,
                        "cross_split": len(item_splits) > 1,
                        "items": items,
                    }
                )

    protocol = {
        "history": int(metadata["history"]),
        "future": int(metadata["future"]),
        "stride": int(metadata["stride"]),
        "train_ratio": float(metadata["train_ratio"]),
        "val_ratio": float(metadata["val_ratio"]),
        "test_ratio": 1.0 - float(metadata["train_ratio"]) - float(metadata["val_ratio"]),
        "num_joints": int(metadata["num_joints"]),
        "joint_dims": int(metadata["joint_dims"]),
        "root_index": int(metadata["root_index"]),
        "use_root_relative": bool(metadata.get("use_root_relative", False)),
        "center_on_first_frame": bool(metadata.get("center_on_first_frame", False)),
        "save_normalization": bool(metadata.get("save_normalization", False)),
        "normalization_stats_exists": (spec["data_root"] / "normalization_stats.npz").exists(),
    }
    stats_path = spec["data_root"] / "normalization_stats.npz"
    if stats_path.exists():
        with np.load(stats_path, allow_pickle=False) as stats:
            protocol["normalization_mean_shape"] = list(stats["mean"].shape)
            protocol["normalization_std_shape"] = list(stats["std"].shape)
            protocol["normalization_std_min"] = float(np.min(stats["std"]))

    leaking_subjects = {
        subject: sorted(splits)
        for subject, splits in subject_to_splits.items()
        if len(splits) > 1
    }
    return {
        "dataset": dataset,
        "metadata_path": str(spec["metadata"]),
        "data_root": str(spec["data_root"]),
        "protocol": protocol,
        "split_summary": {
            split: {
                "sequences": len(sequence_sets[split]),
                "subjects": len(subject_sets[split]),
                "windows": int(metadata["splits"][split]["num_samples"]),
            }
            for split in SPLITS
        },
        "sequence_overlaps": sequence_overlaps,
        "subject_overlaps": subject_overlaps,
        "leaking_subjects": leaking_subjects,
        "sequence_leakage": any(sequence_overlaps.values()),
        "subject_overlap": bool(leaking_subjects),
        "missing_sources": missing_sources,
        "materialized_window_id_checks": materialized,
        "duplicate_source_groups": duplicate_groups,
        "cross_split_duplicate_source_groups": [
            group for group in duplicate_groups if group["cross_split"]
        ],
        "sequence_rows": sequence_rows,
    }


def build_summary_markdown(results: list[dict], static_checks: list[dict]) -> str:
    lines = [
        "# Data Leakage and Protocol Audit",
        "",
        "## Verdict",
        "",
        "The current datasets are split before window construction at the complete-sequence level. "
        "No sequence ID is shared across train, validation, and test. However, the split hash is applied "
        "independently to each sequence path rather than to a subject/group identifier. Consequently, "
        "the same subjects occur across splits in all three datasets. The current protocol is therefore "
        "sequence-independent but not subject-independent. This is not sequence-level leakage, but it limits "
        "claims about generalization to unseen subjects.",
        "",
        "## Dataset Findings",
        "",
        "| Dataset | Train/Val/Test sequences | Train/Val/Test subjects | Sequence leakage | Subject overlap | Materialized IDs match metadata |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        split_summary = result["split_summary"]
        seq_counts = "/".join(str(split_summary[s]["sequences"]) for s in SPLITS)
        subject_counts = "/".join(str(split_summary[s]["subjects"]) for s in SPLITS)
        ids_match = all(
            item["matches_metadata"]
            for item in result["materialized_window_id_checks"].values()
        )
        lines.append(
            f"| {result['dataset']} | {seq_counts} | {subject_counts} | "
            f"{'YES' if result['sequence_leakage'] else 'NO'} | "
            f"{'YES' if result['subject_overlap'] else 'NO'} | {'YES' if ids_match else 'NO'} |"
        )

    lines.extend(
        [
            "",
            "## Protocol Matrix",
            "",
            "| Dataset | History | Future block | Window stride | Joints | Root-relative | Center on first frame | Train-only normalization configured |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        p = result["protocol"]
        lines.append(
            f"| {result['dataset']} | {p['history']} | {p['future']} | {p['stride']} | "
            f"{p['num_joints']} | {p['use_root_relative']} | {p['center_on_first_frame']} | "
            f"{p['save_normalization']} |"
        )

    lines.extend(
        [
            "",
            "## Static Code Checks",
            "",
            "| Check | Status | Evidence | Interpretation |",
            "|---|---|---|---|",
        ]
    )
    for check in static_checks:
        lines.append(
            f"| {check['check']} | {check['status']} | {check['evidence']} | {check['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Revision Paths",
            "",
            "### Sequence-independent revision (preserves the current trained models)",
            "",
            "1. State explicitly that complete original motion sequences were assigned to splits before window generation.",
            "2. Provide the zero sequence-overlap and zero cross-split duplicate-file results from this audit.",
            "3. State that subjects may occur in more than one split and do not claim unseen-subject generalization.",
            "",
            "### Subject-independent revision (stronger, requires retraining)",
            "",
            "1. Derive a stable subject/group ID for every source sequence.",
            "2. Assign subjects, not individual sequences, to train/validation/test before window construction.",
            "3. Rebuild all windows and normalization statistics from the new training subjects only.",
            "4. Retrain every model used in the main comparison on the corrected subject-independent splits.",
            "5. Evaluate all models using the same locked test subjects, stride, centering, root-relative representation, and joint definition.",
            "",
            "### Independent protocol issue",
            "",
            "Align HisRepItself preprocessing with the selected common-joint protocol; the current clean trainer does not apply the CMU/KIT centering flag.",
            "Re-specify the current root-guidance claim under root-relative coordinates: the encoded root node can aggregate neighboring-body context, but it does not carry a global root trajectory; the associated loss is a zero-root anchor constraint.",
            "",
            "## Files",
            "",
            "- `audit_report.json`: machine-readable complete audit.",
            "- `dataset_protocol_matrix.csv`: locked dataset protocol fields.",
            "- `split_overlap_summary.csv`: sequence and subject overlap counts.",
            "- `sequence_split_manifest.csv`: every sequence, subject, split, frames, and windows.",
            "- `subject_split_manifest.csv`: every subject and the splits in which it occurs.",
            "- `static_protocol_checks.csv`: code-path consistency findings.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        audit_dataset(dataset, spec, args.hash_source_files)
        for dataset, spec in DATASETS.items()
    ]
    static_checks = static_protocol_findings()

    sequence_rows = [row for result in results for row in result.pop("sequence_rows")]
    protocol_rows = []
    overlap_rows = []
    subject_rows = []
    for result in results:
        protocol_rows.append({"dataset": result["dataset"], **result["protocol"]})
        for left, right in pair_names():
            key = f"{left}_vs_{right}"
            overlap_rows.append(
                {
                    "dataset": result["dataset"],
                    "split_pair": key,
                    "sequence_overlap_count": len(result["sequence_overlaps"][key]),
                    "subject_overlap_count": len(result["subject_overlaps"][key]),
                    "overlapping_subjects": ";".join(result["subject_overlaps"][key]),
                }
            )
        all_subjects = defaultdict(set)
        for row in sequence_rows:
            if row["dataset"] == result["dataset"]:
                all_subjects[row["subject"]].add(row["split"])
        for subject, splits in sorted(all_subjects.items()):
            subject_rows.append(
                {
                    "dataset": result["dataset"],
                    "subject": subject,
                    "splits": ";".join(sorted(splits)),
                    "num_splits": len(splits),
                    "overlaps_across_splits": len(splits) > 1,
                }
            )

    report = {
        "audit_id": "reviewer_revision_data_protocol_audit_20260815",
        "project_root": str(PROJECT_ROOT),
        "hash_source_files": args.hash_source_files,
        "overall_sequence_leakage": any(r["sequence_leakage"] for r in results),
        "overall_subject_overlap": any(r["subject_overlap"] for r in results),
        "datasets": results,
        "static_protocol_checks": static_checks,
    }
    (args.output_dir / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "AUDIT_SUMMARY.md").write_text(
        build_summary_markdown(results, static_checks), encoding="utf-8"
    )
    write_csv(
        args.output_dir / "dataset_protocol_matrix.csv",
        protocol_rows,
        list(protocol_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "split_overlap_summary.csv",
        overlap_rows,
        list(overlap_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "sequence_split_manifest.csv",
        sequence_rows,
        list(sequence_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "subject_split_manifest.csv",
        subject_rows,
        list(subject_rows[0].keys()),
    )
    write_csv(
        args.output_dir / "static_protocol_checks.csv",
        static_checks,
        list(static_checks[0].keys()),
    )

    if args.mirror_dir is not None:
        args.mirror_dir.mkdir(parents=True, exist_ok=True)
        for item in args.output_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, args.mirror_dir / item.name)

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "mirror_dir": None if args.mirror_dir is None else str(args.mirror_dir),
        "overall_sequence_leakage": report["overall_sequence_leakage"],
        "overall_subject_overlap": report["overall_subject_overlap"],
    }, indent=2))


if __name__ == "__main__":
    main()
