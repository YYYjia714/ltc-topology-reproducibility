from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology import JOINT_NAMES_25, SKELETON_EDGES_25


SPLITS = ("train", "val", "test")
COMMON_JOINTS = list(range(4, 22))

DATASETS = {
    "CMU": {
        "metadata": PROJECT_ROOT / "data/processed/windows_cmu/metadata_cmu.json",
        "hisrep_run": PROJECT_ROOT / "runs/hisrep_cmu_clean_30ep_25in25out_18j_cuda",
        "full_run": PROJECT_ROOT
        / "runs/cmu_ltc_topology_rollout_consistency_centered_full_30ep_cuda_bs128_fast_20260613_152237",
    },
    "KIT": {
        "metadata": PROJECT_ROOT / "data/processed/windows_kit/metadata_kit.json",
        "hisrep_run": PROJECT_ROOT / "runs/hisrep_kit_clean_30ep_25in25out_18j_cuda",
        "full_run": PROJECT_ROOT
        / "runs/kit_ltc_topology_rollout_consistency_centered_full_30ep_cuda_bs128_workers0_20260614_042235",
    },
    "BMLmovi": {
        "metadata": PROJECT_ROOT
        / "data/processed/windows_bmlmovi_root_relative_mincheck/metadata_bmlmovi_root_relative_mincheck.json",
        "hisrep_run": PROJECT_ROOT
        / "runs/hisrep_bmlmovi_root_relative_clean_30ep_25in25out_18j_cuda",
        "full_run": PROJECT_ROOT
        / "runs/bml_root_relative_ltc_topology_rollout_consistency_full_30ep_cuda",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit readiness for the unified Protocol v2.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs/analysis/protocol_v2_readiness_audit_20260815",
    )
    parser.add_argument("--mirror-dir", type=Path, default=None)
    parser.add_argument("--sample-sequences", type=int, default=60)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def canonical_subject(dataset: str, sequence: str) -> str:
    parts = Path(sequence.replace("/", "\\")).parts
    if dataset in {"CMU", "KIT"}:
        return f"{dataset}:{parts[1]}" if len(parts) > 1 else f"{dataset}:UNKNOWN"
    match = re.search(r"Subject_(\d+)", sequence, flags=re.IGNORECASE)
    return f"BMLmovi:Subject_{int(match.group(1)):02d}" if match else "BMLmovi:UNKNOWN"


def balanced_subject_assignment(subject_windows: dict[str, int]) -> dict[str, str]:
    total = sum(subject_windows.values())
    targets = {"train": 0.8 * total, "val": 0.1 * total, "test": 0.1 * total}
    assigned = {split: 0 for split in SPLITS}
    assignment = {}
    ordered = sorted(
        subject_windows,
        key=lambda subject: (
            -subject_windows[subject],
            hashlib.sha256(subject.encode("utf-8")).hexdigest(),
        ),
    )
    for subject in ordered:
        # Fill the split with the largest remaining target. This is deterministic
        # and balances windows while keeping each subject wholly within one split.
        split = max(
            SPLITS,
            key=lambda name: (targets[name] - assigned[name], targets[name], name),
        )
        assignment[subject] = split
        assigned[split] += subject_windows[subject]
    return assignment


def resolve_sequence_path(dataset: str, sequence: str) -> Path:
    root = PROJECT_ROOT / "data/interim/joints"
    candidates = [root / sequence, root / dataset / sequence]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(sequence)


def connected_components(num_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    graph = [[] for _ in range(num_nodes)]
    for left, right in edges:
        graph[left].append(right)
        graph[right].append(left)
    seen = set()
    components = []
    for node in range(num_nodes):
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def common18_graph_audit() -> dict:
    remap = {original: index for index, original in enumerate(COMMON_JOINTS)}
    induced = [
        (remap[left], remap[right])
        for left, right in SKELETON_EDGES_25
        if left in remap and right in remap
    ]
    induced_components = connected_components(len(COMMON_JOINTS), induced)

    # Collapse omitted pelvis/hip/spine1 paths into the nearest retained nodes.
    contracted = sorted(set(induced + [(remap[4], remap[6]), (remap[5], remap[6])]))
    contracted_components = connected_components(len(COMMON_JOINTS), contracted)
    return {
        "common_joints_original_indices": COMMON_JOINTS,
        "common_joint_names": [JOINT_NAMES_25[index] for index in COMMON_JOINTS],
        "induced_edges": [list(edge) for edge in induced],
        "induced_num_edges": len(induced),
        "induced_components": induced_components,
        "induced_is_connected": len(induced_components) == 1,
        "contracted_edges": [list(edge) for edge in contracted],
        "contracted_num_edges": len(contracted),
        "contracted_components": contracted_components,
        "contracted_is_connected": len(contracted_components) == 1,
        "added_contracted_edges": [
            [remap[4], remap[6]],
            [remap[5], remap[6]],
        ],
        "metric_rule": "Use contracted real evaluation edges; visualization-only edges must not enter metrics.",
    }


def sample_preprocessing(dataset: str, metadata: dict, limit: int) -> dict:
    records = sorted(metadata["sequences"], key=lambda item: item["sequence"])
    if len(records) > limit:
        indices = np.linspace(0, len(records) - 1, num=limit, dtype=int)
        records = [records[index] for index in indices]

    root_max = 0.0
    offsets = []
    mismatch_distances = []
    frames_checked = 0
    for record in records:
        path = resolve_sequence_path(dataset, record["sequence"])
        with np.load(path, allow_pickle=False) as payload:
            raw = np.asarray(payload["joints_root_relative"], dtype=np.float32)
        root_max = max(root_max, float(np.max(np.abs(raw[:, 0, :]))))
        first_pose = raw[0:1]
        centered = raw - first_pose
        common_raw = raw[:25, COMMON_JOINTS, :]
        common_centered = centered[:25, COMMON_JOINTS, :]
        distances = np.linalg.norm(common_raw - common_centered, axis=-1)
        mismatch_distances.append(float(np.mean(distances)))
        offsets.append(float(np.mean(np.linalg.norm(first_pose[:, COMMON_JOINTS, :], axis=-1))))
        frames_checked += min(25, raw.shape[0])

    current_center = bool(metadata.get("center_on_first_frame", False))
    return {
        "sample_sequences": len(records),
        "sample_history_frames": frames_checked,
        "max_abs_root_coordinate_m": root_max,
        "mean_first_pose_joint_norm_m": float(np.mean(offsets)),
        "current_ltc_vs_raw_hisrep_input_mpjpe_m": (
            float(np.mean(mismatch_distances)) if current_center else 0.0
        ),
        "current_ltc_vs_raw_hisrep_input_mpjpe_mm": (
            float(np.mean(mismatch_distances)) * 1000.0 if current_center else 0.0
        ),
        "physical_input_match": not current_center,
    }


def config_audit(dataset: str, spec: dict, metadata: dict) -> dict:
    hisrep_config = read_json(spec["hisrep_run"] / "config.json")
    his_cfg = hisrep_config["config"]
    his_joints = hisrep_config["common_joints"]
    full_results = read_json(spec["full_run"] / "results.json")
    return {
        "dataset": dataset,
        "metadata_history": int(metadata["history"]),
        "metadata_future": int(metadata["future"]),
        "metadata_stride": int(metadata["stride"]),
        "metadata_num_joints": int(metadata["num_joints"]),
        "metadata_root_relative": bool(metadata.get("use_root_relative", False)),
        "metadata_center_on_first_frame": bool(metadata.get("center_on_first_frame", False)),
        "hisrep_input_n": int(his_cfg["input_n"]),
        "hisrep_output_n": int(his_cfg["output_n"]),
        "hisrep_stride": int(his_cfg["stride"]),
        "hisrep_num_joints": len(his_joints),
        "hisrep_common_joints_match_locked": list(his_joints) == COMMON_JOINTS,
        "full_num_joints": int(metadata["num_joints"]),
        "full_rollout_horizon": int(full_results["rollout_consistency"]["horizon"]),
        "full_rollout_step": int(full_results["rollout_consistency"]["future_step"]),
        "same_history": int(metadata["history"]) == int(his_cfg["input_n"]),
        "same_future_block": int(metadata["future"]) == int(his_cfg["output_n"]),
        "same_stride": int(metadata["stride"]) == int(his_cfg["stride"]),
        "same_joint_representation": int(metadata["num_joints"]) == len(his_joints),
        "same_physical_preprocessing": not bool(metadata.get("center_on_first_frame", False)),
    }


def build_split_manifests(dataset: str, metadata: dict) -> tuple[list[dict], list[dict], dict]:
    sequence_manifest = []
    subject_manifest = []
    preview = {
        "current_sequence_split": {split: {"subjects": set(), "sequences": 0, "windows": 0} for split in SPLITS},
        "candidate_subject_split": {split: {"subjects": set(), "sequences": 0, "windows": 0} for split in SPLITS},
    }
    subject_records: dict[str, list[dict]] = defaultdict(list)
    for record in metadata["sequences"]:
        subject = canonical_subject(dataset, record["sequence"])
        subject_records[subject].append(record)
        current_split = record["split"]
        sequence_manifest.append(
            {
                "dataset": dataset,
                "subject_id": subject,
                "sequence_id": record["sequence"],
                "split": current_split,
                "frames": int(record["frames"]),
                "num_windows_25in25out_stride5": int(record["num_windows"]),
                "num_windows_25in75out_stride5": max(
                    0, ((int(record["frames"]) - 100) // 5) + 1
                ),
            }
        )
        current = preview["current_sequence_split"][current_split]
        current["subjects"].add(subject)
        current["sequences"] += 1
        current["windows"] += int(record["num_windows"])

    subject_windows = {
        subject: sum(int(record["num_windows"]) for record in records)
        for subject, records in subject_records.items()
    }
    subject_assignment = balanced_subject_assignment(subject_windows)
    for subject, records in sorted(subject_records.items()):
        split = subject_assignment[subject]
        windows = sum(int(record["num_windows"]) for record in records)
        subject_manifest.append(
            {
                "dataset": dataset,
                "subject_id": subject,
                "split": split,
                "num_sequences": len(records),
                "num_windows_25in25out_stride5": windows,
                "num_windows_25in75out_stride5": sum(
                    max(0, ((int(record["frames"]) - 100) // 5) + 1)
                    for record in records
                ),
            }
        )
        target = preview["candidate_subject_split"][split]
        target["subjects"].add(subject)
        target["sequences"] += len(records)
        target["windows"] += windows

    serialized_preview = {}
    for mode, split_data in preview.items():
        total_windows = sum(item["windows"] for item in split_data.values())
        serialized_preview[mode] = {
            split: {
                "subjects": len(item["subjects"]),
                "sequences": item["sequences"],
                "windows": item["windows"],
                "window_fraction": item["windows"] / max(total_windows, 1),
            }
            for split, item in split_data.items()
        }
    return sequence_manifest, subject_manifest, serialized_preview


def retraining_matrix() -> list[dict]:
    rows = []
    model_groups = [
        ("GRU native-25", "native"),
        ("LSTM native-25", "native"),
        ("Original LTC native-25", "native"),
        ("LTC-Topology no-rollout native-25", "native"),
        ("Full LTC-Topology native-25", "native"),
        ("HisRepItself common-18", "hisrep"),
        ("LTC-Topology no-rollout common-18", "common18_new"),
        ("Full LTC-Topology common-18", "common18_new"),
    ]
    for dataset in DATASETS:
        for model, group in model_groups:
            if group == "native":
                sequence_path = "RETRAIN" if dataset in {"CMU", "KIT"} else "REUSE_AFTER_CHECKSUM"
            elif group == "hisrep":
                sequence_path = "REUSE_AND_REEVALUATE"
            else:
                sequence_path = "TRAIN_NEW"
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "sequence_split_rootrel_no_center": sequence_path,
                    "subject_grouped_rootrel_no_center": "RETRAIN" if group != "common18_new" else "TRAIN_NEW",
                    "reason": (
                        "CMU/KIT native checkpoints use first-pose displacement; BMLmovi native checkpoints already use raw root-relative pose. "
                        "Existing HisRep checkpoints use raw root-relative common-18 pose. Equal-shape common-18 LTC checkpoints do not exist."
                    ),
                }
            )
    return rows


def protocol_spec() -> dict:
    return {
        "protocol_name": "Protocol v2: locked root-relative pose protocol",
        "split_options": {
            "minimum_revision": "locked complete-sequence split before window generation",
            "strong_revision": "locked subject-grouped split before window generation",
        },
        "coordinates": "joints_root_relative",
        "center_on_first_frame": False,
        "stored_unit": "meter",
        "history": 25,
        "future_block": 25,
        "rollout_horizon": 75,
        "rollout_context": "latest 25 frames only",
        "evaluation_stride": 5,
        "native_joint_representation": {"num_joints": 25, "indices": list(range(25))},
        "controlled_hisrep_representation": {
            "num_joints": 18,
            "indices": COMMON_JOINTS,
            "graph": "contracted common-18 tree",
        },
        "normalization": {
            "ltc_family": "per-channel mean/std from training split only",
            "hisrep": "same physical coordinates with model-native conversion from meters to millimeters",
            "metric_space": "denormalized physical coordinates in millimeters",
        },
        "bone_metric": "actual contracted skeleton edges only; visualization-only links excluded",
        "timing": {
            "joint_representation": "common-18 for both models",
            "batch_size": "identical",
            "precision": "FP32",
            "data_loading": "excluded for both",
            "cuda_synchronize": "before and after each timed region",
            "warmup_and_repetitions": "identical and reported",
        },
    }


def markdown_report(
    configs: list[dict],
    preprocessing: list[dict],
    graph: dict,
    split_previews: dict,
) -> str:
    lines = [
        "# Protocol v2 Readiness Audit",
        "",
        "## Decision",
        "",
        "The current CMU/KIT Full LTC-Topology checkpoints and the current HisRepItself checkpoints do not use the same physical input representation. CMU and KIT LTC models use per-joint displacement from the first pose of the source sequence, whereas HisRepItself uses raw root-relative poses. BMLmovi uses raw root-relative poses for both model families, but the model inputs remain unequal in shape (25 joints versus 18 joints). A controlled common-18 LTC-Topology model does not currently exist.",
        "",
        "The recommended Protocol v2 uses raw root-relative poses without first-frame subtraction for all datasets. Under this protocol, at least partial retraining is required. A subject-grouped split requires complete retraining; retaining the audited sequence-level split permits reuse of the BMLmovi native checkpoints and potentially the existing HisRepItself checkpoints after checksum/configuration verification.",
        "",
        "## Current Configuration",
        "",
        "| Dataset | LTC center-first | HisRep joints | Full joints | Same history/block/stride | Same joints | Same physical preprocessing |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in configs:
        same_timing = row["same_history"] and row["same_future_block"] and row["same_stride"]
        lines.append(
            f"| {row['dataset']} | {row['metadata_center_on_first_frame']} | {row['hisrep_num_joints']} | "
            f"{row['full_num_joints']} | {same_timing} | {row['same_joint_representation']} | "
            f"{row['same_physical_preprocessing']} |"
        )
    lines.extend(
        [
            "",
            "## Measured Input Mismatch",
            "",
            "| Dataset | Sampled sequences | Root coordinate max (m) | Current LTC-vs-HisRep physical input difference (mm) | Physical inputs match |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in preprocessing:
        lines.append(
            f"| {row['dataset']} | {row['sample_sequences']} | {row['max_abs_root_coordinate_m']:.6g} | "
            f"{row['current_ltc_vs_raw_hisrep_input_mpjpe_mm']:.3f} | {row['physical_input_match']} |"
        )
    lines.extend(
        [
            "",
            "## Common-18 Graph",
            "",
            f"The direct induced common-18 graph has {graph['induced_num_edges']} edges and {len(graph['induced_components'])} connected components, so it is not suitable as a single connected skeleton prior. Contracting the omitted pelvis/hip/spine1 paths adds left-knee-to-spine2 and right-knee-to-spine2 links, producing {graph['contracted_num_edges']} edges and one connected component. These contracted links are part of the common-18 model/evaluation graph, not merely visualization links.",
            "",
            "## Subject-Split Preview",
            "",
            "| Dataset | Split | Subjects | Sequences | Windows | Window fraction |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset, preview in split_previews.items():
        for split in SPLITS:
            item = preview["candidate_subject_split"][split]
            lines.append(
                f"| {dataset} | {split} | {item['subjects']} | {item['sequences']} | {item['windows']} | {item['window_fraction']:.3%} |"
            )
    lines.extend(
        [
            "",
            "## Required Before Training",
            "",
            "1. Choose and freeze either the audited sequence-level manifest or the candidate subject-grouped manifest.",
            "2. Build new data directories; do not overwrite the current processed windows.",
            "3. Set root-relative coordinates on and first-frame subtraction off for every dataset.",
            "4. Recompute LTC-family normalization from training windows only.",
            "5. Train an equal-shape common-18 LTC-Topology model using the contracted graph.",
            "6. Verify that the same raw window produces identical physical coordinates for LTC-Topology and HisRepItself before model-specific numerical scaling.",
            "7. Run one-epoch smoke tests before formal training.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    preprocessing = []
    all_sequence_manifest = []
    all_subject_manifest = []
    split_previews = {}
    for dataset, spec in DATASETS.items():
        metadata = read_json(spec["metadata"])
        configs.append(config_audit(dataset, spec, metadata))
        preprocessing.append({"dataset": dataset, **sample_preprocessing(dataset, metadata, args.sample_sequences)})
        sequence_manifest, subject_manifest, preview = build_split_manifests(dataset, metadata)
        all_sequence_manifest.extend(sequence_manifest)
        all_subject_manifest.extend(subject_manifest)
        split_previews[dataset] = preview

    graph = common18_graph_audit()
    retraining = retraining_matrix()
    spec = protocol_spec()
    subject_assignment = {
        (row["dataset"], row["subject_id"]): row["split"]
        for row in all_subject_manifest
    }
    candidate_subject_grouped_sequence_manifest = [
        {
            **row,
            "split": subject_assignment[(row["dataset"], row["subject_id"])],
        }
        for row in all_sequence_manifest
    ]
    report = {
        "audit_id": "protocol_v2_readiness_audit_20260815",
        "protocol_v2": spec,
        "current_config_audit": configs,
        "preprocessing_measurements": preprocessing,
        "common18_graph_audit": graph,
        "split_previews": split_previews,
        "retraining_matrix": retraining,
        "conclusion": {
            "retraining_required": True,
            "minimum_revision": "Retrain CMU/KIT native models; train common-18 LTC models; re-evaluate audited HisRep checkpoints on raw root-relative windows.",
            "strong_revision": "Retrain all reported models on locked subject-grouped Protocol v2 splits.",
        },
    }

    (args.output_dir / "protocol_v2_readiness_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "PROTOCOL_V2_READINESS_SUMMARY.md").write_text(
        markdown_report(configs, preprocessing, graph, split_previews), encoding="utf-8"
    )
    (args.output_dir / "protocol_v2_spec.json").write_text(
        json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "common18_graph.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "split_preview.json").write_text(
        json.dumps(split_previews, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "current_protocol_matrix.csv", configs)
    write_csv(args.output_dir / "preprocessing_input_mismatch.csv", preprocessing)
    write_csv(args.output_dir / "locked_sequence_split_manifest.csv", all_sequence_manifest)
    write_csv(args.output_dir / "candidate_subject_split_manifest.csv", all_subject_manifest)
    write_csv(
        args.output_dir / "candidate_subject_grouped_sequence_manifest.csv",
        candidate_subject_grouped_sequence_manifest,
    )
    write_csv(args.output_dir / "retraining_matrix.csv", retraining)

    if args.mirror_dir is not None:
        args.mirror_dir.mkdir(parents=True, exist_ok=True)
        for path in args.output_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, args.mirror_dir / path.name)

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "mirror_dir": None if args.mirror_dir is None else str(args.mirror_dir),
                "retraining_required": True,
                "common18_induced_connected": graph["induced_is_connected"],
                "common18_contracted_connected": graph["contracted_is_connected"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
