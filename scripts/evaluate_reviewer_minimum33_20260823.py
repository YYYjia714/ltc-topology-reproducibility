from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_hisrep_cmu_clean as hisrep_train  # noqa: E402
import train_protocol_v2_revision_hisrep_20260821 as revision_hisrep  # noqa: E402
import train_protocol_v2_supplementary_rollout_20260820 as rollout_train  # noqa: E402
from src.models import MotionForecastModel  # noqa: E402
from src.reviewer_baselines import build_reviewer_baseline  # noqa: E402


METRIC_NAMES = (
    "mpjpe_1_25_mm",
    "mpjpe_26_50_mm",
    "mpjpe_51_75_mm",
    "mpjpe_1_75_mm",
    "bone_1_75_mm",
    "bone_51_75_mm",
    "velocity_1_75_mm",
    "velocity_51_75_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reviewer-minimum common18 physical evaluation and statistics.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=["cmu", "kit", "bmlmovi"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-windows-per-sequence", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_ltc_model(entry: dict, metadata: dict, device: torch.device) -> torch.nn.Module:
    model = MotionForecastModel(
        model_type=str(entry["model"]),
        joints=int(metadata["num_joints"]),
        history=25,
        future=25,
        hidden_size=256,
        num_layers=2,
        dropout=0.0,
        use_topology_encoder=bool(entry["use_topology"]),
        use_topology_decoder=bool(entry["use_topology"]),
        use_root_guidance=bool(entry["use_root_guidance"]),
        skeleton_edges=[[int(a), int(b)] for a, b in metadata["skeleton_edges"]],
        guidance_joint_index=int(metadata["guidance_joint_index"]),
    ).to(device)
    checkpoint = torch.load(Path(entry["checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def load_baseline_model(entry: dict, metadata: dict, device: torch.device) -> torch.nn.Module:
    model = build_reviewer_baseline(
        str(entry["baseline"]),
        int(metadata["num_joints"]),
        25,
        25,
        [[int(a), int(b)] for a, b in metadata["skeleton_edges"]],
        256,
        2,
        0.0,
    ).to(device)
    checkpoint = torch.load(Path(entry["checkpoint"]), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def load_hisrep_model(entry: dict, device: torch.device) -> tuple[torch.nn.Module, hisrep_train.Config]:
    checkpoint = torch.load(Path(entry["checkpoint"]), map_location="cpu", weights_only=False)
    raw_config = checkpoint.get("config", {})
    saved = raw_config.get("hisrep_config", raw_config)
    cfg = hisrep_train.Config(**saved)
    model = hisrep_train.make_model(cfg, device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint.get("state_dict")), strict=True)
    model.eval()
    return model, cfg


def hisrep_block(model: torch.nn.Module, history_m: torch.Tensor, cfg: hisrep_train.Config) -> torch.Tensor:
    if history_m.shape[1] != 25:
        raise RuntimeError(f"HisRep evaluation requires 25 observed frames, got {history_m.shape[1]}")
    if cfg.input_n == 25:
        batch_size = history_m.shape[0]
        padded_m = torch.cat([history_m, torch.zeros_like(history_m)], dim=1)
        src_mm = (padded_m * hisrep_train.MM_PER_METER).reshape(
            batch_size, cfg.input_n + cfg.output_n, cfg.in_features
        )
        output = model(src_mm, output_n=cfg.model_output_n, input_n=cfg.input_n, itera=cfg.itera)
        return hisrep_train._hisrep_future_from_output(output, cfg, batch_size) / hisrep_train.MM_PER_METER
    if cfg.input_n == revision_hisrep.HISREP_INTERNAL_INPUT_FRAMES:
        return revision_hisrep.predict_block_mm(model, history_m * 1000.0, cfg) / 1000.0
    raise RuntimeError(f"Unsupported HisRep internal input length: {cfg.input_n}")


def hisrep_rollout(model: torch.nn.Module, history_m: torch.Tensor, cfg: hisrep_train.Config) -> torch.Tensor:
    current = history_m
    chunks = []
    for _ in range(3):
        prediction = hisrep_block(model, current, cfg)
        chunks.append(prediction)
        current = prediction
    return torch.cat(chunks, dim=1)


def metric_values(
    history_m: torch.Tensor,
    prediction_m: torch.Tensor,
    target_m: torch.Tensor,
    edges: torch.Tensor,
) -> dict[str, float]:
    distance = torch.linalg.norm(prediction_m - target_m, dim=-1) * 1000.0
    pred_bones = prediction_m[:, :, edges[:, 0]] - prediction_m[:, :, edges[:, 1]]
    true_bones = target_m[:, :, edges[:, 0]] - target_m[:, :, edges[:, 1]]
    bone = torch.abs(torch.linalg.norm(pred_bones, dim=-1) - torch.linalg.norm(true_bones, dim=-1)) * 1000.0
    pred_full = torch.cat([history_m[:, -1:], prediction_m], dim=1)
    true_full = torch.cat([history_m[:, -1:], target_m], dim=1)
    velocity = torch.linalg.norm(
        (pred_full[:, 1:] - pred_full[:, :-1]) - (true_full[:, 1:] - true_full[:, :-1]), dim=-1
    ) * 1000.0
    return {
        "mpjpe_1_25_mm": float(distance[:, :25].mean()),
        "mpjpe_26_50_mm": float(distance[:, 25:50].mean()),
        "mpjpe_51_75_mm": float(distance[:, 50:75].mean()),
        "mpjpe_1_75_mm": float(distance.mean()),
        "bone_1_75_mm": float(bone.mean()),
        "bone_51_75_mm": float(bone[:, 50:75].mean()),
        "velocity_1_75_mm": float(velocity.mean()),
        "velocity_51_75_mm": float(velocity[:, 50:75].mean()),
    }


def sequence_index(dataset: rollout_train.RolloutWindowDataset, metadata: dict) -> tuple[dict[str, list[int]], dict[str, dict]]:
    path_to_record = {}
    for record in metadata["sequences"]:
        if record["split"] != "test":
            continue
        path = rollout_train.resolve_sequence_path(str(metadata["dataset"]), str(record["sequence"]))
        path_to_record[str(path)] = record
    indices: dict[str, list[int]] = defaultdict(list)
    for index, (path, _) in enumerate(dataset.records):
        indices[path].append(index)
    return dict(indices), path_to_record


@torch.inference_mode()
def evaluate_entry(
    entry: dict,
    dataset: rollout_train.RolloutWindowDataset,
    indices_by_path: dict[str, list[int]],
    path_metadata: dict[str, dict],
    mean: torch.Tensor,
    std: torch.Tensor,
    edges: torch.Tensor,
    metadata: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict]:
    kind = str(entry["kind"])
    if kind == "hisrep":
        model, hisrep_cfg = load_hisrep_model(entry, device)
    elif kind == "baseline":
        model = load_baseline_model(entry, metadata, device)
        hisrep_cfg = None
    else:
        model = load_ltc_model(entry, metadata, device)
        hisrep_cfg = None
    rows = []
    sequence_paths = list(indices_by_path)
    if args.max_sequences > 0:
        sequence_paths = sequence_paths[: args.max_sequences]
    for sequence_path in sequence_paths:
        indices = indices_by_path[sequence_path]
        if args.max_windows_per_sequence > 0:
            indices = indices[: args.max_windows_per_sequence]
        loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
        totals = {name: 0.0 for name in METRIC_NAMES}
        count = 0
        for batch in loader:
            inputs_n = batch["inputs"].to(device)
            targets_n = batch["targets"].to(device)
            history_m = inputs_n * std + mean
            targets_m = targets_n * std + mean
            if kind == "hisrep":
                prediction_m = hisrep_rollout(model, history_m, hisrep_cfg)
            else:
                prediction_n = rollout_train.recursive_rollout(model, inputs_n, horizon=75, step=25)
                prediction_m = prediction_n * std + mean
            values = metric_values(history_m, prediction_m, targets_m, edges)
            batch_size = int(inputs_n.shape[0])
            for name, value in values.items():
                totals[name] += value * batch_size
            count += batch_size
        record = path_metadata[sequence_path]
        row = {
            "dataset": str(entry["dataset"]),
            "method": str(entry["method"]),
            "analysis_group": str(entry.get("analysis_group", "")),
            "seed": int(entry["seed"]),
            "subject_id": str(record["subject_id"]),
            "sequence": str(record["sequence"]),
            "windows": count,
        }
        row.update({name: totals[name] / count for name in METRIC_NAMES})
        rows.append(row)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def clustered_bootstrap_ci(
    differences: np.ndarray,
    subjects: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subjects)
    samples = np.empty(iterations, dtype=np.float64)
    subject_values = np.asarray(
        [float(np.mean(differences[subjects == subject])) for subject in unique_subjects],
        dtype=np.float64,
    )
    for index in range(iterations):
        sampled = rng.choice(subject_values, size=len(subject_values), replace=True)
        samples[index] = sampled.mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def subject_balanced_differences(differences: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    unique_subjects = np.unique(subjects)
    return np.asarray(
        [float(np.mean(differences[subjects == subject])) for subject in unique_subjects],
        dtype=np.float64,
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def paired_statistics(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = str(row.get("analysis_group", ""))
        if group in {"full", "hisrep", "lstm"}:
            grouped[(row["dataset"], row["subject_id"], row["sequence"])][group].append(row)

    output = []
    metrics = ("mpjpe_51_75_mm", "mpjpe_1_75_mm", "bone_1_75_mm", "velocity_1_75_mm")
    for dataset_name in args.datasets:
        dataset_keys = [key for key in grouped if key[0] == dataset_name]
        for comparison_index, (left, right, label, mode) in enumerate(
            (
                ("full", "hisrep", "Full LTC-Topology - HisRepItself", "three_seed_mean"),
                ("full", "lstm", "Full LTC-Topology - controlled LSTM", "seed42"),
            )
        ):
            complete = []
            for key in dataset_keys:
                groups = grouped[key]
                if left not in groups or right not in groups:
                    continue
                if mode == "three_seed_mean" and (
                    sorted(int(row["seed"]) for row in groups[left]) != [42, 123, 456]
                    or sorted(int(row["seed"]) for row in groups[right]) != [42, 123, 456]
                ):
                    continue
                complete.append(key)
            for metric_index, metric in enumerate(metrics):
                if not complete:
                    continue
                differences = []
                for key in complete:
                    groups = grouped[key]
                    if mode == "three_seed_mean":
                        left_value = float(np.mean([row[metric] for row in groups[left]]))
                        right_value = float(np.mean([row[metric] for row in groups[right]]))
                    else:
                        left_value = next(float(row[metric]) for row in groups[left] if int(row["seed"]) == 42)
                        right_value = next(float(row[metric]) for row in groups[right] if int(row["seed"]) == 42)
                    differences.append(left_value - right_value)
                subjects = np.asarray([key[1] for key in complete])
                output.append(
                    difference_statistics(
                        dataset_name,
                        label,
                        metric,
                        np.asarray(differences, dtype=np.float64),
                        subjects,
                        args.bootstrap_iterations,
                        args.bootstrap_seed + comparison_index * 20 + metric_index,
                    )
                )
    if output:
        adjusted = holm_adjust([float(row["wilcoxon_p"]) for row in output])
        for row, value in zip(output, adjusted):
            row["holm_adjusted_p"] = value
    return output


def difference_statistics(
    dataset_name: str,
    comparison: str,
    metric: str,
    differences: np.ndarray,
    subjects: np.ndarray,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict:
    subject_differences = subject_balanced_differences(differences, subjects)
    try:
        wilcoxon = stats.wilcoxon(subject_differences, alternative="two-sided", zero_method="wilcox")
        statistic = float(wilcoxon.statistic)
        p_value = float(wilcoxon.pvalue)
    except ValueError:
        statistic = 0.0
        p_value = 1.0
    ci_low, ci_high = clustered_bootstrap_ci(
        differences, subjects, bootstrap_iterations, bootstrap_seed
    )
    return {
        "dataset": dataset_name,
        "comparison": comparison,
        "metric": metric,
        "n_sequences": int(differences.size),
        "n_subjects": int(subject_differences.size),
        "estimand": "subject_balanced_mean_of_within_subject_sequence_differences",
        "subject_balanced_mean_difference_mm": float(subject_differences.mean()),
        "sequence_weighted_mean_difference_mm": float(differences.mean()),
        "subject_median_difference_mm": float(np.median(subject_differences)),
        "cluster_bootstrap_ci_low_mm": ci_low,
        "cluster_bootstrap_ci_high_mm": ci_high,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": p_value,
        "rank_biserial": rank_biserial(subject_differences),
        "subject_negative_difference_rate": float(np.mean(subject_differences < 0)),
    }


def ablation_statistics(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    seed42 = [row for row in rows if int(row["seed"]) == 42]
    by_key = {
        (row["dataset"], row["subject_id"], row["sequence"], row["method"]): row
        for row in seed42
    }
    methods = {
        "a": "Original LTC no rollout",
        "b": "Original LTC full rollout",
        "c": "LTC-Topology no rollout",
        "d": "Full LTC-Topology",
        "e": "LTC-Topology w/o spine2 anchor guidance",
    }
    output = []
    for dataset_name in args.datasets:
        sequence_keys = sorted(
            {(row["subject_id"], row["sequence"]) for row in seed42 if row["dataset"] == dataset_name}
        )
        complete = [
            key for key in sequence_keys
            if all((dataset_name, key[0], key[1], method) in by_key for method in methods.values())
        ]
        if not complete:
            continue
        subjects = np.asarray([key[0] for key in complete])
        for metric_index, metric in enumerate(("mpjpe_51_75_mm", "mpjpe_1_75_mm", "bone_1_75_mm", "velocity_1_75_mm")):
            values = {
                code: np.asarray(
                    [by_key[(dataset_name, key[0], key[1], method)][metric] for key in complete], dtype=np.float64
                )
                for code, method in methods.items()
            }
            contrasts = {
                "Topology effect without rollout: C-A": values["c"] - values["a"],
                "Topology effect with rollout: D-B": values["d"] - values["b"],
                "Rollout fine-tuning effect for Original LTC: B-A": values["b"] - values["a"],
                "Rollout effect for LTC-Topology: D-C": values["d"] - values["c"],
                "Topology x rollout interaction: (D-C)-(B-A)": (values["d"] - values["c"]) - (values["b"] - values["a"]),
                "Spine2-anchor guidance effect: Full-w/o-anchor": values["d"] - values["e"],
            }
            for contrast_index, (name, differences) in enumerate(contrasts.items()):
                output.append(
                    difference_statistics(
                        dataset_name,
                        name,
                        metric,
                        differences,
                        subjects,
                        args.bootstrap_iterations,
                        args.bootstrap_seed + 100 + metric_index * 10 + contrast_index,
                    )
                )
    if output:
        adjusted = holm_adjust([float(row["wilcoxon_p"]) for row in output])
        for row, value in zip(output, adjusted):
            row["holm_adjusted_p"] = value
    return output


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], int(row["seed"]))].append(row)
    output = []
    for (dataset, method, seed), items in grouped.items():
        row = {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "sequences": len(items),
            "subjects": len({item["subject_id"] for item in items}),
            "windows": sum(int(item["windows"]) for item in items),
        }
        for metric in METRIC_NAMES:
            row[metric] = float(np.mean([float(item[metric]) for item in items]))
        output.append(row)
    return sorted(output, key=lambda item: (item["dataset"], item["method"], item["seed"]))


def multiseed_summary(aggregate: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in aggregate:
        if row["method"] in {"Full LTC-Topology", "HisRepItself"}:
            grouped[(row["dataset"], row["method"])].append(row)
    output = []
    for (dataset, method), items in grouped.items():
        if sorted(int(item["seed"]) for item in items) != [42, 123, 456]:
            continue
        row = {"dataset": dataset, "method": method, "seeds": "42;123;456", "n_seeds": 3}
        for metric in METRIC_NAMES:
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
        output.append(row)
    return sorted(output, key=lambda item: (item["dataset"], item["method"]))


def sensitivity_summary(aggregate: list[dict]) -> list[dict]:
    methods = {
        "Full LTC-Topology",
        "Sensitivity regularizers 0.5x",
        "Sensitivity regularizers 2x",
        "Sensitivity velocity loss 0",
        "Sensitivity equal rollout-stage weights",
    }
    rows = [
        row
        for row in aggregate
        if row["dataset"] == "cmu" and int(row["seed"]) == 42 and row["method"] in methods
    ]
    if len(rows) != 5:
        raise RuntimeError(f"Expected five CMU seed-42 sensitivity rows including the reused default, found {len(rows)}")
    return sorted(rows, key=lambda row: row["method"])


def build_entries(plan: dict) -> list[dict]:
    run_root = Path(plan["run_root"])
    plan_path = run_root / "formal_reviewer_minimum33_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Frozen plan is missing: {plan_path}")
    current_plan_sha256 = sha256_file(plan_path)
    inventory_path = run_root / "reused" / "reused_inventory.json"
    inventory = read_json(inventory_path)
    if inventory.get("status") != "PASS" or inventory.get("plan_sha256") != current_plan_sha256:
        raise RuntimeError(f"Reused-artifact inventory did not pass: {inventory_path}")
    for item in inventory.get("files", []):
        path = Path(item["path"])
        if (
            not path.is_file()
            or sha256_file(path) != item.get("sha256")
            or path.stat().st_size != int(item.get("size", -1))
        ):
            raise RuntimeError(f"Reviewed source is missing or changed: {path}")
    entries = []

    def method_mapping(stage_id: str, kind: str, baseline: str | None = None) -> tuple[str, str]:
        if kind == "hisrep":
            return "HisRepItself", "hisrep"
        if kind == "baseline":
            names = {
                "lstm": ("Controlled LSTM", "lstm"),
                "msrgcn": ("MSR-GCN common18 adaptation", "recent"),
                "st_transformer": ("ST-Transformer common18 adaptation", "recent"),
                "simlpe": ("siMLPe common18 adaptation", "recent"),
                "humanmac": ("HumanMAC common18 adaptation", "recent"),
            }
            return names[str(baseline)]
        if "skeleton_full_rollout" in stage_id:
            return "Full LTC-Topology", "full"
        if "skeleton_no_rollout" in stage_id:
            return "LTC-Topology no rollout", "ablation"
        if "no_anchor_full_rollout" in stage_id:
            return "LTC-Topology w/o spine2 anchor guidance", "ablation"
        if "no_anchor_no_rollout" in stage_id:
            return "LTC-Topology w/o spine2 anchor guidance no rollout", "ablation"
        if "original_ltc_no_rollout" in stage_id:
            return "Original LTC no rollout", "ablation"
        if "original_ltc_full_rollout" in stage_id:
            return "Original LTC full rollout", "ablation"
        if "sensitivity_regularizers_half" in stage_id:
            return "Sensitivity regularizers 0.5x", "sensitivity"
        if "sensitivity_regularizers_double" in stage_id:
            return "Sensitivity regularizers 2x", "sensitivity"
        if "sensitivity_no_velocity" in stage_id:
            return "Sensitivity velocity loss 0", "sensitivity"
        if "sensitivity_equal_stage_weights" in stage_id:
            return "Sensitivity equal rollout-stage weights", "sensitivity"
        raise RuntimeError(f"No analysis method mapping for {stage_id}")

    for record in inventory["records"]:
        role = str(record["role"])
        method = {
            "skeleton_no_rollout": "LTC-Topology no rollout",
            "skeleton_full_rollout": "Full LTC-Topology",
            "hisrep": "HisRepItself",
        }[role]
        group = {"skeleton_no_rollout": "ablation", "skeleton_full_rollout": "full", "hisrep": "hisrep"}[role]
        source_dir = Path(record["source_dir"])
        entries.append(
            {
                "stage_id": str(record["id"]),
                "dataset": str(record["dataset"]),
                "kind": "hisrep" if role == "hisrep" else "ltc",
                "seed": 42,
                "method": method,
                "analysis_group": group,
                "checkpoint": str(source_dir / "best.pt"),
                "history": str(source_dir / "train_history.json"),
                "result_path": str(source_dir / "results.json"),
                "model": "ltc_topology",
                "use_topology": True,
                "use_root_guidance": True,
            }
        )

    for stage in plan["stages"]:
        stage_id = str(stage["id"])
        run_dir = run_root / str(stage["output_rel"])
        result_path = run_dir / "results.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing formal result: {result_path}")
        checkpoint = run_dir / "best.pt"
        history = run_dir / "train_history.json"
        exit_path = run_root / "stage_state" / f"{stage_id}_exit.json"
        exit_state = read_json(exit_path)
        if (
            exit_state.get("status") != "completed"
            or int(exit_state.get("exit_code", -1)) != 0
            or exit_state.get("smoke", False)
            or exit_state.get("plan_sha256") != current_plan_sha256
        ):
            raise RuntimeError(f"Formal stage exit state is invalid or belongs to another plan: {exit_path}")
        artifact = next(item for item in exit_state["artifacts"] if item["name"] == "best.pt")
        if not checkpoint.is_file() or sha256_file(checkpoint) != artifact["sha256"]:
            raise RuntimeError(f"Checkpoint binding failed: {checkpoint}")
        baseline = stage.get("baseline")
        method, group = method_mapping(stage_id, str(stage["kind"]), baseline)
        entries.append(
            {
                "stage_id": stage_id,
                "dataset": str(stage["dataset"]),
                "kind": str(stage["kind"]),
                "seed": int(stage["seed"]),
                "method": method,
                "analysis_group": group,
                "checkpoint": str(checkpoint),
                "history": str(history),
                "result_path": str(result_path),
                "baseline": baseline,
                "model": str(stage.get("model", "ltc_topology")),
                "use_topology": str(stage.get("model", "ltc_topology")) == "ltc_topology",
                "use_root_guidance": bool(stage.get("anchor_guidance", False)),
            }
        )
    return entries


def main() -> None:
    args = parse_args()
    plan = read_json(args.plan)
    entries = [entry for entry in build_entries(plan) if entry["dataset"] in args.datasets]
    if not entries:
        raise RuntimeError("No registry entries matched the requested datasets")
    for entry in entries:
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    dataset_audits = []
    for dataset_name in args.datasets:
        dataset_entries = [entry for entry in entries if entry["dataset"] == dataset_name]
        if not dataset_entries:
            continue
        data_root = Path(plan["data_root"]) / dataset_name / "common18"
        metadata_path = data_root / "metadata.json"
        metadata = read_json(metadata_path)
        if str(metadata["split_manifest_sha256"]).lower() != str(plan["split_manifest_sha256"]).lower():
            raise RuntimeError(f"Manifest mismatch for {dataset_name}")
        dataset = rollout_train.RolloutWindowDataset(
            dataset_name=str(metadata["dataset"]), metadata_path=metadata_path, split="test",
            history=25, horizon=75, stride=5, stats_file=data_root / "normalization_stats.npz",
            max_windows=0, seed=args.bootstrap_seed,
        )
        indices_by_path, path_metadata = sequence_index(dataset, metadata)
        stats_payload = np.load(data_root / "normalization_stats.npz")
        mean = torch.from_numpy(stats_payload["mean"].astype(np.float32)[0]).to(device)
        std = torch.from_numpy(stats_payload["std"].astype(np.float32)[0]).to(device)
        edges = torch.tensor(metadata["skeleton_edges"], dtype=torch.long, device=device)
        dataset_audits.append(
            {
                "dataset": dataset_name,
                "metadata": str(metadata_path),
                "manifest_sha256": metadata["split_manifest_sha256"],
                "test_sequences": len(indices_by_path),
                "test_windows": len(dataset),
            }
        )
        for entry in dataset_entries:
            print(f"START {dataset_name} {entry['method']} seed={entry['seed']}", flush=True)
            all_rows.extend(
                evaluate_entry(
                    entry, dataset, indices_by_path, path_metadata, mean, std, edges, metadata, args, device
                )
            )

    summary = aggregate_rows(all_rows)
    seed_summary = multiseed_summary(summary)
    sensitivity = sensitivity_summary(summary)
    statistics = paired_statistics(all_rows, args)
    ablations = ablation_statistics(all_rows, args)
    write_csv(args.output_dir / "per_sequence_metrics.csv", all_rows)
    write_csv(args.output_dir / "aggregate_metrics.csv", summary)
    write_csv(args.output_dir / "multiseed_summary.csv", seed_summary)
    write_csv(args.output_dir / "sensitivity_summary.csv", sensitivity)
    write_csv(args.output_dir / "paired_statistics.csv", statistics)
    write_csv(args.output_dir / "ablation_contrasts.csv", ablations)
    payload = {
        "status": "completed",
        "plan": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "evaluation_protocol": {
            "split": "test",
            "raw_metric_unit": "sequence",
            "inferential_unit": "subject_id",
            "estimand": "subject-balanced mean of within-subject sequence differences",
            "seed_aggregation": "mean within sequence and method before pairing",
            "primary_endpoint": "mpjpe_51_75_mm",
            "test": "two-sided Wilcoxon signed-rank on subject-mean paired differences",
            "confidence_interval": "subject-level percentile bootstrap",
            "bootstrap_iterations": args.bootstrap_iterations,
            "multiple_comparison_correction": "Holm across emitted comparisons",
            "hisrep_observation_adapter": "legacy direct 25-frame input, exactly matching the reused seed-42 configuration",
            "hisrep_uses_additional_real_history": False,
            "max_sequences": args.max_sequences,
            "max_windows_per_sequence": args.max_windows_per_sequence,
        },
        "dataset_audits": dataset_audits,
        "checkpoint_sha256": {entry["checkpoint"]: sha256_file(Path(entry["checkpoint"])) for entry in entries},
        "aggregate_metrics": summary,
        "multiseed_summary": seed_summary,
        "sensitivity_summary": sensitivity,
        "paired_statistics": statistics,
        "ablation_contrasts": ablations,
    }
    if any(not math.isfinite(float(row[metric])) for row in all_rows for metric in METRIC_NAMES):
        raise RuntimeError("Non-finite sequence metric detected")
    write_json(args.output_dir / "sequence_evaluation_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
