from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


DATASETS = ("cmu", "kit", "bmlmovi")
SEEDS = (42, 123, 456)
CHECKPOINT_SELECTION = "validation_recursive_75_frame_physical_mpjpe_mm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the complete 51-stage revision experiment package.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_csv_issues(path: Path) -> list[str]:
    issues = []
    for row_number, row in enumerate(read_csv(path), start=2):
        for key, value in row.items():
            if value in (None, ""):
                continue
            try:
                parsed = float(value)
            except ValueError:
                continue
            if not math.isfinite(parsed):
                issues.append(f"{path}: row {row_number}, column {key} is non-finite")
    return issues


def verify_artifact_record(record: dict, expected_path: Path) -> list[str]:
    issues = []
    if Path(record.get("path", "")).resolve() != expected_path.resolve():
        issues.append(f"recorded artifact path mismatch for {expected_path.name}")
        return issues
    if not expected_path.is_file():
        issues.append(f"missing {expected_path.name}")
        return issues
    if int(record.get("size", -1)) != expected_path.stat().st_size:
        issues.append(f"artifact size mismatch for {expected_path.name}")
    if str(record.get("sha256", "")).lower() != sha256_file(expected_path):
        issues.append(f"artifact hash mismatch for {expected_path.name}")
    return issues


def verify_result_binding(stage: dict, result: dict, plan: dict) -> list[str]:
    issues = []
    config = result.get("config", {})
    if result.get("status") != "completed":
        issues.append("results.json is not completed")
    if str(result.get("representation")) != "common18":
        issues.append("result representation is not common18")
    if int(result.get("seed", -1)) != int(stage["seed"]):
        issues.append("seed mismatch")
    if str(result.get("checkpoint_selection")) != CHECKPOINT_SELECTION:
        issues.append("checkpoint selection mismatch")
    if str(config.get("manifest_sha256", "")).lower() != str(plan["split_manifest_sha256"]).lower():
        issues.append("split manifest mismatch")
    if not config.get("deterministic_algorithms"):
        issues.append("deterministic algorithms were not recorded")
    if not str(config.get("frozen_inventory_sha256", "")):
        issues.append("missing frozen inventory binding")

    if stage["kind"] == "hisrep":
        hisrep_config = config.get("hisrep_config", {})
        if int(hisrep_config.get("seed", -1)) != int(stage["seed"]):
            issues.append("HisRep config seed mismatch")
        if int(config.get("observation_frames", -1)) != 25 or int(hisrep_config.get("input_n", -1)) != 50:
            issues.append("HisRep 25-observation/internal-50 adapter mismatch")
        if int(hisrep_config.get("output_n", -1)) != 25 or bool(config.get("uses_additional_real_history", True)):
            issues.append("HisRep fair-observation configuration mismatch")
        if int(stage.get("observation_frames", -1)) != 25 or int(stage.get("internal_input_frames", -1)) != 50:
            issues.append("HisRep plan adapter mismatch")
        if str(config.get("observation_adapter")) != str(stage.get("observation_adapter")):
            issues.append("HisRep observation adapter binding mismatch")
    else:
        architecture = result.get("architecture", {})
        if str(result.get("training_objective")) != str(stage["objective"]):
            issues.append("training objective mismatch")
        if str(config.get("loss_coordinate_space")) != "denormalized_meter":
            issues.append("loss coordinate space mismatch")
        if str(architecture.get("model")) != str(stage["model"]):
            issues.append("model family mismatch")
        expected_adjacency = "not_applicable" if stage["model"] == "ltc" else stage["adjacency_mode"]
        if str(architecture.get("adjacency_mode")) != expected_adjacency:
            issues.append("adjacency mode mismatch")
        expected_anchor = bool(stage["anchor_guidance"]) if stage["model"] == "ltc_topology" else False
        if bool(architecture.get("use_anchor_guidance")) != expected_anchor:
            issues.append("anchor-guidance setting mismatch")
        profile = str(stage.get("loss_profile", "default"))
        defaults = plan["defaults"]
        expected = {
            "lambda_bone": float(defaults["lambda_bone"]),
            "lambda_anchor": float(defaults["lambda_anchor"]),
            "lambda_velocity": float(defaults["lambda_velocity"]),
        }
        if profile == "regularizers_half":
            expected = {key: value * 0.5 for key, value in expected.items()}
        elif profile == "regularizers_double":
            expected = {key: value * 2.0 for key, value in expected.items()}
        elif profile == "no_anchor_loss":
            expected["lambda_anchor"] = 0.0
        for key, expected_value in expected.items():
            if not math.isclose(float(config.get(key, math.nan)), expected_value, rel_tol=0.0, abs_tol=1e-12):
                issues.append(f"{key} does not match loss profile {profile}")
    return issues


def expected_matrix_issues(plan: dict) -> list[str]:
    issues = []
    stages = plan["stages"]
    if len(stages) != 51:
        issues.append(f"expected 51 stages, found {len(stages)}")
    if len({stage["id"] for stage in stages}) != len(stages):
        issues.append("duplicate stage IDs")
    observed_order = [stage["dataset"] for stage in stages]
    order_rank = {dataset: index for index, dataset in enumerate(DATASETS)}
    if observed_order != sorted(observed_order, key=lambda dataset: order_rank[dataset]):
        issues.append("dataset order is not strictly CMU, KIT, BMLmovi")
    expected_counts = {"cmu": 19, "kit": 16, "bmlmovi": 16}
    counts = Counter(stage["dataset"] for stage in stages)
    if dict(counts) != expected_counts:
        issues.append(f"dataset stage counts mismatch: {dict(counts)}")
    ids = {stage["id"] for stage in stages}
    for dataset in DATASETS:
        for seed in SEEDS:
            for suffix in ("skeleton_no_rollout", "skeleton_full_rollout", "hisrep"):
                stage_id = f"{dataset}_seed{seed}_{suffix}"
                if stage_id not in ids:
                    issues.append(f"missing primary stage {stage_id}")
        for suffix in (
            "identity_no_rollout",
            "identity_full_rollout",
            "no_anchor_no_rollout",
            "no_anchor_full_rollout",
            "compute_matched_continued_no_rollout",
            "original_ltc_no_rollout",
            "original_ltc_full_rollout",
        ):
            stage_id = f"{dataset}_seed42_{suffix}"
            if stage_id not in ids:
                issues.append(f"missing ablation stage {stage_id}")
    for suffix in ("regularizers_half", "regularizers_double", "no_anchor_loss"):
        stage_id = f"cmu_seed42_sensitivity_{suffix}"
        if stage_id not in ids:
            issues.append(f"missing sensitivity stage {stage_id}")
    return issues


def verify_analysis(run_root: Path) -> tuple[list[str], dict]:
    issues = []
    paths = {
        "sequence_summary": run_root / "analysis/sequence/sequence_evaluation_summary.json",
        "per_sequence": run_root / "analysis/sequence/per_sequence_metrics.csv",
        "aggregate": run_root / "analysis/sequence/aggregate_metrics.csv",
        "multiseed": run_root / "analysis/sequence/multiseed_summary.csv",
        "sensitivity": run_root / "analysis/sequence/sensitivity_summary.csv",
        "paired": run_root / "analysis/sequence/paired_statistics.csv",
        "ablations": run_root / "analysis/sequence/ablation_contrasts.csv",
        "efficiency_json": run_root / "analysis/efficiency/efficiency_benchmark.json",
        "efficiency_csv": run_root / "analysis/efficiency/efficiency_benchmark.csv",
        "curve_manifest": run_root / "analysis/curves/curve_manifest.json",
        "curve_csv": run_root / "analysis/curves/training_curve_data.csv",
    }
    row_expectations = {
        "multiseed": 6,
        "sensitivity": 4,
        "paired": 12,
        "ablations": 120,
        "efficiency_csv": 12,
    }
    summary = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing or empty analysis artifact: {path}")
            continue
        summary[name] = {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}
        if path.suffix.lower() == ".csv":
            issues.extend(finite_csv_issues(path))
            rows = read_csv(path)
            summary[name]["rows"] = len(rows)
            expected = row_expectations.get(name)
            if expected is not None and len(rows) != expected:
                issues.append(f"{name}: expected {expected} rows, found {len(rows)}")

    curve_manifest_path = paths["curve_manifest"]
    if curve_manifest_path.is_file():
        manifest = read_json(curve_manifest_path)
        figures = [Path(path) for path in manifest.get("generated_figures", [])]
        if len(figures) != 12:
            issues.append(f"expected 12 curve figures, found {len(figures)}")
        for figure in figures:
            if not figure.is_file() or figure.stat().st_size == 0:
                issues.append(f"missing or empty curve figure: {figure}")
    return issues, summary


def main() -> None:
    args = parse_args()
    plan = read_json(args.plan)
    run_root = Path(plan["run_root"])
    issues = expected_matrix_issues(plan)
    stage_audits = []
    result_index = {}

    for stage in plan["stages"]:
        stage_id = str(stage["id"])
        run_dir = run_root / str(stage["output_rel"])
        exit_path = run_root / "stage_state" / f"{stage_id}_exit.json"
        stage_issues = []
        if not exit_path.is_file():
            stage_issues.append("missing exit state")
        else:
            state = read_json(exit_path)
            if state.get("status") != "completed" or int(state.get("exit_code", 1)) != 0:
                stage_issues.append(f"invalid exit state {state.get('status')}/{state.get('exit_code')}")
            if state.get("smoke"):
                stage_issues.append("formal exit state is marked as smoke")
            if str(state.get("plan_sha256", "")).lower() != sha256_file(args.plan):
                stage_issues.append("exit state plan hash mismatch")
            records = {record.get("name"): record for record in state.get("artifacts", [])}
            for name in stage["required_outputs"]:
                if name not in records:
                    stage_issues.append(f"missing artifact record for {name}")
                else:
                    stage_issues.extend(verify_artifact_record(records[name], run_dir / name))

        result_path = run_dir / "results.json"
        if result_path.is_file():
            result = read_json(result_path)
            result_index[stage_id] = result
            stage_issues.extend(verify_result_binding(stage, result, plan))
            checkpoint = Path(result.get("best_checkpoint", ""))
            if not checkpoint.is_file():
                stage_issues.append("best checkpoint is missing")
            elif str(result.get("best_checkpoint_sha256", "")).lower() != sha256_file(checkpoint):
                stage_issues.append("best checkpoint result hash mismatch")
            history = Path(result.get("history_file", ""))
            if not history.is_file():
                stage_issues.append("history file is missing")
            elif str(result.get("history_sha256", "")).lower() != sha256_file(history):
                stage_issues.append("history result hash mismatch")
        else:
            stage_issues.append("missing results.json")

        stage_audits.append(
            {"stage_id": stage_id, "status": "PASS" if not stage_issues else "FAIL", "issues": stage_issues}
        )
        issues.extend(f"{stage_id}: {issue}" for issue in stage_issues)

    for stage in plan["stages"]:
        if not stage.get("warm_start_id") or stage["id"] not in result_index:
            continue
        dependency = result_index.get(stage["warm_start_id"])
        current = result_index[stage["id"]]
        if dependency and str(current.get("warm_start_sha256", "")).lower() != str(
            dependency.get("best_checkpoint_sha256", "")
        ).lower():
            issues.append(f"{stage['id']}: warm-start checkpoint hash is not bound to {stage['warm_start_id']}")

    for dataset in DATASETS:
        metadata_path = Path(plan["data_root"]) / dataset / "common18/metadata.json"
        stats_path = Path(plan["data_root"]) / dataset / "common18/normalization_stats.npz"
        metadata = read_json(metadata_path)
        if str(metadata.get("split_manifest_sha256", "")).lower() != str(plan["split_manifest_sha256"]).lower():
            issues.append(f"{dataset}: metadata split manifest mismatch")
        if metadata.get("split_unit") != "subject" or metadata.get("representation") != "common18":
            issues.append(f"{dataset}: protocol metadata mismatch")
        if sha256_file(metadata_path) != plan["metadata_sha256"][dataset]:
            issues.append(f"{dataset}: metadata changed after plan freeze")
        if sha256_file(stats_path) != plan["normalization_sha256"][dataset]:
            issues.append(f"{dataset}: normalization statistics changed after plan freeze")

    analysis_issues, analysis_artifacts = verify_analysis(run_root)
    issues.extend(analysis_issues)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS" if not issues else "FAIL",
        "protocol": plan["protocol"],
        "plan": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "training_stage_count": len(plan["stages"]),
        "checkpoint_selection": CHECKPOINT_SELECTION,
        "loss_coordinate_space": "denormalized_meter",
        "stage_audits": stage_audits,
        "analysis_artifacts": analysis_artifacts,
        "issues": issues,
    }
    report_path = args.output_dir / "revision_51stage_audit.json"
    write_json(report_path, payload)
    if issues:
        raise RuntimeError(f"Revision audit failed with {len(issues)} issue(s); see {report_path}")
    pass_payload = {
        "status": "PASS",
        "audit_report": str(report_path),
        "audit_report_sha256": sha256_file(report_path),
        "plan_sha256": sha256_file(args.plan),
        "training_stage_count": 51,
        "all_formal_stages_completed": True,
        "all_artifact_hashes_verified": True,
        "all_analysis_outputs_verified": True,
    }
    pass_path = args.output_dir / "revision_51stage_audit_pass.json"
    write_json(pass_path, pass_payload)
    print(json.dumps(pass_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
