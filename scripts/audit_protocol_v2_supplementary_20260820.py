from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the completed Protocol v2 supplementary experiment package.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def scan_finite_csv(path: Path) -> list[str]:
    issues = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed = float(value)
                except ValueError:
                    continue
                if not math.isfinite(parsed):
                    issues.append(f"{path}: row {row_number}, column {key} is non-finite")
    return issues


def main() -> None:
    args = parse_args()
    plan = read_json(args.plan)
    registry = read_json(args.registry)
    issues = []
    stage_rows = []
    state_dir = args.output_root / "stage_state"
    for stage in plan["stages"]:
        exit_path = state_dir / f"{stage['id']}_exit.json"
        run_dir = args.output_root / Path(stage["output_rel"])
        stage_issues = []
        if not exit_path.is_file():
            stage_issues.append("missing exit state")
        else:
            state = read_json(exit_path)
            if state.get("status") != "completed" or int(state.get("exit_code", 1)) != 0:
                stage_issues.append(f"invalid exit state: {state.get('status')}/{state.get('exit_code')}")
        for name in ("best.pt", "last.pt", "results.json", "train_history.json"):
            if not (run_dir / name).is_file():
                stage_issues.append(f"missing {name}")
        stage_rows.append({"stage_id": stage["id"], "status": "PASS" if not stage_issues else "FAIL", "issues": stage_issues})
        issues.extend(f"{stage['id']}: {message}" for message in stage_issues)

    for dataset_name in ("cmu", "kit", "bmlmovi"):
        metadata_path = Path(plan["protocol_root"]) / dataset_name / "common18" / "metadata.json"
        metadata = read_json(metadata_path)
        if str(metadata.get("split_manifest_sha256", "")).lower() != str(plan["manifest_sha256"]).lower():
            issues.append(f"{dataset_name}: manifest mismatch")
        if metadata.get("split_unit") != "subject" or metadata.get("center_on_first_frame") is not False:
            issues.append(f"{dataset_name}: protocol metadata mismatch")

    missing_registry = []
    for entry in registry["models"]:
        for field in ("checkpoint", "history"):
            path = Path(entry[field])
            if not path.is_file():
                missing_registry.append(str(path))
    if missing_registry:
        issues.append(f"Registry is missing {len(missing_registry)} artifacts")

    expected_analytics = (
        args.output_root / "analysis" / "sequence" / "sequence_evaluation_summary.json",
        args.output_root / "analysis" / "sequence" / "per_sequence_metrics.csv",
        args.output_root / "analysis" / "sequence" / "multiseed_summary.csv",
        args.output_root / "analysis" / "sequence" / "paired_statistics.csv",
        args.output_root / "analysis" / "sequence" / "ablation_contrasts.csv",
        args.output_root / "analysis" / "efficiency" / "efficiency_benchmark.json",
        args.output_root / "analysis" / "efficiency" / "efficiency_benchmark.csv",
        args.output_root / "analysis" / "curves" / "curve_manifest.json",
        args.output_root / "analysis" / "curves" / "training_curve_data.csv",
    )
    for path in expected_analytics:
        if not path.is_file():
            issues.append(f"Missing analysis artifact: {path}")
        elif path.suffix.lower() == ".csv":
            issues.extend(scan_finite_csv(path))

    seed_coverage = {}
    for dataset_name in ("cmu", "kit", "bmlmovi"):
        seed_coverage[dataset_name] = {}
        for group in ("full", "hisrep"):
            seeds = sorted(
                int(entry["seed"]) for entry in registry["models"]
                if entry["dataset"] == dataset_name and entry.get("analysis_group") == group
            )
            seed_coverage[dataset_name][group] = seeds
            if seeds != [42, 123, 456]:
                issues.append(f"{dataset_name}/{group}: invalid seed coverage {seeds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "PASS" if not issues else "FAIL",
        "experiment_id": plan["experiment_id"],
        "plan": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "registry": str(args.registry),
        "registry_sha256": sha256_file(args.registry),
        "manifest_sha256": plan["manifest_sha256"],
        "training_stage_count": len(plan["stages"]),
        "seed_coverage": seed_coverage,
        "stage_audits": stage_rows,
        "issues": issues,
    }
    report_path = args.output_dir / "supplementary_audit.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if issues:
        raise RuntimeError(f"Supplementary audit failed with {len(issues)} issue(s); see {report_path}")
    pass_payload = {
        "status": "PASS",
        "audit_report": str(report_path),
        "audit_report_sha256": sha256_file(report_path),
        "plan_sha256": sha256_file(args.plan),
        "registry_sha256": sha256_file(args.registry),
    }
    (args.output_dir / "supplementary_audit_pass.json").write_text(
        json.dumps(pass_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(pass_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
