from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete artifact audit for the reviewer-minimum queue")
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


def csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    plan = read_json(args.plan)
    run_root = Path(plan["run_root"])
    checks: list[dict] = []

    def check(name: str, passed: bool, detail) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    stages = plan["stages"]
    check("training_stage_count", len(stages) == 33, len(stages))
    for stage in stages:
        exit_path = run_root / "stage_state" / f"{stage['id']}_exit.json"
        passed = exit_path.is_file()
        details = {"exit_path": str(exit_path)}
        if passed:
            state = read_json(exit_path)
            passed = (
                state.get("status") == "completed"
                and int(state.get("exit_code", -1)) == 0
                and not state.get("smoke", False)
                and state.get("plan_sha256") == sha256_file(args.plan)
            )
            run_dir = run_root / stage["output_rel"]
            for name in stage["required_outputs"]:
                path = run_dir / name
                record = next((item for item in state.get("artifacts", []) if item.get("name") == name), None)
                passed = passed and path.is_file() and record is not None
                if path.is_file() and record is not None:
                    passed = passed and record["sha256"] == sha256_file(path) and int(record["size"]) == path.stat().st_size
        check(f"stage_{stage['id']}", passed, details)

    inventory_path = run_root / "reused" / "reused_inventory.json"
    inventory_ok = inventory_path.is_file()
    if inventory_ok:
        inventory = read_json(inventory_path)
        inventory_ok = inventory.get("status") == "PASS" and inventory.get("plan_sha256") == sha256_file(args.plan)
        for item in inventory.get("files", []):
            path = Path(item["path"])
            inventory_ok = (
                inventory_ok
                and path.is_file()
                and sha256_file(path) == item["sha256"]
                and path.stat().st_size == int(item["size"])
            )
        for record in inventory.get("records", []):
            for item in record.get("source_files", []):
                path = Path(item["path"])
                inventory_ok = inventory_ok and path.is_file() and sha256_file(path) == item["sha256"]
    check("reused_seed42_inventory", inventory_ok, str(inventory_path))

    analysis_root = run_root / "analysis"
    expected_analysis = {
        "aggregate": analysis_root / "sequence" / "aggregate_metrics.csv",
        "multiseed": analysis_root / "sequence" / "multiseed_summary.csv",
        "sensitivity": analysis_root / "sequence" / "sensitivity_summary.csv",
        "statistics": analysis_root / "sequence" / "paired_statistics.csv",
        "ablation": analysis_root / "sequence" / "ablation_contrasts.csv",
        "sequence_summary": analysis_root / "sequence" / "sequence_evaluation_summary.json",
        "efficiency_csv": analysis_root / "efficiency" / "common18_efficiency.csv",
        "efficiency_json": analysis_root / "efficiency" / "common18_efficiency.json",
        "stability": analysis_root / "curves" / "checkpoint_stability.csv",
        "curve_summary": analysis_root / "curves" / "curve_summary.json",
    }
    for name, path in expected_analysis.items():
        check(f"analysis_{name}", path.is_file() and path.stat().st_size > 0, str(path))
    if all(path.is_file() for path in expected_analysis.values()):
        check("aggregate_row_count", len(csv_rows(expected_analysis["aggregate"])) == 42, len(csv_rows(expected_analysis["aggregate"])))
        check("multiseed_row_count", len(csv_rows(expected_analysis["multiseed"])) == 6, len(csv_rows(expected_analysis["multiseed"])))
        check("sensitivity_row_count", len(csv_rows(expected_analysis["sensitivity"])) == 5, len(csv_rows(expected_analysis["sensitivity"])))
        check("paired_statistics_row_count", len(csv_rows(expected_analysis["statistics"])) == 24, len(csv_rows(expected_analysis["statistics"])))
        check("ablation_row_count", len(csv_rows(expected_analysis["ablation"])) == 24, len(csv_rows(expected_analysis["ablation"])))
        check("efficiency_row_count", len(csv_rows(expected_analysis["efficiency_csv"])) == 13, len(csv_rows(expected_analysis["efficiency_csv"])))
        check("stability_row_count", len(csv_rows(expected_analysis["stability"])) == 18, len(csv_rows(expected_analysis["stability"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    passed = all(item["passed"] for item in checks)
    if not passed:
        failure = {
            "status": "BLOCKED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "plan": str(args.plan),
            "plan_sha256": sha256_file(args.plan),
            "checks": checks,
        }
        (args.output_dir / "reviewer_minimum_audit_failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise SystemExit("Reviewer-minimum artifact audit failed")

    files = []
    for path in sorted(analysis_root.rglob("*")):
        if path.is_file():
            files.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "training_stages_completed": 33,
        "reused_seed42_artifacts": 9,
        "checks": checks,
        "artifacts": files,
    }
    pass_path = args.output_dir / "reviewer_minimum_audit_pass.json"
    pass_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
