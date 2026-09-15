from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight for the reviewer-minimum serial queue")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = read_json(args.plan)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    stages = plan.get("stages", [])
    check("training_stage_count", len(stages) == 33, len(stages))
    stage_ids = [item["id"] for item in stages]
    check("unique_stage_ids", len(stage_ids) == len(set(stage_ids)), len(set(stage_ids)))
    dataset_order = [item["dataset"] for item in stages]
    order_map = {name: index for index, name in enumerate(plan["strict_order"])}
    check(
        "strict_dataset_order",
        dataset_order == sorted(dataset_order, key=lambda name: order_map[name]),
        dataset_order,
    )
    known_ids = set(stage_ids)
    dependencies_ok = all(
        dependency in known_ids
        for item in stages
        for dependency in item.get("depends_on", [])
    )
    check("dependencies_resolve", dependencies_ok, "all stage dependencies")
    exact_main = [
        item for item in stages
        if item.get("compatibility_profile") == "legacy_seed42_exact"
    ]
    exact_ablation = [
        item for item in stages
        if item.get("compatibility_profile") == "legacy_ablation_exact"
    ]
    main_roles_ok = (
        len(exact_main) == 18
        and all(item["kind"] in {"ltc", "hisrep"} for item in exact_main)
        and all(int(item["seed"]) in {123, 456} for item in exact_main)
    )
    check(
        "legacy_seed42_exact_compatibility_profiles",
        main_roles_ok,
        {"main_stage_count": len(exact_main), "ablation_stage_count": len(exact_ablation)},
    )
    check(
        "legacy_ablation_compatibility_profiles",
        len(exact_ablation) == 8 and all(item["kind"] == "ltc" for item in exact_ablation),
        [item["id"] for item in exact_ablation],
    )
    recent = [item for item in stages if item.get("family") in {"graph", "transformer", "mlp", "diffusion"}]
    check(
        "representative_cmu_recent_methods",
        len(recent) == 4
        and {item.get("family") for item in recent} == {"graph", "transformer", "mlp", "diffusion"}
        and all(item["dataset"] == "cmu" and int(item["seed"]) == 42 for item in recent),
        [item["id"] for item in recent],
    )
    lstm = [item for item in stages if item.get("baseline") == "lstm"]
    check(
        "controlled_lstm_all_datasets",
        len(lstm) == 3 and {item["dataset"] for item in lstm} == {"cmu", "kit", "bmlmovi"},
        [item["id"] for item in lstm],
    )
    sensitivity = [item for item in stages if "sensitivity_" in item["id"]]
    check(
        "compact_cmu_sensitivity",
        len(sensitivity) == 4
        and {item.get("loss_profile") for item in sensitivity}
        == {"regularizers_half", "regularizers_double", "no_velocity", "equal_stage_weights"}
        and all(item["dataset"] == "cmu" for item in sensitivity),
        [item["id"] for item in sensitivity],
    )
    root_ablation = [item for item in stages if "no_anchor_" in item["id"]]
    topology_2x2 = [item for item in stages if "original_ltc_" in item["id"]]
    check(
        "representative_cmu_root_ablation",
        len(root_ablation) == 2 and all(item["dataset"] == "cmu" for item in root_ablation),
        [item["id"] for item in root_ablation],
    )
    check(
        "representative_cmu_topology_2x2",
        len(topology_2x2) == 2 and all(item["dataset"] == "cmu" for item in topology_2x2),
        [item["id"] for item in topology_2x2],
    )
    check(
        "excluded_out_of_scope_training",
        not any(item["dataset"] == "h36m" or "compute_matched" in item["id"] for item in stages),
        "no additional database and no compute-matched stages",
    )

    data_root = Path(plan["data_root"])
    expected_manifest = plan["split_manifest_sha256"].lower()
    for dataset in ("cmu", "kit", "bmlmovi"):
        metadata_path = data_root / dataset / "common18" / "metadata.json"
        if not metadata_path.is_file():
            check(f"{dataset}_metadata", False, str(metadata_path))
            continue
        metadata = read_json(metadata_path)
        passed = (
            metadata.get("representation") == "common18"
            and int(metadata.get("num_joints", 0)) == 18
            and str(metadata.get("split_manifest_sha256", "")).lower() == expected_manifest
            and metadata["joint_names"][int(metadata["guidance_joint_index"])] == "spine2"
        )
        check(
            f"{dataset}_metadata",
            passed,
            {"path": str(metadata_path), "sha256": sha256_file(metadata_path)},
        )

    source_plan_path = Path(
        r"E:\AMASS_LNN_Project\runs\formal_protocol_v2_subject_grouped_rootrel_pose_20260815\formal_execution_plan.json"
    )
    source_plan = read_json(source_plan_path)
    source_settings_ok = (
        source_plan["manifest_sha256"].lower() == expected_manifest
        and source_plan["training"] == {
            "epochs": 30,
            "batch_size": 128,
            "num_workers": 0,
            "seed": 42,
            "device": "cuda",
            "full_train_windows": True,
            "full_validation_windows": True,
        }
    )
    check(
        "source_18stage_frozen_settings",
        source_settings_ok,
        {"path": str(source_plan_path), "sha256": sha256_file(source_plan_path)},
    )

    reusable_records: list[dict[str, object]] = []
    for item in plan["reused_artifacts"]:
        source_dir = Path(item["source_dir"])
        dataset = item["dataset"]
        audit_path = source_dir.parent / "audit" / "dataset_audit_pass.json"
        artifact_records = []
        passed = True
        for name in item["required_outputs"]:
            path = source_dir / name
            if not path.is_file():
                passed = False
                artifact_records.append({"name": name, "missing": True})
            else:
                artifact_records.append(
                    {
                        "name": name,
                        "path": str(path),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        if not audit_path.is_file():
            passed = False
            audit_status = "MISSING"
        else:
            audit = read_json(audit_path)
            audit_status = audit.get("status")
            passed = passed and audit_status == "PASS"
        hisrep_compatibility = None
        if item["role"] == "hisrep":
            config_path = source_dir / "config.json"
            if not config_path.is_file():
                passed = False
                hisrep_compatibility = {"config_path": str(config_path), "missing": True}
            else:
                hisrep_config = read_json(config_path)["config"]
                expected_hisrep = {
                    "seed": 42,
                    "input_n": 25,
                    "output_n": 25,
                    "model_output_n": 10,
                    "itera": 3,
                    "dct_n": 20,
                    "epochs": 30,
                    "batch_size": 128,
                    "stride": 5,
                }
                actual_hisrep = {name: hisrep_config.get(name) for name in expected_hisrep}
                hisrep_compatibility = {
                    "config_path": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "expected": expected_hisrep,
                    "actual": actual_hisrep,
                }
                passed = passed and actual_hisrep == expected_hisrep
        record = {
            "id": item["id"],
            "dataset": dataset,
            "role": item["role"],
            "passed": passed,
            "dataset_audit": str(audit_path),
            "dataset_audit_status": audit_status,
            "artifacts": artifact_records,
            "hisrep_compatibility": hisrep_compatibility,
        }
        reusable_records.append(record)
        check(f"reuse_{item['id']}", passed, record)

    source_checks = {}
    for path_text, expected in plan["source_sha256"].items():
        path = Path(path_text)
        actual = sha256_file(path) if path.is_file() else None
        source_checks[path_text] = {"expected": expected, "actual": actual}
    source_passed = all(
        record["expected"] is not None and record["expected"] == record["actual"]
        for record in source_checks.values()
    )
    check("source_hashes_frozen", source_passed, source_checks)

    passed = all(item["passed"] for item in checks)
    formal_ready = passed
    payload = {
        "status": "PASS" if passed else "BLOCKED",
        "formal_ready": formal_ready,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "checks": checks,
        "reused_artifact_inventory": reusable_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
