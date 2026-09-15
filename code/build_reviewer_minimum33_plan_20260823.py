from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(r"E:\AMASS_LNN_Project")
RUN_ROOT = PROJECT / "runs" / "formal_reviewer_minimum33_revision_20260823"
SOURCE_18 = PROJECT / "runs" / "formal_protocol_v2_subject_grouped_rootrel_pose_20260815"
DATA_ROOT = PROJECT / "data" / "processed" / "protocol_v2_subject_grouped_rootrel_pose"
MANIFEST_SHA256 = "2e4ed5afb524c328efdd1246fa61fb96e104b39484e17d55784fa3ef2aec879e"
DATASETS = ("cmu", "kit", "bmlmovi")
DISPLAY_NAMES = {"cmu": "CMU", "kit": "KIT", "bmlmovi": "BMLmovi"}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage(
    stage_id: str,
    dataset: str,
    kind: str,
    output_rel: str,
    *,
    depends_on: list[str] | None = None,
    **settings: object,
) -> dict[str, object]:
    required = ["best.pt", "last.pt", "train_history.json", "results.json"]
    if kind == "hisrep":
        required.insert(2, "config.json")
    if settings.get("compatibility_profile") == "legacy_seed42_exact" and settings.get("objective") == "rollout":
        required.remove("last.pt")
    return {
        "id": stage_id,
        "dataset": dataset,
        "dataset_name": DISPLAY_NAMES[dataset],
        "kind": kind,
        "output_rel": output_rel,
        "depends_on": depends_on or [],
        "required_outputs": required,
        **settings,
    }


def reusable_artifacts() -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for dataset in DATASETS:
        for role, run_name in (
            ("skeleton_no_rollout", "common18_no_rollout"),
            ("skeleton_full_rollout", "common18_full_rollout"),
            ("hisrep", "hisrep_common18"),
        ):
            source = SOURCE_18 / dataset / run_name
            required_outputs = ["best.pt", "results.json", "train_history.json"]
            compatibility_requirements: dict[str, object] = {
                "manifest_sha256": MANIFEST_SHA256,
                "representation": "common18",
                "history": 25,
                "future_step": 25,
                "validation_horizon": 75,
                "epochs": 30,
                "batch_size": 128,
                "seed": 42,
                "dataset_audit_status": "PASS",
            }
            if role == "hisrep":
                required_outputs.append("config.json")
                compatibility_requirements.update(
                    {
                        "input_n": 25,
                        "output_n": 25,
                        "model_output_n": 10,
                        "itera": 3,
                        "dct_n": 20,
                        "stride": 5,
                    }
                )
            items.append(
                {
                    "id": f"reuse_{dataset}_seed42_{role}",
                    "dataset": dataset,
                    "seed": 42,
                    "role": role,
                    "source_dir": str(source),
                    "required_outputs": required_outputs,
                    "compatibility_requirements": compatibility_requirements,
                }
            )
    return items


def common_stages() -> list[dict[str, object]]:
    stages: list[dict[str, object]] = []
    for dataset in DATASETS:
        prefix = dataset
        for seed in (123, 456):
            no_rollout = f"{prefix}_seed{seed}_skeleton_no_rollout"
            full = f"{prefix}_seed{seed}_skeleton_full_rollout"
            stages.append(
                stage(
                    no_rollout,
                    dataset,
                    "ltc",
                    f"{dataset}/seed{seed}_skeleton_no_rollout",
                    objective="no_rollout",
                    model="ltc_topology",
                    seed=seed,
                    adjacency_mode="skeleton",
                    anchor_guidance=True,
                    loss_profile="default",
                    compatibility_profile="legacy_seed42_exact",
                )
            )
            stages.append(
                stage(
                    full,
                    dataset,
                    "ltc",
                    f"{dataset}/seed{seed}_skeleton_full_rollout",
                    depends_on=[no_rollout],
                    objective="rollout",
                    model="ltc_topology",
                    seed=seed,
                    adjacency_mode="skeleton",
                    anchor_guidance=True,
                    loss_profile="default",
                    compatibility_profile="legacy_seed42_exact",
                    warm_start_id=no_rollout,
                )
            )
            stages.append(
                stage(
                    f"{prefix}_seed{seed}_hisrep",
                    dataset,
                    "hisrep",
                    f"{dataset}/seed{seed}_hisrep",
                    seed=seed,
                    observation_frames=25,
                    internal_input_frames=25,
                    observation_adapter="legacy_direct_25_frame_input",
                    compatibility_profile="legacy_seed42_exact",
                )
            )

        if dataset == "cmu":
            no_anchor = "cmu_seed42_no_anchor_no_rollout"
            stages.append(
                stage(
                    no_anchor,
                    dataset,
                    "ltc",
                    "cmu/seed42_no_anchor_no_rollout",
                    objective="no_rollout",
                    model="ltc_topology",
                    seed=42,
                    adjacency_mode="skeleton",
                    anchor_guidance=False,
                    loss_profile="default",
                    compatibility_profile="legacy_ablation_exact",
                )
            )
            stages.append(
                stage(
                    "cmu_seed42_no_anchor_full_rollout",
                    dataset,
                    "ltc",
                    "cmu/seed42_no_anchor_full_rollout",
                    depends_on=[no_anchor],
                    objective="rollout",
                    model="ltc_topology",
                    seed=42,
                    adjacency_mode="skeleton",
                    anchor_guidance=False,
                    loss_profile="default",
                    compatibility_profile="legacy_ablation_exact",
                    warm_start_id=no_anchor,
                )
            )
            for profile in (
                "regularizers_half",
                "regularizers_double",
                "no_velocity",
                "equal_stage_weights",
            ):
                stages.append(
                    stage(
                        f"cmu_seed42_sensitivity_{profile}",
                        dataset,
                        "ltc",
                        f"cmu/seed42_sensitivity_{profile}",
                        objective="rollout",
                        model="ltc_topology",
                        seed=42,
                        adjacency_mode="skeleton",
                        anchor_guidance=True,
                        loss_profile=profile,
                        compatibility_profile="legacy_ablation_exact",
                        warm_start_reuse_id="reuse_cmu_seed42_skeleton_no_rollout",
                    )
                )

        stages.append(
            stage(
                f"{prefix}_seed42_lstm",
                dataset,
                "baseline",
                f"{dataset}/seed42_lstm",
                baseline="lstm",
                family="recurrent",
                seed=42,
            )
        )
        if dataset == "cmu":
            for baseline, family, batch_size in (
                ("msrgcn", "graph", 128),
                ("st_transformer", "transformer", 64),
                ("simlpe", "mlp", 128),
                ("humanmac", "diffusion", 32),
            ):
                stages.append(
                    stage(
                        f"cmu_seed42_{baseline}",
                        dataset,
                        "baseline",
                        f"cmu/seed42_{baseline}",
                        baseline=baseline,
                        family=family,
                        seed=42,
                        batch_size_override=batch_size,
                    )
                )

            original_no = "cmu_seed42_original_ltc_no_rollout"
            stages.append(
                stage(
                    original_no,
                    dataset,
                    "ltc",
                    "cmu/seed42_original_ltc_no_rollout",
                    objective="no_rollout",
                    model="ltc",
                    seed=42,
                    adjacency_mode="skeleton",
                    anchor_guidance=False,
                    loss_profile="default",
                    compatibility_profile="legacy_ablation_exact",
                )
            )
            stages.append(
                stage(
                    "cmu_seed42_original_ltc_full_rollout",
                    dataset,
                    "ltc",
                    "cmu/seed42_original_ltc_full_rollout",
                    depends_on=[original_no],
                    objective="rollout",
                    model="ltc",
                    seed=42,
                    adjacency_mode="skeleton",
                    anchor_guidance=False,
                    loss_profile="default",
                    compatibility_profile="legacy_ablation_exact",
                    warm_start_id=original_no,
                )
            )
    return stages


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    stages = common_stages()
    if len(stages) != 33:
        raise RuntimeError(f"Expected 33 new training stages, got {len(stages)}")
    source_files = [
        PROJECT / "build_reviewer_minimum33_plan_20260823.py",
        PROJECT / "src" / "models.py",
        PROJECT / "src" / "reviewer_baselines.py",
        PROJECT / "train.py",
        PROJECT / "train_ltc_topology_rollout_consistency.py",
        PROJECT / "train_protocol_v2_revision_20260821.py",
        PROJECT / "train_protocol_v2_revision_hisrep_20260821.py",
        PROJECT / "train_hisrep_cmu_clean.py",
        PROJECT / "train_protocol_v2_reviewer_baseline_20260822.py",
        PROJECT / "train_reviewer_ltc_legacy_compatible_20260822.py",
        PROJECT / "run_reviewer_minimum33_stage_20260823.ps1",
        PROJECT / "manage_reviewer_minimum33_queue_20260823.ps1",
        PROJECT / "run_reviewer_minimum33_smoke_20260823.ps1",
        PROJECT / "run_reviewer_minimum33_postanalysis_20260823.ps1",
        PROJECT / "audit_reviewer_minimum33_preflight_20260823.py",
        PROJECT / "materialize_reviewer_reuse_20260822.py",
        PROJECT / "scripts" / "evaluate_reviewer_minimum33_20260823.py",
        PROJECT / "scripts" / "benchmark_reviewer_minimum33_efficiency_20260823.py",
        PROJECT / "scripts" / "plot_reviewer_minimum33_curves_20260823.py",
        PROJECT / "scripts" / "audit_reviewer_minimum33_20260823.py",
    ]
    plan = {
        "protocol": "reviewer_required_minimum_revision",
        "version": "20260823-minimum33-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT),
        "run_root": str(RUN_ROOT),
        "data_root": str(DATA_ROOT),
        "new_training_stage_count": len(stages),
        "reused_training_artifact_count": 9,
        "strict_order": ["cmu", "kit", "bmlmovi"],
        "representation": "controlled common18 for CMU/KIT/BMLmovi",
        "scope_policy": (
            "No additional database is introduced. Multi-seed main-model evidence spans all three datasets; "
            "root guidance, compact sensitivity, recent-model coverage, and topology 2x2 are evaluated on representative CMU."
        ),
        "split_manifest_sha256": MANIFEST_SHA256,
        "checkpoint_selection": (
            "legacy-compatible normalized validation MPJPE for reused/common18 LTC comparisons; "
            "all frozen checkpoints are re-evaluated by one physical-coordinate evaluator"
        ),
        "defaults": {
            "history": 25,
            "future_step": 25,
            "validation_horizon": 75,
            "stride": 5,
            "epochs": 30,
            "batch_size": 128,
            "num_workers": 0,
            "ltc_lr": 0.0002,
            "hisrep_lr": 0.0005,
            "baseline_lr": 0.0002,
            "weight_decay": 0.0001,
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.0,
            "lambda_26_50": 0.7,
            "lambda_51_75": 0.5,
            "lambda_bone": 0.1,
            "lambda_anchor": 0.1,
            "lambda_velocity": 0.05,
        },
        "recent_method_provenance": [
            {
                "method": "MSR-GCN",
                "family": "graph",
                "local_source": str(PROJECT / "sota_models" / "MSRGCN"),
                "adaptation": "controlled common18 two-scale residual GCN reimplementation",
            },
            {
                "method": "ST-Transformer",
                "family": "transformer",
                "source_url": "https://github.com/eth-ait/motion-transformer",
                "adaptation": "controlled PyTorch common18 spatial-temporal reimplementation",
            },
            {
                "method": "siMLPe",
                "family": "mlp",
                "local_source": str(PROJECT / "sota_models" / "siMLPe"),
                "adaptation": "controlled common18 DCT residual-MLP reimplementation",
            },
            {
                "method": "HumanMAC",
                "family": "diffusion",
                "source_url": "https://github.com/LinghaoChan/HumanMAC",
                "adaptation": "controlled common18 masked-completion diffusion reimplementation",
            },
        ],
        "source_sha256": {str(path): sha256_file(path) for path in source_files},
        "reused_artifacts": reusable_artifacts(),
        "stages": stages,
        "post_training_analyses": [
            "reused_artifact_compatibility_audit",
            "test_sequence_prediction_export",
            "subject_clustered_full_vs_hisrep_and_full_vs_lstm_statistics",
            "three_seed_mean_sd_and_confidence_intervals",
            "root_guidance_ablation_table",
            "loss_sensitivity_table",
            "recent_method_common18_table",
            "topology_by_rollout_2x2_interaction",
            "common18_efficiency_throughput_latency_peak_memory",
            "training_validation_curves",
            "complete_artifact_hash_audit",
        ],
        "formal_gate": {
            "status": "LOCKED_PENDING_SMOKE_REVIEW_AND_EXPLICIT_AUTHORIZATION",
            "requires": [
                "all source files frozen by SHA-256",
                "all five common18 model kinds pass smoke training",
                "reused seed42 compatibility audit passes",
                "formal static code review status PASS",
                "formal authorization binds plan, inventory, smoke, and review hashes",
            ],
        },
    }
    output = RUN_ROOT / "formal_reviewer_minimum33_plan.json"
    output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(output), "stages": len(stages)}, indent=2))


if __name__ == "__main__":
    main()
