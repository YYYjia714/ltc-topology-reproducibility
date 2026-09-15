# LTC-Topology — Reproducibility Package (Revision v2, 15 September 2026)

This package reproduces the training, evaluation, and analysis behind the revised
manuscript *"<manuscript title>"* and accompanies the response to reviewers. It
contains the pre-registered Protocol v2 plan, the SHA-256-locked artifact audit,
all training histories, the analysis outputs behind every table, and the code
that produced them. Checkpoints are distributed separately (see §7).

- Repository: https://github.com/YYYjia714/ltc-topology-reproducibility
  (private during review; made public upon acceptance)
- Checkpoints: https://zenodo.org/records/22774021
  (draft during review; published upon acceptance, restricted reviewer link available on request)

## Contents

```
code/                     Source code (src/), training entrypoints, queue runners
scripts/                  Canonical evaluation / benchmark / curve / audit scripts
protocol/                 Pre-registered plan, frozen source inventory, audit pass,
                          stage exit states, dataset audits, review records
manifests/                Frozen subject-grouped split manifest (hash-locked),
                          common18 skeleton graph, protocol v2 spec
results/                  Per-run train_history.json + results.json for all runs
                          (both formal batches, folder names preserved)
analysis/outputs/         Audited post-analysis CSVs/JSONs behind the tables
analysis/figure4/         Figure 4 regeneration (script, data, rendered files)
requirements.txt          Locked environment (pip freeze of the execution venv)
package_manifest.json     SHA-256 of every file in this package (generated at build)
```

## 1. Environment

- Python 3.11.9 (the execution environment; other 3.11.x should work)
- `pip install -r requirements.txt`
- The reported runs used `torch==2.12.0+cu130`; any CUDA or CPU build of
  torch 2.x is acceptable for reproduction.

## 2. The pre-registration and audit chain (read this first)

The experiments were executed under a pre-declared protocol. The chain of
evidence, in order of binding strength:

1. `protocol/formal_reviewer_minimum33_plan.json` — the frozen plan: 33 strictly
   serial training stages plus 9 pre-declared reused artifacts (seed 42 primary
   runs), split-manifest hash, checkpoint-selection rule, and post-training
   analyses. Its SHA-256 is recorded inside every artifact below.
2. `protocol/reused_inventory.json` — the 22 source files that implement the
   plan, each with size and SHA-256 (checked before and after every run).
3. `protocol/review/` — preflight reports and the freeze manifest.
4. `protocol/stage_state/` — one exit-state JSON per training stage
   (`status=completed`, `exit_code=0`, artifact hashes re-verified).
5. `protocol/audit/reviewer_minimum_audit_pass.json` — the final automated audit:
   52 checks, all PASS, plus hashes of the audited artifacts.

To re-verify the chain on your machine (PowerShell):

```powershell
# plan hash recorded in the audit pass must match the plan file
Get-FileHash protocol/formal_reviewer_minimum33_plan.json -Algorithm SHA256
# every file in reused_inventory.json must match its recorded sha256/size
python scripts/audit_reviewer_minimum33_20260823.py --plan protocol/formal_reviewer_minimum33_plan.json --verify-only
```

## 3. Data preparation

CMU, KIT, and BMLmovi motion data are redistributed under AMASS licenses and are
**not** included. Download the three datasets (AMASS: cmu, kit, bmlmovi),
then run the pre-processing pipeline:

1. `python scripts/extract_amass_joints.py --config configs/amass_preprocess.example.json`
2. `python scripts/build_protocol_v2_subject_grouped_windows.py ...`
   This assigns **subjects** to train/validation/test before any fixed-length
   window is extracted (the leakage-control design), and consumes
   `manifests/frozen_subject_grouped_sequence_manifest.csv`.

The frozen split manifest is included here. Its SHA-256 must equal
`2e4ed5afb524c328efdd1246fa61fb96e104b39484e17d55784fa3ef2aec879e`
(as pre-declared in the plan). Verify with:

```powershell
Get-FileHash manifests/frozen_subject_grouped_sequence_manifest.csv -Algorithm SHA256
```

## 4. Reproducing training

The plan is executed by two batches; both are required for the paper's tables:

| Batch | What it produced | Runner |
|---|---|---|
| `formal_protocol_v2_subject_grouped_rootrel_pose_20260815` | Seed-42 primary runs (**pre-declared reused artifacts**): `common18_full_rollout`, `hisrep_common18`, `common18_no_rollout`, `native25_*` per dataset | `code/manage_protocol_v2_formal_queue_20260815.ps1` |
| `formal_reviewer_minimum33_revision_20260823` | The 33 serial stages: seed 123/456 primaries, all controlled baselines, ablations, sensitivity variants | `code/manage_reviewer_minimum33_queue_20260823.ps1` |

**Important for reviewers:** the seed-42 Full LTC-Topology and HisRepItself runs
are reused artifacts executed in the first batch. Their run directories are named
`common18_full_rollout` and `hisrep_common18` (no `seed42` prefix) — do not
search for `seed42_skeleton_full_rollout` on disk, and do not retrain them from
scratch; the plan binds their exact locations and hashes
(`protocol/formal_reviewer_minimum33_plan.json`, key `reused_artifacts`).

Per-stage reproduction is `run_reviewer_minimum33_stage_20260823.ps1 -Stage <id>`
with the queue state prepared by the corresponding `manage_*.ps1` script.

## 5. Checkpoint selection (why "best epoch" is what it is)

Each run trains the full 30-epoch budget and keeps its history. The selected
checkpoint is chosen by the **pre-declared method-family-compatible validation
rule**, not by test performance:

- LTC / common18 runs (Full, No-Rollout, ablated LTC variants):
  normalized recursive validation MPJPE, field `val_mpjpe_1_75` in
  `train_history.json`.
- HisRepItself and controlled recent baselines:
  physical recursive validation MPJPE, field `val_mpjpe` (mm).

The audited selected epochs are tabulated in
`analysis/outputs/curves/checkpoint_stability.csv` (e.g., CMU Full: 3 / 3 / 1
for seeds 42 / 123 / 456). All final comparisons use one physical-coordinate
test evaluator (`scripts/evaluate_reviewer_minimum33_20260823.py`).

## 6. Reproducing the tables and Figure 4

Run the post-analysis pipeline (requires the plan, the completed stages, and the
checkpoints from §7):

```powershell
powershell -File code/run_reviewer_minimum33_postanalysis_20260823.ps1
```

This regenerates `analysis/outputs/` (sequence evaluation, efficiency benchmark,
curves, final audit). Mapping to the manuscript:

| Manuscript table | Source |
|---|---|
| Table 1 (dataset construction) | `manifests/` + `protocol/dataset_audit/` |
| Table 2 (main three-seed comparison) | `analysis/outputs/sequence/multiseed_summary.csv`, `aggregate_metrics.csv` |
| Table 3 (controlled baselines) | `analysis/outputs/sequence/aggregate_metrics.csv` |
| Table 4 (ablations) | `analysis/outputs/sequence/ablation_contrasts.csv` |
| Table 5 (rollout fine-tuning) | `analysis/outputs/sequence/multiseed_summary.csv` |
| Table 6 (sensitivity) | `analysis/outputs/sequence/sensitivity_summary.csv` |
| Table 7 (structural consistency) | `analysis/outputs/sequence/aggregate_metrics.csv` |
| Table 8 (subject-level paired stats) | `analysis/outputs/sequence/paired_statistics.csv` |
| Table 9 (efficiency) | `analysis/outputs/efficiency/common18_efficiency.csv` |
| Table 10 (test windows) | `analysis/outputs/sequence/sequence_evaluation_summary.json` + `protocol/dataset_audit/` |
| Figure 4 (training/validation trajectories) | `analysis/figure4/` — `python plot_figure4.py` reads `figure4_data.json` (the 18 retained `train_history.json` files, merged) and renders PNG (600 dpi), PDF, SVG. Hollow markers sit at the audited best epochs from `checkpoint_stability.csv`. |

## 7. Checkpoints (separate archive)

All 48 `best.pt` checkpoints (~1.2 GB total) are provided as a separate archive
`ltc_topology_checkpoints_20260915.zip` on Zenodo
(https://zenodo.org/records/22774021), with a SHA-256 manifest
`checkpoints_manifest.json`. Place the unpacked `checkpoints/` folder
at the package root before running the post-analysis pipeline. Sizes per file:
12–48 MB (HisRepItself and ST-Transformer are the largest at 41–48 MB).

## 8. License

Code: MIT (`LICENSE`). Data manifests and analysis outputs are provided for
reproducibility of this manuscript. AMASS-derived data remains under the
AMASS license terms.

## Notes for reviewers

- Negative and null findings were preserved by design (Protocol v2 §9); no
  dataset, horizon, or seed was dropped after results were seen.
- The audit pass record `reviewer_minimum_audit_pass.json` was produced on
  2026-09-15 and binds the plan, inventory, review, smoke, and preflight hashes.
- Questions about a specific number can be answered from
  `results/` histories plus the audit record; every audited artifact's hash is
  in the audit pass file.
