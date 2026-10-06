# LTC-Topology: Public Code and Analysis Release

Companion to **A topology-guided liquid time-constant framework for recursive
long-horizon 3D human motion prediction**, The Visual Computer,
[doi:10.1007/s00371-026-04773-8](https://doi.org/10.1007/s00371-026-04773-8).

Authors: Yangjia Xiong, Yanke Xie, Xingyuan Song, and Ying Chen.

Repository: https://github.com/YYYjia714/ltc-topology-reproducibility

## Release Scope and Access

This release makes the authors' reproduction code, training histories, numerical
analysis outputs, figures, frozen plans, and audit records publicly available.
**The 48 trained checkpoints are NOT publicly released at this time.** Their
SHA-256 inventory is included in `checkpoints_manifest.json`. The existing Zenodo
upload remains an unpublished draft; its reserved identifier is not a published
dataset DOI. This README supersedes historical reviewer-only access statements.

No raw AMASS motions, extracted joint arrays, training windows, SMPL/SMPL-X/SMPL-H
body-model assets, third-party repository copies, credentials, uploader utilities,
author photographs, or manuscript/proof PDFs are redistributed. Split manifests
contain dataset sequence identifiers and pseudonymous subject IDs, not personal
participant names. Models and data-dependent reevaluation therefore cannot be
downloaded and rerun from this public release alone.

## Package Contents

| Directory/file | Contents |
|---|---|
| `code/` | Authors' model source, training entrypoints, historical queue runners |
| `scripts/` | Data preparation, evaluation, benchmarking, statistics, curve and audit code |
| `protocol/` | Original frozen plans, source inventory, review records, stage exit states and final audit |
| `manifests/` | Frozen subject-grouped splits, common18 graph and protocol specification |
| `results/` | Retained training histories and results from both formal batches |
| `analysis/outputs/` | Sequence metrics, paired tests, multiseed summaries, ablations, sensitivity, efficiency and stability |
| `analysis/final_figures/` | Final manuscript Figures 1-5 in SVG |
| `analysis/figure4/` | Historical filename for the training-trajectory plot, now manuscript Figure 3 |
| `requirements.txt` | Original recorded environment, not a claim of cross-platform equivalence |
| `package_manifest.json` | SHA-256 and sizes for this public package |
| `release_security_report.json` | Scope, exclusion policy and release validation results |

## Artifact Verification Without Data, Models, or a GPU

Python 3.11+ and the standard library are sufficient:

```text
python tools/verify_public_package.py
```

This verifies public-file hashes, the frozen split and plan hashes, source
inventory, 33 successful stage exit records, and 14 original post-analysis
artifacts against the final audit. It does not run training or reevaluation and
does not interpret a historical audit PASS as a new reproduction experiment.

The original final audit records 33 completed new training stages, nine reused
seed-42 artifacts, and 52 passing checks. All original numerical results and
frozen protocol documents are retained byte-for-byte. Historical absolute paths
describe the execution machine; verification maps them to packaged locations
without rewriting the original records. The source inventory and final review
also retain the approved HisRepItself resume fix: the initial plan's older source
hash is historical, not a reason to replace the corrected source silently.

"Frozen/prespecified protocol" here refers to the retained dated execution plan.
This package does not establish an independent external preregistration.

## Environment and Execution Layout

The recorded environment used Python 3.11.9 and `torch==2.12.0+cu130`.
Install the appropriate PyTorch build from its official distribution channel and
then reconcile the recorded requirements. Other CUDA/CPU versions have not been
validated for equivalent numerical results.

The source was originally executed with training files and `src/` at the project
root. Prepare a separate working directory, without starting any experiment:

```text
python tools/prepare_execution_workspace.py --destination /path/to/new_workspace
```

This copies source and scripts into the original import layout. It does not copy
motion data, install dependencies, download models, change frozen plans, or launch
the queue. Historical PowerShell runners target the original Windows project
path; inspect and explicitly relocate them in your own working copy before use.
Never launch a historical queue against an existing experiment directory.

HisRepItself must be obtained separately from
https://github.com/wei-mao-2019/HisRepItself at commit
`0451c84491cf2b3697e373ce26dc75623ccaa89e`, under
`sota_models/HisRepItself` in the prepared working directory. Obtain and comply
with its upstream terms. No upstream body models or pretrained weights are
included. Controlled MSR-GCN, ST-Transformer, siMLPe and HumanMAC implementations
are the authors' adaptations in `code/src/reviewer_baselines.py`, not identical
upstream training pipelines or literature-score reproductions.

## Data Preparation

Obtain CMU, KIT and BMLmovi from AMASS and the required body-model files directly
from their rights holders. See https://amass.is.tue.mpg.de/license.html.
From the prepared working directory, the actual extraction interface is:

```text
python scripts/extract_amass_joints.py --amass-root /path/to/AMASS --output-root data/interim/joints --model-path /path/to/licensed/body_models --save-root-relative
```

Choose the model type matching the separately licensed files. Build each dataset
and representation with the frozen manifest, for example:

```text
python scripts/build_protocol_v2_subject_grouped_windows.py --manifest manifests/frozen_subject_grouped_sequence_manifest.csv --dataset CMU --representation common18 --output-root data/processed/protocol_v2_subject_grouped_rootrel_pose/cmu/common18
```

Repeat for KIT/BMLmovi and native25 when needed. Split assignment is at subject
level before window extraction. The frozen manifest SHA-256 is
`2e4ed5afb524c328efdd1246fa61fb96e104b39484e17d55784fa3ef2aec879e`.

## Checkpoint Selection and Training Provenance

Both formal batches are retained. The first,
`formal_protocol_v2_subject_grouped_rootrel_pose_20260815`, supplies the original
seed-42 runs. The second, `formal_reviewer_minimum33_revision_20260823`, contains
the 33-stage extension. Exact commands are preserved in `protocol/stage_state/`.
The historical per-stage runner argument is `-StageId`, not `-Stage`.

Checkpoint selection uses the prespecified method-compatible validation metric,
never test performance: normalized recursive validation MPJPE for the retained
LTC comparisons, physical recursive validation MPJPE for HisRepItself and
controlled baselines. The selected epochs are recorded in
`analysis/outputs/curves/checkpoint_stability.csv`. Final comparisons use the
shared physical-coordinate evaluator. Retained negative/null findings and the
CMU-only scope of sensitivity and selected ablations are not expanded.

## Current Manuscript Mapping

| Item | Source |
|---|---|
| Table 1: dataset construction | `manifests/` and `protocol/dataset_audit/` |
| Table 2: three-seed comparison | `analysis/outputs/sequence/multiseed_summary.csv` |
| Table 3: controlled CMU baselines | `analysis/outputs/sequence/aggregate_metrics.csv` |
| Table 4: rollout ablations | `analysis/outputs/sequence/ablation_contrasts.csv`, multiseed summary |
| Table 5: loss sensitivity | `analysis/outputs/sequence/sensitivity_summary.csv` |
| Table 6: structural consistency | `analysis/outputs/sequence/aggregate_metrics.csv` |
| Table 7: paired subject statistics | `analysis/outputs/sequence/paired_statistics.csv` |
| Table 8: throughput | `analysis/outputs/efficiency/common18_efficiency.csv` |
| Tables S1-S2 | Retained per-run results, sequence evaluation summary and dataset audits |
| Figure 3: trajectories | `analysis/figure4/figure4_data.json`, `plot_figure4.py` |
| Figure 4: subject-level comparison | Per-sequence metrics and paired statistics |
| Figure 5: accuracy/latency | Aggregate metrics and common18 efficiency |

The trajectory plot can be rendered without models or motion data:

```text
python analysis/figure4/plot_figure4.py
```

Legacy output filenames still say Figure4; the current manuscript labels this
plot Figure3. The public-copy change only resolves its input/output directory
relative to the script; it does not change the numerical plot data.

## Licenses and Limitations

Authors' code retains the existing MIT license in `LICENSE`. This license does
not override third-party data, body-model, dependency or model-weight terms.
See `THIRD_PARTY_NOTICES.md`. Public numerical summaries are supplied as study
reproducibility materials, not as a redistribution of motion-capture datasets.

This release validates the integrity of retained artifacts and package layout.
No new training, GPU benchmark, statistical recomputation or complete
data-dependent reproduction was performed for the public release.
