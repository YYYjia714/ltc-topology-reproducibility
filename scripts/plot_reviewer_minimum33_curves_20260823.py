from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_reviewer_minimum33_20260823 import build_entries, read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reviewer-requested training and validation curves")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def curve_values(entry: dict) -> tuple[list[int], list[float], list[float], str]:
    history = read_json(Path(entry["history"]))
    if entry["method"] == "Full LTC-Topology":
        train_key, val_key, unit = "train_mpjpe_1_75", "val_mpjpe_1_75", "normalized MPJPE"
    else:
        train_key, val_key, unit = "train_mpjpe", "val_mpjpe", "MPJPE (mm)"
    epochs = [int(row["epoch"]) for row in history]
    train = [float(row[train_key]) for row in history]
    val = [float(row[val_key]) for row in history]
    return epochs, train, val, unit


def main() -> None:
    args = parse_args()
    plan = read_json(args.plan)
    entries = [
        entry
        for entry in build_entries(plan)
        if entry["method"] in {"Full LTC-Topology", "HisRepItself"}
    ]
    if len(entries) != 18:
        raise RuntimeError(f"Expected 18 Full/HisRep curve entries, found {len(entries)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stability = []
    figure_paths = []
    colors = {42: "#202020", 123: "#d65f35", 456: "#2a7f9e"}
    for dataset in ("cmu", "kit", "bmlmovi"):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for axis, method in zip(axes, ("Full LTC-Topology", "HisRepItself")):
            selected = sorted(
                [entry for entry in entries if entry["dataset"] == dataset and entry["method"] == method],
                key=lambda entry: int(entry["seed"]),
            )
            for entry in selected:
                epochs, train, val, unit = curve_values(entry)
                seed = int(entry["seed"])
                axis.plot(epochs, train, linestyle="--", color=colors[seed], alpha=0.7, label=f"seed {seed} train")
                axis.plot(epochs, val, color=colors[seed], linewidth=1.8, label=f"seed {seed} validation")
                best_index = int(np.argmin(val))
                axis.scatter([epochs[best_index]], [val[best_index]], color=colors[seed], s=24, zorder=3)
                stability.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "best_epoch": epochs[best_index],
                        "best_validation": val[best_index],
                        "final_validation": val[-1],
                        "final_minus_best": val[-1] - val[best_index],
                        "unit": unit,
                    }
                )
            axis.set_title(method)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(unit)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7, ncol=2)
        figure.suptitle(f"{dataset.upper()} training and validation curves")
        output = args.output_dir / f"{dataset}_full_hisrep_curves.png"
        figure.savefig(output, dpi=220)
        plt.close(figure)
        figure_paths.append(output)
    write_csv(args.output_dir / "checkpoint_stability.csv", stability)
    payload = {
        "status": "completed",
        "plan": str(args.plan),
        "plan_sha256": sha256_file(args.plan),
        "figures": [{"path": str(path), "sha256": sha256_file(path)} for path in figure_paths],
        "stability": stability,
        "note": "Full and HisRep panels use their own legacy-compatible validation units and are not overlaid.",
    }
    (args.output_dir / "curve_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
