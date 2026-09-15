# -*- coding: utf-8 -*-
"""Figure 4. Complete training and validation trajectories for Full LTC-Topology
and HisRepItself across datasets and seeds. Rows retain method-specific
legacy-compatible validation units.

Spec: seed 42 muted blue #4F779F / seed 123 muted orange #C8795C / seed 456 muted
teal #5F9691; validation solid, training dashed (same seed color); selected
checkpoint = hollow marker (42 o, 123 s, 456 ^) at the audited best epoch, no
vertical lines; white background, very light gray horizontal grid only; no
gradients / colored panels / shadows / thick borders / colored titles.
"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.stdout.reconfigure(encoding="utf-8")
BASE = r"C:\Users\1\Desktop\返修提交终稿_20260915"

# ---------------------------------------------------------------- data loading
def load():
    DATA = json.load(open(os.path.join(BASE, "figure4_data.json"), encoding="utf-8"))
    EXTRA_PATH = os.path.join(BASE, "missing6_results.json")
    if os.path.exists(EXTRA_PATH):
        EXTRA = json.load(open(EXTRA_PATH, encoding="utf-8"))
        for key, hist in EXTRA.items():
            ds, run = key.split("/")
            m = "Full LTC-Topology" if run.endswith("_skeleton_full_rollout") else "HisRepItself"
            DATA.setdefault(m, {}).setdefault(ds, {})[run] = hist
    return DATA

# audited best_epoch from checkpoint_stability.csv (authoritative)
AUDITED = {
    ("full", "cmu"): {42: 3, 123: 3, 456: 1},
    ("full", "kit"): {42: 3, 123: 2, 456: 13},
    ("full", "bmlmovi"): {42: 9, 123: 7, 456: 9},
    ("hisrep", "cmu"): {42: 30, 123: 30, 456: 29},
    ("hisrep", "kit"): {42: 25, 123: 30, 456: 26},
    ("hisrep", "bmlmovi"): {42: 27, 123: 30, 456: 29},
}
METHODS = [("full", "Full LTC-Topology", "Normalized MPJPE"),
           ("hisrep", "HisRepItself", "MPJPE (mm)")]
DATASETS = [("cmu", "CMU"), ("kit", "KIT"), ("bmlmovi", "BMLmovi")]
SEED_COLORS = {42: "#4F779F", 123: "#C8795C", 456: "#5F9691"}
SEED_MARKERS = {42: "o", 123: "s", 456: "^"}
GRID = "#E8E8E8"

DATA = load()

# coverage check
missing = []
for meth, mname, unit in METHODS:
    for ds, dsname in DATASETS:
        for seed in (42, 123, 456):
            run = ("seed{s}_skeleton_full_rollout" if meth == "full" else "seed{s}_hisrep").format(s=seed)
            if run not in DATA.get(mname, {}).get(ds, {}):
                missing.append(f"{mname} {ds} {run}")
if missing:
    print("MISSING RUNS:")
    for x in missing:
        print(" ", x)
    sys.exit(1)

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(10.6, 5.7))
gs = fig.add_gridspec(2, 4, width_ratios=[0.09, 1, 1, 1],
                      hspace=0.42, wspace=0.14, left=0.01, right=0.99,
                      top=0.90, bottom=0.13)

axes = {}
for ri, (meth, mname, unit) in enumerate(METHODS):
    for ci, (ds, dsname) in enumerate(DATASETS):
        ax = fig.add_subplot(gs[ri, ci + 1])
        axes[(meth, ds)] = ax
        ax.set_facecolor("white")
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("bottom", "left"):
            ax.spines[sp].set_color("#404040")
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(labelsize=8.5, colors="black")
        ax.set_xlim(1, 30)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        if ri == 0:
            ax.set_title(dsname, loc="left", fontsize=10, fontweight="bold",
                         color="black", pad=6)
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Epoch", fontsize=9, color="black")

# curves + checkpoint markers
for (meth, ds), aud_seeds in AUDITED.items():
    mname = "Full LTC-Topology" if meth == "full" else "HisRepItself"
    for seed in (42, 123, 456):
        run = ("seed{s}_skeleton_full_rollout" if meth == "full" else "seed{s}_hisrep").format(s=seed)
        hist = DATA[mname][ds][run]
        epochs = [h["epoch"] for h in hist]
        if meth == "full":
            train = [h["train_mpjpe_1_75"] for h in hist]
            val = [h["val_mpjpe_1_75"] for h in hist]
        else:
            train = [h["train_mpjpe"] for h in hist]
            val = [h["val_mpjpe"] for h in hist]
        c = SEED_COLORS[seed]
        ax = axes[(meth, ds)]
        ax.plot(epochs, train, color=c, linestyle="--", linewidth=1.3, zorder=3)
        ax.plot(epochs, val, color=c, linestyle="-", linewidth=1.3, zorder=3)
        be = aud_seeds[seed]
        ax.plot(be, val[be - 1], marker=SEED_MARKERS[seed], linestyle="none",
                color=c, markerfacecolor="white", markeredgecolor=c,
                markeredgewidth=1.2, markersize=7, zorder=5)
        # consistency check against audited selection
        argmin = min(range(len(val)), key=lambda i: val[i]) + 1
        if argmin != be:
            print(f"note: {mname} {ds} seed{seed}: curve argmin {argmin} != audited {be}")

# per-row shared y-limits
for ri, (meth, mname, unit) in enumerate(METHODS):
    lo = min(min(h["val_mpjpe_1_75" if meth == "full" else "val_mpjpe"] for h in DATA[
        "Full LTC-Topology" if meth == "full" else "HisRepItself"][ds][
        ("seed{s}_skeleton_full_rollout" if meth == "full" else "seed{s}_hisrep").format(s=s)])
             for s in (42, 123, 456) for ds, _ in DATASETS)
    hi = max(max(h["val_mpjpe_1_75" if meth == "full" else "val_mpjpe"] for h in DATA[
        "Full LTC-Topology" if meth == "full" else "HisRepItself"][ds][
        ("seed{s}_skeleton_full_rollout" if meth == "full" else "seed{s}_hisrep").format(s=s)])
             for s in (42, 123, 456) for ds, _ in DATASETS)
    pad = 0.04 * (hi - lo)
    for ds, _ in DATASETS:
        axes[(meth, ds)].set_ylim(lo - pad, hi + pad)

# rotated row labels
for ri, (meth, mname, unit) in enumerate(METHODS):
    ax = fig.add_subplot(gs[ri, 0])
    ax.set_axis_off()
    ax.text(0.5, 0.5, f"{mname}\n{unit}", rotation=90, ha="center", va="center",
            fontsize=10, color="black", linespacing=1.6)

# figure legend: three seeds + line-style entries
handles = []
for seed in (42, 123, 456):
    handles.append(mlines.Line2D([], [], color=SEED_COLORS[seed], marker=SEED_MARKERS[seed],
                                 markerfacecolor="white", markeredgecolor=SEED_COLORS[seed],
                                 markeredgewidth=1.2, markersize=7, linestyle="-", linewidth=1.3,
                                 label=f"Seed {seed}"))
handles.append(mlines.Line2D([], [], color="#404040", linestyle="-", linewidth=1.3, label="Validation"))
handles.append(mlines.Line2D([], [], color="#404040", linestyle="--", linewidth=1.3, label="Training"))
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
           ncol=5, frameon=False, fontsize=9, handlelength=1.8, columnspacing=1.4)

OUT_PNG = os.path.join(BASE, "Figure4_training_validation_trajectories.png")
OUT_PDF = os.path.join(BASE, "Figure4_training_validation_trajectories.pdf")
OUT_SVG = os.path.join(BASE, "Figure4_training_validation_trajectories.svg")
fig.savefig(OUT_PNG, dpi=600, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
fig.savefig(OUT_SVG, facecolor="white")
print("saved:", OUT_PNG)
print("saved:", OUT_PDF)
print("saved:", OUT_SVG)
