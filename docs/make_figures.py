"""Slide figures for the methodology section, from the Kaggle runs in results/runs.

Usage: python docs/make_figures.py [runs_dir] [out_dir]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "results/runs")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
BAR, BAR_FINAL = "#86b6ef", "#1c5cab"                 # one hue; the final model in a darker step

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})

res = pd.read_csv(RUNS / "results.csv")


def hist(run):
    h = pd.read_csv(RUNS / run / "history.csv")
    return h.dropna(subset=["val_PQ"])


# 1. Stage comparison: validation PQ of every run, in the order the search ran them
labels = {
    "A1_unet_fg": "A1  U-Net, fg only",
    "A2_unet_mt": "A2  U-Net, 5 outputs",
    "A3_r34_mt": "A3  ResNet-34 U-Net (lr 3e-4)",
    "B1_lr0.0001": "B1  lr 1e-4",
    "B1_lr0.001": "B1  lr 1e-3",
    "B2_soft0.5": "B2  soft labels",
    "B2_crop768": "B2  768 crops",
}
single = res[res.name.isin(labels)].set_index("name").loc[list(labels)]
final = res[res.name == "C_final"].val_PQ
rows = [(labels[n], single.loc[n, "val_PQ"], 0.0, BAR) for n in labels]
rows.append(("C  final: 768 crops, 50 ep (3 seeds)", final.mean(), final.std(), BAR_FINAL))
pp = pd.read_csv(next(RUNS.glob("C_final_s*/pp_grid.csv")))  # tuned on the best seed
best_seed = res[res.name == "C_final"].sort_values("val_PQ").iloc[-1].seed
rows.append((f"C  seed {best_seed} + tuned post-processing", pp.PQ.iloc[0], 0.0, BAR_FINAL))

fig, ax = plt.subplots(figsize=(10, 5.6))
y = range(len(rows))[::-1]
for yi, (lab, v, sd, c) in zip(y, rows):
    ax.barh(yi, v, height=0.62, color=c, edgecolor=SURFACE, linewidth=2)
    if sd:
        ax.errorbar(v, yi, xerr=sd, color=INK, capsize=4, lw=1.5)
    ax.text(v + (sd or 0) + 0.004, yi, f"{v:.3f}" + (f" ± {sd:.3f}" if sd else ""),
            va="center", color=INK, fontsize=11)
ax.set_yticks(list(y), [r[0] for r in rows])
ax.set_xlim(0, 0.48)
ax.set_axisbelow(True)
ax.set_xlabel("Validation PQ (107 observations, all annotation sets)")
ax.grid(axis="y", visible=False)
ax.set_title("Validation PQ, stage by stage", loc="left")
fig.text(0.01, 0.01, "Single-seed runs differ by seed noise of about ±0.008 (std over the three final seeds).",
         color=INK2, fontsize=10)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "stage_comparison.png", dpi=200)

# 2. Validation PQ during training: architecture/formulation runs and the crop-size winner
curves = [("A1_unet_fg_s0", "A1 U-Net, fg only"), ("A2_unet_mt_s0", "A2 U-Net, 5 outputs"),
          ("A3_r34_mt_s0", "A3 ResNet-34 U-Net"), ("B2_crop768_s0", "B2 ResNet-34, 768 crops")]
fig, ax = plt.subplots(figsize=(10, 5.2))
for (run, lab), c in zip(curves, SERIES):
    h = hist(run)
    ax.plot(h.epoch, h.val_PQ, color=c, lw=2, marker="o", ms=6, label=lab)
    ax.annotate(f"{h.val_PQ.iloc[-1]:.3f}", (h.epoch.iloc[-1], h.val_PQ.iloc[-1]),
                xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=10)
ax.set(xlabel="Epoch", ylabel="Validation PQ", xticks=[5, 10, 15, 20], xlim=(3, 22.5))
ax.legend(loc="lower right")
ax.set_title("Five outputs beat fg-only; larger crops help most", loc="left")
fig.tight_layout()
fig.savefig(OUT / "val_pq_curves.png", dpi=200)

# 3. Final configuration, three seeds, 50 epochs
fig, ax = plt.subplots(figsize=(10, 4.6))
for s, c in zip(range(3), SERIES):
    h = hist(f"C_final_s{s}")
    ax.plot(h.epoch, h.val_PQ, color=c, lw=2, marker="o", ms=6, label=f"seed {s}")
ax.set(xlabel="Epoch", ylabel="Validation PQ", xticks=range(5, 55, 5))
ax.legend(loc="lower right")
ax.set_title(f"Final model: {final.mean():.3f} ± {final.std():.3f} over 3 seeds, still rising at epoch 50",
             loc="left")
fig.tight_layout()
fig.savefig(OUT / "final_seeds.png", dpi=200)

print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
