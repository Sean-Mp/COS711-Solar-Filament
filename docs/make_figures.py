"""Slide figures for the methodology section, from the Kaggle runs: round 1 in results/runs
(with training histories), round 2 in results/round2/results.csv.

Usage: python docs/make_figures.py [runs_dir] [out_dir] [round2_csv]
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
BAR = "#86b6ef"                                       # light blue band for C_final's spread

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})

res = pd.read_csv(RUNS / "results.csv")
res2 = pd.read_csv(Path(sys.argv[3] if len(sys.argv) > 3 else "results/round2/results.csv"))


def hist(run):
    h = pd.read_csv(RUNS / run / "history.csv")
    return h.dropna(subset=["val_PQ"])


# 1. Stage comparison: every run of both Kaggle rounds, so the run-to-run noise is visible
labels = {
    "A1_unet_fg": "A1  U-Net, fg only",
    "A2_unet_mt": "A2  U-Net, 5 outputs",
    "A3_r34_mt": "A3  ResNet-34 U-Net (lr 3e-4)",
    "B1_lr0.0001": "B1  lr 1e-4",
    "B1_lr0.001": "B1  lr 1e-3",
    "B2_soft0.5": "B2  soft labels",
    "B2_crop768": "B2  768 crops",
    "D1_soft_crop768": "D1  soft labels + 768 crops",
    "D2_crop1024": "D2  1024 crops",
    "D3_pos0.5": "D3  50% filament-centred crops",
    "C_final": "C  final: 768 crops, 50 ep (3 seeds)",
    "D_final": "D  1024 crops, 80 ep (1 seed)",
}
rounds = {1: res, 2: res2}
c_all = pd.concat([r[r.name == "C_final"].val_PQ for r in rounds.values()])


def dot_plot(path, figsize, colors, markers, band, fontsize, surface, title=None, footnote=None, short=False,
             band_label="C_final mean ± sd"):
    "Every run of both rounds as a dot (one colour and marker per round), a tick at the mean."
    with plt.rc_context({"font.size": fontsize, "figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axvspan(c_all.mean() - c_all.std(), c_all.mean() + c_all.std(), color=band, alpha=0.35, lw=0,
                   label=f"{band_label} ({len(c_all)} runs)")
        y = list(range(len(labels)))[::-1]
        for yi, name in zip(y, labels):
            vals = []
            for (rnd, r), c, mk, off in zip(rounds.items(), colors, markers, (0.12, -0.12)):
                v = r[r.name == name].val_PQ
                vals += v.tolist()
                ax.scatter(v, [yi + off] * len(v), s=fontsize * 4, marker=mk, color=c, edgecolor=surface,
                           linewidth=1.2, zorder=3, label=f"round {rnd}" if yi == y[0] else None)
            mean = sum(vals) / len(vals)
            ax.plot([mean, mean], [yi - 0.3, yi + 0.3], color=INK, lw=2, zorder=4)
            ax.text(0.405, yi, f"{mean:.3f}" + (f" (n={len(vals)})" if len(vals) != 2 else ""),
                    va="center", color=INK, fontsize=fontsize - 2)
        names = [v.split("  ", 1)[1] if short else v for v in labels.values()]
        ax.set_yticks(y, names)
        ax.set_xlim(0.28, 0.43)
        ax.set_xlabel("Validation PQ")
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, fontsize=fontsize - 2,
                  handletextpad=0.3, columnspacing=1.2)
        if title:
            ax.set_title(title, loc="left")
        if footnote:
            fig.text(0.01, 0.01, footnote, color=INK2, fontsize=fontsize - 2)
        fig.tight_layout(rect=(0, 0.03 if footnote else 0, 1, 1))
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


# Report version (palette slots 1-2) and slide version (the deck's UP blue and gold, larger text)
dot_plot(OUT / "stage_comparison.png", (10, 6.9), SERIES[:2], ("o", "D"), BAR, 12, SURFACE,
         title="Validation PQ of every run; black tick = mean",
         footnote="Each experiment ran once per round (D-stage runs: round 2 only). "
                  "The same configuration varies by about ±0.007 between runs.")
dot_plot(OUT / "stage_comparison_slide.png", (7.6, 5.3), ("#005BAA", "#C48B3B"), ("o", "D"), "#9DC3E6", 13,
         "#FFFFFF", short=True, band_label="final model, mean ± sd")

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
ax.set_title("Round 1: five outputs beat fg-only; larger crops help most", loc="left")
fig.tight_layout()
fig.savefig(OUT / "val_pq_curves.png", dpi=200)

# 3. Final configuration, three seeds, 50 epochs
fig, ax = plt.subplots(figsize=(10, 4.6))
for s, c in zip(range(3), SERIES):
    h = hist(f"C_final_s{s}")
    ax.plot(h.epoch, h.val_PQ, color=c, lw=2, marker="o", ms=6, label=f"seed {s}")
ax.set(xlabel="Epoch", ylabel="Validation PQ", xticks=range(5, 55, 5))
ax.legend(loc="lower right")
final = res[res.name == "C_final"].val_PQ
ax.set_title(f"C_final, round 1: {final.mean():.3f} ± {final.std():.3f} over 3 seeds",
             loc="left")
fig.tight_layout()
fig.savefig(OUT / "final_seeds.png", dpi=200)

print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
