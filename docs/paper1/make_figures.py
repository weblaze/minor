"""Figures of paper 1.

The plotted values are those of the tables in paper.md (source: docs/run_log.md). Figure 2 is
assembled from the sample grids of three evaluation runs and needs the run directories.

Usage:
  python docs/paper1/make_figures.py [--runs <path to runs>]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).with_name("figures")
BLUE, ORANGE, GREEN, GREY, RED = "#3b6fb6", "#d9822b", "#4a9d5b", "#8a8a8a", "#b8443f"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})

SAMPLE_GRIDS = [
    ("S1 (generated)", "kaggle/abstraction-s1-seed0/runs/s1_eval_s1_seed0_20261112T053557Z/samples.png"),
    ("Q1, mean code (generated)",
     "kaggle/abstraction-code-swap/runs/q1_eval_q1_meancode_seed0_20261112T174353Z/samples.png"),
    ("F1 (retrieved paintings)", "f1_eval_f1_seed0_20261112T213235Z/samples.png"),
]


def box(ax, x, y, w, h, text, colour="#eef2f7"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=colour, ec="#444444", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def arrow(ax, x0, y0, x1, y1, style="-|>", colour="#444444", ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=colour, lw=0.9, ls=ls))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    box(ax, 0.1, 2.0, 1.2, 0.7, "track\n3 x 10 s")
    box(ax, 1.7, 2.0, 1.2, 0.7, "MERT\n(frozen)", "#e3e3e3")
    box(ax, 3.3, 2.0, 1.2, 0.7, "audio code a")
    box(ax, 4.9, 2.0, 1.0, 0.7, "M\n(trained)", "#f6dfc4")
    for x in (1.3, 2.9, 4.5):
        arrow(ax, x, 2.35, x + 0.4, 2.35)
    box(ax, 6.4, 3.3, 1.3, 0.7, "VAE latent z\nS1: 4x16x16\nQ1: 128")
    box(ax, 8.1, 3.3, 0.9, 0.7, "decoder\n(frozen)", "#e3e3e3")
    box(ax, 8.1, 2.0, 0.9, 0.7, "DINOv2\n(frozen)", "#e3e3e3")
    box(ax, 6.4, 0.7, 1.3, 0.7, "feature f\nF1: 384")
    box(ax, 8.1, 0.7, 1.6, 0.7, "nearest training\npainting")
    arrow(ax, 5.9, 2.5, 6.4, 3.5)
    arrow(ax, 5.9, 2.2, 6.4, 1.1)
    arrow(ax, 7.7, 3.65, 8.1, 3.65)
    arrow(ax, 8.55, 3.3, 8.55, 2.7)
    arrow(ax, 7.7, 1.05, 8.1, 1.05)
    ax.text(9.1, 3.65, "image", va="center", fontsize=8)
    ax.text(9.1, 2.35, "features v", va="center", fontsize=8)
    ax.plot([3.9, 3.9, 9.9, 9.9], [2.7, 4.15, 4.15, 2.5], color=BLUE, ls="--", lw=0.9)
    ax.text(6.9, 4.3, "structure term: 1 - Pearson(D_a, D_v)", ha="center", color=BLUE, fontsize=8)
    ax.text(5.0, 0.15, "distribution term: outputs (z, v or f) against real training art", ha="center",
            color=ORANGE, fontsize=8)
    fig.savefig(OUT / "fig1_pipeline.png")
    plt.close(fig)


def fig_samples(runs):
    from PIL import Image
    rows = 2
    fig, axes = plt.subplots(len(SAMPLE_GRIDS), 1, figsize=(7.0, 5.6))
    for ax, (title, rel) in zip(axes, SAMPLE_GRIDS):
        image = Image.open(Path(runs) / rel).convert("RGB")
        tile = image.size[0] // 8
        ax.imshow(image.crop((0, 0, image.size[0], rows * tile + 2)))
        ax.set_title(title, fontsize=9, loc="left")
        ax.axis("off")
    fig.savefig(OUT / "fig2_samples.png")
    plt.close(fig)


def fig_gate():
    kl = [0.02, 0.1, 0.3, 1.0, 3.0]
    recon = [208.5, 307.5, 576.8, 980.5, 1284.4]
    prior = [974.7, 685.2, 773.2, 1078.2, 1261.4]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(kl, recon, "o-", color=BLUE, label="reconstructions")
    ax.plot(kl, prior, "s-", color=ORANGE, label="prior samples")
    ax.axhline(113.3, color=GREY, ls=":", lw=1)
    ax.text(0.021, 135, "training art 113.3", color=GREY, fontsize=7)
    ax.axvspan(0.3, 3.0, color=GREEN, alpha=0.08)
    ax.text(0.33, 1330, "gate passed", color=GREEN, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("KL weight")
    ax.set_ylabel("Frechet distance to test art")
    ax.set_ylim(0, 1450)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    fig.savefig(OUT / "fig3_gate.png")
    plt.close(fig)


def fig_code_method():
    codes = ["mean", "standardised", "whitened"]
    data = {"Mantel r": ([0.861, 0.847, 0.272], [0.825, 0.797, 0.259]),
            "kNN overlap": ([0.268, 0.410, 0.178], [0.282, 0.351, 0.152]),
            "Frechet distance": ([1172.8, 1364.0, 1172.9], [425.4, 422.1, 402.1])}
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for ax, (name, (s1, q1)) in zip(axes, data.items()):
        x = range(3)
        ax.bar([i - 0.2 for i in x], s1, 0.4, color=BLUE, label="S1")
        ax.bar([i + 0.2 for i in x], q1, 0.4, color=ORANGE, label="Q1")
        ax.set_xticks(list(x), codes)
        ax.set_title(name, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_code_method.png")
    plt.close(fig)


def fig_f1_stages():
    stages = ["train,\ndirect", "test,\ndirect", "test,\npainting", "oracle,\npainting"]
    rows = {"mean": [0.965, 0.945, 0.644, 0.650], "standardised": [0.991, 0.988, 0.572, 0.571],
            "whitened": [0.980, 0.966, 0.066, 0.085]}
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for (code, values), colour in zip(rows.items(), (BLUE, ORANGE, GREY)):
        ax.plot(range(4), values, "o-", color=colour, label=code)
    ax.set_xticks(range(4), stages)
    ax.set_ylabel("Mantel r")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(OUT / "fig5_f1_stages.png")
    plt.close(fig)


def fig_tradeoff():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot([854, 651, 484, 396, 384], [0.945, 0.929, 0.868, 0.765, 0.645], "-", color=BLUE, lw=1.2,
            marker=".", label="F1 interpolation, mean code")
    ax.plot([897, 663, 471, 367, 344], [0.988, 0.949, 0.837, 0.696, 0.573], "-", color=ORANGE, lw=1.2,
            marker=".", label="F1 interpolation, standardised code")
    points = [
        ("S1 (mean)", 1172.8, 0.861, "^", RED, (-8, 7)), ("S1 (standardised)", 1364.0, 0.847, "^", RED, (-4, -11)),
        ("Q1 (mean), 3 seeds", 425.4, 0.825, "o", GREEN, (-78, 4)), ("", 409.0, 0.815, "o", GREEN, None),
        ("", 409.7, 0.818, "o", GREEN, None), ("Q1 (standardised)", 422.1, 0.797, "o", GREEN, (-78, -9)),
        ("Q1 no_feat", 675.2, 0.869, "D", GREEN, (6, -9)), ("Q1 shuffled", 405.5, 0.209, "x", GREY, (5, 3)),
        ("S1 s0", 1108.7, 0.082, "x", GREY, (-8, 6)), ("S1 shuffled", 1571.7, 0.140, "x", GREY, (-40, 6)),
        ("F1 best (mean)", 375, 0.747, "s", BLUE, (-62, -12)),
        ("F1 best (standardised)", 393, 0.641, "s", ORANGE, (6, -9)),
        ("PCA alignment", 539.4, 0.420, "P", GREY, (5, 3)), ("retrieval baseline", 248.3, 0.115, "P", GREY, (5, 3)),
    ]
    for label, fd, r, marker, colour, offset in points:
        ax.scatter(fd, r, marker=marker, color=colour, s=28, zorder=3)
        if label:
            ax.annotate(label, (fd, r), textcoords="offset points", xytext=offset, fontsize=6.5)
    ax.axvline(113.3, color=GREY, ls=":", lw=1)
    ax.text(125, 0.02, "training art", color=GREY, fontsize=7, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Frechet distance to test art (log scale)")
    ax.set_ylabel("Mantel r")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7, loc=(0.07, 0.42))
    fig.savefig(OUT / "fig6_tradeoff.png")
    plt.close(fig)


def fig_orientation():
    settings = ["mean\n1024x3", "mean\n256x3", "mean\n1024x1", "mean\n2048x3", "mean\nsoft", "stand.\n1024x3"]
    majority = [40 / 50, 11 / 20, 14 / 20, 13 / 20, 14 / 20, 34 / 50]
    genres = ["Electronic", "Hip-Hop", "Rock", "Pop", "Internat.", "Experim.", "Instrum.", "Folk"]
    seeds = [[0.14, 0.15, 0.30, 0.37, 0.51, 0.51, 0.65, 0.83], [0.10, 0.20, 0.39, 0.37, 0.48, 0.60, 0.72, 0.84],
             [0.20, 0.16, 0.30, 0.34, 0.53, 0.55, 0.74, 0.85], [0.14, 0.16, 0.43, 0.32, 0.49, 0.52, 0.65, 0.87],
             [0.80, 0.68, 0.51, 0.43, 0.34, 0.39, 0.15, 0.09]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
    left.bar(range(6), majority, color=BLUE)
    left.axhline(0.5, color=GREY, ls=":", lw=1)
    left.set_xticks(range(6), settings, fontsize=7)
    left.set_ylim(0, 1)
    left.set_ylabel("runs with majority sign, axis 1", fontsize=8)
    for index, values in enumerate(seeds):
        right.plot(range(8), values, "o-", ms=3, lw=1, color=RED if index == 4 else BLUE,
                   label="seed 4" if index == 4 else ("seeds 0 to 3" if index == 0 else None))
    right.set_xticks(range(8), genres, rotation=40, ha="right", fontsize=7)
    right.set_ylabel("share Colour Field + Minimalism")
    right.set_ylim(0, 1)
    right.legend(frameon=False, fontsize=7, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_orientation.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=None, help="directory holding the run directories (for figure 2)")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    fig_pipeline()
    fig_gate()
    fig_code_method()
    fig_f1_stages()
    fig_tradeoff()
    fig_orientation()
    if args.runs:
        fig_samples(args.runs)
    print("written to", OUT)


if __name__ == "__main__":
    main()
