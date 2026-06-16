"""IEEE-style figures for the headline comparison and the (k, sigma) sweep."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Config, log


def plot_headline(df: pd.DataFrame, cfg: Config):
    log(f"Rendering headline figure -> {cfg.out_png}")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    palette = sns.color_palette("colorblind")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    methods = df["method"].tolist()
    x = np.arange(len(methods))

    ax = axes[0]
    w = 0.38
    ax.bar(x - w / 2, df["recall@5"], w, label="Recall@5", color=palette[0])
    ax.bar(x + w / 2, df["recall@10"], w, label="Recall@10", color=palette[1])
    for xi, (r5, r10) in enumerate(zip(df["recall@5"], df["recall@10"])):
        ax.text(xi - w / 2, r5, f"{r5:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, r10, f"{r10:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("(a) Retrieval Utility (Recall@K)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, max(df["recall@10"].max(), df["recall@5"].max()) * 1.18)
    ax.legend(frameon=True)

    ax = axes[1]
    bars = ax.bar(x, df["latency_ms"], color=palette[2], width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_title("(b) Query Latency")
    ax.set_ylabel("Avg latency per query (ms)")
    ax.set_ylim(0, df["latency_ms"].max() * 1.18)

    ax = axes[2]
    bars = ax.bar(x, df["dram_mb"], color=palette[3], width=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_title("(c) Corpus DRAM Overhead")
    ax.set_ylabel("Corpus matrix size (MB)")
    ax.set_ylim(0, df["dram_mb"].max() * 1.18)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=12)
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("ASPE Distance-Preserving Encryption: Utility-Security Trade-off "
                 f"(FiQA2018, N={cfg.n_corpus} docs, Q={cfg.n_queries})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(cfg.out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(df: pd.DataFrame, headline: pd.DataFrame, cfg: Config):
    log(f"Rendering sweep figure -> {cfg.sweep_png}")
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    raw_recall = float(headline.loc[headline.method == "Raw", "recall@10"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    for sigma in cfg.sweep_sigma:
        sub = df[df.dummy_sigma == sigma].sort_values("dummy_k")
        ax.plot(sub.dummy_k, sub["recall@10"], marker="o", label=f"$\\sigma$={sigma}")
    ax.axhline(raw_recall, ls="--", color="k", lw=1, label="Raw / Vanilla ASPE")
    ax.set_title("(a) Utility vs. Dummy Dimensions")
    ax.set_xlabel("Number of dummy dimensions $k$")
    ax.set_ylabel("Recall@10")
    ax.legend(frameon=True, fontsize=8)

    ax = axes[1]
    for k in cfg.sweep_k:
        sub = df[df.dummy_k == k].sort_values("dummy_sigma")
        ax.plot(sub.dummy_sigma, sub["recall@10"], marker="s", label=f"k={k}")
    ax.axhline(raw_recall, ls="--", color="k", lw=1, label="Raw / Vanilla ASPE")
    ax.set_title("(b) Utility vs. Noise Scale")
    ax.set_xlabel("Dummy noise std $\\sigma$")
    ax.set_ylabel("Recall@10")
    ax.legend(frameon=True, fontsize=8)

    ax = axes[2]
    sc = ax.scatter(df["recall@10"], df["distortion_norm"],
                    c=df["dummy_k"], s=60, cmap="viridis",
                    edgecolor="k", linewidth=0.4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Dummy dimensions $k$")
    ax.axvline(raw_recall, ls="--", color="k", lw=1, label="Raw utility")
    ax.set_title("(c) Utility-Security Trade-off")
    ax.set_xlabel("Recall@10 (utility)")
    ax.set_ylabel("Normalised score distortion (obfuscation)")
    ax.legend(frameon=True, fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.4)

    fig.suptitle("ASPE + Dummy Dimensions: (k, $\\sigma$) Utility-Security Sweep "
                 "(FiQA2018)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(cfg.sweep_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
