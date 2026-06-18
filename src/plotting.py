"""IEEE-style figures: four standalone metric plots for the five-method headline
comparison, plus the (k, sigma) utility-security sweep."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import Config, log

# Shared IEEE-paper styling.
_DPI = 300
_FIGSIZE = (6.4, 4.2)


def _ieee_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.4,
                         "font.size": 11, "savefig.bbox": "tight"})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"Rendered figure -> {path}")


def _suffix(cfg: Config) -> str:
    return f"FiQA2018, N={cfg.n_corpus} docs, Q={cfg.n_queries}"


def plot_recall(df: pd.DataFrame, cfg: Config):
    _ieee_style()
    palette = sns.color_palette("colorblind")
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    w = 0.38
    r5, r10 = df["recall@5"].fillna(0).values, df["recall@10"].fillna(0).values
    ax.bar(x - w / 2, r5, w, label="Recall@5", color=palette[0])
    ax.bar(x + w / 2, r10, w, label="Recall@10", color=palette[1])
    for xi, (a, b) in enumerate(zip(r5, r10)):
        ax.text(xi - w / 2, a, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, max(r10.max(), r5.max()) * 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_title(f"Retrieval Utility (Recall@K)\n{_suffix(cfg)}")
    ax.legend(frameon=True)
    _save(fig, cfg.plot_recall_png)


def plot_latency(df: pd.DataFrame, cfg: Config):
    _ieee_style()
    palette = sns.color_palette("colorblind")
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    vals = df["latency_ms"].values.astype(float)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    finite = vals[np.isfinite(vals) & (vals > 0)]
    lo = (finite.min() if finite.size else 1e-3) * 0.3
    hi = (finite.max() if finite.size else 1.0) * 3.0
    ax.bar(x, np.where(np.isfinite(vals), vals, lo), color=palette[2], width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    for xi, v in enumerate(vals):
        if np.isnan(v):
            ax.text(xi, lo, "skipped", ha="center", va="bottom", fontsize=8)
        else:
            label = f"{v:.3f}" if v < 100 else f"{v:.0f}"
            ax.text(xi, v, label, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Avg query latency (ms, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_title(f"Query Latency\n{_suffix(cfg)}")
    _save(fig, cfg.plot_latency_png)


def plot_dram(df: pd.DataFrame, cfg: Config):
    _ieee_style()
    palette = sns.color_palette("colorblind")
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    vals = df["dram_mb"].values.astype(float)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    finite = vals[np.isfinite(vals) & (vals > 0)]
    lo = (finite.min() if finite.size else 1e-2) * 0.3
    hi = (finite.max() if finite.size else 1.0) * 3.0
    ax.bar(x, np.where(np.isfinite(vals), vals, lo), color=palette[3], width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    for xi, v in enumerate(vals):
        if np.isnan(v):
            ax.text(xi, lo, "skipped", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Encrypted corpus footprint (MB, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_title(f"DRAM Overhead ({cfg.n_corpus} corpus vectors)\n{_suffix(cfg)}")
    _save(fig, cfg.plot_dram_png)


def plot_security(df: pd.DataFrame, cfg: Config):
    _ieee_style()
    palette = sns.color_palette("colorblind")
    methods = df["method"].tolist()
    x = np.arange(len(methods))
    raw = df["security_mse"].values.astype(float)

    # An MSE below BROKEN means the linear KPA reconstructed the plaintext to
    # within floating-point noise -- the scheme is fully broken. Such values
    # differ only by meaningless round-off (e.g. Raw ~1e-32 vs ASPE's matrix
    # solve ~1e-16), so on a log axis their true magnitudes would render as a
    # spurious 16-orders-of-magnitude spread. We clamp every broken scheme to a
    # single floor bar so they read as equally (and totally) broken, keep the
    # genuinely informative values as-is, and cap +inf (FHE) above the top.
    BROKEN = 1e-9
    meaningful = raw[np.isfinite(raw) & (raw >= BROKEN)]
    top = float(meaningful.max()) if meaningful.size else BROKEN
    bottom = float(meaningful.min()) if meaningful.size else BROKEN
    floor = bottom * 1e-3          # shared height for all "broken" bars
    cap = top * 1e2                # sentinel height for the +inf (FHE) bar

    disp = np.empty_like(raw)
    for i, v in enumerate(raw):
        if np.isinf(v):
            disp[i] = cap
        elif np.isnan(v) or v < BROKEN:
            disp[i] = floor
        else:
            disp[i] = v

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.bar(x, disp, color=palette[4], width=0.6)
    ax.set_yscale("log")
    for xi, v in enumerate(raw):
        if np.isinf(v):
            label = "$\\infty$\n(KPA infeasible)"
        elif np.isnan(v):
            label = "n/a\n(skipped)"
        elif v < BROKEN:
            label = "$\\approx$0\n(broken)"
        else:
            label = f"{v:.1e}"
        ax.text(xi, disp[xi], label, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("KPA reconstruction MSE (log scale)")
    ax.set_ylim(floor * 0.3, cap * 8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_title(f"Empirical Security: Known-Plaintext Attack\n"
                 f"{_suffix(cfg)}, {cfg.kpa_pairs} known pairs")
    _save(fig, cfg.plot_security_png)


def plot_all(df: pd.DataFrame, cfg: Config):
    """Render the four standalone IEEE metric figures."""
    plot_recall(df, cfg)
    plot_latency(df, cfg)
    plot_dram(df, cfg)
    plot_security(df, cfg)


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
