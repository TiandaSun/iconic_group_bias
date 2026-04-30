"""Generate Task 2 (description) figures from computed metrics.

Outputs to results/figures/:
  fig_description_length.pdf
  fig_general_ctc.pdf
  fig_correct_ethnic_ctc_heatmap.pdf
  fig_human_ratings.pdf
  fig_human_vs_auto_corr.pdf
  fig_human_vs_cultural_depth_scatter.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.15)

MODEL_DISPLAY = {
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
    "qwen2-vl-7b": "Qwen2-VL-7B",
    "llama-3.2-vision-11b": "LLaMA-3.2-V-11B",
    "gpt-4o-mini": "GPT-4o-mini",
    "claude-haiku-4.5": "Claude-Haiku-4.5",
}
MODEL_ORDER = [
    "qwen2.5-vl-7b", "qwen2-vl-7b",
    "llama-3.2-vision-11b", "gpt-4o-mini", "claude-haiku-4.5",
]
ETHNIC_ORDER = ["Miao", "Dong", "Yi", "Li", "Tibetan"]
ORIGIN_COLORS = {"chinese": "#E67E22", "western": "#3498DB"}
LANG_COLORS = {"zh": "#C0392B", "en": "#16A085"}
ORIGIN = {
    "qwen2.5-vl-7b": "chinese", "qwen2-vl-7b": "chinese",
    "llama-3.2-vision-11b": "western", "gpt-4o-mini": "western",
    "claude-haiku-4.5": "western",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"  saved {name}.{{pdf,png}}")
    plt.close(fig)


def paired_barplot(df, value_col, ylabel, title, fname, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 4.3))
    x = np.arange(len(MODEL_ORDER))
    w = 0.38
    for i, lang in enumerate(["zh", "en"]):
        sub = df[df["language"] == lang].set_index("model").reindex(MODEL_ORDER)
        ax.bar(x + (i - 0.5) * w, sub[value_col], w,
               label=lang, color=LANG_COLORS[lang],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=20, ha="right")
    for tick, m in zip(ax.get_xticklabels(), MODEL_ORDER):
        tick.set_color(ORIGIN_COLORS[ORIGIN[m]])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(title="Prompt", loc="best", fontsize=9)
    if ylim:
        ax.set_ylim(ylim)
    save(fig, fname)


# ---------------------------------------------------------------------------
# Fig: description length
# ---------------------------------------------------------------------------
def fig_length():
    df = pd.read_csv(METRICS / "task2_per_description.csv")
    agg = df.groupby(["model", "language"])["char_len"].mean().reset_index()
    paired_barplot(agg, "char_len",
                   "Mean description length (chars)",
                   "Description length by model × prompt language",
                   "fig_description_length")


# ---------------------------------------------------------------------------
# Fig: general CTC
# ---------------------------------------------------------------------------
def fig_general_ctc():
    df = pd.read_csv(METRICS / "task2_per_description.csv")
    agg = df.groupby(["model", "language"])["general_ctc"].mean().reset_index()
    paired_barplot(agg, "general_ctc",
                   "General Cultural Term Coverage",
                   "General CTC (coverage of generic cultural vocabulary)",
                   "fig_general_ctc", ylim=(0, 0.25))


# ---------------------------------------------------------------------------
# Fig: correct-ethnic CTC heatmap
# ---------------------------------------------------------------------------
def fig_correct_ethnic_heatmap():
    df = pd.read_csv(METRICS / "task2_per_description.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, lang in zip(axes, ["zh", "en"]):
        sub = df[df["language"] == lang]
        pivot = sub.groupby(["model", "ethnic_group"])["correct_ethnic_ctc"].mean().unstack()
        pivot = pivot.reindex(MODEL_ORDER)[ETHNIC_ORDER]
        pivot.index = [MODEL_DISPLAY[m] for m in pivot.index]
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap="YlOrRd",
                    vmin=0, vmax=0.1, cbar=(lang == "en"),
                    linewidths=0.5, linecolor="white",
                    annot_kws={"fontsize": 9})
        ax.set_title(f"Prompt language: {lang}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Ethnic group")
        if lang == "zh":
            ax.set_ylabel("")
    fig.suptitle("Correct-ethnic CTC: fraction of group-specific terms mentioned (values near 0 indicate models avoid specific cultural vocabulary)",
                 fontsize=10, y=1.03)
    save(fig, "fig_correct_ethnic_ctc_heatmap")


# ---------------------------------------------------------------------------
# Fig: human ratings
# ---------------------------------------------------------------------------
def fig_human_ratings():
    df = pd.read_csv(METRICS / "task2_human_vs_auto.csv")
    metrics = ["Rating_1to5", "Accuracy_1to5", "Cultural_Depth_1to5"]
    # Completeness ceiling-effect dropped.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    for ax, m in zip(axes, metrics):
        agg = df.groupby(["model", "language"])[m].mean().reset_index()
        x = np.arange(len(MODEL_ORDER))
        w = 0.38
        for i, lang in enumerate(["zh", "en"]):
            sub = agg[agg["language"] == lang].set_index("model").reindex(MODEL_ORDER)
            ax.bar(x + (i - 0.5) * w, sub[m], w,
                   label=lang, color=LANG_COLORS[lang],
                   edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[mm] for mm in MODEL_ORDER], rotation=25, ha="right", fontsize=8)
        for tick, mm in zip(ax.get_xticklabels(), MODEL_ORDER):
            tick.set_color(ORIGIN_COLORS[ORIGIN[mm]])
        pretty = m.replace("_1to5", "").replace("_", " ")
        ax.set_title(pretty, fontsize=11, fontweight="bold")
        ax.set_ylim(1, 5)
        ax.axhline(3, color="gray", linestyle=":", linewidth=0.8)
        if ax is axes[0]:
            ax.set_ylabel("Mean human rating (1-5)")
    axes[0].legend(title="Prompt", loc="lower left", fontsize=8)
    fig.suptitle("Human-rated description quality (n=150, 15 per model × language)",
                 fontsize=11, y=1.02)
    save(fig, "fig_human_ratings")


# ---------------------------------------------------------------------------
# Fig: human vs auto correlation heatmap
# ---------------------------------------------------------------------------
def fig_correlation_heatmap():
    with open(METRICS / "task2_correlations.json") as f:
        corr = json.load(f)
    human_metrics = ["Rating_1to5", "Accuracy_1to5", "Completeness_1to5", "Cultural_Depth_1to5"]
    auto_metrics = ["char_len", "token_count", "ttr", "general_ctc", "correct_ethnic_ctc", "confusion_term_rate"]
    M = np.zeros((len(human_metrics), len(auto_metrics)))
    P = np.zeros_like(M, dtype=bool)
    for i, h in enumerate(human_metrics):
        for j, a in enumerate(auto_metrics):
            M[i, j] = corr[h][a]["rho"] or 0
            P[i, j] = (corr[h][a]["p_value"] is not None) and (corr[h][a]["p_value"] < 0.05)
    annot = np.array([[f"{M[i,j]:+.2f}" + ("*" if P[i,j] else "")
                      for j in range(M.shape[1])] for i in range(M.shape[0])])
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    sns.heatmap(M, ax=ax, annot=annot, fmt="", cmap="RdBu_r",
                vmin=-0.5, vmax=0.5, center=0,
                xticklabels=auto_metrics,
                yticklabels=[h.replace("_1to5", "") for h in human_metrics],
                linewidths=0.5, linecolor="white", cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Human rating × automated metric correlation (* p<0.05, n=150)",
                 fontsize=11, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    save(fig, "fig_human_vs_auto_corr")


# ---------------------------------------------------------------------------
# Fig: scatter of char_len vs Cultural_Depth (top correlation)
# ---------------------------------------------------------------------------
def fig_depth_scatter():
    df = pd.read_csv(METRICS / "task2_human_vs_auto.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    for m in MODEL_ORDER:
        sub = df[df["model"] == m]
        ax.scatter(sub["char_len"], sub["Cultural_Depth_1to5"],
                   alpha=0.6, s=42, label=MODEL_DISPLAY[m],
                   edgecolor="black", linewidth=0.3,
                   color=ORIGIN_COLORS[ORIGIN[m]],
                   marker="o" if ORIGIN[m] == "chinese" else "s")
    # Add jitter for visualization of discrete ratings
    ax.set_xlabel("Description length (chars)")
    ax.set_ylabel("Cultural Depth rating (1-5)")
    ax.set_title("Cultural-Depth rating vs. description length (ρ = +0.37)",
                 fontsize=11, fontweight="bold")
    # Trend line
    x = df["char_len"].values
    y = df["Cultural_Depth_1to5"].values
    mask = ~np.isnan(x) & ~np.isnan(y)
    z = np.polyfit(x[mask], y[mask], 1)
    xs = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(xs, np.polyval(z, xs), "k--", linewidth=1.5, alpha=0.7, label="Linear fit")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig_human_vs_cultural_depth_scatter")


def main():
    print("Generating Task 2 figures ...")
    fig_length()
    fig_general_ctc()
    fig_correct_ethnic_heatmap()
    fig_human_ratings()
    fig_correlation_heatmap()
    fig_depth_scatter()
    print("Done. Output dir:", FIGS)


if __name__ == "__main__":
    main()
