"""Generate Task 1 figures: mode collapse, OBI per group, interaction plot, confusion matrices.

Reads metrics from results/metrics/ and writes PDF + PNG to results/figures/.
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
# Ordered: Chinese first, then Western
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


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    print(f"  saved {name}.{{pdf,png}}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 1: Prediction distribution (mode collapse)
# ---------------------------------------------------------------------------
def fig_mode_collapse() -> None:
    df = pd.read_csv(METRICS / "task1_prediction_dist.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, lang in zip(axes, ["zh", "en"]):
        sub = df[df["language"] == lang]
        pivot = sub.pivot(index="model", columns="predicted_group", values="fraction")
        pivot = pivot.reindex(MODEL_ORDER)[ETHNIC_ORDER]
        pivot.index = [MODEL_DISPLAY[m] for m in pivot.index]
        pivot.plot(kind="bar", stacked=True, ax=ax,
                   colormap="viridis", width=0.75, edgecolor="white", linewidth=0.5)
        # True base rate is uniform (20% each); 5 equal-size classes
        ax.axhline(0.2, color="red", linestyle="--", linewidth=2.2, alpha=0.85, zorder=5)
        ax.set_title(f"Prompt language: {lang}", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Fraction of predictions" if lang == "zh" else "")
        ax.set_ylim(0, 1.05)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(handles, labels, title="Predicted", loc="center right",
               bbox_to_anchor=(1.05, 0.5), fontsize=9)
    fig.suptitle("Mode collapse: models concentrate predictions on Miao and Tibetan\n(red dashed line = uniform base rate 20%)",
                 fontsize=12, y=1.02)
    save(fig, "fig_mode_collapse")


# ---------------------------------------------------------------------------
# Fig 2: OBI per ethnic group
# ---------------------------------------------------------------------------
def fig_obi_per_group() -> None:
    with open(METRICS / "task1_bias_metrics.json") as f:
        bias = json.load(f)
    obi_g = bias["obi_by_ethnic_group_pooled"]
    rows = []
    for g in ETHNIC_ORDER:
        rows.append({
            "group": g,
            "chinese": obi_g[g]["chinese_mean_acc"],
            "western": obi_g[g]["western_mean_acc"],
            "obi": obi_g[g]["obi"],
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Left: per-group accuracy grouped bar
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w/2, df["chinese"], w, label="Chinese-origin", color=ORIGIN_COLORS["chinese"], edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, df["western"], w, label="Western-origin", color=ORIGIN_COLORS["western"], edgecolor="black", linewidth=0.4)
    ax.axhline(0.2, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Chance (0.20)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["group"])
    ax.set_ylabel("Per-class accuracy (pooled over zh + en)")
    ax.set_title("(a) Accuracy by model origin and ethnic group", fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", fontsize=9)
    ax.set_ylim(0, 1.0)

    # Right: OBI per group
    ax = axes[1]
    colors = ["#C0392B" if v < 0 else "#27AE60" for v in df["obi"]]
    bars = ax.barh(df["group"], df["obi"], color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("OBI = (CN − W) / all")
    ax.set_title("(b) Origin Bias Index per ethnic group", fontsize=11, fontweight="bold")
    for bar, val, g in zip(bars, df["obi"], df["group"]):
        xoff = 0.03 if val >= 0 else -0.03
        ha = "left" if val >= 0 else "right"
        # Annotate Li OBI with a caveat marker — near-zero denominator inflates ratio
        label = f"{val:+.2f}" + (" †" if g == "Li" else "")
        ax.text(val + xoff, bar.get_y() + bar.get_height()/2, label,
                va="center", ha=ha, fontsize=9)
    ax.set_xlim(-1.0, 1.6)
    ax.invert_yaxis()
    ax.text(1.55, 4.35, "† Li OBI inflated:\n   both accuracies\n   near zero",
            fontsize=7, color="gray", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    save(fig, "fig_obi_per_group")


# ---------------------------------------------------------------------------
# Fig 3: Origin × Language interaction
# ---------------------------------------------------------------------------
def fig_interaction() -> None:
    df = pd.read_csv(METRICS / "task1_overall.csv")
    grp = df.groupby(["origin", "language"])["accuracy"].agg(["mean", "std", "count"]).reset_index()
    grp["se"] = grp["std"] / np.sqrt(grp["count"])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for origin in ["chinese", "western"]:
        sub = grp[grp["origin"] == origin].set_index("language").loc[["en", "zh"]]
        ax.errorbar(["en", "zh"], sub["mean"], yerr=sub["se"],
                    marker="o", markersize=10, linewidth=2.2, capsize=5,
                    color=ORIGIN_COLORS[origin], label=f"{origin.capitalize()}-origin")
        for lang, val in sub["mean"].items():
            ax.text(lang, val + 0.008, f"{val:.3f}", ha="center", fontsize=9,
                    color=ORIGIN_COLORS[origin], fontweight="bold")
    ax.axhline(0.2, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Chance")
    ax.set_xlabel("Prompt language")
    ax.set_ylabel("Overall accuracy (mean across models)")
    ax.set_title("Origin × Language interaction on classification accuracy",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.18, 0.45)
    save(fig, "fig_origin_language_interaction")


# ---------------------------------------------------------------------------
# Fig 4: Confusion matrices (5 models × 2 languages)
# ---------------------------------------------------------------------------
def fig_confusion() -> None:
    with open(METRICS / "task1_confusion.json") as f:
        conf = json.load(f)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2))
    for col, model in enumerate(MODEL_ORDER):
        for row, lang in enumerate(["zh", "en"]):
            ax = axes[row, col]
            M = np.array(conf[f"{model}|{lang}"]["row_normalized"])
            sns.heatmap(M, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                        vmin=0, vmax=1, cbar=(col == 4),
                        xticklabels=ETHNIC_ORDER, yticklabels=ETHNIC_ORDER,
                        annot_kws={"fontsize": 7}, linewidths=0.3, linecolor="white")
            if row == 0:
                title_color = ORIGIN_COLORS[ORIGIN[model]]
                ax.set_title(MODEL_DISPLAY[model], fontsize=10, fontweight="bold", color=title_color)
            if col == 0:
                ax.set_ylabel(f"lang={lang}\nTrue", fontsize=10)
            else:
                ax.set_ylabel("")
            if row == 1:
                ax.set_xlabel("Predicted", fontsize=9)
            else:
                ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.tick_params(axis="y", labelsize=7, rotation=0)
    fig.suptitle("Confusion matrices (row-normalized). Chinese-origin models in orange, Western in blue.",
                 fontsize=11, y=1.00)
    save(fig, "fig_confusion_matrices")


# ---------------------------------------------------------------------------
# Fig 5 (bonus): Overall accuracy grouped bar with CI
# ---------------------------------------------------------------------------
def fig_overall_accuracy() -> None:
    df = pd.read_csv(METRICS / "task1_overall.csv")
    df["model_display"] = df["model"].map(MODEL_DISPLAY)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("model")
    df["err_lo"] = df["accuracy"] - df["accuracy_ci_lo"]
    df["err_hi"] = df["accuracy_ci_hi"] - df["accuracy"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(MODEL_ORDER))
    w = 0.38
    for i, lang in enumerate(["zh", "en"]):
        sub = df[df["language"] == lang].set_index("model").loc[MODEL_ORDER]
        ax.bar(x + (i - 0.5) * w, sub["accuracy"], w,
               yerr=[sub["err_lo"], sub["err_hi"]],
               label=f"{lang}", color=LANG_COLORS[lang],
               edgecolor="black", linewidth=0.4, capsize=3)
    ax.axhline(0.2, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Chance (0.20)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=20, ha="right")
    # Color x-ticks by origin
    for tick, m in zip(ax.get_xticklabels(), MODEL_ORDER):
        tick.set_color(ORIGIN_COLORS[ORIGIN[m]])
    ax.set_ylabel("Overall accuracy (95% bootstrap CI)")
    ax.set_title("Classification accuracy by model and prompt language",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Prompt", loc="upper right", fontsize=9)
    ax.set_ylim(0, 0.5)
    save(fig, "fig_overall_accuracy")


def main() -> None:
    print("Generating Task 1 figures ...")
    fig_overall_accuracy()
    fig_mode_collapse()
    fig_obi_per_group()
    fig_interaction()
    fig_confusion()
    print("Done. Output dir:", FIGS)


if __name__ == "__main__":
    main()
