"""In-text cross-group confusion analysis.

Scans every description for mentions of:
  - Each ethnic-group name (English + Chinese)
  - Geo-mislocation terms (Thailand, Southeast Asia, 东南亚, 泰国)
Builds a (true_group × mentioned_term) matrix per (model, language).

Outputs:
  results/metrics/task2_intext_confusion.csv
  results/figures/fig_intext_confusion.pdf
  results/figures/fig_failure_taxonomy.pdf
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
METRICS = ROOT / "results" / "metrics"
FIGS = ROOT / "results" / "figures"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.1)

ETHNIC_GROUPS = ["Miao", "Dong", "Yi", "Li", "Tibetan"]
MODEL_CANONICAL: dict[str, str] = {}
MODEL_DISPLAY = {
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
    "qwen2-vl-7b": "Qwen2-VL-7B",
    "llama-3.2-vision-11b": "LLaMA-3.2-V-11B",
    "gpt-4o-mini": "GPT-4o-mini",
    "claude-haiku-4.5": "Claude-Haiku-4.5",
}
MODEL_ORDER = ["qwen2.5-vl-7b", "qwen2-vl-7b",
               "llama-3.2-vision-11b", "gpt-4o-mini", "claude-haiku-4.5"]
ORIGIN = {
    "qwen2.5-vl-7b": "chinese", "qwen2-vl-7b": "chinese",
    "llama-3.2-vision-11b": "western", "gpt-4o-mini": "western",
    "claude-haiku-4.5": "western",
}
ORIGIN_COLORS = {"chinese": "#E67E22", "western": "#3498DB"}

# Group-name tokens and geo-mislocation markers
GROUP_TOKENS = {
    "en": {
        "Miao": [r"\bmiao\b", r"\bhmong\b"],
        "Dong": [r"\bdong\b(?!\s*quixote)"],       # avoid false match
        "Yi": [r"\byi\s+(people|ethnic|minority|nationality|women|costume|dress|attire)"],
        "Li": [r"\bli\s+(people|ethnic|minority|nationality|women|costume|dress|attire)"],
        "Tibetan": [r"\btibetan\b", r"\btibet\b"],
    },
    "zh": {
        "Miao":    ["苗族"],
        "Dong":    ["侗族"],
        "Yi":      ["彝族"],
        "Li":      ["黎族"],
        "Tibetan": ["藏族"],
    },
}
GEO_MISLOCATION = {
    "en": [r"\bthailand\b", r"\bthai\b", r"\bsoutheast asia\b", r"\bvietnam\b",
           r"\blaos\b", r"\bcambodia\b", r"\bindigenous\b"],
    "zh": ["泰国", "东南亚", "越南", "老挝"],
}


def match_any(text: str, patterns: list[str], lang: str) -> bool:
    if lang == "en":
        return any(re.search(p, text.lower()) for p in patterns)
    return any(p in text for p in patterns)


def load_descriptions() -> pd.DataFrame:
    rows = []
    for fp in sorted(glob.glob(str(RAW / "task2_*_results.json"))):
        with open(fp) as f:
            d = json.load(f)
        model = MODEL_CANONICAL.get(d["model_name"], d["model_name"])
        lang = d["language"]
        for img_id, v in d["results"].items():
            rows.append({
                "image_id": img_id,
                "model": model,
                "language": lang,
                "true_group": v.get("ethnic_group", ""),
                "description": v.get("description", "") or "",
            })
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Mark whether each description mentions each group name + geo-mislocation."""
    records = []
    for _, row in df.iterrows():
        lang = row["language"]
        text = row["description"]
        entry = {"image_id": row["image_id"], "model": row["model"],
                 "language": lang, "true_group": row["true_group"]}
        for g in ETHNIC_GROUPS:
            entry[f"mentions_{g}"] = match_any(text, GROUP_TOKENS[lang][g], lang)
        entry["mentions_geo_mislocation"] = match_any(text, GEO_MISLOCATION[lang], lang)
        records.append(entry)
    return pd.DataFrame(records)


def confusion_matrix_from_mentions(ann: pd.DataFrame) -> pd.DataFrame:
    """Fraction of descriptions of true_group X that mention group Y."""
    mat = pd.DataFrame(index=ETHNIC_GROUPS, columns=ETHNIC_GROUPS, dtype=float)
    for true_g in ETHNIC_GROUPS:
        sub = ann[ann["true_group"] == true_g]
        if len(sub) == 0:
            continue
        for mention_g in ETHNIC_GROUPS:
            mat.loc[true_g, mention_g] = sub[f"mentions_{mention_g}"].mean()
    return mat


def geo_mislocation_by_group(ann: pd.DataFrame) -> pd.DataFrame:
    """Rate of geo-mislocation term per true_group × (model, language)."""
    return (ann.groupby(["true_group", "model", "language"])["mentions_geo_mislocation"]
               .mean().reset_index())


# ---------------------------------------------------------------------------
# Figure: in-text confusion heatmap (pooled across models)
# ---------------------------------------------------------------------------
def fig_intext_confusion_pooled(ann: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, lang in zip(axes, ["zh", "en"]):
        sub = ann[ann["language"] == lang]
        mat = confusion_matrix_from_mentions(sub)
        sns.heatmap(mat.astype(float), ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=0.5, cbar=(lang == "en"),
                    linewidths=0.5, linecolor="white", annot_kws={"fontsize": 10})
        ax.set_title(f"Prompt language: {lang}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Group mentioned in description")
        if lang == "zh":
            ax.set_ylabel("True ethnic group")
        # Highlight diagonal with thick green borders (correct = what we want to see)
        for i in range(5):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                       edgecolor="#006400", lw=2.8, zorder=10))
    fig.suptitle("In-text group-name confusion: fraction of descriptions mentioning each group\n(green diagonal boxes = correct group self-mention; red off-diagonals = confusion)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_intext_confusion.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_intext_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_intext_confusion.{pdf,png}")


# ---------------------------------------------------------------------------
# Figure: failure taxonomy — in-text confusion per model for two diagnostic cells
# ---------------------------------------------------------------------------
def fig_failure_taxonomy(ann: pd.DataFrame) -> None:
    """For each true_group, show per-model rates (pooled over languages):
       (a) correct self-mention, (b) 'Miao' mislabel, (c) geo-mislocation."""
    rows = []
    for g in ETHNIC_GROUPS:
        for m in MODEL_ORDER:
            sub = ann[(ann["true_group"] == g) & (ann["model"] == m)]
            if len(sub) == 0:
                continue
            rows.append({
                "true_group": g, "model": m,
                "self_rate": sub[f"mentions_{g}"].mean(),
                "miao_rate": sub["mentions_Miao"].mean(),
                "geo_rate": sub["mentions_geo_mislocation"].mean(),
            })
    dfm = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8), sharey=True)
    x = np.arange(len(MODEL_ORDER))
    w = 0.27
    for col, g in enumerate(ETHNIC_GROUPS):
        ax = axes[col]
        sub = dfm[dfm["true_group"] == g].set_index("model").reindex(MODEL_ORDER)
        ax.bar(x - w, sub["self_rate"], w, label="correct group name",
               color="#27AE60", edgecolor="black", linewidth=0.3)
        ax.bar(x, sub["miao_rate"], w, label="mentions 'Miao'",
               color="#E74C3C", edgecolor="black", linewidth=0.3)
        ax.bar(x + w, sub["geo_rate"], w, label="geo-mislocation",
               color="#8E44AD", edgecolor="black", linewidth=0.3)
        ax.set_title(g, fontsize=12, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Fraction of descriptions\n(pooled over zh+en)", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m].replace("-VL-", "-").replace("-Vision-", "-")
                            for m in MODEL_ORDER], rotation=55, ha="right", fontsize=8)
        for tick, m in zip(ax.get_xticklabels(), MODEL_ORDER):
            tick.set_color(ORIGIN_COLORS[ORIGIN[m]])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.suptitle("Failure taxonomy per ethnic group: green=correctly names true group, red=mentions 'Miao', purple=references SE Asia/Thailand",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_failure_taxonomy.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_failure_taxonomy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved fig_failure_taxonomy.{pdf,png}")


def main():
    print("Loading descriptions ...")
    df = load_descriptions()
    print(f"  {len(df)} descriptions")

    print("Annotating mentions ...")
    ann = annotate(df)
    ann.to_csv(METRICS / "task2_intext_mentions.csv", index=False)

    # Pooled confusion matrices
    for lang in ["zh", "en"]:
        mat = confusion_matrix_from_mentions(ann[ann["language"] == lang])
        print(f"\n=== In-text confusion, pooled over models [{lang}] ===")
        print((mat * 100).round(1))

    # Save to CSV (long format)
    long_rows = []
    for model in MODEL_ORDER:
        for lang in ["zh", "en"]:
            sub = ann[(ann["model"] == model) & (ann["language"] == lang)]
            for true_g in ETHNIC_GROUPS:
                s = sub[sub["true_group"] == true_g]
                if len(s) == 0:
                    continue
                for mention_g in ETHNIC_GROUPS:
                    long_rows.append({
                        "model": model, "language": lang,
                        "true_group": true_g, "mentioned_group": mention_g,
                        "rate": s[f"mentions_{mention_g}"].mean(),
                        "n": len(s),
                    })
                long_rows.append({
                    "model": model, "language": lang,
                    "true_group": true_g, "mentioned_group": "geo_mislocation",
                    "rate": s["mentions_geo_mislocation"].mean(),
                    "n": len(s),
                })
    pd.DataFrame(long_rows).to_csv(METRICS / "task2_intext_confusion.csv", index=False)

    # Geo-mislocation summary
    print("\n=== Geo-mislocation rate (any SEA/Thailand term) by true_group × language ===")
    geo = ann.groupby(["true_group", "language"])["mentions_geo_mislocation"].mean().unstack()
    print((geo * 100).round(1))

    # Figures
    print("\nGenerating figures ...")
    fig_intext_confusion_pooled(ann)
    fig_failure_taxonomy(ann)

    print("\nDone.")


if __name__ == "__main__":
    main()
