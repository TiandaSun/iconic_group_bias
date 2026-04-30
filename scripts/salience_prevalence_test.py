"""Test the salience-bias hypothesis: does per-group accuracy track training-data prevalence?

Proxy for training-data prevalence (no direct access to model training corpora):
  - Wikipedia (en) page size: bytes of the main article
  - Wikipedia (en) page views: via REST API (last 60 days average)
  - Wikipedia (zh) page size: Chinese Wikipedia
  - Number of images on the English Wikipedia article (via MediaWiki API)

Correlates each proxy with per-group classification accuracy (pooled over models/languages)
and writes a scatter figure.

Outputs:
  results/metrics/salience_prevalence.json
  results/metrics/salience_prevalence.csv
  results/figures/fig_salience_scatter.pdf / .png
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"
FIGS = ROOT / "results" / "figures"

plt.style.use("seaborn-v0_8-whitegrid")

# Wikipedia page titles per group
TITLES = {
    "Miao":    {"en": "Miao_people",    "zh": "苗族"},
    "Dong":    {"en": "Dong_people",    "zh": "侗族"},
    "Yi":      {"en": "Yi_people",      "zh": "彝族"},
    "Li":      {"en": "Li_people",      "zh": "黎族"},
    "Tibetan": {"en": "Tibetan_people", "zh": "藏族"},
}

UA = "vlm-cultural-eval-research/1.0 (academic contact)"


def fetch(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def page_bytes(lang: str, title: str) -> int:
    url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    return len(fetch(url))


def page_image_count(lang: str, title: str) -> int:
    """Count images on the page via MediaWiki API."""
    api = (f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=images"
           f"&titles={urllib.parse.quote(title)}&imlimit=500&format=json")
    data = json.loads(fetch(api))
    pages = data.get("query", {}).get("pages", {})
    total = 0
    for _, p in pages.items():
        total += len(p.get("images", []))
    return total


def page_views(lang: str, title: str, days: int = 60) -> int:
    end = datetime.utcnow().date() - timedelta(days=1)
    start = end - timedelta(days=days)
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
           f"{lang}.wikipedia.org/all-access/user/{urllib.parse.quote(title)}/daily/"
           f"{start:%Y%m%d}/{end:%Y%m%d}")
    try:
        data = json.loads(fetch(url))
        return sum(item.get("views", 0) for item in data.get("items", []))
    except Exception as e:
        print(f"  warn: pageviews failed for {lang}:{title} — {e}")
        return -1


def collect_prevalence() -> pd.DataFrame:
    rows = []
    for g, titles in TITLES.items():
        print(f"  fetching {g} ...")
        r = {"ethnic_group": g}
        for lang in ["en", "zh"]:
            title = titles[lang]
            r[f"bytes_{lang}"] = page_bytes(lang, title)
            time.sleep(0.3)
            r[f"images_{lang}"] = page_image_count(lang, title)
            time.sleep(0.3)
            r[f"views_60d_{lang}"] = page_views(lang, title)
            time.sleep(0.3)
        rows.append(r)
    return pd.DataFrame(rows)


def load_per_group_accuracy() -> pd.DataFrame:
    """Mean classification accuracy per ethnic group (pooled over models + languages)."""
    df = pd.read_csv(METRICS / "task1_per_cell.csv")
    out = df.groupby("ethnic_group")["accuracy"].mean().reset_index()
    out.columns = ["ethnic_group", "mean_accuracy"]
    # Also per-language
    by_lang = df.groupby(["ethnic_group", "language"])["accuracy"].mean().unstack()
    by_lang.columns = [f"mean_acc_{c}" for c in by_lang.columns]
    return out.merge(by_lang.reset_index(), on="ethnic_group")


def make_figure(df: pd.DataFrame, corrs: dict) -> None:
    proxies = [
        ("bytes_en",     "EN Wikipedia page size (bytes)"),
        ("bytes_zh",     "ZH Wikipedia page size (bytes)"),
        ("images_en",    "EN Wikipedia image count"),
        ("views_60d_en", "EN Wikipedia page views (60d)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9))
    group_colors = {"Miao": "#8E44AD", "Dong": "#2980B9", "Yi": "#16A085",
                    "Li": "#F39C12", "Tibetan": "#C0392B"}
    for ax, (col, label) in zip(axes, proxies):
        for _, row in df.iterrows():
            ax.scatter(row[col], row["mean_accuracy"],
                       s=140, color=group_colors[row["ethnic_group"]],
                       edgecolor="black", linewidth=0.6, zorder=3)
            ax.annotate(row["ethnic_group"],
                        (row[col], row["mean_accuracy"]),
                        xytext=(6, 6), textcoords="offset points", fontsize=9)
        rho = corrs[col]["spearman_rho"]
        p = corrs[col]["spearman_p"]
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("Mean classification accuracy" if col == "bytes_en" else "")
        ax.set_title(f"ρ = {rho:+.2f}  (p={p:.2f})", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.axhline(0.2, color="red", linestyle="--", linewidth=1, alpha=0.5)
    fig.suptitle("Salience hypothesis: per-group accuracy vs. training-data prevalence proxies (n=5)",
                 fontsize=11, y=1.03)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_salience_scatter.pdf", bbox_inches="tight")
    fig.savefig(FIGS / "fig_salience_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Fetching Wikipedia prevalence proxies ...")
    prev = collect_prevalence()
    print(prev.to_string(index=False))

    acc = load_per_group_accuracy()
    merged = prev.merge(acc, on="ethnic_group")
    merged.to_csv(METRICS / "salience_prevalence.csv", index=False)

    print("\nComputing correlations with mean accuracy ...")
    proxy_cols = ["bytes_en", "bytes_zh", "images_en", "images_zh",
                  "views_60d_en", "views_60d_zh"]
    corrs = {}
    for col in proxy_cols:
        vals = merged[col]
        if (vals < 0).any():
            corrs[col] = {"error": "missing data"}
            continue
        rho, p_s = spearmanr(merged[col], merged["mean_accuracy"])
        r, p_p = pearsonr(merged[col], merged["mean_accuracy"])
        corrs[col] = {"spearman_rho": float(rho), "spearman_p": float(p_s),
                      "pearson_r": float(r), "pearson_p": float(p_p)}
        print(f"  {col:<16}  Spearman ρ={rho:+.2f} (p={p_s:.3f})  "
              f"Pearson r={r:+.2f} (p={p_p:.3f})")

    with open(METRICS / "salience_prevalence.json", "w") as f:
        json.dump({"data": merged.to_dict(orient="records"), "correlations": corrs},
                  f, indent=2, ensure_ascii=False)

    make_figure(merged, corrs)
    print(f"\nSaved figure + CSV + JSON to {METRICS} and {FIGS}")


if __name__ == "__main__":
    main()
