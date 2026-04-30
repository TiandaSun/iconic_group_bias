"""Compute Task 2 description-quality metrics.

Metrics:
  - Length (chars, words)
  - Type-token ratio (TTR)
  - Cultural Term Coverage (CTC) — general + ethnic-specific
  - Correct-ethnic-term hit rate
  - Cross-ethnic confusion term rate (description mentions wrong ethnic group's terms)
Cross-reference with human eval ratings (Spearman correlations).

Outputs to results/metrics/:
  task2_per_description.csv      — per-description automated metrics
  task2_by_cell.csv              — aggregated by (model, language, ethnic)
  task2_human_vs_auto.csv        — merged with human ratings
  task2_correlations.json        — Spearman ρ human vs automated metrics
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "metrics"
VOCAB_PATH = ROOT / "configs" / "cultural_vocabulary.yaml"
HUMAN_EVAL = ROOT / "results" / "human_eval" / "result" / "description_quality_rating.csv"
OUT.mkdir(parents=True, exist_ok=True)

ETHNIC_GROUPS = ["Miao", "Dong", "Yi", "Li", "Tibetan"]
MODEL_CANONICAL: dict[str, str] = {}
ORIGIN = {
    "qwen2.5-vl-7b": "chinese", "qwen2-vl-7b": "chinese",
    "llama-3.2-vision-11b": "western", "gpt-4o-mini": "western",
    "claude-haiku-4.5": "western",
}


def load_vocabulary() -> dict:
    with open(VOCAB_PATH) as f:
        v = yaml.safe_load(f)
    return {
        "general": {"zh": list(v["zh"]), "en": list(v["en"])},
        "ethnic": {g: v["ethnic_specific"][g] for g in ETHNIC_GROUPS},
    }


def count_term_hits(text: str, terms: list[str], language: str) -> tuple[int, int, list[str]]:
    """Return (n_matched_terms, total_terms, matched_list)."""
    if not text:
        return 0, len(terms), []
    if language == "zh":
        matched = [t for t in terms if t in text]
    else:  # en: word-boundary case-insensitive
        lower = text.lower()
        matched = [t for t in terms if re.search(rf"\b{re.escape(t.lower())}\b", lower)]
    return len(matched), len(terms), matched


def word_tokens(text: str, language: str) -> list[str]:
    if language == "zh":
        # crude: drop whitespace/punct, use chars (for zh TTR we use char-level as proxy)
        return [c for c in text if c.strip() and not re.match(r"[\W_]", c)]
    return re.findall(r"[A-Za-z]+", text.lower())


def describe_text(text: str, language: str) -> dict:
    tokens = word_tokens(text, language)
    n = len(tokens)
    ttr = len(set(tokens)) / n if n > 0 else 0.0
    return {
        "char_len": len(text),
        "token_count": n,
        "ttr": ttr,
    }


def compute_per_description() -> pd.DataFrame:
    vocab = load_vocabulary()
    rows = []
    for fp in sorted(glob.glob(str(RAW / "task2_*_results.json"))):
        with open(fp) as f:
            d = json.load(f)
        model = MODEL_CANONICAL.get(d["model_name"], d["model_name"])
        lang = d["language"]
        for img_id, v in d["results"].items():
            text = v.get("description", "") or ""
            true_g = v.get("ethnic_group", "")
            stats = describe_text(text, lang)
            # General CTC
            gen_hit, gen_tot, _ = count_term_hits(text, vocab["general"][lang], lang)
            # Correct-ethnic terms
            true_terms = vocab["ethnic"].get(true_g, {}).get(lang, [])
            true_hit, true_tot, true_matched = count_term_hits(text, true_terms, lang)
            # Confusion terms (other ethnic groups)
            conf_hits = 0
            conf_tot = 0
            for g in ETHNIC_GROUPS:
                if g == true_g:
                    continue
                other_terms = vocab["ethnic"].get(g, {}).get(lang, [])
                h, t, _ = count_term_hits(text, other_terms, lang)
                conf_hits += h
                conf_tot += t
            rows.append({
                "image_id": img_id,
                "model": model,
                "origin": ORIGIN[model],
                "language": lang,
                "ethnic_group": true_g,
                **stats,
                "general_ctc": gen_hit / gen_tot if gen_tot else 0.0,
                "general_terms_matched": gen_hit,
                "correct_ethnic_ctc": true_hit / true_tot if true_tot else 0.0,
                "correct_ethnic_terms_matched": true_hit,
                "correct_ethnic_terms_total": true_tot,
                "confusion_terms_matched": conf_hits,
                "confusion_term_rate": conf_hits / conf_tot if conf_tot else 0.0,
                "matched_ethnic_terms": ";".join(true_matched),
            })
    return pd.DataFrame(rows)


def aggregate_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = ["char_len", "token_count", "ttr", "general_ctc",
                "correct_ethnic_ctc", "correct_ethnic_terms_matched",
                "confusion_terms_matched", "confusion_term_rate"]
    grp = df.groupby(["model", "origin", "language", "ethnic_group"])[agg_cols].mean().reset_index()
    counts = df.groupby(["model", "origin", "language", "ethnic_group"]).size().rename("n").reset_index()
    return grp.merge(counts, on=["model", "origin", "language", "ethnic_group"])


def load_human_eval() -> pd.DataFrame:
    df = pd.read_csv(HUMAN_EVAL, skiprows=[1], encoding="gbk")
    # Drop obviously empty rows
    df = df.dropna(subset=["Image_ID", "Model", "Language"]).copy()
    df["model"] = df["Model"].astype(str)
    df["language"] = df["Language"].astype(str)
    df["image_id"] = df["Image_ID"].astype(str)
    for c in ["Rating_1to5", "Accuracy_1to5", "Completeness_1to5", "Cultural_Depth_1to5"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["image_id", "model", "language", "Ethnic_Group",
               "Rating_1to5", "Accuracy_1to5", "Completeness_1to5", "Cultural_Depth_1to5"]]


def correlate_human_auto(auto_df: pd.DataFrame, human_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    merged = human_df.merge(
        auto_df,
        on=["image_id", "model", "language"],
        how="inner",
    )
    auto_metrics = ["char_len", "token_count", "ttr", "general_ctc",
                    "correct_ethnic_ctc", "confusion_term_rate"]
    human_metrics = ["Rating_1to5", "Accuracy_1to5", "Completeness_1to5", "Cultural_Depth_1to5"]
    corr = {}
    for h in human_metrics:
        corr[h] = {}
        for a in auto_metrics:
            sub = merged[[h, a]].dropna()
            if len(sub) >= 10:
                rho, pval = spearmanr(sub[h], sub[a])
                corr[h][a] = {"rho": float(rho), "p_value": float(pval), "n": len(sub)}
            else:
                corr[h][a] = {"rho": None, "p_value": None, "n": len(sub)}
    return merged, corr


def main() -> None:
    print("Computing per-description metrics ...")
    per_desc = compute_per_description()
    per_desc.to_csv(OUT / "task2_per_description.csv", index=False)
    print(f"  {len(per_desc)} descriptions")

    print("Aggregating by cell ...")
    by_cell = aggregate_by_cell(per_desc)
    by_cell.to_csv(OUT / "task2_by_cell.csv", index=False)

    print("Loading human eval ...")
    human = load_human_eval()
    print(f"  {len(human)} rated rows")

    print("Correlating human vs automated ...")
    merged, corr = correlate_human_auto(per_desc, human)
    merged.to_csv(OUT / "task2_human_vs_auto.csv", index=False)
    with open(OUT / "task2_correlations.json", "w") as f:
        json.dump(corr, f, indent=2)
    print(f"  merged {len(merged)} rows")

    # --- Summary printout ---
    print("\n=== Mean metrics by model × language (pooled over ethnic groups) ===")
    summary = per_desc.groupby(["model", "language"])[
        ["char_len", "token_count", "ttr", "general_ctc", "correct_ethnic_ctc", "confusion_term_rate"]
    ].mean().round(3)
    print(summary)

    print("\n=== Human rating vs automated metric (Spearman ρ) ===")
    for h, d in corr.items():
        line = f"  {h:<20} "
        for a, s in d.items():
            if s["rho"] is None:
                continue
            sig = "*" if s["p_value"] < 0.05 else " "
            line += f"{a}={s['rho']:+.2f}{sig}  "
        print(line)

    print("\nSaved to:", OUT)


if __name__ == "__main__":
    main()
