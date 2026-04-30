"""C4 analysis — does the seeded Miao-coded vocabulary in the original
Task 2 prompt drive the Dong->Miao in-text confusion finding?

Compares the Dong->Miao confusion rate (and Miao vocabulary usage) between:

  * Default Task 2 results (results/raw/task2_*_results.json) —
    prompt seeds "embroidery, batik, brocade" and similar Miao-coded
    accessory/occasion terms.
  * Neutral Task 2 results (results/raw/neutral_prompt/task2_neutral_*_results.json) —
    prompt asks for the same 5 aspects but without seed words.

Output:
  results/metrics/prompt_leak_comparison.csv
  results/metrics/prompt_leak_comparison.md

Decision rule (for revision response):
  * If Dong->Miao confusion rate persists under the neutral prompt
    (delta < 0.10), the original finding is robust to the prompt leak.
  * If the rate drops substantially (delta >= 0.20), the Dong->Miao
    finding is partially a prompt artefact and must be reframed.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

ETHNIC_GROUPS = ["Miao", "Dong", "Yi", "Li", "Tibetan"]

# Tokens imported / mirrored from analyze_text_confusion.py
GROUP_TOKENS = {
    "en": {
        "Miao": [r"\bmiao\b", r"\bhmong\b"],
        "Dong": [r"\bdong\b(?!\s*quixote)"],
        "Yi":   [r"\byi\s+(people|ethnic|minority|nationality|women|costume|dress|attire)"],
        "Li":   [r"\bli\s+(people|ethnic|minority|nationality|women|costume|dress|attire)"],
        "Tibetan": [r"\btibetan\b", r"\btibet\b"],
    },
    "zh": {
        "Miao": ["苗族"], "Dong": ["侗族"], "Yi": ["彝族"],
        "Li": ["黎族"], "Tibetan": ["藏族"],
    },
}

# Miao-coded vocabulary seeded by the DEFAULT prompt — track whether
# models still use these specific words under the NEUTRAL prompt
MIAO_SEED_WORDS = {
    "en": [
        r"\bembroidery\b", r"\bembroider",
        r"\bbatik\b", r"\bbrocade\b",
        r"\bsilver\s+ornament", r"\bheaddress\b",
    ],
    "zh": ["刺绣", "蜡染", "织锦", "银饰", "头饰"],
}


def match_any(text: str, patterns: List[str], lang: str) -> bool:
    if lang == "en":
        return any(re.search(p, text.lower()) for p in patterns)
    return any(p in text for p in patterns)


def count_any(text: str, patterns: List[str], lang: str) -> int:
    if lang == "en":
        total = 0
        for p in patterns:
            total += len(re.findall(p, text.lower()))
        return total
    total = 0
    for p in patterns:
        total += text.count(p)
    return total


def load_runs(pattern: str, variant_name: str) -> pd.DataFrame:
    rows = []
    for fp in sorted(glob.glob(pattern)):
        with open(fp) as f:
            d = json.load(f)
        model = d["model_name"]
        # Normalise claude-3.5-sonnet -> note this was original label
        model_canonical = (
            "claude-haiku-4.5" if model == "claude-3.5-sonnet" else model
        )
        lang = d["language"]
        for img_id, v in d["results"].items():
            text = v.get("description", "") or ""
            rows.append({
                "image_id": img_id,
                "model": model,
                "model_canonical": model_canonical,
                "language": lang,
                "true_group": v.get("ethnic_group", ""),
                "variant": variant_name,
                "description": text,
                "char_len": len(text),
            })
    return pd.DataFrame(rows)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        lang = row["language"]
        text = row["description"]
        entry = {
            "image_id": row["image_id"],
            "model": row["model_canonical"],
            "language": lang,
            "true_group": row["true_group"],
            "variant": row["variant"],
            "char_len": row["char_len"],
        }
        for g in ETHNIC_GROUPS:
            entry[f"mentions_{g}"] = int(match_any(text, GROUP_TOKENS[lang][g], lang))
        entry["miao_seed_count"] = count_any(text, MIAO_SEED_WORDS[lang], lang)
        entry["miao_seed_any"] = int(entry["miao_seed_count"] > 0)
        rows.append(entry)
    return pd.DataFrame(rows)


def compare(ann: pd.DataFrame) -> pd.DataFrame:
    """Per (true_group × language × variant) summary."""
    grp = ann.groupby(["true_group", "language", "variant"])
    out = grp.agg(
        n=("image_id", "count"),
        frac_mentions_miao=("mentions_Miao", "mean"),
        frac_mentions_dong=("mentions_Dong", "mean"),
        frac_mentions_yi=("mentions_Yi", "mean"),
        frac_mentions_li=("mentions_Li", "mean"),
        frac_mentions_tibetan=("mentions_Tibetan", "mean"),
        frac_miao_seed=("miao_seed_any", "mean"),
        miao_seed_count_mean=("miao_seed_count", "mean"),
        mean_char_len=("char_len", "mean"),
    ).reset_index()
    return out


def write_markdown(comp: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# C4: Prompt-leak sanity check",
        "",
        "Compares Task 2 in-text confusion rates between the **default** "
        "prompt (seeds Miao-coded technique words: "
        "*embroidery / batik / brocade / silver ornaments*) and the "
        "**neutral** prompt (same 5 aspects but without seed terms).",
        "",
        "## Key metric: Dong→Miao in-text confusion rate",
        "",
        "Fraction of Dong-costume descriptions that mention the word "
        "\"Miao/苗族\".",
        "",
    ]
    # Pivot for Dong->Miao
    dong = comp[comp["true_group"] == "Dong"].copy()
    if not dong.empty:
        pv = dong.pivot_table(
            index="language", columns="variant",
            values="frac_mentions_miao", aggfunc="mean"
        )
        lines.append("| Language | Default (seeded) | Neutral | Δ |")
        lines.append("|---|---|---|---|")
        for lang in pv.index:
            default = pv.loc[lang].get("default", float("nan"))
            neutral = pv.loc[lang].get("neutral", float("nan"))
            delta = neutral - default if (
                pd.notna(default) and pd.notna(neutral)
            ) else float("nan")
            lines.append(
                f"| {lang} | {default:.3f} | {neutral:.3f} "
                f"| {delta:+.3f} |"
            )
    else:
        lines.append("_(No Dong-costume rows found.)_")

    lines.extend([
        "",
        "## Seed-vocabulary usage rate (any Miao-coded word)",
        "",
        "Fraction of descriptions of EACH true group that use at least "
        "one of \"embroidery / batik / brocade / silver ornaments\" "
        "(or their Chinese equivalents).",
        "",
    ])
    if not comp.empty:
        pv2 = comp.pivot_table(
            index=["true_group", "language"],
            columns="variant",
            values="frac_miao_seed",
            aggfunc="mean",
        )
        lines.append(
            "| Group | Lang | Default | Neutral | Δ |"
        )
        lines.append("|---|---|---|---|---|")
        for (g, l_), row in pv2.iterrows():
            default = row.get("default", float("nan"))
            neutral = row.get("neutral", float("nan"))
            delta = neutral - default if (
                pd.notna(default) and pd.notna(neutral)
            ) else float("nan")
            lines.append(
                f"| {g} | {l_} "
                f"| {default:.3f} | {neutral:.3f} | {delta:+.3f} |"
            )

    lines.extend([
        "",
        "## Decision rule",
        "",
        "- **Robust** (|Δ Dong->Miao| < 0.10): original finding stands; "
        "the prompt leak is acknowledged as a limitation but does not "
        "drive the confusion.",
        "- **Partially artefactual** (Δ <= -0.20): report the neutral-"
        "prompt numbers as primary; reframe §4.4 to acknowledge prompt "
        "contribution.",
        "- **Intermediate**: discuss both variants side-by-side.",
        "",
        "Seed-vocabulary delta additionally shows whether models are "
        "simply echoing the prompt verbatim: a large drop (Δ << 0) in "
        "seed-word usage means models do follow the prompt vocabulary.",
    ])
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--default-glob",
        default=str(RAW / "task2_*_results.json"),
        help="Glob for default-prompt Task 2 JSONs"
    )
    ap.add_argument(
        "--neutral-glob",
        default=str(RAW / "neutral_prompt" / "task2_neutral_*_results.json"),
        help="Glob for neutral-prompt Task 2 JSONs"
    )
    ap.add_argument(
        "--out-csv",
        default=str(METRICS / "prompt_leak_comparison.csv"),
    )
    ap.add_argument(
        "--out-md",
        default=str(METRICS / "prompt_leak_comparison.md"),
    )
    args = ap.parse_args()

    df_default = load_runs(args.default_glob, "default")
    df_neutral = load_runs(args.neutral_glob, "neutral")

    if df_neutral.empty:
        print(
            f"No neutral-prompt results found at {args.neutral_glob}. "
            f"Run the C4 sanity job first.",
            flush=True,
        )
        return

    ann = annotate(pd.concat([df_default, df_neutral], ignore_index=True))
    comp = compare(ann)

    comp.to_csv(args.out_csv, index=False)
    write_markdown(comp, Path(args.out_md))

    print(f"Saved: {args.out_csv}")
    print(f"       {args.out_md}")


if __name__ == "__main__":
    main()
