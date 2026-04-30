"""Seed cultural-vocabulary candidates from existing descriptions via distinctiveness.

For each (ethnic_group, language), find n-grams that are:
  - Frequent in descriptions of THAT group
  - Rare in descriptions of OTHER groups
Using log-odds-ratio with informative Dirichlet prior (Monroe et al. 2008),
which is robust to frequency differences.

Outputs:
  results/metrics/vocab_candidates_{lang}.csv  — ranked candidates per group
For expert curation before updating configs/cultural_vocabulary.yaml.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PER_DESC = ROOT / "results" / "metrics" / "task2_per_description.csv"
RAW_TEXTS_OUT = ROOT / "results" / "metrics"
EXISTING_VOCAB = ROOT / "configs" / "cultural_vocabulary.yaml"

ETHNIC_GROUPS = ["Miao", "Dong", "Yi", "Li", "Tibetan"]

# Minimal stopwords
EN_STOP = set("""
a an the and or but if then of to in on at for from by with without into onto upon over under between
is are was were be been being have has had do does did this that these those it its their there here
which who whom whose what when where why how as so such than too very can could should would may might
also not no nor only own same other another more most some any each every all both few many several
image photo picture depicted shown shows figure style design feature featured features traditional
ethnic minority costume costumes garment garments clothing clothes attire wear worn dress dressed
person woman man figure figures individuals people young old
color colors colored bright dark light rich vibrant deep
pattern patterns motif motifs detail details element elements piece pieces layer layers section sections
overall general appear appears appearance looks look seen seems
chinese china minority minorities group groups cultural culture
""".split())

# Chinese stopwords (chars/bigrams that are functional, not cultural)
ZH_STOP_SET = set("的了是在和与及或等及其这那其有为以上下一二三四五六七八九十个种类型样式件条")


def tokenize_en(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text.lower())


def char_ngrams_zh(text: str, n_range=(2, 4)) -> list[str]:
    # Keep Chinese chars only
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    s = "".join(chars)
    ngrams = []
    for n in range(n_range[0], n_range[1] + 1):
        ngrams.extend(s[i:i + n] for i in range(len(s) - n + 1))
    return ngrams


def word_ngrams_en(tokens: list[str], n_range=(1, 3)) -> list[str]:
    ngrams = []
    for n in range(n_range[0], n_range[1] + 1):
        ngrams.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return ngrams


def contains_only_stop(ngram: str, lang: str) -> bool:
    if lang == "en":
        return all(w in EN_STOP for w in ngram.split())
    return all(c in ZH_STOP_SET for c in ngram)


def load_existing_terms(lang: str) -> set[str]:
    with open(EXISTING_VOCAB) as f:
        v = yaml.safe_load(f)
    terms = set(v.get(lang, []))
    for g_info in v.get("ethnic_specific", {}).values():
        terms.update(g_info.get(lang, []))
    if lang == "en":
        terms = {t.lower() for t in terms}
    return terms


def log_odds_with_prior(counts_target: Counter, counts_other: Counter,
                       n_target: int, n_other: int,
                       prior_alpha: float = 0.01) -> dict[str, tuple[float, float]]:
    """Return {ngram: (log_odds_ratio, z_score)} (Monroe et al. 2008)."""
    vocab = set(counts_target) | set(counts_other)
    # Build prior from background counts
    bg = {w: counts_target[w] + counts_other[w] for w in vocab}
    a0 = sum(bg.values()) * prior_alpha
    alpha_w = {w: bg[w] * prior_alpha + 0.01 for w in vocab}
    out = {}
    for w in vocab:
        y_t = counts_target[w]
        y_o = counts_other[w]
        # Smoothed log-odds for each corpus
        log_odds_t = np.log((y_t + alpha_w[w]) / (n_target + a0 - y_t - alpha_w[w]))
        log_odds_o = np.log((y_o + alpha_w[w]) / (n_other + a0 - y_o - alpha_w[w]))
        delta = log_odds_t - log_odds_o
        # Variance
        var = 1.0 / (y_t + alpha_w[w]) + 1.0 / (y_o + alpha_w[w])
        z = delta / np.sqrt(var)
        out[w] = (delta, z)
    return out


def candidates_for_language(df: pd.DataFrame, lang: str, top_k: int = 50) -> pd.DataFrame:
    """Return top candidate n-grams per ethnic group for this language."""
    sub = df[df["language"] == lang]
    existing = load_existing_terms(lang)

    # Build per-group n-gram document-frequency counts (document = description)
    # Using document frequency (num descriptions containing ngram) is more stable than raw counts.
    group_docs = {g: [] for g in ETHNIC_GROUPS}
    for _, row in sub.iterrows():
        g = row["ethnic_group"]
        if g not in group_docs:
            continue
        text = row.get("matched_ethnic_terms", "")  # placeholder, we need the actual description
        # We need to reload actual text — per_description doesn't store it.

    return pd.DataFrame()  # stub


def load_descriptions(lang: str) -> pd.DataFrame:
    """Reload descriptions from raw files (per_description.csv doesn't store full text)."""
    import glob, json
    MODEL_CANONICAL = {"claude-3.5-sonnet": "claude-haiku-4.5"}
    rows = []
    for fp in sorted(glob.glob(str(ROOT / "results" / "raw" / f"task2_*_{lang}_*_results.json"))):
        with open(fp) as f:
            d = json.load(f)
        model = MODEL_CANONICAL.get(d["model_name"], d["model_name"])
        for img_id, v in d["results"].items():
            rows.append({
                "image_id": img_id,
                "model": model,
                "language": lang,
                "ethnic_group": v.get("ethnic_group", ""),
                "description": v.get("description", "") or "",
            })
    return pd.DataFrame(rows)


def extract_candidates(lang: str, top_k: int = 50, min_df_frac: float = 0.05) -> pd.DataFrame:
    print(f"Loading {lang} descriptions ...")
    df = load_descriptions(lang)
    existing = load_existing_terms(lang)
    print(f"  {len(df)} descriptions; {len(existing)} existing terms")

    # Build document-frequency counts per group
    group_doc_freq = {g: Counter() for g in ETHNIC_GROUPS}
    group_n_docs = {g: 0 for g in ETHNIC_GROUPS}

    for _, row in df.iterrows():
        g = row["ethnic_group"]
        if g not in group_doc_freq:
            continue
        text = row["description"]
        if lang == "en":
            tokens = tokenize_en(text)
            ngrams = set(word_ngrams_en(tokens, (1, 3)))
        else:
            ngrams = set(char_ngrams_zh(text, (2, 4)))
        group_doc_freq[g].update(ngrams)
        group_n_docs[g] += 1

    all_candidates = []
    for target in ETHNIC_GROUPS:
        # Target vs all-other
        target_counts = group_doc_freq[target]
        n_target = group_n_docs[target]
        other_counts = Counter()
        n_other = 0
        for g in ETHNIC_GROUPS:
            if g == target:
                continue
            other_counts.update(group_doc_freq[g])
            n_other += group_n_docs[g]

        # Filter: must appear in >= min_df_frac of target docs
        min_df = max(3, int(min_df_frac * n_target))
        vocab_filtered = {w for w, c in target_counts.items()
                          if c >= min_df and not contains_only_stop(w, lang)}

        # Drop items already in vocabulary
        if lang == "en":
            vocab_filtered = {w for w in vocab_filtered if w not in existing}
        else:
            vocab_filtered = {w for w in vocab_filtered if w not in existing}

        # Drop ngrams that are substrings contained in a longer ngram in the set (keep longest)
        # Only for zh (en trigrams are meaningful on their own)
        if lang == "zh":
            sorted_ngrams = sorted(vocab_filtered, key=len, reverse=True)
            kept = []
            for ng in sorted_ngrams:
                # Keep if not a substring of an already-kept ngram with similar count
                subsumed = False
                for k in kept:
                    if ng in k and target_counts[ng] <= target_counts[k] * 1.3:
                        subsumed = True
                        break
                if not subsumed:
                    kept.append(ng)
            vocab_filtered = set(kept)

        target_sub = Counter({w: target_counts[w] for w in vocab_filtered})
        other_sub = Counter({w: other_counts[w] for w in vocab_filtered})
        scores = log_odds_with_prior(target_sub, other_sub, n_target, n_other)

        ranked = sorted(scores.items(), key=lambda x: -x[1][1])[:top_k]
        for ngram, (delta, z) in ranked:
            all_candidates.append({
                "language": lang,
                "ethnic_group": target,
                "ngram": ngram,
                "target_df": target_counts[ngram],
                "other_df": other_counts[ngram],
                "target_df_frac": target_counts[ngram] / n_target,
                "other_df_frac": other_counts[ngram] / n_other if n_other else 0,
                "log_odds_delta": delta,
                "z_score": z,
            })
    return pd.DataFrame(all_candidates)


def main() -> None:
    for lang in ["en", "zh"]:
        cand = extract_candidates(lang, top_k=40, min_df_frac=0.05)
        out_path = ROOT / "results" / "metrics" / f"vocab_candidates_{lang}.csv"
        cand.to_csv(out_path, index=False)
        print(f"Saved {len(cand)} candidates → {out_path.name}")
        print(f"\n=== Top 10 per group [{lang}] ===")
        for g in ETHNIC_GROUPS:
            sub = cand[cand["ethnic_group"] == g].head(10)
            print(f"\n{g}:")
            for _, r in sub.iterrows():
                print(f"  {r['ngram']:<40} z={r['z_score']:+5.1f}  "
                      f"target={r['target_df_frac']:.2f}  other={r['other_df_frac']:.3f}")


if __name__ == "__main__":
    main()
