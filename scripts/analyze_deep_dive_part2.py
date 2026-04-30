"""Deep-dive analyses part 2 (reviewer items 5 and 6).

5. Per-image iconicness regression (n=3000)
   For each image i, compute leave-one-out ensemble agreement across the
   other 9 (model, language) cells. Regress per-image correctness on
   iconicness, pooled and within-group. Moves the salience claim from
   n=5 groups to n=3000 images.

6. Tokenizer-coverage probe
   For each model's tokenizer, count tokens for 5 group-name forms in
   zh and en (e.g., "Miao" / "苗族"). Multi-token names => less direct
   semantic grounding. Tests Decouples-2025 style mechanism.
   - Qwen2-VL-7B: local HF
   - Qwen2.5-VL-7B: local HF
   - LLaMA-3.2-V-11B: local HF
   - GPT-4o-mini: tiktoken o200k_base
   - Claude-Haiku-4.5: skipped (tokenizer not publicly available)

Outputs:
  results/metrics/deep_dive_iconicness.csv        per-image + per-cell regression summary
  results/metrics/deep_dive_iconicness_pergroup.csv
  results/metrics/deep_dive_tokenizer_probe.csv
  results/metrics/deep_dive_part2_summary.md
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"
RAW = ROOT / "results" / "raw"

GROUPS = ["Miao", "Dong", "Yi", "Li", "Tibetan"]

MODELS = [
    "qwen2.5-vl-7b",
    "qwen2-vl-7b",
    "llama-3.2-vision-11b",
    "gpt-4o-mini",
    "claude-haiku-4.5",
]


def load_predictions() -> pd.DataFrame:
    """Return long-form DataFrame with columns:
       image_id, group, model, language, correct (0/1).
    """
    rows = []
    for path in sorted(RAW.glob("task1_*_results.json")):
        fname = path.name
        if "/archive" in str(path) or "/shuffled" in str(path):
            continue
        with open(path) as f:
            d = json.load(f)
        model = d["model_name"]
        lang = d["language"]
        for img_id, rec in d["results"].items():
            # image id is e.g. "Dong_0001"
            group = img_id.rsplit("_", 1)[0]
            if group not in GROUPS:
                continue
            rows.append({
                "image_id": img_id,
                "group": group,
                "model": model,
                "language": lang,
                "correct": int(bool(rec["correct"])),
            })
    df = pd.DataFrame(rows)
    return df


# =============================================================================
# 5. Per-image iconicness regression
# =============================================================================


def iconicness_analysis() -> None:
    df = load_predictions()
    print(f"Loaded {len(df):,} (image, model, language) observations "
          f"across {df['image_id'].nunique()} images")

    # Pivot to 10-column per-image matrix of correctness
    df["cell"] = df["model"] + "|" + df["language"]
    wide = df.pivot_table(index=["image_id", "group"],
                           columns="cell",
                           values="correct",
                           aggfunc="first").reset_index()
    cell_cols = [c for c in wide.columns if c not in ("image_id", "group")]
    assert len(cell_cols) == 10, f"expected 10 cells, got {len(cell_cols)}"

    # Leave-one-out iconicness for each (image, cell) pair:
    # iconicness_loo = mean of the OTHER 9 cells' correctness on this image.
    M = wide[cell_cols].to_numpy(dtype=float)  # (n_images, 10)
    n_img, n_cells = M.shape
    row_sum = M.sum(axis=1, keepdims=True)
    # iconicness_loo[i, c] = (row_sum[i] - M[i,c]) / (n_cells - 1)
    iconicness_loo = (row_sum - M) / (n_cells - 1)

    # Long-form pool for regression
    long_rows = []
    for c_idx, c in enumerate(cell_cols):
        for i, img in enumerate(wide["image_id"]):
            long_rows.append({
                "image_id": img,
                "group": wide["group"].iloc[i],
                "cell": c,
                "iconicness_loo": float(iconicness_loo[i, c_idx]),
                "correct": int(M[i, c_idx]),
            })
    long_df = pd.DataFrame(long_rows)

    # Pooled Pearson r
    r_pool, p_pool = pearsonr(long_df["iconicness_loo"], long_df["correct"])
    print(f"POOLED (n={len(long_df):,}): "
          f"Pearson r(iconicness_loo, correct) = {r_pool:+.3f}, p = {p_pool:.2e}")

    # Per-group (within-group variation in iconicness)
    per_group_rows = []
    for g in GROUPS:
        sub = long_df[long_df["group"] == g]
        if sub["iconicness_loo"].std() < 1e-6:
            r_g, p_g = float("nan"), float("nan")
        else:
            r_g, p_g = pearsonr(sub["iconicness_loo"], sub["correct"])
        # Also within-group spread: how many images have LOO agreement
        # 0, 0-0.33, 0.33-0.66, 0.66-1.0
        img_loo = long_df[long_df["group"] == g].drop_duplicates("image_id")
        # within-group per-image mean iconicness (across the 10 cells)
        per_img_mean = (
            long_df[long_df["group"] == g]
            .groupby("image_id")["iconicness_loo"]
            .mean()
        )
        per_group_rows.append({
            "group": g,
            "n_obs": len(sub),
            "n_images": sub["image_id"].nunique(),
            "mean_acc": round(sub["correct"].mean(), 3),
            "mean_iconicness": round(sub["iconicness_loo"].mean(), 3),
            "iconicness_std_within_group": round(per_img_mean.std(), 3),
            "pearson_r_within_group": round(r_g, 3),
            "p_within_group": round(p_g, 4),
            "frac_images_iconicness_above_half": round(
                (per_img_mean > 0.5).mean(), 3),
        })
    per_group = pd.DataFrame(per_group_rows)
    per_group.to_csv(METRICS / "deep_dive_iconicness_pergroup.csv", index=False)
    print("\nWithin-group iconicness -> correctness:")
    print(per_group.to_string(index=False))

    # Per-cell (per model x lang): does iconicness from OTHER cells predict
    # THIS cell's accuracy? If yes, models agree on which images are "easy."
    per_cell_rows = []
    for c in cell_cols:
        sub = long_df[long_df["cell"] == c]
        r_c, p_c = pearsonr(sub["iconicness_loo"], sub["correct"])
        per_cell_rows.append({
            "cell": c,
            "pearson_r": round(r_c, 3),
            "p_value": round(p_c, 4),
            "n_images": len(sub),
        })
    per_cell = pd.DataFrame(per_cell_rows).sort_values("pearson_r", ascending=False)
    print("\nPer-cell: does other-model ensemble predict my correctness?")
    print(per_cell.to_string(index=False))

    # Variance decomposition: R^2 of group-only vs group + iconicness model
    # logit(P(correct)) ~ group + iconicness
    y = long_df["correct"].to_numpy(dtype=float)
    # Design: group fixed effects (4 dummies) + iconicness
    X_group = pd.get_dummies(long_df["group"], drop_first=True).to_numpy(dtype=float)
    X_full = np.column_stack([
        np.ones(len(long_df)),
        X_group,
        long_df["iconicness_loo"].to_numpy(dtype=float),
    ])
    X_group_only = np.column_stack([np.ones(len(long_df)), X_group])

    beta_full, *_ = np.linalg.lstsq(X_full, y, rcond=None)
    beta_g, *_ = np.linalg.lstsq(X_group_only, y, rcond=None)
    pred_full = X_full @ beta_full
    pred_g = X_group_only @ beta_g
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2_full = 1.0 - ((y - pred_full) ** 2).sum() / ss_tot
    r2_group = 1.0 - ((y - pred_g) ** 2).sum() / ss_tot
    r2_marginal_iconicness = r2_full - r2_group
    iconicness_beta = beta_full[-1]

    summary = {
        "pooled_pearson_r": r_pool,
        "pooled_p_value": p_pool,
        "n_obs_pooled": len(long_df),
        "r2_group_only": round(r2_group, 4),
        "r2_group_plus_iconicness": round(r2_full, 4),
        "r2_marginal_from_iconicness": round(r2_marginal_iconicness, 4),
        "iconicness_coef_OLS": round(iconicness_beta, 4),
    }
    with open(METRICS / "deep_dive_iconicness.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nVariance decomposition:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save the long-form pool (sampled) for downstream plots if wanted
    long_df.to_csv(METRICS / "deep_dive_iconicness.csv", index=False)

    return per_group, per_cell, summary


# =============================================================================
# 6. Tokenizer-coverage probe
# =============================================================================


ZH_NAMES = {
    "Miao": "苗族",
    "Dong": "侗族",
    "Yi": "彝族",
    "Li": "黎族",
    "Tibetan": "藏族",
}
EN_NAMES = {
    "Miao": "Miao",
    "Dong": "Dong",
    "Yi": "Yi",
    "Li": "Li",
    "Tibetan": "Tibetan",
}


def tokenize_hf(model_id: str, texts: list) -> list:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    out = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        out.append(len(ids))
    return out


def tokenize_tiktoken(encoding_name: str, texts: list) -> list:
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    return [len(enc.encode(t)) for t in texts]


def tokenizer_probe() -> None:
    configs = [
        {
            "model_alias": "qwen2-vl-7b",
            "tokenizer_src": "Qwen/Qwen2-VL-7B-Instruct",
            "backend": "hf",
        },
        {
            "model_alias": "qwen2.5-vl-7b",
            "tokenizer_src": "Qwen/Qwen2.5-VL-7B-Instruct",
            "backend": "hf",
        },
        {
            "model_alias": "llama-3.2-vision-11b",
            "tokenizer_src": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "backend": "hf",
        },
        {
            "model_alias": "gpt-4o-mini",
            "tokenizer_src": "o200k_base",
            "backend": "tiktoken",
        },
    ]

    groups = GROUPS
    zh_texts = [ZH_NAMES[g] for g in groups]
    en_texts = [EN_NAMES[g] for g in groups]

    rows = []
    for cfg in configs:
        try:
            if cfg["backend"] == "hf":
                zh_counts = tokenize_hf(cfg["tokenizer_src"], zh_texts)
                en_counts = tokenize_hf(cfg["tokenizer_src"], en_texts)
            else:
                zh_counts = tokenize_tiktoken(cfg["tokenizer_src"], zh_texts)
                en_counts = tokenize_tiktoken(cfg["tokenizer_src"], en_texts)
        except Exception as e:
            print(f"  [{cfg['model_alias']}] FAILED: {e}")
            continue

        for i, g in enumerate(groups):
            rows.append({
                "model": cfg["model_alias"],
                "group": g,
                "name_zh": zh_texts[i],
                "tokens_zh": zh_counts[i],
                "name_en": en_texts[i],
                "tokens_en": en_counts[i],
            })
        print(f"  [{cfg['model_alias']}] zh: {dict(zip(groups, zh_counts))} "
              f"en: {dict(zip(groups, en_counts))}")

    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "deep_dive_tokenizer_probe.csv", index=False)

    # Correlation of tokens_zh with per-group accuracy (from salience csv)
    sal = pd.read_csv(METRICS / "salience_prevalence.csv").set_index("ethnic_group")
    acc = sal.loc[groups, "mean_accuracy"].to_numpy()

    print("\nCorrelation of token count with per-group accuracy:")
    corr_rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model].set_index("group").loc[groups]
        zh_tokens = sub["tokens_zh"].to_numpy(dtype=float)
        en_tokens = sub["tokens_en"].to_numpy(dtype=float)
        if zh_tokens.std() > 0:
            rho_zh = spearmanr(zh_tokens, acc).statistic
        else:
            rho_zh = float("nan")
        if en_tokens.std() > 0:
            rho_en = spearmanr(en_tokens, acc).statistic
        else:
            rho_en = float("nan")
        corr_rows.append({
            "model": model,
            "rho_zh_tokens_vs_acc": round(rho_zh, 3),
            "rho_en_tokens_vs_acc": round(rho_en, 3),
            "zh_tokens_vary": bool(zh_tokens.std() > 0),
            "en_tokens_vary": bool(en_tokens.std() > 0),
        })
        print(f"  {model}: rho(tokens_zh, acc)={rho_zh:+.3f}, "
              f"rho(tokens_en, acc)={rho_en:+.3f}")
    pd.DataFrame(corr_rows).to_csv(
        METRICS / "deep_dive_tokenizer_correlation.csv", index=False)
    return df


# =============================================================================
# Main — write summary
# =============================================================================


def main() -> None:
    print("\n[5] Per-image iconicness regression (n=3000)\n")
    per_group, per_cell, summary = iconicness_analysis()

    print("\n[6] Tokenizer-coverage probe\n")
    tok_df = tokenizer_probe()

    # Markdown summary
    def md_table(df):
        cols = list(df.columns)
        rows = ["| " + " | ".join(str(c) for c in cols) + " |",
                "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in df.iterrows():
            rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(rows)

    lines = []
    lines.append("# Deep-dive analyses part 2 (reviewer items 5-6)\n")
    lines.append("Generated by `scripts/analyze_deep_dive_part2.py`.\n")

    lines.append("## 5. Per-image iconicness regression\n")
    lines.append(f"- Pooled Pearson r across n={summary['n_obs_pooled']:,} "
                 f"(image, model, language) observations: **"
                 f"{summary['pooled_pearson_r']:+.3f}** "
                 f"(p = {summary['pooled_p_value']:.2e}).\n")
    lines.append(f"- R^2 with group fixed effects only: "
                 f"{summary['r2_group_only']:.3f}\n")
    lines.append(f"- R^2 with group + iconicness: "
                 f"{summary['r2_group_plus_iconicness']:.3f}\n")
    lines.append(f"- Marginal R^2 added by within-group iconicness: **"
                 f"{summary['r2_marginal_from_iconicness']:+.3f}**\n")
    lines.append("\nPer-group within-group correlation:\n")
    lines.append(md_table(per_group))
    lines.append("\n\nPer-(model, language) cell: other-models' agreement -> "
                 "my correctness:\n")
    lines.append(md_table(per_cell))
    lines.append("\n")

    lines.append("## 6. Tokenizer-coverage probe\n")
    lines.append(md_table(tok_df))
    lines.append("\n")

    (METRICS / "deep_dive_part2_summary.md").write_text("\n".join(lines))
    print(f"\nWrote: {METRICS / 'deep_dive_part2_summary.md'}")


if __name__ == "__main__":
    main()
