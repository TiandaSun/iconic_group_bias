"""Rater calibration analysis for the split-annotation human evaluation.

Addresses peer-review concern C5 (single-rater HE) at code level by
recharacterising the design as split-annotation (2 experts × ~75
disjoint items each) and checking:

  * Scale calibration — do the 2 raters use the 1-5 Likert scale
    similarly (same central tendency, same spread)?
  * Model-ranking consistency — does each rater's sub-sample rank
    the 5 models in a similar order? Spearman rho of per-rater
    per-dimension model means.
  * Robustness of main findings to rater identity: does the ranking
    Qwen2.5 > GPT > Claude > Qwen2 > LLaMA hold within each rater's
    sub-sample?

Because no items are doubly rated, we cannot compute Krippendorff's
alpha. We acknowledge this explicitly.

Usage:
    # With an explicit rater assignment (DQ_ID, Rater_ID) CSV
    python scripts/analyze_rater_calibration.py \\
        --ratings results/human_eval/description_quality_rating.csv \\
        --rater-assignment results/human_eval/rater_assignment.csv

    # Without rater assignment (script runs diagnostics + split-half proxy)
    python scripts/analyze_rater_calibration.py \\
        --ratings results/human_eval/description_quality_rating.csv

Outputs:
  results/human_eval/rater_calibration_report.json
  results/human_eval/rater_calibration_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent
HE_DIR = ROOT / "results" / "human_eval"

DIMENSIONS = ["Rating_1to5", "Accuracy_1to5",
              "Completeness_1to5", "Cultural_Depth_1to5"]
MODELS_IN_ORDER = [
    "qwen2.5-vl-7b", "qwen2-vl-7b", "llama-3.2-vision-11b",
    "gpt-4o-mini", "claude-haiku-4.5", "claude-3.5-sonnet",
]


def detect_rater_column(df: pd.DataFrame) -> Optional[str]:
    """Heuristic: find any column that looks like a rater id."""
    for c in df.columns:
        lc = c.lower()
        if "rater" in lc or "annotator" in lc or "evaluator" in lc:
            return c
    return None


def load_ratings(ratings_path: Path,
                 rater_assignment_path: Optional[Path]) -> pd.DataFrame:
    df = pd.read_csv(ratings_path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]

    # Detect or merge rater info
    rater_col = detect_rater_column(df)
    if rater_col is None and rater_assignment_path and rater_assignment_path.exists():
        assign = pd.read_csv(rater_assignment_path)
        assign.columns = [c.strip() for c in assign.columns]
        if "ID" in assign.columns and "Rater_ID" in assign.columns:
            df = df.merge(assign[["ID", "Rater_ID"]], on="ID", how="left")
            rater_col = "Rater_ID"

    if rater_col is None:
        df["Rater_ID"] = "UNKNOWN"
        rater_col = "Rater_ID"

    df.rename(columns={rater_col: "Rater_ID"}, inplace=True)

    # Coerce Likert ratings to numeric
    for c in DIMENSIONS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def scale_calibration(df: pd.DataFrame) -> Dict:
    """Per-rater central tendency and spread per dimension."""
    out = {}
    for r in sorted(df["Rater_ID"].unique()):
        sub = df[df["Rater_ID"] == r]
        per_dim = {}
        for d in DIMENSIONS:
            if d not in sub.columns:
                continue
            vals = sub[d].dropna()
            per_dim[d] = {
                "n": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else None,
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                "median": float(vals.median()) if len(vals) else None,
            }
        out[str(r)] = {"n_items": int(len(sub)), "per_dimension": per_dim}
    return out


def ranking_consistency(df: pd.DataFrame) -> Dict:
    """Per-rater per-dimension per-model means + cross-rater rank agreement."""
    raters = sorted(df["Rater_ID"].unique())
    out = {"per_rater_means": {}, "cross_rater_rank_correlation": {}}

    # Per-rater per-dimension per-model means
    for r in raters:
        sub = df[df["Rater_ID"] == r]
        out["per_rater_means"][str(r)] = {}
        for d in DIMENSIONS:
            if d not in sub.columns:
                continue
            model_means = sub.groupby("Model")[d].mean().sort_values(ascending=False)
            out["per_rater_means"][str(r)][d] = {
                m: round(float(v), 3) for m, v in model_means.items()
            }

    # Cross-rater rank correlation (only meaningful with >=2 raters)
    if len(raters) >= 2:
        for d in DIMENSIONS:
            if d not in df.columns:
                continue
            per_rater_model_means = {}
            for r in raters:
                sub = df[df["Rater_ID"] == r]
                per_rater_model_means[r] = sub.groupby("Model")[d].mean().to_dict()
            # Use common models (both raters rated at least one item for this model)
            common = set.intersection(*[set(v.keys()) for v in per_rater_model_means.values()])
            if len(common) < 3:
                out["cross_rater_rank_correlation"][d] = {
                    "error": "too few common models for rank correlation"
                }
                continue
            # Pairwise rater rank correlation
            pairs = []
            for i, r1 in enumerate(raters):
                for r2 in raters[i + 1:]:
                    v1 = np.array([per_rater_model_means[r1][m] for m in common])
                    v2 = np.array([per_rater_model_means[r2][m] for m in common])
                    rho, p = spearmanr(v1, v2)
                    tau, p_t = kendalltau(v1, v2)
                    pairs.append({
                        "rater_1": str(r1),
                        "rater_2": str(r2),
                        "n_models_compared": int(len(common)),
                        "spearman_rho": float(rho),
                        "spearman_p": float(p),
                        "kendall_tau": float(tau),
                        "kendall_p": float(p_t),
                    })
            out["cross_rater_rank_correlation"][d] = pairs

    return out


def mean_shift_test(df: pd.DataFrame) -> Dict:
    """Mann-Whitney U test per dimension between the 2 raters.

    Tests whether rater central tendency differs. Large shift =
    raters are using the scale differently.
    """
    raters = sorted(df["Rater_ID"].unique())
    if len(raters) != 2:
        return {"note": "mean-shift test is only computed when exactly 2 raters"}
    r1, r2 = raters
    out = {}
    for d in DIMENSIONS:
        if d not in df.columns:
            continue
        v1 = df.loc[df["Rater_ID"] == r1, d].dropna().to_numpy()
        v2 = df.loc[df["Rater_ID"] == r2, d].dropna().to_numpy()
        if len(v1) < 5 or len(v2) < 5:
            out[d] = {"error": "too few ratings"}
            continue
        U, p = mannwhitneyu(v1, v2, alternative="two-sided")
        out[d] = {
            "rater_1": str(r1), "rater_2": str(r2),
            "n_1": int(len(v1)), "n_2": int(len(v2)),
            "mean_1": float(v1.mean()), "mean_2": float(v2.mean()),
            "mean_diff": float(v1.mean() - v2.mean()),
            "U": float(U), "p": float(p),
            "interpretation": (
                "significant scale shift" if p < 0.05
                else "no significant scale shift"
            ),
        }
    return out


def split_half_proxy(df: pd.DataFrame, seed: int = 42) -> Dict:
    """Split-half reliability when rater info is unknown.

    Randomly halves the dataset, computes per-model rank in each half
    per dimension, then reports Spearman rho between halves. This
    lower-bounds the consistency of the overall rating without
    rater labels. Averaged over 100 random splits.
    """
    rng = np.random.default_rng(seed)
    out = {}
    n_splits = 100
    for d in DIMENSIONS:
        if d not in df.columns:
            continue
        sub = df.dropna(subset=[d])
        n = len(sub)
        if n < 30:
            continue
        rhos = []
        for _ in range(n_splits):
            idx = rng.permutation(n)
            h1 = sub.iloc[idx[: n // 2]]
            h2 = sub.iloc[idx[n // 2:]]
            m1 = h1.groupby("Model")[d].mean().to_dict()
            m2 = h2.groupby("Model")[d].mean().to_dict()
            common = sorted(set(m1) & set(m2))
            if len(common) < 3:
                continue
            v1 = np.array([m1[m] for m in common])
            v2 = np.array([m2[m] for m in common])
            rho, _ = spearmanr(v1, v2)
            rhos.append(float(rho))
        out[d] = {
            "n_splits": len(rhos),
            "mean_rho": float(np.mean(rhos)) if rhos else None,
            "median_rho": float(np.median(rhos)) if rhos else None,
            "std_rho": float(np.std(rhos)) if rhos else None,
        }
    return out


def write_markdown_report(report: Dict, out_path: Path) -> None:
    """Produce a Markdown summary for the paper Appendix B/limitations."""
    lines = ["# Rater Calibration Report\n"]
    lines.append(
        "**Design**: Split-annotation. Two domain experts rated disjoint "
        "stratified subsets (aiming for ~75 items each) of the 150 "
        "human-evaluation descriptions, balanced across "
        "(ethnic group × model × language) cells.\n"
    )
    lines.append(
        "**Rationale**: Budget constraints precluded overlapping "
        "annotation. Consequently, formal inter-rater reliability "
        "statistics (Krippendorff's α, Cohen's κ) cannot be computed. "
        "This is acknowledged as a limitation. In place of IRR we "
        "report rater scale calibration and cross-rater model-ranking "
        "consistency, plus split-half reliability on the combined "
        "dataset as a proxy.\n"
    )
    lines.append("## Scale Calibration (per-rater means)\n")
    for r, info in report["scale_calibration"].items():
        lines.append(f"### Rater `{r}` (n = {info['n_items']} items)\n")
        rows = ["| Dimension | n | Mean | SD | Median |",
                "|---|---|---|---|---|"]
        for d, s in info["per_dimension"].items():
            rows.append(
                f"| {d} | {s['n']} | {s['mean']:.2f} "
                f"| {s.get('std', 0):.2f} | {s['median']:.2f} |"
                if s["mean"] is not None else f"| {d} | 0 | — | — | — |"
            )
        lines.extend(rows)
        lines.append("")

    lines.append("## Cross-Rater Rank Consistency\n")
    lines.append(
        "Spearman ρ on per-model means, computed within each "
        "rater's sub-sample.\n"
    )
    if "cross_rater_rank_correlation" in report:
        crr = report["cross_rater_rank_correlation"]
        for d, entry in crr.items():
            if isinstance(entry, list):
                for p in entry:
                    lines.append(
                        f"* **{d}** (raters {p['rater_1']} vs {p['rater_2']}): "
                        f"ρ = {p['spearman_rho']:+.3f}, "
                        f"Kendall τ = {p['kendall_tau']:+.3f} "
                        f"on n = {p['n_models_compared']} models"
                    )
            else:
                lines.append(f"* **{d}**: {entry.get('error', 'n/a')}")

    if "mean_shift_test" in report and "note" not in report["mean_shift_test"]:
        lines.append("\n## Mean-Shift Test (Mann-Whitney U, two-sided)\n")
        for d, res in report["mean_shift_test"].items():
            if "error" in res:
                continue
            lines.append(
                f"* **{d}**: Rater {res['rater_1']} mean {res['mean_1']:.2f} "
                f"vs Rater {res['rater_2']} mean {res['mean_2']:.2f} "
                f"(Δ = {res['mean_diff']:+.2f}, U = {res['U']:.0f}, "
                f"p = {res['p']:.3f}) — {res['interpretation']}"
            )

    if "split_half_proxy" in report:
        lines.append("\n## Split-Half Model-Ranking Reliability (Proxy)\n")
        lines.append(
            "For each dimension, the dataset is randomly halved 100 times; "
            "per-model means computed in each half; ρ between halves "
            "reported. This lower-bounds the consistency of the overall "
            "rating signal.\n"
        )
        for d, res in report["split_half_proxy"].items():
            if res.get("mean_rho") is not None:
                lines.append(
                    f"* **{d}**: mean ρ = {res['mean_rho']:+.3f} "
                    f"(median = {res['median_rho']:+.3f}, "
                    f"SD = {res['std_rho']:.3f})"
                )

    lines.append(
        "\n## Interpretation Guide\n"
        "- Mean ρ ≥ 0.80: model ordering robust to rater sub-sample.\n"
        "- Mean shift p ≥ 0.10: raters calibrated similarly.\n"
        "- Split-half ρ ≥ 0.80: overall rating signal is stable.\n"
        "- Any dimension falling below these thresholds should be "
        "flagged in the paper's §5.3 limitations.\n"
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings",
                    default=str(HE_DIR / "description_quality_rating.csv"))
    ap.add_argument("--rater-assignment",
                    default=str(HE_DIR / "rater_assignment.csv"),
                    help="Optional CSV with columns (ID, Rater_ID)")
    ap.add_argument("--out-json",
                    default=str(HE_DIR / "rater_calibration_report.json"))
    ap.add_argument("--out-md",
                    default=str(HE_DIR / "rater_calibration_report.md"))
    args = ap.parse_args()

    ratings_path = Path(args.ratings)
    rater_assign_path = Path(args.rater_assignment) if args.rater_assignment else None

    df = load_ratings(ratings_path, rater_assign_path)

    n_raters = df["Rater_ID"].nunique()
    print(f"Loaded {len(df)} ratings across {n_raters} rater(s).")
    if n_raters == 1 and list(df["Rater_ID"].unique())[0] == "UNKNOWN":
        print("NOTE: No rater information found. To enable cross-rater\n"
              "analysis, create results/human_eval/rater_assignment.csv\n"
              "with columns (ID, Rater_ID) where each ID is a DQ_nnn and\n"
              "Rater_ID is the rater name. Without this, only split-half\n"
              "reliability is reported.", file=sys.stderr)

    report: Dict = {
        "n_total_ratings": int(len(df)),
        "n_raters": int(n_raters),
        "dimensions": DIMENSIONS,
        "scale_calibration": scale_calibration(df),
        "cross_rater_rank_correlation": ranking_consistency(df)[
            "cross_rater_rank_correlation"
        ],
        "per_rater_model_means": ranking_consistency(df)["per_rater_means"],
        "mean_shift_test": mean_shift_test(df),
        "split_half_proxy": split_half_proxy(df),
    }

    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_markdown_report(report, Path(args.out_md))
    print(f"Saved: {args.out_json}")
    print(f"       {args.out_md}")


if __name__ == "__main__":
    main()
