"""Robust statistical re-analysis of the salience-accuracy relationship.

Addresses peer-review concerns C1 (n=5 statistical fragility) at code level:

  * Exact Spearman permutation p-values (asymptotic approximation is
    inappropriate at n=5) — reports both one- and two-tailed.
  * Holm-Bonferroni correction across the 4 Wikipedia proxies tested.
  * Log-linear frequency->accuracy model (Udandarao 2024 style):
    logit(acc) = beta * log(prevalence) + alpha; reports beta, R^2,
    leave-one-out R^2.
  * Jackknife / leave-one-out robustness of Spearman rho (shows how
    sensitive the rho=0.90 result is to a single group).

Reads the per-group accuracy from results/metrics/task1_per_cell.csv and
the Wikipedia proxies from results/metrics/salience_prevalence.csv
(already written by scripts/salience_prevalence_test.py). Does NOT
re-fetch Wikipedia — that is the expensive part of the parent script.

Outputs:
  results/metrics/salience_prevalence_robust.json
  results/metrics/salience_prevalence_robust.csv  (per-proxy summary)
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"

PROXY_COLS = ["bytes_en", "bytes_zh", "images_en", "images_zh",
              "views_60d_en", "views_60d_zh"]


# =============================================================================
# Exact Spearman p-value via permutation enumeration
# =============================================================================


def spearman_exact(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Exact two-tailed and one-tailed Spearman p-values via enumeration.

    For n=5 there are 5! = 120 permutations; exact computation is trivial.
    At large n (say n>10) we fall back to the asymptotic scipy value.

    Returns:
        (rho, p_one_tailed_greater, p_two_tailed)
        where p_one_tailed_greater = Pr(rho* >= rho_observed | null).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    rho_obs, _ = spearmanr(x, y)

    if n > 8:  # exact enumeration infeasible beyond ~8
        _, p_two = spearmanr(x, y)
        # scipy's p is two-sided; one-sided = p/2 if same sign as rho
        return float(rho_obs), float(p_two / 2.0), float(p_two)

    # Enumerate permutations of y, compute rho vs x
    rhos = []
    for perm in permutations(range(n)):
        y_perm = y[list(perm)]
        r, _ = spearmanr(x, y_perm)
        rhos.append(r)
    rhos = np.asarray(rhos)
    total = len(rhos)
    p_ge = float(np.sum(rhos >= rho_obs) / total)
    p_two = float(np.sum(np.abs(rhos) >= abs(rho_obs)) / total)
    return float(rho_obs), p_ge, p_two


# =============================================================================
# Holm-Bonferroni correction
# =============================================================================


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni step-down adjustment on a dict of named p-values.

    Returns a dict with the same keys and adjusted p-values.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    prev_adj = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev_adj, (m - i) * p))
        adjusted[k] = adj
        prev_adj = adj
    return adjusted


# =============================================================================
# Log-linear frequency->accuracy model (Udandarao 2024 style)
# =============================================================================


def loglinear_fit(prev: np.ndarray, acc: np.ndarray,
                  eps: float = 1e-3) -> Dict[str, float]:
    """Fit logit(acc) = alpha + beta * log(prev).

    Reports beta, intercept, full-sample R^2, leave-one-out CV R^2,
    and Pearson rho of log(prev) vs logit(acc).
    """
    p = np.asarray(prev, dtype=float)
    a = np.asarray(acc, dtype=float)
    # Guard against 0/1 in accuracy
    a_clip = np.clip(a, eps, 1.0 - eps)
    # Guard against 0 in prevalence (some proxies are 0 for low-salience
    # groups, e.g. EN-Wiki image count for Dong/Li/Tibetan in our data).
    # We use log1p which maps 0 -> 0 rather than -inf.
    log_p = np.log1p(p)
    logit_a = np.log(a_clip / (1.0 - a_clip))

    # OLS in log-log logit space
    X = np.column_stack([np.ones_like(log_p), log_p])
    beta_hat, *_ = np.linalg.lstsq(X, logit_a, rcond=None)
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    pred = X @ beta_hat
    ss_res = float(np.sum((logit_a - pred) ** 2))
    ss_tot = float(np.sum((logit_a - logit_a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Pearson r of log_p vs logit_a (interpretable on scatter)
    if len(p) >= 3:
        r_pearson, p_pearson = pearsonr(log_p, logit_a)
    else:
        r_pearson, p_pearson = float("nan"), float("nan")

    # Leave-one-out R^2
    if len(p) >= 3:
        preds_loo = np.zeros_like(logit_a)
        for i in range(len(p)):
            mask = np.ones(len(p), dtype=bool)
            mask[i] = False
            Xtr = X[mask]
            ytr = logit_a[mask]
            b_loo, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
            preds_loo[i] = X[i] @ b_loo
        ss_res_loo = float(np.sum((logit_a - preds_loo) ** 2))
        r2_loo = 1.0 - ss_res_loo / ss_tot if ss_tot > 0 else float("nan")
    else:
        r2_loo = float("nan")

    return {
        "alpha": alpha,
        "beta": beta,
        "r2": float(r2),
        "r2_loo": float(r2_loo),
        "pearson_r_log_logit": float(r_pearson),
        "pearson_p_log_logit": float(p_pearson),
        "n": int(len(p)),
    }


# =============================================================================
# Jackknife (leave-one-out) Spearman rho
# =============================================================================


def jackknife_spearman(x: np.ndarray, y: np.ndarray,
                       names: List[str]) -> Dict[str, float]:
    """Leave-one-out Spearman rho for each omitted observation."""
    out = {}
    for i, name in enumerate(names):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        rho, _ = spearmanr(x[mask], y[mask])
        out[name] = float(rho)
    out["min"] = float(min(out.values()))
    out["max"] = float(max(out.values()))
    out["full"] = float(spearmanr(x, y).statistic)
    return out


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    df = pd.read_csv(METRICS / "salience_prevalence.csv")
    y = df["mean_accuracy"].to_numpy()
    names = df["ethnic_group"].tolist()

    per_proxy: Dict[str, dict] = {}
    raw_one_tailed: Dict[str, float] = {}
    raw_two_tailed: Dict[str, float] = {}

    print("=" * 70)
    print("Exact Spearman permutation p-values (n=5, 5!=120 permutations)")
    print("=" * 70)
    for col in PROXY_COLS:
        if col not in df.columns:
            continue
        x = df[col].to_numpy(dtype=float)
        if (x < 0).any():
            per_proxy[col] = {"error": "missing data"}
            continue

        rho, p_ge, p_two = spearman_exact(x, y)
        ll = loglinear_fit(x, y)
        jk = jackknife_spearman(x, y, names)

        per_proxy[col] = {
            "spearman_rho": rho,
            "p_one_tailed_greater_exact": p_ge,
            "p_two_tailed_exact": p_two,
            "loglinear": ll,
            "jackknife": jk,
        }
        raw_one_tailed[col] = p_ge
        raw_two_tailed[col] = p_two

        print(f"  {col:<15}  rho = {rho:+.3f}  "
              f"one-tailed p = {p_ge:.4f}  two-tailed p = {p_two:.4f}  "
              f"beta={ll['beta']:+.2f}  R2_LOO={ll['r2_loo']:+.2f}  "
              f"jackknife rho in [{jk['min']:+.2f}, {jk['max']:+.2f}]")

    # Holm across the 4 proxies reviewers care about (excluding those with missing data)
    core_proxies = [c for c in PROXY_COLS if c in raw_one_tailed]
    holm_1 = holm_adjust({k: raw_one_tailed[k] for k in core_proxies})
    holm_2 = holm_adjust({k: raw_two_tailed[k] for k in core_proxies})

    print()
    print("=" * 70)
    print(f"Holm-Bonferroni across {len(core_proxies)} proxies tested")
    print("=" * 70)
    print(f"  {'proxy':<15}  {'raw (1t)':>10}  {'Holm (1t)':>10}  "
          f"{'raw (2t)':>10}  {'Holm (2t)':>10}")
    for col in core_proxies:
        p1 = raw_one_tailed[col]
        p2 = raw_two_tailed[col]
        print(f"  {col:<15}  {p1:>10.4f}  {holm_1[col]:>10.4f}  "
              f"{p2:>10.4f}  {holm_2[col]:>10.4f}")

    # Add Holm-adjusted values into per_proxy record
    for col in core_proxies:
        per_proxy[col]["holm_adjusted"] = {
            "one_tailed": float(holm_1[col]),
            "two_tailed": float(holm_2[col]),
        }

    # Save
    out = {
        "method": "Exact permutation Spearman; Holm correction across proxies; "
                  "log-linear model logit(acc) ~ log(prevalence); jackknife LOO",
        "n_groups": int(len(df)),
        "per_proxy": per_proxy,
    }
    (METRICS / "salience_prevalence_robust.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )

    # Wide table for the paper
    rows = []
    for col in core_proxies:
        p = per_proxy[col]
        rows.append({
            "proxy": col,
            "spearman_rho": round(p["spearman_rho"], 3),
            "p_one_tailed_exact": round(p["p_one_tailed_greater_exact"], 4),
            "p_two_tailed_exact": round(p["p_two_tailed_exact"], 4),
            "p_one_holm": round(p["holm_adjusted"]["one_tailed"], 4),
            "p_two_holm": round(p["holm_adjusted"]["two_tailed"], 4),
            "loglinear_beta": round(p["loglinear"]["beta"], 3),
            "loglinear_r2": round(p["loglinear"]["r2"], 3),
            "loglinear_r2_loo": round(p["loglinear"]["r2_loo"], 3),
            "jackknife_min_rho": round(p["jackknife"]["min"], 3),
            "jackknife_max_rho": round(p["jackknife"]["max"], 3),
        })
    pd.DataFrame(rows).to_csv(METRICS / "salience_prevalence_robust.csv", index=False)

    print()
    print(f"Saved: {METRICS / 'salience_prevalence_robust.json'}")
    print(f"       {METRICS / 'salience_prevalence_robust.csv'}")


if __name__ == "__main__":
    main()
