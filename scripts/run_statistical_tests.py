"""Statistical significance tests for Task 1 classification.

Tests:
  1. McNemar's test pairwise between models (per language), Holm-corrected.
  2. Permutation test for Origin × Language interaction (image-level).
  3. Bootstrap CI for OBI and LES.

Outputs:
  results/metrics/task1_significance.json
"""

from __future__ import annotations

import glob
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_CANONICAL: dict[str, str] = {}
ORIGIN = {
    "qwen2.5-vl-7b": "chinese", "qwen2-vl-7b": "chinese",
    "llama-3.2-vision-11b": "western", "gpt-4o-mini": "western",
    "claude-haiku-4.5": "western",
}
CH_MODELS = [m for m, o in ORIGIN.items() if o == "chinese"]
W_MODELS = [m for m, o in ORIGIN.items() if o == "western"]


def load_correct_vectors() -> dict[tuple[str, str], dict[str, int]]:
    """Return {(model, lang): {image_id: 1_if_correct_else_0}}."""
    out = {}
    for fp in sorted(glob.glob(str(RAW / "task1_*_results.json"))):
        with open(fp) as f:
            d = json.load(f)
        model = MODEL_CANONICAL.get(d["model_name"], d["model_name"])
        lang = d["language"]
        out[(model, lang)] = {
            img_id: int(v["ground_truth"] == v["predicted"])
            for img_id, v in d["results"].items()
        }
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Two-sided binomial with p=0.5
    return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)


def holm_correct(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni correction."""
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = [0.0] * m
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, pvals[idx] * (m - rank))
        adj = max(adj, prev)
        corrected[idx] = adj
        prev = adj
    return corrected


def pairwise_mcnemar(correct: dict) -> list[dict]:
    results = []
    models = sorted({m for m, _ in correct})
    for lang in ["zh", "en"]:
        tests = []
        for m1, m2 in combinations(models, 2):
            v1 = correct[(m1, lang)]
            v2 = correct[(m2, lang)]
            ids = sorted(set(v1) & set(v2))
            b = sum(1 for i in ids if v1[i] == 1 and v2[i] == 0)
            c = sum(1 for i in ids if v1[i] == 0 and v2[i] == 1)
            p = mcnemar_exact(b, c)
            tests.append({
                "language": lang, "model_a": m1, "model_b": m2,
                "a_correct_b_wrong": b, "b_correct_a_wrong": c,
                "n_discordant": b + c, "p_value": p,
            })
        # Holm correction within language
        corrected = holm_correct([t["p_value"] for t in tests])
        for t, pc in zip(tests, corrected):
            t["p_holm"] = pc
            t["significant_05"] = pc < 0.05
        results.extend(tests)
    return results


def interaction_permutation(correct: dict, n_perm: int = 2000, seed: int = 42) -> dict:
    """Permutation test for Origin × Language interaction at image level.

    Statistic: difference of accuracy differences
      D = (mean_ch_zh - mean_w_zh) - (mean_ch_en - mean_w_en)
    Under H0 (no interaction), origin labels are exchangeable across models
    WITHIN each language. Permute model-to-origin assignment.
    """
    models = sorted({m for m, _ in correct})
    # Per-model accuracy per language
    acc = {}
    for m in models:
        for lang in ["zh", "en"]:
            acc[(m, lang)] = float(np.mean(list(correct[(m, lang)].values())))

    def compute_D(origin_map: dict) -> float:
        ch = [m for m in models if origin_map[m] == "chinese"]
        w = [m for m in models if origin_map[m] == "western"]
        ch_zh = np.mean([acc[(m, "zh")] for m in ch])
        w_zh = np.mean([acc[(m, "zh")] for m in w])
        ch_en = np.mean([acc[(m, "en")] for m in ch])
        w_en = np.mean([acc[(m, "en")] for m in w])
        return (ch_zh - w_zh) - (ch_en - w_en)

    D_obs = compute_D(ORIGIN)

    # Permute origin labels (fixed counts: 2 chinese, 3 western)
    rng = np.random.default_rng(seed)
    n_ch = sum(1 for o in ORIGIN.values() if o == "chinese")
    D_null = []
    for _ in range(n_perm):
        perm = models.copy()
        rng.shuffle(perm)
        permuted = {m: ("chinese" if i < n_ch else "western") for i, m in enumerate(perm)}
        D_null.append(compute_D(permuted))
    D_null = np.array(D_null)
    p = float(np.mean(np.abs(D_null) >= abs(D_obs)))

    return {
        "statistic_D": D_obs,
        "interpretation": "D = (CN-W)_zh - (CN-W)_en; negative means W benefits more from zh prompts",
        "n_permutations": n_perm,
        "p_value": p,
        "null_mean": float(np.mean(D_null)),
        "null_std": float(np.std(D_null)),
    }


def bootstrap_obi_les(correct: dict, n_boot: int = 1000, seed: int = 42) -> dict:
    """Bootstrap CIs for OBI (combined) and LES (per model) by resampling images."""
    rng = np.random.default_rng(seed)
    models = sorted({m for m, _ in correct})
    # Image IDs are shared across models within a language (same dataset)
    ids_zh = sorted(correct[(models[0], "zh")].keys())
    ids_en = sorted(correct[(models[0], "en")].keys())

    def per_model_acc(ids: list[str], lang: str) -> dict[str, float]:
        return {m: float(np.mean([correct[(m, lang)][i] for i in ids])) for m in models}

    obi_vals = []
    obi_logit_vals = []
    cohens_h_vals = []
    les_vals = {m: [] for m in models}

    for _ in range(n_boot):
        zh_sample = [ids_zh[i] for i in rng.integers(0, len(ids_zh), len(ids_zh))]
        en_sample = [ids_en[i] for i in rng.integers(0, len(ids_en), len(ids_en))]
        acc_zh = per_model_acc(zh_sample, "zh")
        acc_en = per_model_acc(en_sample, "en")

        # Three OBI variants (ratio / logit / Cohen's h). Ratio form
        # is the original definition; logit and h are stable near 0/1
        # (fixes the Li OBI=+1.29 pathology flagged by reviewers).
        ch_vals = [acc_zh[m] for m in CH_MODELS] + [acc_en[m] for m in CH_MODELS]
        w_vals = [acc_zh[m] for m in W_MODELS] + [acc_en[m] for m in W_MODELS]
        ch_mean = float(np.mean(ch_vals))
        w_mean = float(np.mean(w_vals))
        all_mean = float(np.mean(ch_vals + w_vals))
        obi_vals.append((ch_mean - w_mean) / all_mean if all_mean > 0 else np.nan)

        # Logit OBI
        eps = 1e-3
        p_ch = float(np.clip(ch_mean, eps, 1 - eps))
        p_w = float(np.clip(w_mean, eps, 1 - eps))
        obi_logit_vals.append(
            float(np.log(p_ch / (1 - p_ch)) - np.log(p_w / (1 - p_w)))
        )
        # Cohen's h
        phi_ch = 2.0 * np.arcsin(np.sqrt(np.clip(ch_mean, 0.0, 1.0)))
        phi_w = 2.0 * np.arcsin(np.sqrt(np.clip(w_mean, 0.0, 1.0)))
        cohens_h_vals.append(float(phi_ch - phi_w))

        # LES per model
        for m in models:
            les_vals[m].append((acc_zh[m] - acc_en[m]) / acc_en[m] if acc_en[m] > 0 else np.nan)

    def ci(vals):
        arr = np.array(vals)
        arr = arr[~np.isnan(arr)]
        return {"mean": float(np.mean(arr)),
                "ci_lo": float(np.percentile(arr, 2.5)),
                "ci_hi": float(np.percentile(arr, 97.5))}

    return {
        "obi_combined": ci(obi_vals),
        "obi_logit": ci(obi_logit_vals),
        "cohens_h": ci(cohens_h_vals),
        "les_per_model": {m: ci(les_vals[m]) for m in models},
        "n_bootstrap": n_boot,
    }


def main() -> None:
    print("Loading correct vectors ...")
    correct = load_correct_vectors()

    print("Running pairwise McNemar tests ...")
    mcnemar = pairwise_mcnemar(correct)

    print("Running Origin × Language interaction permutation test ...")
    interaction = interaction_permutation(correct, n_perm=5000)

    print("Bootstrapping OBI and LES CIs ...")
    bootstrap = bootstrap_obi_les(correct, n_boot=1000)

    output = {
        "pairwise_mcnemar": mcnemar,
        "origin_language_interaction": interaction,
        "bootstrap_ci": bootstrap,
    }
    with open(OUT / "task1_significance.json", "w") as f:
        json.dump(output, f, indent=2)

    # --- Summary ---
    print("\n=== Pairwise McNemar (significant after Holm correction) ===")
    for lang in ["zh", "en"]:
        print(f"\n[{lang}]")
        for t in mcnemar:
            if t["language"] != lang:
                continue
            flag = "***" if t["significant_05"] else "   "
            print(f"  {flag} {t['model_a']:<22} vs {t['model_b']:<22} "
                  f"p_holm={t['p_holm']:.4f}  (b={t['a_correct_b_wrong']}, c={t['b_correct_a_wrong']})")

    print("\n=== Origin × Language Interaction ===")
    print(f"  D_observed = {interaction['statistic_D']:+.4f}")
    print(f"  p (permutation) = {interaction['p_value']:.4f}")
    print(f"  interpretation: {interaction['interpretation']}")

    print("\n=== Bootstrap 95% CIs ===")
    b = bootstrap
    print(f"  OBI ratio    : {b['obi_combined']['mean']:+.3f} "
          f"[{b['obi_combined']['ci_lo']:+.3f}, {b['obi_combined']['ci_hi']:+.3f}]")
    print(f"  OBI logit    : {b['obi_logit']['mean']:+.3f} "
          f"[{b['obi_logit']['ci_lo']:+.3f}, {b['obi_logit']['ci_hi']:+.3f}]  "
          f"(log odds-ratio, stable near 0/1)")
    print(f"  Cohen's h    : {b['cohens_h']['mean']:+.3f} "
          f"[{b['cohens_h']['ci_lo']:+.3f}, {b['cohens_h']['ci_hi']:+.3f}]  "
          f"(small ~0.2, medium ~0.5, large ~0.8)")
    print("  LES per model:")
    for m, ci in b["les_per_model"].items():
        sig = "*" if (ci["ci_lo"] > 0 or ci["ci_hi"] < 0) else " "
        print(f"    {sig} {m:<25} {ci['mean']:+.3f} [{ci['ci_lo']:+.3f}, {ci['ci_hi']:+.3f}]")

    print("\nSaved to:", OUT / "task1_significance.json")


if __name__ == "__main__":
    main()
