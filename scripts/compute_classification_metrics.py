"""Compute Task 1 classification metrics (accuracy, F1, OBI, LES, confusion, fairness).

Outputs:
    results/metrics/task1_per_cell.csv       — per (model, language, ethnic) metrics
    results/metrics/task1_overall.csv        — per (model, language) overall metrics
    results/metrics/task1_bias_metrics.json  — OBI, LES summaries
    results/metrics/task1_confusion.json     — confusion matrices
    results/metrics/task1_prediction_dist.csv — prediction distribution
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["A", "B", "C", "D", "E"]
LABEL = {"A": "Miao", "B": "Dong", "C": "Yi", "D": "Li", "E": "Tibetan"}

# Canonical model-name mapping. After the 2026-04-23 full-n=3000 API re-run
# on claude-haiku-4.5, there is no longer a rename to perform — the Task 1
# files genuinely carry their model's real name. We keep the dict for
# backward compatibility with any archived files that might be re-ingested.
MODEL_CANONICAL: dict[str, str] = {}
ORIGIN = {
    "qwen2.5-vl-7b": "chinese",
    "qwen2-vl-7b": "chinese",
    "llama-3.2-vision-11b": "western",
    "gpt-4o-mini": "western",
    "claude-haiku-4.5": "western",
}


def load_task1() -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Load all task1 result files, returning {(model, lang): [(gt, pred), ...]}."""
    records: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for fp in sorted(glob.glob(str(RAW / "task1_*_results.json"))):
        with open(fp) as f:
            d = json.load(f)
        model = MODEL_CANONICAL.get(d["model_name"], d["model_name"])
        lang = d["language"]
        pairs = [(v["ground_truth"], v["predicted"]) for v in d["results"].values()]
        records[(model, lang)] = pairs
    return records


def accuracy(pairs: list[tuple[str, str]]) -> float:
    return sum(1 for g, p in pairs if g == p) / len(pairs)


def per_class_accuracy(pairs: list[tuple[str, str]]) -> dict[str, float]:
    out = {}
    for c in CLASSES:
        rel = [p for p in pairs if p[0] == c]
        out[c] = sum(1 for g, p in rel if g == p) / len(rel) if rel else float("nan")
    return out


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    f1s = []
    for c in CLASSES:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def bootstrap_ci(pairs: list[tuple[str, str]], metric_fn, n: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(pairs)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(arr), len(arr))
        sample = [tuple(arr[i]) for i in idx]
        vals.append(metric_fn(sample))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def confusion_matrix(pairs: list[tuple[str, str]]) -> np.ndarray:
    M = np.zeros((5, 5), dtype=int)
    for gt, pr in pairs:
        if gt in CLASSES and pr in CLASSES:
            M[CLASSES.index(gt), CLASSES.index(pr)] += 1
    return M


def prediction_distribution(pairs: list[tuple[str, str]]) -> dict[str, float]:
    c = Counter(p for _, p in pairs)
    total = len(pairs)
    return {LABEL[k]: c.get(k, 0) / total for k in CLASSES}


def main() -> None:
    records = load_task1()
    models = sorted({m for m, _ in records})

    # ---- Per-cell (model × language × ethnic) table ----
    cell_rows = []
    for (m, lang), pairs in records.items():
        pc = per_class_accuracy(pairs)
        for c in CLASSES:
            cell_rows.append({
                "model": m,
                "origin": ORIGIN[m],
                "language": lang,
                "ethnic_group": LABEL[c],
                "class_code": c,
                "accuracy": pc[c],
                "n": sum(1 for g, _ in pairs if g == c),
            })
    pd.DataFrame(cell_rows).to_csv(OUT / "task1_per_cell.csv", index=False)

    # ---- Overall (model × language) table with CI ----
    overall_rows = []
    for (m, lang), pairs in records.items():
        acc = accuracy(pairs)
        acc_lo, acc_hi = bootstrap_ci(pairs, accuracy)
        f1 = macro_f1(pairs)
        f1_lo, f1_hi = bootstrap_ci(pairs, macro_f1)
        overall_rows.append({
            "model": m,
            "origin": ORIGIN[m],
            "language": lang,
            "n": len(pairs),
            "accuracy": acc,
            "accuracy_ci_lo": acc_lo,
            "accuracy_ci_hi": acc_hi,
            "macro_f1": f1,
            "macro_f1_ci_lo": f1_lo,
            "macro_f1_ci_hi": f1_hi,
        })
    pd.DataFrame(overall_rows).to_csv(OUT / "task1_overall.csv", index=False)

    # ---- Prediction distribution ----
    dist_rows = []
    for (m, lang), pairs in records.items():
        dist = prediction_distribution(pairs)
        for g, frac in dist.items():
            dist_rows.append({"model": m, "language": lang, "predicted_group": g, "fraction": frac})
    pd.DataFrame(dist_rows).to_csv(OUT / "task1_prediction_dist.csv", index=False)

    # ---- Confusion matrices ----
    confusion = {}
    for (m, lang), pairs in records.items():
        M = confusion_matrix(pairs)
        confusion[f"{m}|{lang}"] = {
            "classes": [LABEL[c] for c in CLASSES],
            "counts": M.tolist(),
            "row_normalized": (M / M.sum(axis=1, keepdims=True)).tolist(),
        }
    with open(OUT / "task1_confusion.json", "w") as f:
        json.dump(confusion, f, indent=2)

    # ---- Bias metrics (OBI, LES) ----
    ch_models = [m for m in models if ORIGIN[m] == "chinese"]
    w_models = [m for m in models if ORIGIN[m] == "western"]

    def mean_acc(models_: list[str], lang: str | None) -> float:
        accs = []
        for m in models_:
            if lang:
                accs.append(accuracy(records[(m, lang)]))
            else:
                accs.extend([accuracy(records[(m, l)]) for l in ["zh", "en"]])
        return float(np.mean(accs))

    def obi(lang: str | None) -> dict:
        ch = mean_acc(ch_models, lang)
        w = mean_acc(w_models, lang)
        all_ = mean_acc(ch_models + w_models, lang)
        return {
            "chinese_mean_acc": ch,
            "western_mean_acc": w,
            "overall_mean_acc": all_,
            "obi": (ch - w) / all_ if all_ > 0 else None,
        }

    def les_per_model() -> dict:
        out = {}
        for m in models:
            zh = accuracy(records[(m, "zh")])
            en = accuracy(records[(m, "en")])
            out[m] = {
                "origin": ORIGIN[m],
                "acc_zh": zh,
                "acc_en": en,
                "les": (zh - en) / en if en > 0 else None,
            }
        return out

    # OBI per ethnic group (languages pooled)
    obi_per_group = {}
    for c in CLASSES:
        ch_a = [per_class_accuracy(records[(m, l)])[c] for m in ch_models for l in ["zh", "en"]]
        w_a = [per_class_accuracy(records[(m, l)])[c] for m in w_models for l in ["zh", "en"]]
        ch_m = float(np.nanmean(ch_a))
        w_m = float(np.nanmean(w_a))
        all_m = float(np.nanmean(ch_a + w_a))
        obi_per_group[LABEL[c]] = {
            "chinese_mean_acc": ch_m,
            "western_mean_acc": w_m,
            "obi": (ch_m - w_m) / all_m if all_m > 0 else None,
        }

    # Fairness (CV across ethnic groups)
    fairness = {}
    for m in models:
        for lang in ["zh", "en"]:
            pc = per_class_accuracy(records[(m, lang)])
            vals = [pc[c] for c in CLASSES]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            fairness[f"{m}|{lang}"] = {"mean": mean, "std": std, "cv": std / mean if mean > 0 else None}

    bias = {
        "chinese_models": ch_models,
        "western_models": w_models,
        "obi_by_language": {
            "zh": obi("zh"),
            "en": obi("en"),
            "combined": obi(None),
        },
        "obi_by_ethnic_group_pooled": obi_per_group,
        "les_per_model": les_per_model(),
        "fairness_cv": fairness,
        "notes": {
            "obi_definition": "(mean_acc_chinese_models - mean_acc_western_models) / mean_acc_all",
            "les_definition": "(acc_chinese_prompt - acc_english_prompt) / acc_english_prompt",
            "model_naming": "As of 2026-04-24 all Task 1 raw files carry their real model names. Pre-revision runs used claude-3.5-sonnet for Claude Task 1 (n=500); those have been archived in results/raw/archive_old_api_n500/.",
            "unbalanced_warning": "OBI computed over 2 chinese vs 3 western models; interpret with care.",
        },
    }
    with open(OUT / "task1_bias_metrics.json", "w") as f:
        json.dump(bias, f, indent=2)

    print("Saved to:", OUT)
    for p in sorted(OUT.glob("task1_*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
