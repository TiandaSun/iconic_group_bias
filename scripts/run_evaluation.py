#!/usr/bin/env python3
"""Evaluation script for computing all metrics from raw results.

Loads raw results, computes classification metrics, bias metrics,
runs statistical tests, and generates summary tables.

Usage:
    python scripts/run_evaluation.py --results-dir results/raw --output-dir results/metrics
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.evaluation import (
    calculate_all_classification_metrics,
    origin_bias_index,
    language_effect_score,
    cultural_term_coverage,
    batch_cultural_term_coverage,
    load_cultural_vocabulary,
    two_sample_ttest,
    anova_2x2,
    mcnemar_test,
    bootstrap_confidence_interval,
    cohens_d,
)
from src.evaluation.confusion_analysis import (
    test_hypothesis_h3,
    generate_confusion_report,
    aggregate_confusion_by_origin,
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# Model metadata
MODEL_ORIGINS = {
    "qwen2.5-vl-72b": "chinese",
    "qwen2.5-vl-7b": "chinese",
    "llama-3.2-vision-11b": "western",
    "gpt-4o-mini": "western",
    "gemini-2.5-flash": "western",
    "claude-haiku-4.5": "western",
}


def load_all_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all result JSON files from directory.

    Args:
        results_dir: Directory containing JSON result files.

    Returns:
        Dictionary organized by task, model, and language.
    """
    results = {
        "classification": {},
        "description": {},
    }

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            task = data.get("task", "unknown")
            model = data.get("model_name", "unknown")
            language = data.get("language", "unknown")

            key = f"{model}_{language}"

            if task == "classification":
                results["classification"][key] = data
            elif task == "description":
                results["description"][key] = data

        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")

    logger.info(
        f"Loaded {len(results['classification'])} classification results, "
        f"{len(results['description'])} description results"
    )

    return results


def compute_classification_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute all classification metrics for a single result.

    Args:
        results: Single classification result dictionary.

    Returns:
        Dictionary with all computed metrics.
    """
    # Extract predictions and ground truth
    y_true = []
    y_pred = []

    for img_id, result in results.get("results", {}).items():
        if "ground_truth" in result and "predicted" in result:
            y_true.append(result["ground_truth"])
            y_pred.append(result["predicted"])

    if not y_true:
        return {"error": "No valid predictions"}

    return calculate_all_classification_metrics(y_true, y_pred)


def compute_bias_metrics(
    classification_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute OBI and LES metrics across all models.

    Args:
        classification_results: All classification results.

    Returns:
        Dictionary with bias metrics.
    """
    # Organize accuracies by origin and language
    accuracies = defaultdict(lambda: defaultdict(list))

    for key, data in classification_results.items():
        model = data.get("model_name", "")
        language = data.get("language", "")
        origin = MODEL_ORIGINS.get(model, "unknown")

        acc = data.get("summary", {}).get("accuracy", 0)

        accuracies[origin][language].append(acc)
        accuracies["all"][language].append(acc)
        accuracies[origin]["all"].append(acc)
        accuracies["all"]["all"].append(acc)

    # Calculate OBI
    obi_results = {}

    for lang in ["zh", "en", "all"]:
        chinese_accs = accuracies["chinese"].get(lang, [])
        western_accs = accuracies["western"].get(lang, [])

        if chinese_accs and western_accs:
            obi = origin_bias_index(chinese_accs, western_accs)
            obi_results[f"obi_{lang}"] = obi

            # Bootstrap CI
            all_accs = chinese_accs + western_accs
            if len(all_accs) >= 4:
                _, ci_low, ci_high = bootstrap_confidence_interval(
                    all_accs,
                    statistic_func=lambda x: origin_bias_index(
                        list(x[:len(chinese_accs)]),
                        list(x[len(chinese_accs):])
                    ),
                    n_bootstrap=1000,
                )
                obi_results[f"obi_{lang}_ci"] = (ci_low, ci_high)

    # Calculate LES per model
    les_results = {}

    models_by_lang = defaultdict(dict)
    for key, data in classification_results.items():
        model = data.get("model_name", "")
        language = data.get("language", "")
        acc = data.get("summary", {}).get("accuracy", 0)
        models_by_lang[model][language] = acc

    for model, lang_accs in models_by_lang.items():
        if "zh" in lang_accs and "en" in lang_accs:
            les = language_effect_score(lang_accs["zh"], lang_accs["en"])
            les_results[model] = les

    # Average LES by origin
    les_by_origin = defaultdict(list)
    for model, les in les_results.items():
        origin = MODEL_ORIGINS.get(model, "unknown")
        les_by_origin[origin].append(les)

    les_results["avg_chinese"] = np.mean(les_by_origin["chinese"]) if les_by_origin["chinese"] else 0
    les_results["avg_western"] = np.mean(les_by_origin["western"]) if les_by_origin["western"] else 0

    return {
        "obi": obi_results,
        "les": les_results,
    }


def compute_description_metrics(
    description_results: Dict[str, Dict[str, Any]],
    vocab_config_path: str = "configs/cultural_vocabulary.yaml",
) -> Dict[str, Any]:
    """Compute description quality metrics.

    Args:
        description_results: All description results.
        vocab_config_path: Path to vocabulary config.

    Returns:
        Dictionary with description metrics.
    """
    vocabulary = load_cultural_vocabulary(vocab_config_path)

    metrics = {}

    for key, data in description_results.items():
        model = data.get("model_name", "")
        language = data.get("language", "")

        descriptions = []
        for img_id, result in data.get("results", {}).items():
            desc = result.get("description", "")
            if desc and desc != "ERROR":
                descriptions.append(desc)

        if descriptions:
            vocab = vocabulary.get(language, vocabulary.get("zh", []))
            coverage_stats = batch_cultural_term_coverage(descriptions, vocab, language)

            metrics[f"{model}_{language}"] = {
                "model": model,
                "language": language,
                "n_descriptions": len(descriptions),
                "coverage_stats": coverage_stats,
            }

    return metrics


def run_statistical_tests(
    classification_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Run all statistical tests.

    Args:
        classification_results: All classification results.

    Returns:
        Dictionary with test results.
    """
    tests = {}

    # Organize data for tests
    accuracies_by_origin = defaultdict(list)
    accuracies_by_language = defaultdict(list)
    data_for_anova = []

    for key, data in classification_results.items():
        model = data.get("model_name", "")
        language = data.get("language", "")
        origin = MODEL_ORIGINS.get(model, "unknown")
        acc = data.get("summary", {}).get("accuracy", 0)

        accuracies_by_origin[origin].append(acc)
        accuracies_by_language[language].append(acc)

        data_for_anova.append({
            "model": model,
            "origin": origin,
            "language": language,
            "accuracy": acc,
        })

    # T-test: Chinese vs Western origin
    if len(accuracies_by_origin["chinese"]) >= 2 and len(accuracies_by_origin["western"]) >= 2:
        t_stat, p_value = two_sample_ttest(
            accuracies_by_origin["chinese"],
            accuracies_by_origin["western"],
        )
        effect = cohens_d(
            accuracies_by_origin["chinese"],
            accuracies_by_origin["western"],
        )
        tests["origin_ttest"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": effect,
            "significant": p_value < 0.05,
        }

    # T-test: Chinese vs English prompt
    if len(accuracies_by_language["zh"]) >= 2 and len(accuracies_by_language["en"]) >= 2:
        t_stat, p_value = two_sample_ttest(
            accuracies_by_language["zh"],
            accuracies_by_language["en"],
        )
        effect = cohens_d(
            accuracies_by_language["zh"],
            accuracies_by_language["en"],
        )
        tests["language_ttest"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": effect,
            "significant": p_value < 0.05,
        }

    # 2x2 ANOVA: Origin × Language
    if len(data_for_anova) >= 4:
        anova_results = anova_2x2(data_for_anova, "origin", "language", "accuracy")
        tests["anova_origin_language"] = anova_results

    # H3 Hypothesis Test (Miao-Dong, Li-Yi confusion)
    confusion_matrices = {}
    for key, data in classification_results.items():
        if "results" in data:
            y_true = [r.get("ground_truth") for r in data["results"].values()]
            y_pred = [r.get("predicted") for r in data["results"].values()]

            from src.evaluation import build_confusion_matrix
            cm = build_confusion_matrix(y_true, y_pred)
            confusion_matrices[key] = cm

    if confusion_matrices:
        tests["h3_confusion"] = test_hypothesis_h3(confusion_matrices)

    return tests


def generate_summary_tables(
    classification_results: Dict[str, Dict[str, Any]],
    bias_metrics: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate summary tables as CSV files.

    Args:
        classification_results: All classification results.
        bias_metrics: Computed bias metrics.
        output_dir: Output directory.

    Returns:
        Dictionary mapping table name to file path.
    """
    tables = {}

    # Table 1: Overall results
    rows = []
    for key, data in classification_results.items():
        model = data.get("model_name", "")
        language = data.get("language", "")
        origin = MODEL_ORIGINS.get(model, "unknown")
        summary = data.get("summary", {})

        rows.append({
            "Model": model,
            "Origin": origin,
            "Language": language,
            "Accuracy": summary.get("accuracy", 0),
            "Macro_F1": summary.get("macro_f1", 0) if "macro_f1" in summary else None,
            "Total_Images": summary.get("total_images", 0),
            "Errors": summary.get("errors", 0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["Origin", "Model", "Language"])

    path = output_dir / "table1_overall_results.csv"
    df.to_csv(path, index=False)
    tables["overall_results"] = path

    # Table 2: Per-class accuracy
    per_class_rows = []
    for key, data in classification_results.items():
        model = data.get("model_name", "")
        language = data.get("language", "")
        per_class = data.get("summary", {}).get("per_class_accuracy", {})

        row = {"Model": model, "Language": language}
        for cls, acc in per_class.items():
            row[f"Acc_{cls}"] = acc
        per_class_rows.append(row)

    if per_class_rows:
        df = pd.DataFrame(per_class_rows)
        path = output_dir / "table2_per_class_accuracy.csv"
        df.to_csv(path, index=False)
        tables["per_class_accuracy"] = path

    # Table 3: Bias metrics
    obi = bias_metrics.get("obi", {})
    les = bias_metrics.get("les", {})

    bias_rows = [
        {"Metric": "OBI (Chinese prompts)", "Value": obi.get("obi_zh", "N/A")},
        {"Metric": "OBI (English prompts)", "Value": obi.get("obi_en", "N/A")},
        {"Metric": "OBI (Overall)", "Value": obi.get("obi_all", "N/A")},
        {"Metric": "Avg LES (Chinese models)", "Value": les.get("avg_chinese", "N/A")},
        {"Metric": "Avg LES (Western models)", "Value": les.get("avg_western", "N/A")},
    ]

    # Add per-model LES
    for model, les_value in les.items():
        if model not in ["avg_chinese", "avg_western"]:
            bias_rows.append({"Metric": f"LES ({model})", "Value": les_value})

    df = pd.DataFrame(bias_rows)
    path = output_dir / "table3_bias_metrics.csv"
    df.to_csv(path, index=False)
    tables["bias_metrics"] = path

    # Table 4: Confusion analysis
    confusion_report = generate_confusion_report(
        classification_results,
        model_metadata={m: {"origin": o} for m, o in MODEL_ORIGINS.items()}
    )

    if not confusion_report.empty:
        path = output_dir / "table4_confusion_analysis.csv"
        confusion_report.to_csv(path, index=False)
        tables["confusion_analysis"] = path

    return tables


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics from raw results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results/raw",
        help="Directory containing raw result JSON files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/metrics",
        help="Output directory for metrics",
    )
    parser.add_argument(
        "--vocab-config",
        type=str,
        default="configs/cultural_vocabulary.yaml",
        help="Path to cultural vocabulary config",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Setup
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1

    # Load results
    logger.info(f"Loading results from {results_dir}")
    all_results = load_all_results(results_dir)

    # Compute metrics
    logger.info("Computing classification metrics...")
    classification_metrics = {}
    for key, data in all_results["classification"].items():
        classification_metrics[key] = compute_classification_metrics(data)

    logger.info("Computing bias metrics (OBI, LES)...")
    bias_metrics = compute_bias_metrics(all_results["classification"])

    logger.info("Computing description metrics (CTC)...")
    description_metrics = compute_description_metrics(
        all_results["description"],
        args.vocab_config,
    )

    logger.info("Running statistical tests...")
    statistical_tests = run_statistical_tests(all_results["classification"])

    # Save all metrics
    all_metrics = {
        "timestamp": datetime.now().isoformat(),
        "classification_metrics": classification_metrics,
        "bias_metrics": bias_metrics,
        "description_metrics": description_metrics,
        "statistical_tests": statistical_tests,
    }

    metrics_path = output_dir / "all_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Saved metrics to {metrics_path}")

    # Generate summary tables
    logger.info("Generating summary tables...")
    tables = generate_summary_tables(
        all_results["classification"],
        bias_metrics,
        output_dir,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"\nSummary tables generated:")
    for name, path in tables.items():
        print(f"  - {name}: {path}")

    # Print key findings
    print(f"\nKey Findings:")
    obi = bias_metrics.get("obi", {})
    print(f"  OBI (Overall): {obi.get('obi_all', 'N/A'):.4f}" if isinstance(obi.get('obi_all'), float) else f"  OBI: N/A")

    if "origin_ttest" in statistical_tests:
        t_result = statistical_tests["origin_ttest"]
        sig = "significant" if t_result.get("significant") else "not significant"
        print(f"  Origin effect: {sig} (p={t_result.get('p_value', 0):.4f})")

    if "language_ttest" in statistical_tests:
        t_result = statistical_tests["language_ttest"]
        sig = "significant" if t_result.get("significant") else "not significant"
        print(f"  Language effect: {sig} (p={t_result.get('p_value', 0):.4f})")

    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
