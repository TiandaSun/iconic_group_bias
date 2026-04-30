#!/usr/bin/env python3
"""Generate publication-ready figures from evaluation metrics.

Loads metrics from results/metrics/ and generates all paper figures.

Usage:
    python scripts/generate_figures.py --metrics-dir results/metrics --output-dir results/figures
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.visualization import (
    plot_accuracy_comparison,
    plot_confusion_heatmap,
    plot_language_effect,
    plot_obi_summary,
    plot_model_scaling,
    plot_per_class_accuracy,
    plot_cultural_coverage_distribution,
)
from src.evaluation import build_confusion_matrix

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


def load_metrics(metrics_dir: Path) -> Dict[str, Any]:
    """Load all metrics from directory.

    Args:
        metrics_dir: Directory containing metrics files.

    Returns:
        Dictionary with all metrics.
    """
    metrics = {}

    # Load main metrics file
    metrics_file = metrics_dir / "all_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    # Load any CSV tables
    for csv_file in metrics_dir.glob("*.csv"):
        key = csv_file.stem
        metrics[f"table_{key}"] = pd.read_csv(csv_file)

    return metrics


def load_raw_results(results_dir: Path) -> Dict[str, Any]:
    """Load raw results for detailed figures.

    Args:
        results_dir: Directory containing raw result JSON files.

    Returns:
        Dictionary with raw results.
    """
    results = {"classification": {}, "description": {}}

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

    return results


def generate_figure1_accuracy_comparison(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Generate Figure 1: Accuracy comparison across models.

    Args:
        metrics: Loaded metrics.
        output_dir: Output directory.

    Returns:
        Path to generated figure.
    """
    # Build DataFrame from metrics
    rows = []

    class_metrics = metrics.get("classification_metrics", {})

    for key, data in class_metrics.items():
        if "_" in key:
            parts = key.rsplit("_", 1)
            model = parts[0]
            language = parts[1]
        else:
            continue

        origin = MODEL_ORIGINS.get(model, "unknown")
        accuracy = data.get("accuracy", 0)

        rows.append({
            "model": model,
            "origin": origin,
            "language": language,
            "accuracy": accuracy,
        })

    if not rows:
        logger.warning("No data for accuracy comparison figure")
        return None

    df = pd.DataFrame(rows)

    output_path = output_dir / "fig1_accuracy_comparison"
    plot_accuracy_comparison(df, output_path)

    return output_path


def generate_figure2_confusion_matrices(
    raw_results: Dict[str, Any],
    output_dir: Path,
    models_to_plot: list = None,
) -> Dict[str, Path]:
    """Generate Figure 2: Confusion matrices for selected models.

    Args:
        raw_results: Raw classification results.
        output_dir: Output directory.
        models_to_plot: List of models to generate matrices for.

    Returns:
        Dictionary mapping model to figure path.
    """
    paths = {}

    if models_to_plot is None:
        # Default: one Chinese, one Western model
        models_to_plot = ["qwen2.5-vl-72b", "gpt-4o-mini"]

    for key, data in raw_results.get("classification", {}).items():
        model = data.get("model_name", "")
        language = data.get("language", "")

        # Only plot selected models and Chinese prompts
        if model not in models_to_plot or language != "zh":
            continue

        # Build confusion matrix
        y_true = []
        y_pred = []
        for img_id, result in data.get("results", {}).items():
            if "ground_truth" in result and "predicted" in result:
                y_true.append(result["ground_truth"])
                y_pred.append(result["predicted"])

        if y_true:
            cm = build_confusion_matrix(y_true, y_pred)

            safe_name = model.replace(".", "_").replace("-", "_")
            output_path = output_dir / f"fig2_confusion_{safe_name}"

            plot_confusion_heatmap(
                cm,
                model_name=model,
                output_path=output_path,
                language="zh",
            )

            paths[model] = output_path

    return paths


def generate_figure3_language_effect(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Generate Figure 3: Language Effect Score visualization.

    Args:
        metrics: Loaded metrics.
        output_dir: Output directory.

    Returns:
        Path to generated figure.
    """
    les_scores = metrics.get("bias_metrics", {}).get("les", {})

    # Filter out aggregated values
    model_les = {
        k: v for k, v in les_scores.items()
        if k not in ["avg_chinese", "avg_western"]
    }

    if not model_les:
        logger.warning("No LES data for figure")
        return None

    output_path = output_dir / "fig3_language_effect"
    plot_language_effect(model_les, output_path)

    return output_path


def generate_figure4_obi_summary(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Generate Figure 4: Origin Bias Index summary.

    Args:
        metrics: Loaded metrics.
        output_dir: Output directory.

    Returns:
        Path to generated figure.
    """
    obi_data = metrics.get("bias_metrics", {}).get("obi", {})

    obi_by_language = {}
    ci_data = {}

    for key, value in obi_data.items():
        if key.startswith("obi_") and not key.endswith("_ci"):
            lang = key.replace("obi_", "")
            obi_by_language[lang] = value

            ci_key = f"{key}_ci"
            if ci_key in obi_data:
                ci_data[lang] = obi_data[ci_key]

    if not obi_by_language:
        logger.warning("No OBI data for figure")
        return None

    output_path = output_dir / "fig4_obi_summary"
    plot_obi_summary(
        obi_by_language,
        output_path,
        confidence_intervals=ci_data if ci_data else None,
    )

    return output_path


def generate_figure5_model_scaling(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Generate Figure 5: Model scaling analysis (7B vs 72B).

    Args:
        metrics: Loaded metrics.
        output_dir: Output directory.

    Returns:
        Path to generated figure.
    """
    class_metrics = metrics.get("classification_metrics", {})

    # Find Qwen 7B and 72B results
    results_7b = {}
    results_72b = {}

    for key, data in class_metrics.items():
        if "qwen2.5-vl-7b" in key:
            lang = key.split("_")[-1]
            results_7b[f"accuracy_{lang}"] = data.get("accuracy", 0)
            results_7b["macro_f1"] = data.get("macro_f1", 0)
        elif "qwen2.5-vl-72b" in key:
            lang = key.split("_")[-1]
            results_72b[f"accuracy_{lang}"] = data.get("accuracy", 0)
            results_72b["macro_f1"] = data.get("macro_f1", 0)

    if not results_7b or not results_72b:
        logger.warning("Missing Qwen 7B or 72B data for scaling figure")
        return None

    output_path = output_dir / "fig5_model_scaling"
    plot_model_scaling(results_7b, results_72b, output_path)

    return output_path


def generate_figure6_per_class_accuracy(
    metrics: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Generate Figure 6: Per-class accuracy across models.

    Args:
        metrics: Loaded metrics.
        output_dir: Output directory.

    Returns:
        Path to generated figure.
    """
    class_metrics = metrics.get("classification_metrics", {})

    # Aggregate per-class accuracy (use Chinese prompts)
    results_dict = {}

    for key, data in class_metrics.items():
        if "_zh" in key:
            model = key.replace("_zh", "")
            per_class = data.get("per_class_accuracy", {})

            if per_class:
                # Convert label letters to ethnic group names
                label_map = {"A": "Miao", "B": "Dong", "C": "Yi", "D": "Li", "E": "Tibetan"}
                results_dict[model] = {
                    label_map.get(k, k): v for k, v in per_class.items()
                }

    if not results_dict:
        logger.warning("No per-class accuracy data")
        return None

    output_path = output_dir / "fig6_per_class_accuracy"
    plot_per_class_accuracy(results_dict, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication figures from metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--metrics-dir", "-m",
        type=str,
        default="results/metrics",
        help="Directory containing metrics files",
    )
    parser.add_argument(
        "--raw-dir", "-r",
        type=str,
        default="results/raw",
        help="Directory containing raw result files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--figures",
        type=str,
        nargs="+",
        default=["all"],
        help="Figures to generate (1-6 or 'all')",
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

    metrics_dir = Path(args.metrics_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading metrics from {metrics_dir}")
    metrics = load_metrics(metrics_dir)

    logger.info(f"Loading raw results from {raw_dir}")
    raw_results = load_raw_results(raw_dir)

    # Determine which figures to generate
    if "all" in args.figures:
        figures_to_gen = ["1", "2", "3", "4", "5", "6"]
    else:
        figures_to_gen = args.figures

    generated = {}

    # Generate figures
    if "1" in figures_to_gen:
        logger.info("Generating Figure 1: Accuracy Comparison")
        path = generate_figure1_accuracy_comparison(metrics, output_dir)
        if path:
            generated["fig1_accuracy_comparison"] = path

    if "2" in figures_to_gen:
        logger.info("Generating Figure 2: Confusion Matrices")
        paths = generate_figure2_confusion_matrices(raw_results, output_dir)
        generated.update({f"fig2_{k}": v for k, v in paths.items()})

    if "3" in figures_to_gen:
        logger.info("Generating Figure 3: Language Effect Score")
        path = generate_figure3_language_effect(metrics, output_dir)
        if path:
            generated["fig3_language_effect"] = path

    if "4" in figures_to_gen:
        logger.info("Generating Figure 4: OBI Summary")
        path = generate_figure4_obi_summary(metrics, output_dir)
        if path:
            generated["fig4_obi_summary"] = path

    if "5" in figures_to_gen:
        logger.info("Generating Figure 5: Model Scaling")
        path = generate_figure5_model_scaling(metrics, output_dir)
        if path:
            generated["fig5_model_scaling"] = path

    if "6" in figures_to_gen:
        logger.info("Generating Figure 6: Per-Class Accuracy")
        path = generate_figure6_per_class_accuracy(metrics, output_dir)
        if path:
            generated["fig6_per_class_accuracy"] = path

    # Summary
    print(f"\n{'=' * 60}")
    print("FIGURE GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated {len(generated)} figure(s):")
    for name, path in generated.items():
        print(f"  - {name}: {path}.png, {path}.pdf")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
