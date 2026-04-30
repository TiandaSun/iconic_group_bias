"""Confusion pattern analysis for VLM evaluation.

This module provides functions to analyze systematic confusion patterns
in classification results, including hypothesis testing for specific
ethnic group confusion pairs.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.metrics import build_confusion_matrix, LABEL_TO_ETHNIC_GROUP

logger = logging.getLogger(__name__)

# Class labels
CLASSES = ["A", "B", "C", "D", "E"]

# Ethnic group names for readability
ETHNIC_GROUPS = ["Miao", "Dong", "Yi", "Li", "Tibetan"]

# Label to index mapping
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}
ETHNIC_TO_LABEL = {v: k for k, v in LABEL_TO_ETHNIC_GROUP.items()}


def identify_confusion_pairs(
    confusion_matrix: np.ndarray,
    threshold: float = 0.1,
    normalize: bool = True,
    exclude_diagonal: bool = True,
) -> List[Tuple[str, str, float]]:
    """Identify pairs of classes with high confusion rates.

    Args:
        confusion_matrix: Confusion matrix (rows=true, cols=predicted).
        threshold: Minimum confusion rate to include (0.0 to 1.0).
        normalize: If True, normalize by row (true class) totals.
        exclude_diagonal: If True, exclude correct predictions.

    Returns:
        List of (true_class, pred_class, confusion_rate) tuples,
        sorted by confusion rate descending.
    """
    cm = np.array(confusion_matrix, dtype=float)

    if normalize:
        # Normalize by row sums (true class totals)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm / row_sums

    confusion_pairs = []

    for i, true_class in enumerate(CLASSES):
        for j, pred_class in enumerate(CLASSES):
            if exclude_diagonal and i == j:
                continue

            rate = cm[i, j]
            if rate >= threshold:
                # Convert to ethnic group names for readability
                true_name = LABEL_TO_ETHNIC_GROUP.get(true_class, true_class)
                pred_name = LABEL_TO_ETHNIC_GROUP.get(pred_class, pred_class)
                confusion_pairs.append((true_name, pred_name, float(rate)))

    # Sort by confusion rate descending
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    logger.debug(f"Found {len(confusion_pairs)} confusion pairs above threshold {threshold}")

    return confusion_pairs


def confusion_asymmetry(
    confusion_matrix: np.ndarray,
    class_a: str,
    class_b: str,
) -> float:
    """Calculate asymmetry of confusion between two classes.

    Asymmetry = (P(pred=B|true=A) - P(pred=A|true=B)) / max(both)

    Interpretation:
        - Positive: More confusion from A to B than B to A
        - Negative: More confusion from B to A than A to B
        - Zero: Symmetric confusion

    Args:
        confusion_matrix: Confusion matrix array.
        class_a: First class label (letter or ethnic group name).
        class_b: Second class label (letter or ethnic group name).

    Returns:
        Asymmetry value (-1.0 to 1.0).
    """
    cm = np.array(confusion_matrix, dtype=float)

    # Convert ethnic group names to indices if needed
    if class_a in ETHNIC_TO_LABEL:
        class_a = ETHNIC_TO_LABEL[class_a]
    if class_b in ETHNIC_TO_LABEL:
        class_b = ETHNIC_TO_LABEL[class_b]

    idx_a = LABEL_TO_IDX.get(class_a)
    idx_b = LABEL_TO_IDX.get(class_b)

    if idx_a is None or idx_b is None:
        raise ValueError(f"Invalid class labels: {class_a}, {class_b}")

    # Get row totals for normalization
    total_a = cm[idx_a].sum()
    total_b = cm[idx_b].sum()

    if total_a == 0 or total_b == 0:
        return 0.0

    # P(pred=B | true=A)
    conf_a_to_b = cm[idx_a, idx_b] / total_a

    # P(pred=A | true=B)
    conf_b_to_a = cm[idx_b, idx_a] / total_b

    max_conf = max(conf_a_to_b, conf_b_to_a)
    if max_conf == 0:
        return 0.0

    asymmetry = (conf_a_to_b - conf_b_to_a) / max_conf

    logger.debug(
        f"Confusion asymmetry {class_a}-{class_b}: "
        f"A→B={conf_a_to_b:.3f}, B→A={conf_b_to_a:.3f}, asymmetry={asymmetry:.3f}"
    )

    return float(asymmetry)


def get_confusion_rate(
    confusion_matrix: np.ndarray,
    class_a: str,
    class_b: str,
    bidirectional: bool = True,
) -> float:
    """Get confusion rate between two classes.

    Args:
        confusion_matrix: Confusion matrix array.
        class_a: First class.
        class_b: Second class.
        bidirectional: If True, return average of both directions.

    Returns:
        Confusion rate.
    """
    cm = np.array(confusion_matrix, dtype=float)

    # Convert names to indices
    if class_a in ETHNIC_TO_LABEL:
        class_a = ETHNIC_TO_LABEL[class_a]
    if class_b in ETHNIC_TO_LABEL:
        class_b = ETHNIC_TO_LABEL[class_b]

    idx_a = LABEL_TO_IDX.get(class_a)
    idx_b = LABEL_TO_IDX.get(class_b)

    if idx_a is None or idx_b is None:
        return 0.0

    # Normalize rows
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    if bidirectional:
        return float((cm_norm[idx_a, idx_b] + cm_norm[idx_b, idx_a]) / 2)
    else:
        return float(cm_norm[idx_a, idx_b])


def aggregate_confusion_by_origin(
    results_dict: Dict[str, Dict[str, Any]],
    model_origins: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate confusion patterns by model origin (Chinese vs Western).

    Args:
        results_dict: Dictionary mapping model_name to results containing
                     'y_true', 'y_pred' or 'confusion_matrix'.
        model_origins: Dictionary mapping model_name to origin ('chinese' or 'western').

    Returns:
        Dictionary with aggregated confusion analysis for each origin.
    """
    origin_results = {
        "chinese": {"matrices": [], "models": []},
        "western": {"matrices": [], "models": []},
    }

    for model_name, results in results_dict.items():
        origin = model_origins.get(model_name, "unknown")
        if origin not in origin_results:
            logger.warning(f"Unknown origin '{origin}' for model {model_name}")
            continue

        # Get or build confusion matrix
        if "confusion_matrix" in results:
            cm = np.array(results["confusion_matrix"])
        elif "y_true" in results and "y_pred" in results:
            cm = build_confusion_matrix(results["y_true"], results["y_pred"])
        else:
            logger.warning(f"No confusion data for model {model_name}")
            continue

        origin_results[origin]["matrices"].append(cm)
        origin_results[origin]["models"].append(model_name)

    # Aggregate for each origin
    aggregated = {}

    for origin, data in origin_results.items():
        if not data["matrices"]:
            continue

        # Average confusion matrix
        avg_cm = np.mean(data["matrices"], axis=0)

        # Identify top confusion pairs
        top_pairs = identify_confusion_pairs(avg_cm, threshold=0.05)

        # Calculate asymmetry for top pairs
        asymmetries = {}
        for true_cls, pred_cls, rate in top_pairs[:5]:
            key = f"{true_cls}-{pred_cls}"
            asymmetries[key] = confusion_asymmetry(avg_cm, true_cls, pred_cls)

        aggregated[origin] = {
            "models": data["models"],
            "n_models": len(data["models"]),
            "avg_confusion_matrix": avg_cm.tolist(),
            "top_confusion_pairs": top_pairs[:10],
            "asymmetries": asymmetries,
        }

    # Compare origins
    if "chinese" in aggregated and "western" in aggregated:
        chinese_cm = np.array(aggregated["chinese"]["avg_confusion_matrix"])
        western_cm = np.array(aggregated["western"]["avg_confusion_matrix"])

        # Difference matrix
        diff_cm = chinese_cm - western_cm

        aggregated["comparison"] = {
            "difference_matrix": diff_cm.tolist(),
            "chinese_higher_confusion": [],
            "western_higher_confusion": [],
        }

        # Find where each origin has higher confusion
        for i, true_cls in enumerate(CLASSES):
            for j, pred_cls in enumerate(CLASSES):
                if i == j:
                    continue
                diff = diff_cm[i, j]
                true_name = LABEL_TO_ETHNIC_GROUP[true_cls]
                pred_name = LABEL_TO_ETHNIC_GROUP[pred_cls]

                if diff > 0.02:  # 2% threshold
                    aggregated["comparison"]["chinese_higher_confusion"].append(
                        (true_name, pred_name, float(diff))
                    )
                elif diff < -0.02:
                    aggregated["comparison"]["western_higher_confusion"].append(
                        (true_name, pred_name, float(-diff))
                    )

    return aggregated


def generate_confusion_report(
    all_results: Dict[str, Dict[str, Any]],
    model_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> pd.DataFrame:
    """Generate summary report of confusion patterns across all models.

    Args:
        all_results: Dictionary mapping (model, language) or model to results.
        model_metadata: Optional metadata with 'origin' for each model.

    Returns:
        DataFrame with columns: Model, Origin, Language, Top_Confusion_Pair,
        Confusion_Rate, Second_Pair, Second_Rate.
    """
    rows = []

    for key, results in all_results.items():
        # Parse key
        if isinstance(key, tuple):
            model_name, language = key
        else:
            model_name = key
            language = results.get("language", "unknown")

        # Get origin from metadata or results
        origin = "unknown"
        if model_metadata and model_name in model_metadata:
            origin = model_metadata[model_name].get("origin", "unknown")
        elif "origin" in results:
            origin = results["origin"]

        # Get or build confusion matrix
        if "confusion_matrix" in results:
            cm = np.array(results["confusion_matrix"])
        elif "y_true" in results and "y_pred" in results:
            cm = build_confusion_matrix(results["y_true"], results["y_pred"])
        elif "results" in results:
            # Extract from nested results
            y_true = []
            y_pred = []
            for img_id, res in results["results"].items():
                if "ground_truth" in res and "predicted" in res:
                    y_true.append(res["ground_truth"])
                    y_pred.append(res["predicted"])
            if y_true:
                cm = build_confusion_matrix(y_true, y_pred)
            else:
                continue
        else:
            continue

        # Get top confusion pairs
        pairs = identify_confusion_pairs(cm, threshold=0.01)

        if len(pairs) >= 2:
            top_pair = f"{pairs[0][0]}→{pairs[0][1]}"
            top_rate = pairs[0][2]
            second_pair = f"{pairs[1][0]}→{pairs[1][1]}"
            second_rate = pairs[1][2]
        elif len(pairs) == 1:
            top_pair = f"{pairs[0][0]}→{pairs[0][1]}"
            top_rate = pairs[0][2]
            second_pair = "N/A"
            second_rate = 0.0
        else:
            top_pair = "N/A"
            top_rate = 0.0
            second_pair = "N/A"
            second_rate = 0.0

        # Get specific pair confusion rates for H3 hypothesis
        miao_dong_rate = get_confusion_rate(cm, "Miao", "Dong")
        li_yi_rate = get_confusion_rate(cm, "Li", "Yi")

        rows.append({
            "Model": model_name,
            "Origin": origin,
            "Language": language,
            "Top_Confusion_Pair": top_pair,
            "Confusion_Rate": round(top_rate, 4),
            "Second_Pair": second_pair,
            "Second_Rate": round(second_rate, 4),
            "Miao_Dong_Confusion": round(miao_dong_rate, 4),
            "Li_Yi_Confusion": round(li_yi_rate, 4),
        })

    df = pd.DataFrame(rows)

    # Sort by confusion rate
    if not df.empty:
        df = df.sort_values("Confusion_Rate", ascending=False)

    return df


def test_hypothesis_h3(
    confusion_matrices: Dict[str, np.ndarray],
    target_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """Test Hypothesis H3: Miao-Dong and Li-Yi show higher confusion.

    H3 posits that geographically and culturally proximate ethnic groups
    (Miao-Dong in Guizhou, Li-Yi in Hainan/Yunnan) exhibit higher
    inter-group confusion rates than other pairs.

    Args:
        confusion_matrices: Dictionary mapping model_name to confusion matrix.
        target_pairs: Pairs to test. Defaults to [("Miao", "Dong"), ("Li", "Yi")].

    Returns:
        Dictionary with statistical evidence including:
        - target_pair_rates: Confusion rates for target pairs
        - other_pair_rates: Confusion rates for other pairs
        - t_test: T-test results comparing target vs other pairs
        - effect_size: Cohen's d
        - conclusion: String summary
    """
    if target_pairs is None:
        target_pairs = [("Miao", "Dong"), ("Li", "Yi")]

    # Generate all non-diagonal pairs
    all_pairs = []
    for i, cls_a in enumerate(ETHNIC_GROUPS):
        for j, cls_b in enumerate(ETHNIC_GROUPS):
            if i < j:  # Avoid duplicates and diagonal
                all_pairs.append((cls_a, cls_b))

    # Separate target and other pairs
    target_set = set((tuple(sorted(p)) for p in target_pairs))
    other_pairs = [p for p in all_pairs if tuple(sorted(p)) not in target_set]

    # Collect confusion rates across all models
    target_rates = []
    other_rates = []

    for model_name, cm in confusion_matrices.items():
        cm = np.array(cm)

        # Target pairs
        for pair in target_pairs:
            rate = get_confusion_rate(cm, pair[0], pair[1], bidirectional=True)
            target_rates.append(rate)

        # Other pairs
        for pair in other_pairs:
            rate = get_confusion_rate(cm, pair[0], pair[1], bidirectional=True)
            other_rates.append(rate)

    # Statistical tests
    results = {
        "hypothesis": "H3: Miao-Dong and Li-Yi pairs show higher confusion",
        "target_pairs": [f"{p[0]}-{p[1]}" for p in target_pairs],
        "n_models": len(confusion_matrices),
        "target_pair_stats": {
            "mean": float(np.mean(target_rates)),
            "std": float(np.std(target_rates)),
            "min": float(np.min(target_rates)),
            "max": float(np.max(target_rates)),
            "n_samples": len(target_rates),
        },
        "other_pair_stats": {
            "mean": float(np.mean(other_rates)),
            "std": float(np.std(other_rates)),
            "min": float(np.min(other_rates)),
            "max": float(np.max(other_rates)),
            "n_samples": len(other_rates),
        },
    }

    # T-test: target pairs vs other pairs
    if len(target_rates) >= 2 and len(other_rates) >= 2:
        t_stat, p_value = stats.ttest_ind(target_rates, other_rates, equal_var=False)
        results["t_test"] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_at_005": p_value < 0.05,
            "significant_at_001": p_value < 0.01,
        }

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(target_rates) - 1) * np.var(target_rates, ddof=1) +
             (len(other_rates) - 1) * np.var(other_rates, ddof=1)) /
            (len(target_rates) + len(other_rates) - 2)
        )
        if pooled_std > 0:
            cohens_d = (np.mean(target_rates) - np.mean(other_rates)) / pooled_std
        else:
            cohens_d = 0.0

        results["effect_size"] = {
            "cohens_d": float(cohens_d),
            "interpretation": _interpret_cohens_d(cohens_d),
        }

        # Mann-Whitney U (non-parametric alternative)
        u_stat, u_pvalue = stats.mannwhitneyu(
            target_rates, other_rates, alternative="greater"
        )
        results["mann_whitney_u"] = {
            "u_statistic": float(u_stat),
            "p_value": float(u_pvalue),
            "significant_at_005": u_pvalue < 0.05,
        }

    # Per-pair breakdown
    pair_breakdown = {}
    for pair in target_pairs:
        pair_key = f"{pair[0]}-{pair[1]}"
        rates = []
        for model_name, cm in confusion_matrices.items():
            rate = get_confusion_rate(np.array(cm), pair[0], pair[1])
            rates.append({"model": model_name, "rate": rate})
        pair_breakdown[pair_key] = {
            "rates": rates,
            "mean": float(np.mean([r["rate"] for r in rates])),
            "std": float(np.std([r["rate"] for r in rates])),
        }
    results["pair_breakdown"] = pair_breakdown

    # Generate conclusion
    target_mean = results["target_pair_stats"]["mean"]
    other_mean = results["other_pair_stats"]["mean"]
    p_value = results.get("t_test", {}).get("p_value", 1.0)

    if p_value < 0.05 and target_mean > other_mean:
        conclusion = (
            f"H3 SUPPORTED: Target pairs (Miao-Dong, Li-Yi) show significantly higher "
            f"confusion ({target_mean:.3f}) compared to other pairs ({other_mean:.3f}), "
            f"p={p_value:.4f}, d={results.get('effect_size', {}).get('cohens_d', 0):.2f}"
        )
    elif target_mean > other_mean:
        conclusion = (
            f"H3 PARTIALLY SUPPORTED: Target pairs show higher confusion "
            f"({target_mean:.3f} vs {other_mean:.3f}) but not statistically significant "
            f"(p={p_value:.4f})"
        )
    else:
        conclusion = (
            f"H3 NOT SUPPORTED: Target pairs ({target_mean:.3f}) do not show higher "
            f"confusion than other pairs ({other_mean:.3f})"
        )

    results["conclusion"] = conclusion

    logger.info(f"H3 Test: {conclusion}")

    return results


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def analyze_geographic_confusion(
    confusion_matrices: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Analyze confusion patterns based on geographic proximity.

    Geographic groupings:
    - Southwest China: Miao, Dong, Yi (Guizhou, Yunnan, Sichuan)
    - South China: Li (Hainan) and Tibetan (Tibet plateau)

    Args:
        confusion_matrices: Dictionary mapping model_name to confusion matrix.

    Returns:
        Analysis of within-region vs between-region confusion.
    """
    # Geographic groupings
    southwest = ["Miao", "Dong", "Yi"]
    south = ["Li", "Tibetan"]

    within_region_rates = []
    between_region_rates = []

    for model_name, cm in confusion_matrices.items():
        cm = np.array(cm)

        # Within Southwest
        for i, cls_a in enumerate(southwest):
            for j, cls_b in enumerate(southwest):
                if i < j:
                    rate = get_confusion_rate(cm, cls_a, cls_b)
                    within_region_rates.append(rate)

        # Within South
        for i, cls_a in enumerate(south):
            for j, cls_b in enumerate(south):
                if i < j:
                    rate = get_confusion_rate(cm, cls_a, cls_b)
                    within_region_rates.append(rate)

        # Between regions
        for cls_a in southwest:
            for cls_b in south:
                rate = get_confusion_rate(cm, cls_a, cls_b)
                between_region_rates.append(rate)

    # Statistical comparison
    t_stat, p_value = stats.ttest_ind(
        within_region_rates, between_region_rates, equal_var=False
    )

    return {
        "within_region": {
            "mean": float(np.mean(within_region_rates)),
            "std": float(np.std(within_region_rates)),
            "n": len(within_region_rates),
        },
        "between_region": {
            "mean": float(np.mean(between_region_rates)),
            "std": float(np.std(between_region_rates)),
            "n": len(between_region_rates),
        },
        "t_test": {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        },
        "interpretation": (
            "Within-region confusion is higher"
            if np.mean(within_region_rates) > np.mean(between_region_rates)
            else "Between-region confusion is higher"
        ),
    }


def export_confusion_analysis(
    analysis_results: Dict[str, Any],
    output_path: str,
    format: str = "json",
) -> None:
    """Export confusion analysis results.

    Args:
        analysis_results: Analysis results dictionary.
        output_path: Output file path.
        format: Output format ('json' or 'csv').
    """
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
    elif format == "csv":
        if "pair_breakdown" in analysis_results:
            rows = []
            for pair, data in analysis_results["pair_breakdown"].items():
                for rate_info in data["rates"]:
                    rows.append({
                        "pair": pair,
                        "model": rate_info["model"],
                        "confusion_rate": rate_info["rate"],
                    })
            pd.DataFrame(rows).to_csv(output_path, index=False)

    logger.info(f"Exported confusion analysis to {output_path}")
