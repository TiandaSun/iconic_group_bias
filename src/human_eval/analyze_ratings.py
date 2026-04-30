"""Analyze human evaluation ratings and compute inter-rater reliability.

This script processes collected evaluation data and computes:
- Inter-rater reliability (Krippendorff's Alpha)
- Mean scores by model, ethnic group, and dimension
- Statistical comparisons between models
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse


def load_ratings(ratings_file: str) -> pd.DataFrame:
    """Load ratings from JSON or CSV file.

    Expected format (JSON):
    [
        {
            "sample_id": "...",
            "evaluator_id": "...",
            "cultural_accuracy": 4,
            "visual_completeness": 3,
            "terminology": 4,
            "factual_correctness": 3,
            "overall_quality": 4,
            ...
        },
        ...
    ]

    Args:
        ratings_file: Path to ratings file.

    Returns:
        DataFrame with ratings data.
    """
    path = Path(ratings_file)

    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif path.suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def krippendorff_alpha(
    data: np.ndarray,
    level: str = "ordinal",
) -> float:
    """Compute Krippendorff's Alpha for inter-rater reliability.

    Args:
        data: 2D array of shape (n_raters, n_items).
              Missing values should be np.nan.
        level: Measurement level ('nominal', 'ordinal', 'interval', 'ratio').

    Returns:
        Krippendorff's Alpha coefficient.
    """

    def nominal_metric(a, b):
        return 0 if a == b else 1

    def ordinal_metric(a, b, values):
        a_idx = values.index(a)
        b_idx = values.index(b)
        return sum(1 for v in values[min(a_idx, b_idx):max(a_idx, b_idx) + 1]) - 1

    def interval_metric(a, b):
        return (a - b) ** 2

    # Remove columns with all NaN
    valid_cols = ~np.all(np.isnan(data), axis=0)
    data = data[:, valid_cols]

    if data.shape[1] == 0:
        return np.nan

    n_raters, n_items = data.shape

    # Get all unique values
    all_values = sorted(set(data[~np.isnan(data)].flatten()))

    if len(all_values) < 2:
        return 1.0  # Perfect agreement if only one value

    # Compute observed disagreement
    Do = 0
    n_pairs_o = 0

    for j in range(n_items):
        col = data[:, j]
        valid = col[~np.isnan(col)]
        n_valid = len(valid)

        if n_valid < 2:
            continue

        for i in range(n_valid):
            for k in range(i + 1, n_valid):
                if level == "nominal":
                    Do += nominal_metric(valid[i], valid[k])
                elif level == "ordinal":
                    Do += ordinal_metric(valid[i], valid[k], all_values) ** 2
                elif level in ["interval", "ratio"]:
                    Do += interval_metric(valid[i], valid[k])
                n_pairs_o += 1

    if n_pairs_o == 0:
        return np.nan

    Do /= n_pairs_o

    # Compute expected disagreement
    all_valid = data[~np.isnan(data)].flatten()
    n_total = len(all_valid)

    De = 0
    n_pairs_e = 0

    for i, vi in enumerate(all_valid):
        for vj in all_valid[i + 1:]:
            if level == "nominal":
                De += nominal_metric(vi, vj)
            elif level == "ordinal":
                De += ordinal_metric(vi, vj, all_values) ** 2
            elif level in ["interval", "ratio"]:
                De += interval_metric(vi, vj)
            n_pairs_e += 1

    if n_pairs_e == 0 or De == 0:
        return 1.0

    De /= n_pairs_e

    alpha = 1 - Do / De
    return alpha


def compute_reliability(
    df: pd.DataFrame,
    dimensions: List[str],
) -> Dict[str, float]:
    """Compute inter-rater reliability for each dimension.

    Args:
        df: DataFrame with ratings.
        dimensions: List of dimension column names.

    Returns:
        Dictionary mapping dimension names to alpha values.
    """
    reliability = {}

    evaluators = df["evaluator_id"].unique()
    samples = df["sample_id"].unique()

    for dim in dimensions:
        # Create rater × item matrix
        matrix = np.full((len(evaluators), len(samples)), np.nan)

        for i, evaluator in enumerate(evaluators):
            for j, sample in enumerate(samples):
                row = df[(df["evaluator_id"] == evaluator) & (df["sample_id"] == sample)]
                if len(row) > 0 and dim in row.columns:
                    value = row[dim].values[0]
                    if pd.notna(value):
                        matrix[i, j] = value

        alpha = krippendorff_alpha(matrix, level="ordinal")
        reliability[dim] = alpha

    return reliability


def compute_mean_scores(
    df: pd.DataFrame,
    dimensions: List[str],
    group_by: str,
) -> pd.DataFrame:
    """Compute mean scores grouped by a variable.

    Args:
        df: DataFrame with ratings.
        dimensions: List of dimension column names.
        group_by: Column to group by (e.g., 'model_name', 'ethnic_group').

    Returns:
        DataFrame with mean scores and standard deviations.
    """
    results = []

    for group in df[group_by].unique():
        group_df = df[df[group_by] == group]

        row = {group_by: group}
        for dim in dimensions:
            if dim in group_df.columns:
                values = group_df[dim].dropna()
                row[f"{dim}_mean"] = values.mean()
                row[f"{dim}_std"] = values.std()
                row[f"{dim}_n"] = len(values)

        results.append(row)

    return pd.DataFrame(results)


def friedman_test(
    df: pd.DataFrame,
    dimension: str,
    group_var: str = "model_name",
) -> Tuple[float, float]:
    """Perform Friedman test for comparing multiple groups.

    Args:
        df: DataFrame with ratings.
        dimension: Dimension to compare.
        group_var: Variable defining groups.

    Returns:
        Tuple of (chi-square statistic, p-value).
    """
    try:
        from scipy.stats import friedmanchisquare

        groups = df[group_var].unique()
        samples = df["sample_id"].unique()

        # Build data for Friedman test
        data = []
        for sample in samples:
            sample_df = df[df["sample_id"] == sample]
            row = []
            for group in groups:
                group_sample = sample_df[sample_df[group_var] == group]
                if len(group_sample) > 0 and dimension in group_sample.columns:
                    row.append(group_sample[dimension].mean())
                else:
                    row.append(np.nan)
            data.append(row)

        data = np.array(data)

        # Remove rows with NaN
        valid_rows = ~np.any(np.isnan(data), axis=1)
        data = data[valid_rows]

        if len(data) < 3:
            return np.nan, np.nan

        stat, p_value = friedmanchisquare(*data.T)
        return stat, p_value

    except ImportError:
        print("Warning: scipy not available for Friedman test")
        return np.nan, np.nan


def generate_report(
    df: pd.DataFrame,
    samples_metadata: List[Dict],
    output_dir: str,
) -> None:
    """Generate comprehensive evaluation report.

    Args:
        df: DataFrame with ratings.
        samples_metadata: Original sample metadata.
        output_dir: Output directory for report.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dimensions = [
        "cultural_accuracy",
        "visual_completeness",
        "terminology",
        "factual_correctness",
        "overall_quality",
    ]

    # Add metadata to ratings
    samples_df = pd.DataFrame(samples_metadata)
    if "sample_id" in samples_df.columns and "sample_id" in df.columns:
        df = df.merge(
            samples_df[["sample_id", "model_name", "ethnic_group", "language"]],
            on="sample_id",
            how="left",
        )

    report = []
    report.append("=" * 70)
    report.append("HUMAN EVALUATION ANALYSIS REPORT")
    report.append("=" * 70)
    report.append("")

    # Basic statistics
    report.append("1. BASIC STATISTICS")
    report.append("-" * 40)
    report.append(f"Total ratings: {len(df)}")
    report.append(f"Unique samples: {df['sample_id'].nunique()}")
    report.append(f"Unique evaluators: {df['evaluator_id'].nunique()}")
    report.append("")

    # Inter-rater reliability
    report.append("2. INTER-RATER RELIABILITY (Krippendorff's Alpha)")
    report.append("-" * 40)
    reliability = compute_reliability(df, dimensions)
    for dim, alpha in reliability.items():
        status = "✓ Good" if alpha > 0.6 else "⚠ Low"
        report.append(f"  {dim}: α = {alpha:.3f} {status}")
    report.append("")

    # Mean scores by model
    if "model_name" in df.columns:
        report.append("3. MEAN SCORES BY MODEL")
        report.append("-" * 40)
        model_scores = compute_mean_scores(df, dimensions, "model_name")
        for _, row in model_scores.iterrows():
            report.append(f"\n  {row['model_name']}:")
            for dim in dimensions:
                mean_col = f"{dim}_mean"
                std_col = f"{dim}_std"
                if mean_col in row:
                    report.append(f"    {dim}: {row[mean_col]:.2f} ± {row[std_col]:.2f}")
        report.append("")

    # Mean scores by ethnic group
    if "ethnic_group" in df.columns:
        report.append("4. MEAN SCORES BY ETHNIC GROUP")
        report.append("-" * 40)
        ethnic_scores = compute_mean_scores(df, dimensions, "ethnic_group")
        for _, row in ethnic_scores.iterrows():
            report.append(f"\n  {row['ethnic_group']}:")
            for dim in dimensions:
                mean_col = f"{dim}_mean"
                std_col = f"{dim}_std"
                if mean_col in row:
                    report.append(f"    {dim}: {row[mean_col]:.2f} ± {row[std_col]:.2f}")
        report.append("")

    # Statistical tests
    report.append("5. STATISTICAL COMPARISONS (Friedman Test)")
    report.append("-" * 40)
    if "model_name" in df.columns:
        for dim in dimensions:
            stat, p = friedman_test(df, dim, "model_name")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            report.append(f"  {dim}: χ² = {stat:.2f}, p = {p:.4f} {sig}")
    report.append("")

    # Save report
    report_text = "\n".join(report)
    report_file = output_path / "evaluation_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to {report_file}")

    # Save detailed results as CSV
    if "model_name" in df.columns:
        model_scores.to_csv(output_path / "scores_by_model.csv", index=False)
    if "ethnic_group" in df.columns:
        ethnic_scores.to_csv(output_path / "scores_by_ethnic_group.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze human evaluation ratings")
    parser.add_argument("--ratings", type=str, required=True,
                        help="Path to ratings file (JSON or CSV)")
    parser.add_argument("--samples", type=str, default=None,
                        help="Path to samples metadata JSON")
    parser.add_argument("--output-dir", type=str, default="results/human_eval/analysis",
                        help="Output directory for report")

    args = parser.parse_args()

    print("Loading ratings...")
    df = load_ratings(args.ratings)
    print(f"Loaded {len(df)} ratings")

    samples_metadata = []
    if args.samples:
        with open(args.samples) as f:
            samples_metadata = json.load(f)
        print(f"Loaded {len(samples_metadata)} sample metadata entries")

    print("\nGenerating report...")
    generate_report(df, samples_metadata, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
