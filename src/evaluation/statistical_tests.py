"""Statistical tests for VLM evaluation analysis."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# =============================================================================
# Basic Statistical Tests
# =============================================================================


def two_sample_ttest(
    group1_scores: List[float],
    group2_scores: List[float],
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Perform two-sample t-test (Welch's t-test by default).

    Args:
        group1_scores: Scores from first group.
        group2_scores: Scores from second group.
        equal_var: If True, assume equal variance (Student's t-test).
                   If False, use Welch's t-test (recommended).
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        Tuple of (t_statistic, p_value).
    """
    if len(group1_scores) < 2 or len(group2_scores) < 2:
        logger.warning("Insufficient samples for t-test")
        return (np.nan, np.nan)

    result = stats.ttest_ind(
        group1_scores,
        group2_scores,
        equal_var=equal_var,
        alternative=alternative,
    )

    logger.debug(
        f"T-test: t={result.statistic:.4f}, p={result.pvalue:.4f}, "
        f"n1={len(group1_scores)}, n2={len(group2_scores)}"
    )

    return (float(result.statistic), float(result.pvalue))


def paired_ttest(
    scores1: List[float],
    scores2: List[float],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Perform paired t-test for dependent samples.

    Useful for comparing same model with different prompts (zh vs en).

    Args:
        scores1: First set of paired scores.
        scores2: Second set of paired scores.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        Tuple of (t_statistic, p_value).
    """
    if len(scores1) != len(scores2):
        raise ValueError("Paired t-test requires equal length arrays")

    if len(scores1) < 2:
        logger.warning("Insufficient samples for paired t-test")
        return (np.nan, np.nan)

    result = stats.ttest_rel(scores1, scores2, alternative=alternative)

    return (float(result.statistic), float(result.pvalue))


def mann_whitney_u(
    group1_scores: List[float],
    group2_scores: List[float],
    alternative: str = "two-sided",
) -> Tuple[float, float]:
    """Perform Mann-Whitney U test (non-parametric).

    Use when normality assumption is violated.

    Args:
        group1_scores: Scores from first group.
        group2_scores: Scores from second group.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        Tuple of (U_statistic, p_value).
    """
    if len(group1_scores) < 2 or len(group2_scores) < 2:
        logger.warning("Insufficient samples for Mann-Whitney U test")
        return (np.nan, np.nan)

    result = stats.mannwhitneyu(
        group1_scores,
        group2_scores,
        alternative=alternative,
    )

    return (float(result.statistic), float(result.pvalue))


# =============================================================================
# ANOVA Tests
# =============================================================================


def one_way_anova(*groups: List[float]) -> Tuple[float, float]:
    """Perform one-way ANOVA.

    Args:
        *groups: Variable number of score lists (one per group).

    Returns:
        Tuple of (F_statistic, p_value).
    """
    # Filter out groups with insufficient samples
    valid_groups = [g for g in groups if len(g) >= 2]

    if len(valid_groups) < 2:
        logger.warning("Insufficient groups for ANOVA")
        return (np.nan, np.nan)

    result = stats.f_oneway(*valid_groups)

    return (float(result.statistic), float(result.pvalue))


def kruskal_wallis(*groups: List[float]) -> Tuple[float, float]:
    """Perform Kruskal-Wallis H-test (non-parametric ANOVA).

    Args:
        *groups: Variable number of score lists.

    Returns:
        Tuple of (H_statistic, p_value).
    """
    valid_groups = [g for g in groups if len(g) >= 2]

    if len(valid_groups) < 2:
        logger.warning("Insufficient groups for Kruskal-Wallis test")
        return (np.nan, np.nan)

    result = stats.kruskal(*valid_groups)

    return (float(result.statistic), float(result.pvalue))


def anova_2x2(
    data: List[Dict[str, Any]],
    factor1: str,
    factor2: str,
    value_key: str = "accuracy",
) -> Dict[str, Tuple[float, float]]:
    """Perform 2x2 ANOVA for analyzing factor interactions.

    Useful for analyzing Origin × Language interaction.

    Args:
        data: List of dictionaries with factor levels and values.
              Example: [{"origin": "chinese", "language": "zh", "accuracy": 0.75}, ...]
        factor1: Name of first factor (e.g., "origin").
        factor2: Name of second factor (e.g., "language").
        value_key: Key for the dependent variable.

    Returns:
        Dictionary with F-statistics and p-values for:
        - factor1: Main effect of factor 1
        - factor2: Main effect of factor 2
        - interaction: Interaction effect

    Note:
        This is a simplified implementation using manual calculation.
        For production use, consider statsmodels for full ANOVA table.
    """
    # Extract values by factor combinations
    groups = {}
    for item in data:
        f1 = item.get(factor1)
        f2 = item.get(factor2)
        value = item.get(value_key)

        if f1 is None or f2 is None or value is None:
            continue

        key = (f1, f2)
        if key not in groups:
            groups[key] = []
        groups[key].append(value)

    # Get unique levels
    f1_levels = sorted(set(k[0] for k in groups.keys()))
    f2_levels = sorted(set(k[1] for k in groups.keys()))

    if len(f1_levels) != 2 or len(f2_levels) != 2:
        logger.warning(f"Expected 2x2 design, got {len(f1_levels)}x{len(f2_levels)}")

    # Calculate main effects using marginal means
    # Factor 1 main effect
    f1_group1 = []
    f1_group2 = []
    for key, values in groups.items():
        if key[0] == f1_levels[0]:
            f1_group1.extend(values)
        else:
            f1_group2.extend(values)

    f1_result = two_sample_ttest(f1_group1, f1_group2)

    # Factor 2 main effect
    f2_group1 = []
    f2_group2 = []
    for key, values in groups.items():
        if key[1] == f2_levels[0]:
            f2_group1.extend(values)
        else:
            f2_group2.extend(values)

    f2_result = two_sample_ttest(f2_group1, f2_group2)

    # Interaction effect (simplified: compare differences)
    # Calculate difference scores for each level of factor1
    try:
        diff1 = [
            np.mean(groups.get((f1_levels[0], f2_levels[0]), [0])) -
            np.mean(groups.get((f1_levels[0], f2_levels[1]), [0]))
        ]
        diff2 = [
            np.mean(groups.get((f1_levels[1], f2_levels[0]), [0])) -
            np.mean(groups.get((f1_levels[1], f2_levels[1]), [0]))
        ]

        # Simple interaction test: compare the differences
        interaction_diff = abs(diff1[0] - diff2[0])
        # Approximate p-value using permutation (simplified)
        all_values = [v for vals in groups.values() for v in vals]
        grand_mean = np.mean(all_values)
        grand_std = np.std(all_values)

        if grand_std > 0:
            interaction_z = interaction_diff / (grand_std / np.sqrt(len(all_values)))
            interaction_p = 2 * (1 - stats.norm.cdf(abs(interaction_z)))
        else:
            interaction_z = 0
            interaction_p = 1.0

        interaction_result = (float(interaction_z), float(interaction_p))
    except Exception as e:
        logger.warning(f"Could not calculate interaction: {e}")
        interaction_result = (np.nan, np.nan)

    return {
        factor1: f1_result,
        factor2: f2_result,
        "interaction": interaction_result,
    }


def full_factorial_anova(
    data: List[Dict[str, Any]],
    factors: List[str],
    value_key: str = "accuracy",
) -> Dict[str, Any]:
    """Perform full factorial ANOVA using statsmodels (if available).

    Args:
        data: List of dictionaries with factor levels and values.
        factors: List of factor names.
        value_key: Key for the dependent variable.

    Returns:
        ANOVA table as dictionary.
    """
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        df = pd.DataFrame(data)

        # Build formula
        formula = f"{value_key} ~ " + " * ".join([f"C({f})" for f in factors])

        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        return {
            "anova_table": anova_table.to_dict(),
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
        }

    except ImportError:
        logger.warning("statsmodels not available, using simplified ANOVA")
        if len(factors) == 2:
            return anova_2x2(data, factors[0], factors[1], value_key)
        else:
            return {"error": "Full factorial ANOVA requires statsmodels"}


# =============================================================================
# Post-hoc Tests
# =============================================================================


def tukey_hsd(
    *groups: List[float],
    group_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Perform Tukey's HSD post-hoc test.

    Args:
        *groups: Variable number of score lists.
        group_names: Names for each group.

    Returns:
        Dictionary with pairwise comparisons.
    """
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        # Prepare data
        all_values = []
        all_groups = []

        if group_names is None:
            group_names = [f"Group_{i}" for i in range(len(groups))]

        for name, values in zip(group_names, groups):
            all_values.extend(values)
            all_groups.extend([name] * len(values))

        result = pairwise_tukeyhsd(all_values, all_groups)

        return {
            "summary": str(result),
            "reject": result.reject.tolist(),
            "meandiffs": result.meandiffs.tolist(),
            "pvalues": result.pvalues.tolist() if hasattr(result, "pvalues") else None,
        }

    except ImportError:
        logger.warning("statsmodels not available for Tukey HSD")
        return {"error": "statsmodels required for Tukey HSD"}


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values to correct.
        alpha: Significance level.

    Returns:
        Dictionary with corrected p-values and significance.
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    corrected_p = [min(p * n, 1.0) for p in p_values]
    significant = [p < corrected_alpha for p in p_values]

    return {
        "original_p_values": p_values,
        "corrected_p_values": corrected_p,
        "corrected_alpha": corrected_alpha,
        "significant": significant,
        "n_comparisons": n,
    }


# =============================================================================
# Effect Size Measures
# =============================================================================


def cohens_d(
    group1_scores: List[float],
    group2_scores: List[float],
) -> float:
    """Calculate Cohen's d effect size.

    Args:
        group1_scores: Scores from first group.
        group2_scores: Scores from second group.

    Returns:
        Cohen's d value.

    Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
    """
    n1 = len(group1_scores)
    n2 = len(group2_scores)

    if n1 < 2 or n2 < 2:
        return np.nan

    mean1 = np.mean(group1_scores)
    mean2 = np.mean(group2_scores)
    var1 = np.var(group1_scores, ddof=1)
    var2 = np.var(group2_scores, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean1 - mean2) / pooled_std

    return float(d)


def eta_squared(
    ss_between: float,
    ss_total: float,
) -> float:
    """Calculate eta-squared effect size for ANOVA.

    Args:
        ss_between: Sum of squares between groups.
        ss_total: Total sum of squares.

    Returns:
        Eta-squared value.
    """
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


# =============================================================================
# Bootstrap Methods
# =============================================================================


def bootstrap_confidence_interval(
    data: List[float],
    statistic_func: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Calculate bootstrap confidence interval for a statistic.

    Args:
        data: Sample data.
        statistic_func: Function to compute statistic (default: mean).
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (e.g., 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, lower_ci, upper_ci).
    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.array(data)
    point_estimate = statistic_func(data)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(sample))

    # Calculate CI using percentile method
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return (float(point_estimate), float(lower), float(upper))


def bootstrap_difference_test(
    group1: List[float],
    group2: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Test difference between groups using bootstrap.

    Args:
        group1: First group scores.
        group2: Second group scores.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level.

    Returns:
        Dictionary with difference statistics.
    """
    observed_diff = np.mean(group1) - np.mean(group2)

    # Bootstrap the difference
    diff_samples = []
    for _ in range(n_bootstrap):
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        diff_samples.append(np.mean(sample1) - np.mean(sample2))

    alpha = 1 - confidence_level
    lower = np.percentile(diff_samples, 100 * alpha / 2)
    upper = np.percentile(diff_samples, 100 * (1 - alpha / 2))

    # P-value: proportion of bootstrap samples with opposite sign
    if observed_diff >= 0:
        p_value = 2 * np.mean(np.array(diff_samples) <= 0)
    else:
        p_value = 2 * np.mean(np.array(diff_samples) >= 0)

    return {
        "observed_difference": float(observed_diff),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "p_value": float(min(p_value, 1.0)),
        "significant": lower > 0 or upper < 0,  # CI doesn't include 0
    }


# =============================================================================
# McNemar's Test for Paired Comparisons
# =============================================================================


def mcnemar_test(
    y_pred1: List[str],
    y_pred2: List[str],
    y_true: List[str],
) -> Tuple[float, float]:
    """Perform McNemar's test for comparing two classifiers.

    Tests whether two classifiers have significantly different error rates.

    Args:
        y_pred1: Predictions from first classifier.
        y_pred2: Predictions from second classifier.
        y_true: Ground truth labels.

    Returns:
        Tuple of (chi_squared, p_value).
    """
    if len(y_pred1) != len(y_pred2) != len(y_true):
        raise ValueError("All input lists must have same length")

    # Build contingency table
    # b: pred1 correct, pred2 wrong
    # c: pred1 wrong, pred2 correct
    b = 0
    c = 0

    for p1, p2, true in zip(y_pred1, y_pred2, y_true):
        p1_correct = (p1 == true)
        p2_correct = (p2 == true)

        if p1_correct and not p2_correct:
            b += 1
        elif not p1_correct and p2_correct:
            c += 1

    # McNemar's test with continuity correction
    if b + c == 0:
        return (0.0, 1.0)

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    logger.debug(f"McNemar's test: b={b}, c={c}, chi2={chi2:.4f}, p={p_value:.4f}")

    return (float(chi2), float(p_value))


# =============================================================================
# Chi-Square Tests
# =============================================================================


def chi_square_independence(
    observed: np.ndarray,
) -> Tuple[float, float, int]:
    """Perform chi-square test of independence.

    Args:
        observed: Observed frequency table (2D array).

    Returns:
        Tuple of (chi_squared, p_value, degrees_of_freedom).
    """
    result = stats.chi2_contingency(observed)

    return (float(result[0]), float(result[1]), int(result[2]))


def chi_square_goodness_of_fit(
    observed: List[int],
    expected: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """Perform chi-square goodness of fit test.

    Args:
        observed: Observed frequencies.
        expected: Expected frequencies (uniform if None).

    Returns:
        Tuple of (chi_squared, p_value).
    """
    if expected is None:
        # Assume uniform distribution
        total = sum(observed)
        expected = [total / len(observed)] * len(observed)

    result = stats.chisquare(observed, expected)

    return (float(result.statistic), float(result.pvalue))
