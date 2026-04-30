"""Evaluation metrics and statistical tests."""

from src.evaluation.confusion_analysis import (
    aggregate_confusion_by_origin,
    analyze_geographic_confusion,
    confusion_asymmetry,
    export_confusion_analysis,
    generate_confusion_report,
    get_confusion_rate,
    identify_confusion_pairs,
    test_hypothesis_h3,
)
from src.evaluation.metrics import (
    batch_cultural_term_coverage,
    build_confusion_matrix,
    calculate_all_classification_metrics,
    classification_accuracy,
    confusion_matrix_to_dict,
    cultural_term_coverage,
    language_effect_score,
    load_cultural_vocabulary,
    macro_f1_score,
    origin_bias_index,
    per_class_accuracy,
    precision_recall_per_class,
    term_frequency_analysis,
    weighted_f1_score,
)
from src.evaluation.statistical_tests import (
    anova_2x2,
    bonferroni_correction,
    bootstrap_confidence_interval,
    bootstrap_difference_test,
    chi_square_goodness_of_fit,
    chi_square_independence,
    cohens_d,
    kruskal_wallis,
    mann_whitney_u,
    mcnemar_test,
    one_way_anova,
    paired_ttest,
    tukey_hsd,
    two_sample_ttest,
)

__all__ = [
    # Classification metrics
    "classification_accuracy",
    "per_class_accuracy",
    "macro_f1_score",
    "weighted_f1_score",
    "precision_recall_per_class",
    "build_confusion_matrix",
    "confusion_matrix_to_dict",
    "calculate_all_classification_metrics",
    # Bias metrics
    "origin_bias_index",
    "language_effect_score",
    # Cultural term coverage
    "cultural_term_coverage",
    "batch_cultural_term_coverage",
    "term_frequency_analysis",
    "load_cultural_vocabulary",
    # Confusion analysis
    "identify_confusion_pairs",
    "confusion_asymmetry",
    "get_confusion_rate",
    "aggregate_confusion_by_origin",
    "generate_confusion_report",
    "test_hypothesis_h3",
    "analyze_geographic_confusion",
    "export_confusion_analysis",
    # Statistical tests
    "two_sample_ttest",
    "paired_ttest",
    "mann_whitney_u",
    "one_way_anova",
    "kruskal_wallis",
    "anova_2x2",
    "mcnemar_test",
    "chi_square_independence",
    "chi_square_goodness_of_fit",
    # Post-hoc tests
    "tukey_hsd",
    "bonferroni_correction",
    # Effect size
    "cohens_d",
    # Bootstrap methods
    "bootstrap_confidence_interval",
    "bootstrap_difference_test",
]
