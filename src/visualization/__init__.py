"""Visualization and figure generation for paper."""

from src.visualization.figures import (
    create_all_figures,
    plot_accuracy_comparison,
    plot_confusion_heatmap,
    plot_cultural_coverage_distribution,
    plot_language_effect,
    plot_model_scaling,
    plot_obi_summary,
    plot_per_class_accuracy,
)

__all__ = [
    "plot_accuracy_comparison",
    "plot_confusion_heatmap",
    "plot_language_effect",
    "plot_obi_summary",
    "plot_model_scaling",
    "plot_per_class_accuracy",
    "plot_cultural_coverage_distribution",
    "create_all_figures",
]
