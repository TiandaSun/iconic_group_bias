"""Publication-ready figure generation for VLM evaluation.

All figures follow academic publication standards with:
- Seaborn style for clean aesthetics
- Appropriate font sizes for readability
- Dual output: PNG (300 DPI) and PDF
- Colorblind-friendly palettes where possible
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set global style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.2)

# Color palettes
ORIGIN_COLORS = {
    "chinese": "#E67E22",  # Orange for Chinese-origin models
    "western": "#3498DB",  # Blue for Western-origin models
}

LANGUAGE_COLORS = {
    "zh": "#E74C3C",  # Red for Chinese prompts
    "en": "#2ECC71",  # Green for English prompts
}

# Model display names (shorter for plots)
MODEL_DISPLAY_NAMES = {
    "qwen2.5-vl-72b": "Qwen-72B",
    "qwen2.5-vl-7b": "Qwen-7B",
    "llama-3.2-vision-11b": "LLaMA-11B",
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-2.5-flash": "Gemini-Flash",
    "claude-3.5-sonnet": "Claude-Sonnet",
    "claude-haiku-4.5": "Claude-Haiku",
}

# Ethnic group labels
ETHNIC_GROUPS = ["Miao", "Dong", "Yi", "Li", "Tibetan"]
ETHNIC_GROUPS_ZH = ["苗族", "侗族", "彝族", "黎族", "藏族"]


def _save_figure(
    fig: plt.Figure,
    output_path: Union[str, Path],
    dpi: int = 300,
    save_pdf: bool = True,
) -> None:
    """Save figure as PNG and optionally PDF.

    Args:
        fig: Matplotlib figure object.
        output_path: Base output path (without extension).
        dpi: DPI for PNG output.
        save_pdf: Whether to also save as PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove extension if present
    if output_path.suffix in [".png", ".pdf"]:
        output_path = output_path.with_suffix("")

    # Save PNG
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved figure: {png_path}")

    # Save PDF
    if save_pdf:
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved figure: {pdf_path}")


def _get_display_name(model_name: str) -> str:
    """Get display name for model."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def plot_accuracy_comparison(
    results_df: pd.DataFrame,
    output_path: Union[str, Path],
    figsize: Tuple[float, float] = (12, 6),
    show_error_bars: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """Create grouped bar chart comparing model accuracies.

    Args:
        results_df: DataFrame with columns: model, origin, language, accuracy
                    Optional: accuracy_std for error bars
        output_path: Path to save the figure.
        figsize: Figure size (width, height).
        show_error_bars: Whether to show error bars if std available.
        title: Optional custom title.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    df = results_df.copy()
    df["model_display"] = df["model"].apply(_get_display_name)

    # Get unique models in order (Chinese first, then Western)
    chinese_models = df[df["origin"] == "chinese"]["model_display"].unique()
    western_models = df[df["origin"] == "western"]["model_display"].unique()
    model_order = list(chinese_models) + list(western_models)

    # Create positions
    n_models = len(model_order)
    n_languages = df["language"].nunique()
    bar_width = 0.35
    positions = np.arange(n_models)

    # Plot bars for each language
    languages = sorted(df["language"].unique())
    for i, lang in enumerate(languages):
        lang_data = df[df["language"] == lang].set_index("model_display")

        # Reorder to match model_order
        accuracies = [lang_data.loc[m, "accuracy"] if m in lang_data.index else 0
                      for m in model_order]

        # Error bars
        if show_error_bars and "accuracy_std" in df.columns:
            errors = [lang_data.loc[m, "accuracy_std"] if m in lang_data.index else 0
                      for m in model_order]
        else:
            errors = None

        # Determine colors based on origin
        colors = []
        for m in model_order:
            origin = df[df["model_display"] == m]["origin"].iloc[0]
            # Lighter shade for English
            base_color = ORIGIN_COLORS.get(origin, "#888888")
            if lang == "en":
                colors.append(_lighten_color(base_color, 0.3))
            else:
                colors.append(base_color)

        offset = (i - (n_languages - 1) / 2) * bar_width
        bars = ax.bar(
            positions + offset,
            accuracies,
            bar_width * 0.9,
            label=f"{'Chinese' if lang == 'zh' else 'English'} Prompt",
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            yerr=errors,
            capsize=3,
        )

    # Customize plot
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Add legend
    ax.legend(loc="upper right", framealpha=0.9)

    # Add origin labels
    ax.axvline(x=len(chinese_models) - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(
        len(chinese_models) / 2 - 0.5, 0.02,
        "Chinese-origin", ha="center", fontsize=9, style="italic", alpha=0.7
    )
    ax.text(
        len(chinese_models) + len(western_models) / 2 - 0.5, 0.02,
        "Western-origin", ha="center", fontsize=9, style="italic", alpha=0.7
    )

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    else:
        ax.set_title(
            "Classification Accuracy by Model and Prompt Language",
            fontsize=14, fontweight="bold", pad=15
        )

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


def _lighten_color(color: str, factor: float = 0.3) -> str:
    """Lighten a hex color."""
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color

    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return mc.to_hex(colorsys.hls_to_rgb(c[0], min(1, c[1] + factor), c[2]))


def plot_confusion_heatmap(
    confusion_matrix: np.ndarray,
    classes: Optional[List[str]] = None,
    model_name: str = "",
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 7),
    annotate: bool = True,
    normalize: bool = True,
    highlight_threshold: float = 0.1,
    cmap: str = "Blues",
    language: str = "en",
) -> plt.Figure:
    """Create confusion matrix heatmap.

    Args:
        confusion_matrix: 5×5 confusion matrix array.
        classes: Class labels (default: ethnic group names).
        model_name: Model name for title.
        output_path: Path to save figure (optional).
        figsize: Figure size.
        annotate: Whether to annotate cells with values.
        normalize: Whether to normalize by row (true class).
        highlight_threshold: Highlight off-diagonal cells above this value.
        cmap: Colormap name.
        language: Language for labels ('en' or 'zh').

    Returns:
        Matplotlib figure object.
    """
    cm = np.array(confusion_matrix, dtype=float)

    if classes is None:
        classes = ETHNIC_GROUPS_ZH if language == "zh" else ETHNIC_GROUPS

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion" if normalize else "Count", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    if annotate:
        thresh = cm.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                value = cm[i, j]

                # Highlight high off-diagonal confusion
                if i != j and value >= highlight_threshold:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="red", linewidth=2
                    ))

                text_color = "white" if value > thresh else "black"
                ax.text(
                    j, i, f"{value:.2f}" if normalize else f"{int(value)}",
                    ha="center", va="center", color=text_color, fontsize=10
                )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")

    title = f"Confusion Matrix"
    if model_name:
        title += f" - {_get_display_name(model_name)}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_language_effect(
    les_scores: Dict[str, float],
    output_path: Union[str, Path],
    figsize: Tuple[float, float] = (10, 6),
    sort_by_value: bool = True,
) -> plt.Figure:
    """Create bar chart showing Language Effect Score per model.

    Args:
        les_scores: Dictionary mapping model_name to LES value.
        output_path: Path to save figure.
        figsize: Figure size.
        sort_by_value: Whether to sort bars by LES value.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    models = list(les_scores.keys())
    scores = list(les_scores.values())
    display_names = [_get_display_name(m) for m in models]

    if sort_by_value:
        sorted_idx = np.argsort(scores)[::-1]
        display_names = [display_names[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]
        models = [models[i] for i in sorted_idx]

    # Determine colors (positive = Chinese better, negative = English better)
    colors = [LANGUAGE_COLORS["zh"] if s >= 0 else LANGUAGE_COLORS["en"] for s in scores]

    # Create bars
    positions = np.arange(len(models))
    bars = ax.bar(positions, scores, color=colors, edgecolor="black", linewidth=0.5)

    # Add reference line at 0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Add value labels
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 0.01 if height >= 0 else -0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{score:+.2f}",
            ha="center", va=va, fontsize=9
        )

    # Customize plot
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Language Effect Score (LES)", fontsize=12, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=10)

    # Set y limits symmetrically
    max_abs = max(abs(min(scores)), abs(max(scores))) * 1.2
    ax.set_ylim(-max_abs, max_abs)

    # Add interpretation labels
    ax.text(
        0.98, 0.95, "← Chinese prompt better",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, style="italic", color=LANGUAGE_COLORS["zh"]
    )
    ax.text(
        0.98, 0.05, "← English prompt better",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, style="italic", color=LANGUAGE_COLORS["en"]
    )

    ax.set_title(
        "Language Effect Score by Model\n(LES = (Acc_CN - Acc_EN) / Acc_EN)",
        fontsize=14, fontweight="bold", pad=15
    )

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


def plot_obi_summary(
    obi_by_language: Dict[str, float],
    output_path: Union[str, Path],
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Create bar chart showing OBI for different prompt languages.

    Args:
        obi_by_language: Dictionary mapping language to OBI value.
        output_path: Path to save figure.
        confidence_intervals: Optional CI for each language.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    languages = list(obi_by_language.keys())
    obi_values = list(obi_by_language.values())

    # Display names for languages
    lang_display = {
        "zh": "Chinese Prompt",
        "en": "English Prompt",
        "overall": "Overall",
    }
    display_names = [lang_display.get(l, l) for l in languages]

    # Colors based on OBI direction
    colors = [ORIGIN_COLORS["chinese"] if v >= 0 else ORIGIN_COLORS["western"]
              for v in obi_values]

    # Error bars
    if confidence_intervals:
        errors = [
            [obi_values[i] - confidence_intervals[l][0],
             confidence_intervals[l][1] - obi_values[i]]
            for i, l in enumerate(languages)
        ]
        errors = np.array(errors).T
    else:
        errors = None

    # Create bars
    positions = np.arange(len(languages))
    bars = ax.bar(
        positions, obi_values,
        color=colors, edgecolor="black", linewidth=0.5,
        yerr=errors, capsize=5
    )

    # Reference line at 0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)

    # Value labels
    for bar, value in zip(bars, obi_values):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 0.01 if height >= 0 else -0.01
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{value:+.3f}",
            ha="center", va=va, fontsize=11, fontweight="bold"
        )

    # Customize plot
    ax.set_xlabel("Prompt Language", fontsize=12, fontweight="bold")
    ax.set_ylabel("Origin Bias Index (OBI)", fontsize=12, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(display_names, fontsize=11)

    # Set y limits
    max_abs = max(abs(min(obi_values)), abs(max(obi_values))) * 1.3
    ax.set_ylim(-max_abs, max_abs)

    # Add interpretation
    ax.text(
        0.02, 0.95, "Chinese-origin models\nperform better ↑",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, style="italic", color=ORIGIN_COLORS["chinese"]
    )
    ax.text(
        0.02, 0.05, "Western-origin models\nperform better ↓",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=9, style="italic", color=ORIGIN_COLORS["western"]
    )

    ax.set_title(
        "Origin Bias Index (OBI) by Prompt Language\n"
        "(OBI = (μ_Chinese - μ_Western) / μ_All)",
        fontsize=14, fontweight="bold", pad=15
    )

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


def plot_model_scaling(
    results_7b: Dict[str, float],
    results_72b: Dict[str, float],
    output_path: Union[str, Path],
    figsize: Tuple[float, float] = (10, 6),
    metrics: Optional[List[str]] = None,
) -> plt.Figure:
    """Compare Qwen 7B vs 72B performance to show scaling effects.

    Args:
        results_7b: Metrics for 7B model {metric_name: value}.
        results_72b: Metrics for 72B model {metric_name: value}.
        output_path: Path to save figure.
        figsize: Figure size.
        metrics: List of metrics to plot (default: all common metrics).

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Determine metrics to plot
    if metrics is None:
        metrics = list(set(results_7b.keys()) & set(results_72b.keys()))
        # Prioritize these if available
        priority = ["accuracy", "accuracy_zh", "accuracy_en", "macro_f1"]
        metrics = sorted(metrics, key=lambda x: priority.index(x) if x in priority else 100)
        metrics = metrics[:6]  # Limit to 6 metrics

    # Left plot: Bar comparison
    ax1 = axes[0]
    x = np.arange(len(metrics))
    width = 0.35

    values_7b = [results_7b.get(m, 0) for m in metrics]
    values_72b = [results_72b.get(m, 0) for m in metrics]

    bars1 = ax1.bar(x - width / 2, values_7b, width, label="Qwen-7B",
                    color="#3498DB", edgecolor="black", linewidth=0.5)
    bars2 = ax1.bar(x + width / 2, values_72b, width, label="Qwen-72B",
                    color="#E67E22", edgecolor="black", linewidth=0.5)

    ax1.set_xlabel("Metric", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Performance Comparison", fontsize=12, fontweight="bold")

    # Right plot: Improvement percentage
    ax2 = axes[1]

    improvements = []
    for v7, v72 in zip(values_7b, values_72b):
        if v7 > 0:
            imp = (v72 - v7) / v7 * 100
        else:
            imp = 0
        improvements.append(imp)

    colors = ["#2ECC71" if imp >= 0 else "#E74C3C" for imp in improvements]
    bars = ax2.bar(x, improvements, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 1 if height >= 0 else -1
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            f"{imp:+.1f}%",
            ha="center", va=va, fontsize=9
        )

    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Metric", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Improvement (%)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax2.set_title("7B → 72B Improvement", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Model Scaling Analysis: Qwen 7B vs 72B",
        fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


def plot_per_class_accuracy(
    results_dict: Dict[str, Dict[str, float]],
    output_path: Union[str, Path],
    figsize: Tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Plot per-class accuracy for all models as grouped bars.

    Args:
        results_dict: {model_name: {class_label: accuracy}}.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = list(results_dict.keys())
    classes = ETHNIC_GROUPS
    n_models = len(models)
    n_classes = len(classes)

    # Create positions
    x = np.arange(n_classes)
    width = 0.8 / n_models

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, model in enumerate(models):
        accuracies = [results_dict[model].get(c, 0) for c in classes]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset, accuracies, width,
            label=_get_display_name(model),
            color=colors[i], edgecolor="black", linewidth=0.3
        )

    ax.set_xlabel("Ethnic Group", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="upper right", ncol=2, fontsize=9)

    ax.set_title(
        "Per-Class Classification Accuracy by Model",
        fontsize=14, fontweight="bold", pad=15
    )

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


def plot_cultural_coverage_distribution(
    coverage_scores: Dict[str, List[float]],
    output_path: Union[str, Path],
    figsize: Tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Plot distribution of cultural term coverage scores.

    Args:
        coverage_scores: {model_name: [coverage_scores]}.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Prepare data
    models = list(coverage_scores.keys())
    display_names = [_get_display_name(m) for m in models]

    # Left: Box plot
    ax1 = axes[0]
    data = [coverage_scores[m] for m in models]
    bp = ax1.boxplot(data, labels=display_names, patch_artist=True)

    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax1.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Cultural Term Coverage", fontsize=11, fontweight="bold")
    ax1.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax1.set_title("Coverage Distribution", fontsize=12, fontweight="bold")

    # Right: Violin plot
    ax2 = axes[1]
    positions = np.arange(1, len(models) + 1)

    parts = ax2.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax2.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cultural Term Coverage", fontsize=11, fontweight="bold")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(display_names, rotation=45, ha="right", fontsize=9)
    ax2.set_title("Coverage Density", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Cultural Term Coverage in Generated Descriptions",
        fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    _save_figure(fig, output_path)

    return fig


def create_all_figures(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Generate all publication figures from results.

    Args:
        results: Complete results dictionary with all metrics.
        output_dir: Directory to save figures.

    Returns:
        Dictionary mapping figure name to output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = {}

    logger.info(f"Generating figures in {output_dir}")

    # Figure 1: Accuracy comparison
    if "accuracy_df" in results:
        path = output_dir / "fig1_accuracy_comparison"
        plot_accuracy_comparison(results["accuracy_df"], path)
        generated["accuracy_comparison"] = path

    # Figure 2: Confusion matrices (one per model or aggregated)
    if "confusion_matrices" in results:
        for model_name, cm in results["confusion_matrices"].items():
            path = output_dir / f"fig2_confusion_{model_name.replace('.', '_')}"
            plot_confusion_heatmap(cm, model_name=model_name, output_path=path)
            generated[f"confusion_{model_name}"] = path

    # Figure 3: Language effect
    if "les_scores" in results:
        path = output_dir / "fig3_language_effect"
        plot_language_effect(results["les_scores"], path)
        generated["language_effect"] = path

    # Figure 4: OBI summary
    if "obi_by_language" in results:
        path = output_dir / "fig4_obi_summary"
        plot_obi_summary(
            results["obi_by_language"],
            path,
            confidence_intervals=results.get("obi_ci")
        )
        generated["obi_summary"] = path

    # Figure 5: Model scaling
    if "results_7b" in results and "results_72b" in results:
        path = output_dir / "fig5_model_scaling"
        plot_model_scaling(results["results_7b"], results["results_72b"], path)
        generated["model_scaling"] = path

    logger.info(f"Generated {len(generated)} figures")

    return generated
