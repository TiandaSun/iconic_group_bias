"""Evaluation metrics for VLM cultural evaluation."""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

# Default ethnic group classes
DEFAULT_CLASSES = ["A", "B", "C", "D", "E"]

LABEL_TO_ETHNIC_GROUP = {
    "A": "Miao",
    "B": "Dong",
    "C": "Yi",
    "D": "Li",
    "E": "Tibetan",
}


# =============================================================================
# Classification Metrics
# =============================================================================


def classification_accuracy(
    y_true: List[str],
    y_pred: List[str],
    normalize: bool = True,
) -> float:
    """Calculate overall classification accuracy.

    Args:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels.
        normalize: If True, return fraction; if False, return count.

    Returns:
        Accuracy score (0.0 to 1.0 if normalized).
    """
    # Filter out None/invalid predictions
    valid_pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t is not None and p is not None
    ]

    if not valid_pairs:
        logger.warning("No valid predictions for accuracy calculation")
        return 0.0

    y_true_valid, y_pred_valid = zip(*valid_pairs)
    return accuracy_score(y_true_valid, y_pred_valid, normalize=normalize)


def per_class_accuracy(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate per-class accuracy.

    Args:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels.
        classes: List of class labels. Defaults to A-E.

    Returns:
        Dictionary mapping class label to accuracy.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    # Filter valid predictions
    valid_pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t is not None and p is not None
    ]

    if not valid_pairs:
        return {c: 0.0 for c in classes}

    y_true_valid, y_pred_valid = zip(*valid_pairs)

    # Calculate per-class accuracy
    class_accuracies = {}
    for cls in classes:
        # Get indices where true label is this class
        class_mask = [t == cls for t in y_true_valid]
        class_true = [t for t, m in zip(y_true_valid, class_mask) if m]
        class_pred = [p for p, m in zip(y_pred_valid, class_mask) if m]

        if class_true:
            correct = sum(1 for t, p in zip(class_true, class_pred) if t == p)
            class_accuracies[cls] = correct / len(class_true)
        else:
            class_accuracies[cls] = 0.0

    return class_accuracies


def macro_f1_score(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
) -> float:
    """Calculate macro-averaged F1 score.

    Args:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels.
        classes: List of class labels for ordering.

    Returns:
        Macro F1 score.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    # Filter valid predictions
    valid_pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t is not None and p is not None
    ]

    if not valid_pairs:
        logger.warning("No valid predictions for F1 calculation")
        return 0.0

    y_true_valid, y_pred_valid = zip(*valid_pairs)

    return f1_score(
        y_true_valid,
        y_pred_valid,
        labels=classes,
        average="macro",
        zero_division=0,
    )


def weighted_f1_score(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
) -> float:
    """Calculate weighted F1 score.

    Args:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels.
        classes: List of class labels.

    Returns:
        Weighted F1 score.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    valid_pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t is not None and p is not None
    ]

    if not valid_pairs:
        return 0.0

    y_true_valid, y_pred_valid = zip(*valid_pairs)

    return f1_score(
        y_true_valid,
        y_pred_valid,
        labels=classes,
        average="weighted",
        zero_division=0,
    )


def precision_recall_per_class(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Calculate precision and recall for each class.

    Args:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels.
        classes: List of class labels.

    Returns:
        Dictionary mapping class to {precision, recall, f1, support}.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    valid_pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t is not None and p is not None
    ]

    if not valid_pairs:
        return {c: {"precision": 0, "recall": 0, "f1": 0, "support": 0} for c in classes}

    y_true_valid, y_pred_valid = zip(*valid_pairs)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_valid,
        y_pred_valid,
        labels=classes,
        zero_division=0,
    )

    return {
        cls: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for cls, p, r, f, s in zip(classes, precision, recall, f1, support)
    }


def build_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Build confusion matrix.

    Args:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels.
        classes: List of class labels for ordering.
        normalize: 'true', 'pred', 'all', or None.

    Returns:
        Confusion matrix as numpy array.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    # Filter valid predictions
    valid_pairs = [
        (t, p) for t, p in zip(y_true, y_pred)
        if t is not None and p is not None
    ]

    if not valid_pairs:
        return np.zeros((len(classes), len(classes)))

    y_true_valid, y_pred_valid = zip(*valid_pairs)

    cm = confusion_matrix(
        y_true_valid,
        y_pred_valid,
        labels=classes,
        normalize=normalize,
    )

    return cm


def confusion_matrix_to_dict(
    cm: np.ndarray,
    classes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Convert confusion matrix to nested dictionary.

    Args:
        cm: Confusion matrix array.
        classes: Class labels.

    Returns:
        Nested dictionary {true_label: {pred_label: count}}.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    return {
        true_cls: {
            pred_cls: float(cm[i, j])
            for j, pred_cls in enumerate(classes)
        }
        for i, true_cls in enumerate(classes)
    }


# =============================================================================
# Bias Metrics
# =============================================================================


def origin_bias_index(
    chinese_model_accs: List[float],
    western_model_accs: List[float],
) -> float:
    """Calculate Origin Bias Index (OBI).

    Measures performance gap between Chinese-origin and Western-origin VLMs.

    Formula: (mean(chinese) - mean(western)) / mean(all)

    Interpretation:
        - Positive: Chinese-origin models perform better
        - Negative: Western-origin models perform better
        - Zero: No origin-based performance difference

    Args:
        chinese_model_accs: Accuracies of Chinese-origin models (Qwen).
        western_model_accs: Accuracies of Western-origin models (LLaMA, GPT, Gemini, Claude).

    Returns:
        OBI value.
    """
    if not chinese_model_accs or not western_model_accs:
        logger.warning("Empty accuracy list for OBI calculation")
        return 0.0

    mean_chinese = np.mean(chinese_model_accs)
    mean_western = np.mean(western_model_accs)
    mean_all = np.mean(chinese_model_accs + western_model_accs)

    if mean_all == 0:
        logger.warning("Mean accuracy is zero, cannot calculate OBI")
        return 0.0

    obi = (mean_chinese - mean_western) / mean_all

    logger.debug(
        f"OBI calculation: Chinese={mean_chinese:.4f}, Western={mean_western:.4f}, "
        f"All={mean_all:.4f}, OBI={obi:.4f}"
    )

    return float(obi)


def language_effect_score(
    acc_chinese_prompt: float,
    acc_english_prompt: float,
) -> float:
    """Calculate Language Effect Score (LES).

    Measures how prompt language affects model performance.

    Formula: (acc_chinese_prompt - acc_english_prompt) / acc_english_prompt

    Interpretation:
        - Positive: Chinese prompts lead to better performance
        - Negative: English prompts lead to better performance
        - Zero: No language effect

    Args:
        acc_chinese_prompt: Accuracy with Chinese prompts.
        acc_english_prompt: Accuracy with English prompts.

    Returns:
        LES value.
    """
    if acc_english_prompt == 0:
        logger.warning("English prompt accuracy is zero, cannot calculate LES")
        return 0.0

    les = (acc_chinese_prompt - acc_english_prompt) / acc_english_prompt

    logger.debug(
        f"LES calculation: Chinese={acc_chinese_prompt:.4f}, "
        f"English={acc_english_prompt:.4f}, LES={les:.4f}"
    )

    return float(les)


def calculate_obi_with_ci(
    chinese_model_accs: List[float],
    western_model_accs: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, Tuple[float, float]]:
    """Calculate OBI with bootstrap confidence interval.

    Bootstrap preserves the origin grouping: we resample WITH REPLACEMENT
    separately within each origin group. The previous implementation pooled
    all accuracies and randomly re-split, which destroyed the structure we
    are trying to estimate and produced artificially narrow CIs.

    Args:
        chinese_model_accs: Chinese model accuracies.
        western_model_accs: Western model accuracies.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Optional RNG seed for reproducibility.

    Returns:
        Tuple of (OBI, (lower_ci, upper_ci)).
    """
    obi = origin_bias_index(chinese_model_accs, western_model_accs)
    rng = np.random.default_rng(seed)
    ch = np.asarray(chinese_model_accs, dtype=float)
    w = np.asarray(western_model_accs, dtype=float)
    if len(ch) == 0 or len(w) == 0:
        return obi, (float("nan"), float("nan"))

    # Stratified bootstrap: resample within each origin group
    bootstrap_obis = []
    for _ in range(n_bootstrap):
        b_ch = rng.choice(ch, size=len(ch), replace=True)
        b_w = rng.choice(w, size=len(w), replace=True)
        bootstrap_obis.append(
            origin_bias_index(b_ch.tolist(), b_w.tolist())
        )

    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_obis, 100 * alpha / 2)
    upper = np.percentile(bootstrap_obis, 100 * (1 - alpha / 2))

    return obi, (float(lower), float(upper))


def origin_bias_index_logit(
    chinese_model_accs: List[float],
    western_model_accs: List[float],
    eps: float = 1e-3,
) -> float:
    """Logit-scale Origin Bias Index.

    OBI_logit = logit(mean_chinese) - logit(mean_western)

    Interpretation: log odds-ratio between Chinese- and Western-origin
    mean accuracy. Unlike the ratio form, this is stable when either
    origin mean is near 0 or 1 (fixing the Li case where OBI = +1.29
    due to a near-zero denominator).

    Positive: Chinese-origin models have higher odds of correct
    classification. Zero: parity. Negative: Western-origin higher.

    Args:
        chinese_model_accs: Chinese-origin model accuracies in [0, 1].
        western_model_accs: Western-origin model accuracies in [0, 1].
        eps: Clipping tolerance to avoid logit(0) / logit(1).

    Returns:
        Logit-scale OBI (unbounded real number).
    """
    if not chinese_model_accs or not western_model_accs:
        logger.warning("Empty accuracy list for OBI_logit calculation")
        return 0.0

    mc = float(np.clip(np.mean(chinese_model_accs), eps, 1.0 - eps))
    mw = float(np.clip(np.mean(western_model_accs), eps, 1.0 - eps))
    logit = lambda p: float(np.log(p / (1.0 - p)))
    return logit(mc) - logit(mw)


def cohens_h(
    chinese_model_accs: List[float],
    western_model_accs: List[float],
) -> float:
    """Cohen's h effect size between two proportions.

    h = 2*arcsin(sqrt(p1)) - 2*arcsin(sqrt(p2))

    Reported alongside OBI (ratio) and OBI_logit as a sensitivity
    analysis. Cohen's h is a standard effect-size measure for
    differences of proportions, stable near 0 and 1, with conventional
    interpretation: small ~0.2, medium ~0.5, large ~0.8.

    Args:
        chinese_model_accs: Chinese-origin model accuracies.
        western_model_accs: Western-origin model accuracies.

    Returns:
        Cohen's h (unbounded; same sign convention as OBI).
    """
    if not chinese_model_accs or not western_model_accs:
        return 0.0
    p_ch = float(np.clip(np.mean(chinese_model_accs), 0.0, 1.0))
    p_w = float(np.clip(np.mean(western_model_accs), 0.0, 1.0))
    phi_ch = 2.0 * np.arcsin(np.sqrt(p_ch))
    phi_w = 2.0 * np.arcsin(np.sqrt(p_w))
    return float(phi_ch - phi_w)


def calculate_obi_variants_with_ci(
    chinese_model_accs: List[float],
    western_model_accs: List[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute OBI (ratio), OBI_logit, and Cohen's h with stratified
    bootstrap CIs.

    Used for Table 5 in the revised manuscript, which reports all three
    variants side-by-side to demonstrate robustness to the Li
    near-zero-denominator pathology.
    """
    rng = np.random.default_rng(seed)
    ch = np.asarray(chinese_model_accs, dtype=float)
    w = np.asarray(western_model_accs, dtype=float)

    point = {
        "obi_ratio": origin_bias_index(chinese_model_accs, western_model_accs),
        "obi_logit": origin_bias_index_logit(chinese_model_accs, western_model_accs),
        "cohens_h": cohens_h(chinese_model_accs, western_model_accs),
    }
    boot = {k: [] for k in point}
    for _ in range(n_bootstrap):
        b_ch = rng.choice(ch, size=len(ch), replace=True).tolist()
        b_w = rng.choice(w, size=len(w), replace=True).tolist()
        boot["obi_ratio"].append(origin_bias_index(b_ch, b_w))
        boot["obi_logit"].append(origin_bias_index_logit(b_ch, b_w))
        boot["cohens_h"].append(cohens_h(b_ch, b_w))

    alpha = 1 - confidence_level
    out = {}
    for k, v in point.items():
        arr = np.asarray(boot[k])
        out[k] = {
            "point": float(v),
            "ci_low": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return out


# =============================================================================
# Cultural Term Coverage
# =============================================================================


def load_cultural_vocabulary(
    vocab_path: Optional[Union[str, Path]] = None,
    config_path: str = "configs/cultural_vocabulary.yaml",
) -> Dict[str, List[str]]:
    """Load cultural vocabulary from config file.

    Args:
        vocab_path: Direct path to vocabulary file.
        config_path: Default config file path.

    Returns:
        Dictionary with 'zh' and 'en' vocabulary lists.
    """
    if vocab_path:
        path = Path(vocab_path)
    else:
        path = Path(config_path)

    if not path.exists():
        logger.warning(f"Vocabulary file not found: {path}, using default")
        return _get_default_vocabulary()

    with open(path, "r", encoding="utf-8") as f:
        vocab = yaml.safe_load(f)

    return vocab


def _get_default_vocabulary() -> Dict[str, List[str]]:
    """Get default cultural vocabulary for Chinese minority costumes."""
    return {
        "zh": [
            # General terms
            "刺绣", "蜡染", "织锦", "挑花", "银饰", "头饰", "百褶裙",
            # Craft techniques
            "盘绣", "锁绣", "打籽绣", "贴布绣", "十字绣",
            "蓝靛", "枫香染", "扎染",
            # Materials
            "棉布", "麻布", "丝绸", "蚕丝", "银片", "银泡",
            # Patterns
            "龙纹", "凤纹", "蝴蝶纹", "牛角纹", "太阳纹", "几何纹",
            "铜鼓纹", "鱼纹", "花卉纹", "水波纹",
            # Colors
            "靛蓝", "朱红", "明黄", "藏青", "黑色", "白色",
            # Accessories
            "项圈", "手镯", "耳环", "发簪", "腰带", "绑腿",
            # Garment parts
            "衣襟", "袖口", "领口", "下摆", "裙边", "围腰",
        ],
        "en": [
            # General terms
            "embroidery", "batik", "brocade", "cross-stitch", "silver ornament",
            "headdress", "pleated skirt",
            # Craft techniques
            "coiling stitch", "chain stitch", "satin stitch", "applique",
            "indigo dyeing", "wax-resist dyeing", "tie-dye",
            # Materials
            "cotton", "linen", "silk", "hemp", "silver", "thread",
            # Patterns
            "dragon pattern", "phoenix pattern", "butterfly motif",
            "horn pattern", "sun motif", "geometric pattern",
            "bronze drum pattern", "fish pattern", "floral pattern",
            # Colors
            "indigo blue", "vermillion", "bright yellow", "navy blue",
            "black", "white", "red", "blue",
            # Accessories
            "necklace", "bracelet", "earring", "hairpin", "belt", "legging",
            # Garment parts
            "collar", "sleeve", "cuff", "hem", "waistband", "apron",
        ],
    }


def cultural_term_coverage(
    description: str,
    vocabulary: Optional[List[str]] = None,
    language: str = "zh",
    vocab_config_path: Optional[str] = None,
) -> float:
    """Calculate cultural term coverage in a description.

    Coverage = matched_terms / total_vocabulary_terms

    Args:
        description: Generated description text.
        vocabulary: List of cultural terms. If None, loads from config.
        language: Language of description ('zh' or 'en').
        vocab_config_path: Path to vocabulary config.

    Returns:
        Coverage score (0.0 to 1.0).
    """
    if not description or description == "ERROR":
        return 0.0

    # Load vocabulary if not provided
    if vocabulary is None:
        vocab_dict = load_cultural_vocabulary(vocab_config_path)
        vocabulary = vocab_dict.get(language, vocab_dict.get("zh", []))

    if not vocabulary:
        logger.warning("Empty vocabulary, cannot calculate coverage")
        return 0.0

    # Normalize description for matching
    description_lower = description.lower()

    # Count matched terms
    matched = 0
    matched_terms = []

    for term in vocabulary:
        term_lower = term.lower()
        if term_lower in description_lower:
            matched += 1
            matched_terms.append(term)

    coverage = matched / len(vocabulary)

    logger.debug(
        f"Cultural coverage: {matched}/{len(vocabulary)} terms "
        f"({coverage:.2%}): {matched_terms[:5]}..."
    )

    return coverage


def batch_cultural_term_coverage(
    descriptions: List[str],
    vocabulary: Optional[List[str]] = None,
    language: str = "zh",
) -> Dict[str, float]:
    """Calculate coverage statistics for batch of descriptions.

    Args:
        descriptions: List of description texts.
        vocabulary: Cultural vocabulary list.
        language: Description language.

    Returns:
        Dictionary with coverage statistics.
    """
    if vocabulary is None:
        vocab_dict = load_cultural_vocabulary()
        vocabulary = vocab_dict.get(language, [])

    coverages = [
        cultural_term_coverage(desc, vocabulary, language)
        for desc in descriptions
    ]

    valid_coverages = [c for c in coverages if c > 0]

    return {
        "mean_coverage": np.mean(coverages) if coverages else 0.0,
        "std_coverage": np.std(coverages) if coverages else 0.0,
        "min_coverage": min(coverages) if coverages else 0.0,
        "max_coverage": max(coverages) if coverages else 0.0,
        "median_coverage": np.median(coverages) if coverages else 0.0,
        "zero_coverage_count": len([c for c in coverages if c == 0]),
        "total_descriptions": len(descriptions),
    }


def term_frequency_analysis(
    descriptions: List[str],
    vocabulary: Optional[List[str]] = None,
    language: str = "zh",
    top_n: int = 20,
) -> Dict[str, int]:
    """Analyze frequency of cultural terms across descriptions.

    Args:
        descriptions: List of description texts.
        vocabulary: Cultural vocabulary list.
        language: Description language.
        top_n: Number of top terms to return.

    Returns:
        Dictionary mapping term to frequency.
    """
    if vocabulary is None:
        vocab_dict = load_cultural_vocabulary()
        vocabulary = vocab_dict.get(language, [])

    term_counts = Counter()

    for desc in descriptions:
        if not desc or desc == "ERROR":
            continue
        desc_lower = desc.lower()
        for term in vocabulary:
            if term.lower() in desc_lower:
                term_counts[term] += 1

    return dict(term_counts.most_common(top_n))


# =============================================================================
# Aggregate Metrics
# =============================================================================


def calculate_all_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate all classification metrics at once.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        classes: Class labels.

    Returns:
        Dictionary with all metrics.
    """
    if classes is None:
        classes = DEFAULT_CLASSES

    cm = build_confusion_matrix(y_true, y_pred, classes)

    return {
        "accuracy": classification_accuracy(y_true, y_pred),
        "macro_f1": macro_f1_score(y_true, y_pred, classes),
        "weighted_f1": weighted_f1_score(y_true, y_pred, classes),
        "per_class_accuracy": per_class_accuracy(y_true, y_pred, classes),
        "precision_recall": precision_recall_per_class(y_true, y_pred, classes),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": build_confusion_matrix(
            y_true, y_pred, classes, normalize="true"
        ).tolist(),
        "total_samples": len(y_true),
        "valid_predictions": len([p for p in y_pred if p is not None]),
    }
