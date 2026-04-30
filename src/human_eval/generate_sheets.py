"""Generate expert evaluation materials for human evaluation.

This module creates Excel sheets for expert annotators to evaluate:
1. Description quality (50 samples)
2. Error categorization (30 samples)
3. Metric validation (20 samples)
"""

import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.evaluation.metrics import cultural_term_coverage, load_cultural_vocabulary

logger = logging.getLogger(__name__)

# Ethnic groups
ETHNIC_GROUPS = ["Miao", "Dong", "Yi", "Li", "Tibetan"]

# Model origins
MODEL_ORIGINS = {
    "qwen2.5-vl-7b": "chinese",
    "qwen2-vl-7b": "chinese",
    "llama-3.2-vision-11b": "western",
    "gpt-4o-mini": "western",
    "gemini-2.5-flash": "western",
    "claude-3.5-sonnet": "western",
    "claude-haiku-4.5": "western",
}


def load_results(results_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load all results from results directory.

    Args:
        results_dir: Directory containing result JSON files.

    Returns:
        Dictionary with loaded results organized by task and model.
    """
    results_dir = Path(results_dir)
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


def _create_rating_guidelines_df() -> pd.DataFrame:
    """Create DataFrame with rating guidelines for description quality."""
    guidelines = [
        {
            "Aspect": "Overall Quality Rating (1-5)",
            "Score 1": "Poor - Major errors, irrelevant content, or very superficial",
            "Score 2": "Below Average - Several inaccuracies or missing key aspects",
            "Score 3": "Average - Adequate description with some gaps",
            "Score 4": "Good - Accurate and reasonably comprehensive",
            "Score 5": "Excellent - Accurate, comprehensive, and culturally insightful",
        },
        {
            "Aspect": "Accuracy",
            "Score 1": "Major factual errors about the costume",
            "Score 2": "Several inaccuracies",
            "Score 3": "Mostly accurate with minor errors",
            "Score 4": "Accurate with negligible issues",
            "Score 5": "Completely accurate",
        },
        {
            "Aspect": "Completeness",
            "Score 1": "Covers only 1 aspect (style/pattern/material/accessories/usage)",
            "Score 2": "Covers 2 aspects",
            "Score 3": "Covers 3 aspects",
            "Score 4": "Covers 4 aspects",
            "Score 5": "Covers all 5 required aspects",
        },
        {
            "Aspect": "Cultural Depth",
            "Score 1": "No cultural context or understanding shown",
            "Score 2": "Superficial cultural mentions without depth",
            "Score 3": "Basic cultural understanding demonstrated",
            "Score 4": "Good cultural insight and appropriate terminology",
            "Score 5": "Expert-level cultural knowledge and nuanced analysis",
        },
    ]

    return pd.DataFrame(guidelines)


def _create_error_categories_df() -> pd.DataFrame:
    """Create DataFrame with error category definitions."""
    categories = [
        {
            "Category": "visual_similarity",
            "Definition": "The two ethnic groups have visually similar costume features",
            "Examples": "Similar color schemes, comparable embroidery patterns, alike silver ornaments",
            "Code": "VS",
        },
        {
            "Category": "regional_variation",
            "Definition": "Sub-regional variations within an ethnic group cause confusion",
            "Examples": "Different Miao sub-groups have distinct styles; model confuses with another group",
            "Code": "RV",
        },
        {
            "Category": "accessory_focus",
            "Definition": "Model over-relies on accessories rather than overall costume",
            "Examples": "Mistakes identity based on silver jewelry alone",
            "Code": "AF",
        },
        {
            "Category": "color_pattern_error",
            "Definition": "Misinterpretation of color schemes or pattern motifs",
            "Examples": "Confuses indigo-based costumes, mistakes geometric patterns",
            "Code": "CP",
        },
        {
            "Category": "incomplete_visibility",
            "Definition": "Key costume features not fully visible in image",
            "Examples": "Cropped image, obscured embroidery, partial view",
            "Code": "IV",
        },
        {
            "Category": "modern_adaptation",
            "Definition": "Modern or adapted costumes cause confusion",
            "Examples": "Contemporary festival wear differs from traditional",
            "Code": "MA",
        },
        {
            "Category": "knowledge_gap",
            "Definition": "Model lacks specific cultural knowledge to distinguish",
            "Examples": "Cannot recognize ethnic-specific techniques or motifs",
            "Code": "KG",
        },
        {
            "Category": "other",
            "Definition": "Does not fit above categories",
            "Examples": "Please describe in Notes column",
            "Code": "OT",
        },
    ]

    return pd.DataFrame(categories)


def generate_description_quality_sheet(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    n_samples: int = 50,
    seed: int = 42,
) -> Path:
    """Generate description quality rating sheet.

    Samples stratified by model, then balanced across ethnic group and
    language within each model. Ensures no duplicate images and guarantees
    at least 1 sample per (model, ethnic_group, language) cell when
    available.

    Args:
        results: Loaded results dictionary.
        output_path: Path for output Excel file.
        n_samples: Total number of samples (default: 50).
        seed: Random seed for reproducibility.

    Returns:
        Path to generated Excel file.
    """
    random.seed(seed)
    output_path = Path(output_path)

    # Collect all descriptions
    all_descriptions = []

    for key, data in results.get("description", {}).items():
        model = data.get("model_name", "unknown")
        language = data.get("language", "unknown")
        origin = MODEL_ORIGINS.get(model, "unknown")

        for img_id, result in data.get("results", {}).items():
            desc = result.get("description", "")
            if desc and desc != "ERROR":
                all_descriptions.append({
                    "image_id": img_id,
                    "image_path": result.get("image_path", ""),
                    "ethnic_group": result.get("ethnic_group", ""),
                    "model": model,
                    "origin": origin,
                    "language": language,
                    "description": desc,
                })

    if not all_descriptions:
        logger.warning("No descriptions found in results")
        return output_path

    # Stratified sampling: balanced per model, then per (ethnic, language)
    # 1. Determine available models and allocate equal quota per model
    available_models = sorted(set(d["model"] for d in all_descriptions))
    n_models = len(available_models)
    n_per_model = n_samples // n_models
    n_extra = n_samples % n_models  # distribute remainder to first models

    selected = []
    used_image_paths: set = set()  # track images to prevent duplicates

    for model_idx, model in enumerate(available_models):
        model_quota = n_per_model + (1 if model_idx < n_extra else 0)
        model_descs = [d for d in all_descriptions if d["model"] == model]

        # 2. Group by (ethnic_group, language) within this model
        cells: Dict[Tuple[str, str], List] = defaultdict(list)
        for d in model_descs:
            cells[(d["ethnic_group"], d["language"])].append(d)

        # 3. Phase 1: guarantee at least 1 sample per cell (unique images)
        model_selected: List[Dict] = []
        for cell_key in sorted(cells.keys()):
            candidates = [
                d for d in cells[cell_key]
                if d["image_path"] not in used_image_paths
            ]
            if candidates and len(model_selected) < model_quota:
                pick = random.choice(candidates)
                model_selected.append(pick)
                used_image_paths.add(pick["image_path"])

        # 4. Phase 2: fill remaining quota with round-robin across cells
        remaining = model_quota - len(model_selected)
        if remaining > 0:
            # Build pool of unused candidates per cell
            cell_pools: Dict[Tuple[str, str], List] = {}
            for cell_key in sorted(cells.keys()):
                cell_pools[cell_key] = [
                    d for d in cells[cell_key]
                    if d["image_path"] not in used_image_paths
                    and d not in model_selected
                ]

            # Round-robin across cells for even distribution
            cell_keys = sorted(cell_pools.keys())
            rr_idx = 0
            rounds_without_pick = 0
            while remaining > 0 and rounds_without_pick < len(cell_keys):
                cell_key = cell_keys[rr_idx % len(cell_keys)]
                pool = cell_pools[cell_key]
                if pool:
                    pick = pool.pop(random.randrange(len(pool)))
                    model_selected.append(pick)
                    used_image_paths.add(pick["image_path"])
                    remaining -= 1
                    rounds_without_pick = 0
                else:
                    rounds_without_pick += 1
                rr_idx += 1

        selected.extend(model_selected)

    # Shuffle final selection
    random.shuffle(selected)
    selected = selected[:n_samples]

    # Log balance summary
    model_counts = defaultdict(int)
    ethnic_counts = defaultdict(int)
    lang_counts = defaultdict(int)
    for s in selected:
        model_counts[s["model"]] += 1
        ethnic_counts[s["ethnic_group"]] += 1
        lang_counts[s["language"]] += 1
    logger.info(
        f"Sample balance - Models: {dict(model_counts)}, "
        f"Ethnic: {dict(ethnic_counts)}, Lang: {dict(lang_counts)}, "
        f"Unique images: {len(set(s['image_path'] for s in selected))}/{len(selected)}"
    )

    # Create main data sheet
    data_rows = []
    for i, item in enumerate(selected, 1):
        data_rows.append({
            "ID": f"DQ_{i:03d}",
            "Image_Path": item["image_path"],
            "Ethnic_Group": item["ethnic_group"],
            "Model": item["model"],
            "Model_Origin": item["origin"],
            "Language": item["language"],
            "Description": item["description"],
            "Rating_1to5": "",  # To be filled by expert
            "Accuracy_1to5": "",
            "Completeness_1to5": "",
            "Cultural_Depth_1to5": "",
            "Notes": "",
        })

    data_df = pd.DataFrame(data_rows)
    guidelines_df = _create_rating_guidelines_df()

    # Instructions
    instructions = pd.DataFrame({
        "Instructions": [
            "DESCRIPTION QUALITY EVALUATION",
            "",
            f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Samples: {len(selected)}",
            "",
            "TASK:",
            "Please rate each costume description on a scale of 1-5.",
            "",
            "STEPS:",
            "1. View the image at the given Image_Path",
            "2. Read the model-generated Description",
            "3. Rate the overall quality (Rating_1to5) from 1 (poor) to 5 (excellent)",
            "4. Optionally rate sub-aspects: Accuracy, Completeness, Cultural_Depth",
            "5. Add any notes in the Notes column",
            "",
            "IMPORTANT:",
            "- Focus on factual accuracy about the costume",
            "- Consider coverage of: style, patterns, materials, accessories, occasions",
            "- Evaluate cultural understanding and appropriate terminology",
            "- Do not penalize for language style, only content quality",
            "",
            "See 'Rating_Guidelines' sheet for detailed rubric.",
        ]
    })

    # Write Excel file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False, header=False)
        data_df.to_excel(writer, sheet_name="Evaluation_Data", index=False)
        guidelines_df.to_excel(writer, sheet_name="Rating_Guidelines", index=False)

        # Adjust column widths
        worksheet = writer.sheets["Evaluation_Data"]
        worksheet.column_dimensions["G"].width = 80  # Description column
        worksheet.column_dimensions["L"].width = 40  # Notes column

    logger.info(f"Generated description quality sheet: {output_path} ({len(selected)} samples)")

    return output_path


def generate_error_categorization_sheet(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    n_samples: int = 30,
    seed: int = 42,
) -> Path:
    """Generate error categorization sheet for misclassified examples.

    Args:
        results: Loaded results dictionary.
        output_path: Path for output Excel file.
        n_samples: Number of error samples (default: 30).
        seed: Random seed for reproducibility.

    Returns:
        Path to generated Excel file.
    """
    random.seed(seed)
    output_path = Path(output_path)

    # Collect all misclassifications
    errors = []

    for key, data in results.get("classification", {}).items():
        model = data.get("model_name", "unknown")
        language = data.get("language", "unknown")
        origin = MODEL_ORIGINS.get(model, "unknown")

        for img_id, result in data.get("results", {}).items():
            predicted = result.get("predicted")
            ground_truth = result.get("ground_truth")

            if predicted and ground_truth and predicted != ground_truth:
                errors.append({
                    "image_id": img_id,
                    "image_path": result.get("image_path", ""),
                    "true_label": ground_truth,
                    "predicted_label": predicted,
                    "model": model,
                    "origin": origin,
                    "language": language,
                    "confusion_pair": f"{ground_truth}->{predicted}",
                })

    if not errors:
        logger.warning("No misclassifications found in results")
        return output_path

    # Balance sampling by confusion type
    confusion_groups = defaultdict(list)
    for e in errors:
        confusion_groups[e["confusion_pair"]].append(e)

    # Sample from each confusion type
    selected = []
    n_per_type = max(1, n_samples // len(confusion_groups))

    for pair, items in confusion_groups.items():
        sampled = random.sample(items, min(n_per_type, len(items)))
        selected.extend(sampled)

    # Fill remaining with random samples
    remaining = n_samples - len(selected)
    if remaining > 0:
        available = [e for e in errors if e not in selected]
        if available:
            extra = random.sample(available, min(remaining, len(available)))
            selected.extend(extra)

    random.shuffle(selected)
    selected = selected[:n_samples]

    # Create label mapping for display
    label_to_ethnic = {
        "A": "Miao", "B": "Dong", "C": "Yi", "D": "Li", "E": "Tibetan"
    }

    # Create main data sheet
    data_rows = []
    for i, item in enumerate(selected, 1):
        true_ethnic = label_to_ethnic.get(item["true_label"], item["true_label"])
        pred_ethnic = label_to_ethnic.get(item["predicted_label"], item["predicted_label"])

        data_rows.append({
            "ID": f"EC_{i:03d}",
            "Image_Path": item["image_path"],
            "True_Label": f"{item['true_label']} ({true_ethnic})",
            "Predicted_Label": f"{item['predicted_label']} ({pred_ethnic})",
            "Model": item["model"],
            "Model_Origin": item["origin"],
            "Language": item["language"],
            "Error_Type": "",  # To be filled: VS/RV/AF/CP/IV/MA/KG/OT
            "Secondary_Type": "",  # Optional secondary category
            "Confidence": "",  # How confident in categorization: High/Medium/Low
            "Notes": "",
        })

    data_df = pd.DataFrame(data_rows)
    categories_df = _create_error_categories_df()

    # Instructions
    instructions = pd.DataFrame({
        "Instructions": [
            "ERROR CATEGORIZATION EVALUATION",
            "",
            f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Error Samples: {len(selected)}",
            "",
            "TASK:",
            "Categorize why the model misclassified each costume image.",
            "",
            "STEPS:",
            "1. View the image at the given Image_Path",
            "2. Note the True_Label (correct ethnic group) vs Predicted_Label (model's guess)",
            "3. Analyze WHY the model might have made this error",
            "4. Select the primary Error_Type from the codes below:",
            "   - VS: Visual Similarity",
            "   - RV: Regional Variation",
            "   - AF: Accessory Focus",
            "   - CP: Color/Pattern Error",
            "   - IV: Incomplete Visibility",
            "   - MA: Modern Adaptation",
            "   - KG: Knowledge Gap",
            "   - OT: Other (explain in Notes)",
            "5. Optionally add a Secondary_Type if applicable",
            "6. Rate your Confidence: High / Medium / Low",
            "7. Add explanatory Notes",
            "",
            "See 'Error_Categories' sheet for detailed definitions.",
        ]
    })

    # Summary of confusion pairs in sample
    confusion_summary = pd.DataFrame(
        [(pair, len([s for s in selected if s["confusion_pair"] == pair]))
         for pair in set(s["confusion_pair"] for s in selected)],
        columns=["Confusion_Pair", "Count"]
    ).sort_values("Count", ascending=False)

    # Write Excel file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False, header=False)
        data_df.to_excel(writer, sheet_name="Evaluation_Data", index=False)
        categories_df.to_excel(writer, sheet_name="Error_Categories", index=False)
        confusion_summary.to_excel(writer, sheet_name="Sample_Summary", index=False)

        # Adjust column widths
        worksheet = writer.sheets["Evaluation_Data"]
        worksheet.column_dimensions["B"].width = 50  # Image path
        worksheet.column_dimensions["K"].width = 40  # Notes

    logger.info(f"Generated error categorization sheet: {output_path} ({len(selected)} samples)")

    return output_path


def generate_metric_validation_sheet(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    n_samples: int = 20,
    seed: int = 42,
    vocab_config_path: Optional[str] = None,
) -> Path:
    """Generate metric validation sheet for correlating CTC with expert ratings.

    Args:
        results: Loaded results dictionary.
        output_path: Path for output Excel file.
        n_samples: Number of samples (default: 20).
        seed: Random seed.
        vocab_config_path: Path to cultural vocabulary config.

    Returns:
        Path to generated Excel file.
    """
    random.seed(seed)
    output_path = Path(output_path)

    # Load vocabulary
    try:
        vocabulary = load_cultural_vocabulary(vocab_config_path)
    except Exception:
        vocabulary = {"zh": [], "en": []}

    # Collect descriptions with CTC scores
    descriptions_with_ctc = []

    for key, data in results.get("description", {}).items():
        model = data.get("model_name", "unknown")
        language = data.get("language", "unknown")

        vocab = vocabulary.get(language, vocabulary.get("zh", []))

        for img_id, result in data.get("results", {}).items():
            desc = result.get("description", "")
            if desc and desc != "ERROR":
                ctc = cultural_term_coverage(desc, vocab, language)
                descriptions_with_ctc.append({
                    "image_id": img_id,
                    "ethnic_group": result.get("ethnic_group", ""),
                    "model": model,
                    "language": language,
                    "description": desc,
                    "ctc_score": ctc,
                })

    if not descriptions_with_ctc:
        logger.warning("No descriptions found for metric validation")
        return output_path

    # Sample to cover range of CTC scores
    # Sort by CTC and sample from different quintiles
    sorted_descs = sorted(descriptions_with_ctc, key=lambda x: x["ctc_score"])
    n_quintiles = 5
    quintile_size = len(sorted_descs) // n_quintiles
    n_per_quintile = max(1, n_samples // n_quintiles)

    selected = []
    for i in range(n_quintiles):
        start = i * quintile_size
        end = start + quintile_size if i < n_quintiles - 1 else len(sorted_descs)
        quintile = sorted_descs[start:end]
        if quintile:
            sampled = random.sample(quintile, min(n_per_quintile, len(quintile)))
            selected.extend(sampled)

    # Fill remaining
    remaining = n_samples - len(selected)
    if remaining > 0:
        available = [d for d in descriptions_with_ctc if d not in selected]
        if available:
            extra = random.sample(available, min(remaining, len(available)))
            selected.extend(extra)

    random.shuffle(selected)
    selected = selected[:n_samples]

    # Create main data sheet
    data_rows = []
    for i, item in enumerate(selected, 1):
        data_rows.append({
            "ID": f"MV_{i:03d}",
            "Ethnic_Group": item["ethnic_group"],
            "Model": item["model"],
            "Language": item["language"],
            "Description": item["description"],
            "CTC_Score": round(item["ctc_score"], 4),
            "Expert_Quality_1to5": "",  # To be filled by expert
            "Expert_Cultural_Depth_1to5": "",
            "Expert_Term_Richness_1to5": "",
            "Notes": "",
        })

    data_df = pd.DataFrame(data_rows)

    # Instructions
    instructions = pd.DataFrame({
        "Instructions": [
            "METRIC VALIDATION EVALUATION",
            "",
            f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Samples: {len(selected)}",
            "",
            "PURPOSE:",
            "Validate automated Cultural Term Coverage (CTC) metric against expert judgment.",
            "Your ratings will be correlated with the CTC_Score to assess metric validity.",
            "",
            "TASK:",
            "Rate each description WITHOUT looking at the CTC_Score first.",
            "",
            "STEPS:",
            "1. Read the Description carefully",
            "2. Rate Expert_Quality_1to5: Overall description quality (1=poor, 5=excellent)",
            "3. Rate Expert_Cultural_Depth_1to5: Level of cultural understanding shown",
            "4. Rate Expert_Term_Richness_1to5: Use of appropriate cultural terminology",
            "5. Add any Notes about the description",
            "",
            "RATING SCALE:",
            "1 = Very Poor - Missing cultural content, generic description",
            "2 = Poor - Minimal cultural terminology, superficial",
            "3 = Average - Some cultural terms, basic understanding",
            "4 = Good - Rich cultural vocabulary, good depth",
            "5 = Excellent - Expert-level terminology, comprehensive cultural insight",
            "",
            "IMPORTANT:",
            "- Rate descriptions BEFORE checking CTC_Score",
            "- CTC_Score is the automated metric (0.0 to 1.0)",
            "- Your ratings help validate whether CTC reflects true quality",
        ]
    })

    # CTC score distribution in sample
    ctc_stats = pd.DataFrame({
        "Statistic": ["Min CTC", "Max CTC", "Mean CTC", "Median CTC", "Std CTC"],
        "Value": [
            round(min(s["ctc_score"] for s in selected), 4),
            round(max(s["ctc_score"] for s in selected), 4),
            round(sum(s["ctc_score"] for s in selected) / len(selected), 4),
            round(sorted(s["ctc_score"] for s in selected)[len(selected) // 2], 4),
            round(pd.Series([s["ctc_score"] for s in selected]).std(), 4),
        ]
    })

    # Write Excel file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False, header=False)
        data_df.to_excel(writer, sheet_name="Evaluation_Data", index=False)
        ctc_stats.to_excel(writer, sheet_name="CTC_Statistics", index=False)

        # Adjust column widths
        worksheet = writer.sheets["Evaluation_Data"]
        worksheet.column_dimensions["E"].width = 80  # Description
        worksheet.column_dimensions["J"].width = 40  # Notes

    logger.info(f"Generated metric validation sheet: {output_path} ({len(selected)} samples)")

    return output_path


def generate_expert_evaluation_materials(
    results_dir: Union[str, Path],
    output_dir: Union[str, Path],
    seed: int = 42,
    vocab_config_path: Optional[str] = None,
) -> Dict[str, Path]:
    """Generate all expert evaluation materials.

    Args:
        results_dir: Directory containing result JSON files.
        output_dir: Directory to save Excel files.
        seed: Random seed for reproducibility.
        vocab_config_path: Path to cultural vocabulary config.

    Returns:
        Dictionary mapping sheet type to output path.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading results from {results_dir}")
    results = load_results(results_dir)

    generated = {}

    # 1. Description Quality Rating Sheet (150 samples = 30 per model x 5 models)
    desc_quality_path = output_dir / "description_quality_rating.xlsx"
    generate_description_quality_sheet(
        results, desc_quality_path, n_samples=150, seed=seed
    )
    generated["description_quality"] = desc_quality_path

    # 2. Error Categorization Sheet (30 samples)
    error_cat_path = output_dir / "error_categorization.xlsx"
    generate_error_categorization_sheet(
        results, error_cat_path, n_samples=30, seed=seed
    )
    generated["error_categorization"] = error_cat_path

    # 3. Metric Validation Sheet (20 samples)
    metric_val_path = output_dir / "metric_validation.xlsx"
    generate_metric_validation_sheet(
        results, metric_val_path, n_samples=20, seed=seed,
        vocab_config_path=vocab_config_path
    )
    generated["metric_validation"] = metric_val_path

    logger.info(f"Generated {len(generated)} evaluation sheets in {output_dir}")

    return generated


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate expert evaluation materials",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        required=True,
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results/human_eval",
        help="Output directory for Excel files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--vocab-config",
        type=str,
        default="configs/cultural_vocabulary.yaml",
        help="Path to cultural vocabulary config",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Generate materials
    generated = generate_expert_evaluation_materials(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        vocab_config_path=args.vocab_config,
    )

    print("\nGenerated evaluation materials:")
    for sheet_type, path in generated.items():
        print(f"  - {sheet_type}: {path}")


if __name__ == "__main__":
    main()
