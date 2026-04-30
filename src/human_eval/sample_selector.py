"""Sample selection for human evaluation.

This script selects a stratified sample of images and descriptions
for human evaluation, ensuring balanced representation across:
- Models
- Ethnic groups
- Correct/incorrect classifications
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import shutil


def load_results(results_dir: str) -> Dict[str, Dict]:
    """Load all result files from directory.

    Args:
        results_dir: Path to results directory.

    Returns:
        Dictionary mapping model names to their results.
    """
    results = {}
    results_path = Path(results_dir)

    for f in results_path.glob("task2_*_results.json"):
        with open(f) as fp:
            data = json.load(fp)

        model_name = data["model_name"]
        language = data["language"]
        key = f"{model_name}_{language}"
        results[key] = data

    return results


def select_samples(
    results: Dict[str, Dict],
    samples_per_cell: int = 3,
    ethnic_groups: List[str] = None,
    seed: int = 42,
) -> List[Dict]:
    """Select stratified samples for evaluation.

    Args:
        results: Dictionary of results by model.
        samples_per_cell: Number of samples per model×ethnic_group combination.
        ethnic_groups: List of ethnic groups to include.
        seed: Random seed for reproducibility.

    Returns:
        List of selected samples with metadata.
    """
    if ethnic_groups is None:
        ethnic_groups = ["Miao", "Dong", "Yi", "Li", "Tibetan"]

    random.seed(seed)
    selected = []

    for model_key, data in results.items():
        model_name = data["model_name"]
        language = data["language"]

        for ethnic in ethnic_groups:
            # Get all samples for this ethnic group
            ethnic_samples = [
                (img_id, result)
                for img_id, result in data["results"].items()
                if result.get("ethnic_group") == ethnic
            ]

            if len(ethnic_samples) < samples_per_cell:
                print(f"Warning: Only {len(ethnic_samples)} samples for {model_key}/{ethnic}")
                sample_ids = ethnic_samples
            else:
                sample_ids = random.sample(ethnic_samples, samples_per_cell)

            for img_id, result in sample_ids:
                selected.append({
                    "sample_id": f"{model_key}_{img_id}",
                    "model_name": model_name,
                    "language": language,
                    "image_id": img_id,
                    "ethnic_group": ethnic,
                    "image_path": result.get("image_path"),
                    "description": result.get("description"),
                    "description_length": len(result.get("description", "")),
                })

    return selected


def load_classification_results(results_dir: str) -> Dict[str, Dict]:
    """Load classification results to get correct/incorrect labels.

    Args:
        results_dir: Path to results directory.

    Returns:
        Dictionary mapping (model, language, image_id) to classification result.
    """
    classification = {}
    results_path = Path(results_dir)

    for f in results_path.glob("task1_*_results.json"):
        with open(f) as fp:
            data = json.load(fp)

        model_name = data["model_name"]
        language = data["language"]

        for img_id, result in data["results"].items():
            key = (model_name, language, img_id)
            classification[key] = {
                "predicted": result.get("predicted"),
                "ground_truth": result.get("ground_truth"),
                "correct": result.get("correct"),
            }

    return classification


def enrich_with_classification(
    samples: List[Dict],
    classification: Dict[str, Dict],
) -> List[Dict]:
    """Add classification results to samples.

    Args:
        samples: List of selected samples.
        classification: Classification results dictionary.

    Returns:
        Enriched samples with classification data.
    """
    for sample in samples:
        key = (sample["model_name"], sample["language"], sample["image_id"])
        if key in classification:
            sample.update(classification[key])
        else:
            sample["predicted"] = None
            sample["ground_truth"] = None
            sample["correct"] = None

    return samples


def export_evaluation_package(
    samples: List[Dict],
    output_dir: str,
    copy_images: bool = True,
) -> None:
    """Export evaluation package with samples and images.

    Args:
        samples: List of selected samples.
        output_dir: Output directory for evaluation package.
        copy_images: Whether to copy images to output directory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save samples JSON
    samples_file = output_path / "evaluation_samples.json"
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(samples)} samples to {samples_file}")

    # Create summary statistics
    summary = {
        "total_samples": len(samples),
        "by_model": {},
        "by_ethnic_group": {},
        "by_language": {},
    }

    for sample in samples:
        model = sample["model_name"]
        ethnic = sample["ethnic_group"]
        lang = sample["language"]

        summary["by_model"][model] = summary["by_model"].get(model, 0) + 1
        summary["by_ethnic_group"][ethnic] = summary["by_ethnic_group"].get(ethnic, 0) + 1
        summary["by_language"][lang] = summary["by_language"].get(lang, 0) + 1

    summary_file = output_path / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_file}")

    # Copy images if requested
    if copy_images:
        images_dir = output_path / "images"
        images_dir.mkdir(exist_ok=True)

        copied = 0
        for sample in samples:
            src_path = sample.get("image_path")
            if src_path and Path(src_path).exists():
                # Create unique filename
                dst_name = f"{sample['sample_id']}{Path(src_path).suffix}"
                dst_path = images_dir / dst_name
                shutil.copy2(src_path, dst_path)
                sample["local_image_path"] = str(dst_path)
                copied += 1

        print(f"Copied {copied} images to {images_dir}")

    # Re-save samples with local paths
    with open(samples_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


def generate_evaluation_forms_html(
    samples: List[Dict],
    output_dir: str,
    samples_per_page: int = 5,
) -> None:
    """Generate HTML evaluation forms.

    Args:
        samples: List of selected samples.
        output_dir: Output directory.
        samples_per_page: Number of samples per HTML page.
    """
    output_path = Path(output_dir)
    forms_dir = output_path / "forms"
    forms_dir.mkdir(exist_ok=True)

    # HTML template
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Evaluation Form - Page {page_num}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        .sample {{ border: 2px solid #333; margin: 20px 0; padding: 20px; page-break-inside: avoid; }}
        .sample-header {{ background: #f0f0f0; padding: 10px; margin: -20px -20px 20px -20px; }}
        .image-container {{ text-align: center; margin: 20px 0; }}
        .image-container img {{ max-width: 400px; max-height: 400px; border: 1px solid #ccc; }}
        .description {{ background: #ffffd0; padding: 15px; margin: 15px 0; white-space: pre-wrap; font-size: 14px; }}
        .rating-section {{ margin: 15px 0; }}
        .rating-row {{ display: flex; align-items: center; margin: 10px 0; }}
        .rating-label {{ width: 200px; font-weight: bold; }}
        .rating-options {{ display: flex; gap: 15px; }}
        .rating-options label {{ cursor: pointer; }}
        .comments {{ width: 100%; height: 60px; margin-top: 5px; }}
        .aspects {{ margin: 10px 0; }}
        .aspects label {{ margin-right: 15px; }}
        h2 {{ color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .info-row {{ margin: 5px 0; }}
        .info-label {{ font-weight: bold; display: inline-block; width: 150px; }}
        @media print {{ .sample {{ page-break-inside: avoid; }} }}
    </style>
</head>
<body>
    <h1>Human Evaluation Form - Page {page_num}</h1>
    <p><strong>Evaluator ID:</strong> _________________ &nbsp;&nbsp;&nbsp; <strong>Date:</strong> _________________</p>

    {samples_html}

</body>
</html>"""

    sample_template = """
    <div class="sample">
        <div class="sample-header">
            <h2>Sample {idx}: {sample_id}</h2>
            <div class="info-row"><span class="info-label">Model:</span> {model_name}</div>
            <div class="info-row"><span class="info-label">Language:</span> {language}</div>
            <div class="info-row"><span class="info-label">True Ethnic Group:</span> {ethnic_group}</div>
            <div class="info-row"><span class="info-label">Classification:</span> {classification_info}</div>
        </div>

        <div class="image-container">
            <img src="../images/{image_filename}" alt="Costume image">
        </div>

        <h3>Generated Description:</h3>
        <div class="description">{description}</div>

        <h3>Evaluation Ratings:</h3>

        <div class="rating-section">
            <div class="rating-row">
                <span class="rating-label">1. Cultural Accuracy:</span>
                <div class="rating-options">
                    <label><input type="radio" name="cultural_{sample_id}" value="1"> 1</label>
                    <label><input type="radio" name="cultural_{sample_id}" value="2"> 2</label>
                    <label><input type="radio" name="cultural_{sample_id}" value="3"> 3</label>
                    <label><input type="radio" name="cultural_{sample_id}" value="4"> 4</label>
                    <label><input type="radio" name="cultural_{sample_id}" value="5"> 5</label>
                </div>
            </div>
            <textarea class="comments" placeholder="Comments on cultural accuracy..."></textarea>
        </div>

        <div class="rating-section">
            <div class="rating-row">
                <span class="rating-label">2. Visual Completeness:</span>
                <div class="rating-options">
                    <label><input type="radio" name="complete_{sample_id}" value="1"> 1</label>
                    <label><input type="radio" name="complete_{sample_id}" value="2"> 2</label>
                    <label><input type="radio" name="complete_{sample_id}" value="3"> 3</label>
                    <label><input type="radio" name="complete_{sample_id}" value="4"> 4</label>
                    <label><input type="radio" name="complete_{sample_id}" value="5"> 5</label>
                </div>
            </div>
            <div class="aspects">
                <strong>Aspects covered:</strong>
                <label><input type="checkbox" name="aspect_style_{sample_id}"> Style/Color</label>
                <label><input type="checkbox" name="aspect_pattern_{sample_id}"> Patterns</label>
                <label><input type="checkbox" name="aspect_material_{sample_id}"> Materials</label>
                <label><input type="checkbox" name="aspect_accessory_{sample_id}"> Accessories</label>
                <label><input type="checkbox" name="aspect_usage_{sample_id}"> Usage</label>
            </div>
        </div>

        <div class="rating-section">
            <div class="rating-row">
                <span class="rating-label">3. Terminology:</span>
                <div class="rating-options">
                    <label><input type="radio" name="terminology_{sample_id}" value="1"> 1</label>
                    <label><input type="radio" name="terminology_{sample_id}" value="2"> 2</label>
                    <label><input type="radio" name="terminology_{sample_id}" value="3"> 3</label>
                    <label><input type="radio" name="terminology_{sample_id}" value="4"> 4</label>
                    <label><input type="radio" name="terminology_{sample_id}" value="5"> 5</label>
                </div>
            </div>
        </div>

        <div class="rating-section">
            <div class="rating-row">
                <span class="rating-label">4. Factual Correctness:</span>
                <div class="rating-options">
                    <label><input type="radio" name="factual_{sample_id}" value="1"> 1</label>
                    <label><input type="radio" name="factual_{sample_id}" value="2"> 2</label>
                    <label><input type="radio" name="factual_{sample_id}" value="3"> 3</label>
                    <label><input type="radio" name="factual_{sample_id}" value="4"> 4</label>
                    <label><input type="radio" name="factual_{sample_id}" value="5"> 5</label>
                </div>
            </div>
            <textarea class="comments" placeholder="Note any factual errors..."></textarea>
        </div>

        <div class="rating-section">
            <div class="rating-row">
                <span class="rating-label">5. Overall Quality:</span>
                <div class="rating-options">
                    <label><input type="radio" name="overall_{sample_id}" value="1"> 1</label>
                    <label><input type="radio" name="overall_{sample_id}" value="2"> 2</label>
                    <label><input type="radio" name="overall_{sample_id}" value="3"> 3</label>
                    <label><input type="radio" name="overall_{sample_id}" value="4"> 4</label>
                    <label><input type="radio" name="overall_{sample_id}" value="5"> 5</label>
                </div>
            </div>
        </div>

        <div class="rating-section">
            <strong>Useful for documentation?</strong>
            <label><input type="radio" name="useful_{sample_id}" value="yes"> Yes, as-is</label>
            <label><input type="radio" name="useful_{sample_id}" value="minor"> Yes, with minor edits</label>
            <label><input type="radio" name="useful_{sample_id}" value="major"> Needs major revision</label>
            <label><input type="radio" name="useful_{sample_id}" value="no"> Not useful</label>
        </div>

        <div class="rating-section">
            <strong>Additional Comments:</strong>
            <textarea class="comments" style="height: 80px;" placeholder="What did the model do well? What did it miss?"></textarea>
        </div>
    </div>
    """

    # Generate pages
    num_pages = (len(samples) + samples_per_page - 1) // samples_per_page

    for page in range(num_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(samples))
        page_samples = samples[start_idx:end_idx]

        samples_html = ""
        for i, sample in enumerate(page_samples, start=start_idx + 1):
            image_filename = f"{sample['sample_id']}{Path(sample.get('image_path', '.jpg')).suffix}"

            classification_info = f"Predicted: {sample.get('predicted', 'N/A')}, "
            classification_info += f"Correct: {'✓' if sample.get('correct') else '✗'}"

            samples_html += sample_template.format(
                idx=i,
                sample_id=sample["sample_id"],
                model_name=sample["model_name"],
                language=sample["language"],
                ethnic_group=sample["ethnic_group"],
                classification_info=classification_info,
                image_filename=image_filename,
                description=sample.get("description", "No description available"),
            )

        html_content = html_template.format(
            page_num=page + 1,
            samples_html=samples_html,
        )

        output_file = forms_dir / f"evaluation_form_page_{page + 1:02d}.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

    print(f"Generated {num_pages} evaluation form pages in {forms_dir}")


def main():
    parser = argparse.ArgumentParser(description="Select samples for human evaluation")
    parser.add_argument("--results-dir", type=str, default="results/raw",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", type=str, default="results/human_eval",
                        help="Output directory for evaluation package")
    parser.add_argument("--samples-per-cell", type=int, default=3,
                        help="Samples per model×ethnic_group combination")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-copy-images", action="store_true",
                        help="Don't copy images to output directory")
    parser.add_argument("--generate-forms", action="store_true",
                        help="Generate HTML evaluation forms")
    parser.add_argument("--exclude-models", type=str, nargs="+", default=[],
                        help="Model names to exclude (e.g., internvl2.5-8b)")

    args = parser.parse_args()

    print("Loading results...")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} model results")

    # Filter out excluded models
    if args.exclude_models:
        print(f"Excluding models: {args.exclude_models}")
        results = {k: v for k, v in results.items()
                   if not any(excl in k for excl in args.exclude_models)}
        print(f"After filtering: {len(results)} model results")

    print("\nSelecting samples...")
    samples = select_samples(
        results,
        samples_per_cell=args.samples_per_cell,
        seed=args.seed,
    )
    print(f"Selected {len(samples)} samples")

    print("\nLoading classification results...")
    classification = load_classification_results(args.results_dir)
    print(f"Loaded classification for {len(classification)} samples")

    print("\nEnriching samples with classification data...")
    samples = enrich_with_classification(samples, classification)

    print("\nExporting evaluation package...")
    export_evaluation_package(
        samples,
        args.output_dir,
        copy_images=not args.no_copy_images,
    )

    if args.generate_forms:
        print("\nGenerating HTML evaluation forms...")
        generate_evaluation_forms_html(samples, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
