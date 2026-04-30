#!/usr/bin/env python3
"""Standalone human evaluation sheet generator (no pandas/openpyxl dependency).

Generates CSV files for expert evaluation using only stdlib modules.
"""

import csv
import json
import os
import random
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ETHNIC_GROUPS = ["Miao", "Dong", "Yi", "Li", "Tibetan"]

MODEL_ORIGINS = {
    "qwen2.5-vl-7b": "chinese",
    "qwen2-vl-7b": "chinese",
    "llama-3.2-vision-11b": "western",
    "gpt-4o-mini": "western",
    "claude-haiku-4.5": "western",
    "claude-3.5-sonnet": "western",
}

LABEL_TO_ETHNIC = {
    "A": "Miao", "B": "Dong", "C": "Yi", "D": "Li", "E": "Tibetan"
}


def load_results(results_dir):
    """Load all results from results directory."""
    results_dir = Path(results_dir)
    results = {"classification": {}, "description": {}}

    for json_file in sorted(results_dir.glob("*.json")):
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
            print(f"  Warning: could not load {json_file.name}: {e}")

    print(f"  Loaded {len(results['classification'])} classification, "
          f"{len(results['description'])} description results")
    return results


def generate_description_quality_csv(results, output_path, n_samples=150, seed=42):
    """Generate description quality rating CSV (stratified by model)."""
    random.seed(seed)

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
        print("  WARNING: No descriptions found")
        return

    # Stratified sampling: equal per model, then balanced across (ethnic, language)
    available_models = sorted(set(d["model"] for d in all_descriptions))
    n_models = len(available_models)
    n_per_model = n_samples // n_models
    n_extra = n_samples % n_models

    selected = []
    used_image_paths = set()

    for model_idx, model in enumerate(available_models):
        model_quota = n_per_model + (1 if model_idx < n_extra else 0)
        model_descs = [d for d in all_descriptions if d["model"] == model]

        # Group by (ethnic_group, language)
        cells = defaultdict(list)
        for d in model_descs:
            cells[(d["ethnic_group"], d["language"])].append(d)

        # Phase 1: guarantee at least 1 per cell
        model_selected = []
        for cell_key in sorted(cells.keys()):
            candidates = [d for d in cells[cell_key]
                          if d["image_path"] not in used_image_paths]
            if candidates and len(model_selected) < model_quota:
                pick = random.choice(candidates)
                model_selected.append(pick)
                used_image_paths.add(pick["image_path"])

        # Phase 2: fill remaining with round-robin
        remaining = model_quota - len(model_selected)
        if remaining > 0:
            cell_pools = {}
            for cell_key in sorted(cells.keys()):
                cell_pools[cell_key] = [
                    d for d in cells[cell_key]
                    if d["image_path"] not in used_image_paths
                    and d not in model_selected
                ]

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

    random.shuffle(selected)
    selected = selected[:n_samples]

    # Log balance
    model_counts = defaultdict(int)
    ethnic_counts = defaultdict(int)
    lang_counts = defaultdict(int)
    for s in selected:
        model_counts[s["model"]] += 1
        ethnic_counts[s["ethnic_group"]] += 1
        lang_counts[s["language"]] += 1
    print(f"  Balance - Models: {dict(model_counts)}")
    print(f"  Ethnic: {dict(ethnic_counts)}, Lang: {dict(lang_counts)}")
    print(f"  Unique images: {len(set(s['image_path'] for s in selected))}/{len(selected)}")

    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Image_ID", "Image_Path", "Ethnic_Group", "Model",
            "Model_Origin", "Language", "Description",
            "Rating_1to5", "Accuracy_1to5", "Completeness_1to5",
            "Cultural_Depth_1to5", "Notes"
        ])
        for i, item in enumerate(selected, 1):
            writer.writerow([
                f"DQ_{i:03d}",
                item["image_id"],
                item["image_path"],
                item["ethnic_group"],
                item["model"],
                item["origin"],
                item["language"],
                item["description"],
                "", "", "", "", ""
            ])

    print(f"  -> {output_path} ({len(selected)} samples)")
    return output_path


def generate_error_categorization_csv(results, output_path, n_samples=30, seed=42):
    """Generate error categorization CSV for misclassified examples."""
    random.seed(seed)

    # Collect misclassifications
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
        print("  WARNING: No misclassifications found")
        return

    # Balance by confusion type
    confusion_groups = defaultdict(list)
    for e in errors:
        confusion_groups[e["confusion_pair"]].append(e)

    selected = []
    n_per_type = max(1, n_samples // len(confusion_groups))
    for pair, items in confusion_groups.items():
        sampled = random.sample(items, min(n_per_type, len(items)))
        selected.extend(sampled)

    remaining = n_samples - len(selected)
    if remaining > 0:
        available = [e for e in errors if e not in selected]
        if available:
            extra = random.sample(available, min(remaining, len(available)))
            selected.extend(extra)

    random.shuffle(selected)
    selected = selected[:n_samples]

    # Write CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ID", "Image_ID", "Image_Path", "True_Label", "Predicted_Label",
            "Model", "Model_Origin", "Language",
            "Error_Type", "Secondary_Type", "Confidence", "Notes"
        ])
        for i, item in enumerate(selected, 1):
            true_ethnic = LABEL_TO_ETHNIC.get(item["true_label"], item["true_label"])
            pred_ethnic = LABEL_TO_ETHNIC.get(item["predicted_label"], item["predicted_label"])
            writer.writerow([
                f"EC_{i:03d}",
                item["image_id"],
                item["image_path"],
                f"{item['true_label']} ({true_ethnic})",
                f"{item['predicted_label']} ({pred_ethnic})",
                item["model"],
                item["origin"],
                item["language"],
                "", "", "", ""
            ])

    # Summary
    confusion_summary = defaultdict(int)
    for s in selected:
        confusion_summary[s["confusion_pair"]] += 1
    print(f"  -> {output_path} ({len(selected)} samples)")
    print(f"  Confusion pairs: {dict(sorted(confusion_summary.items(), key=lambda x: -x[1]))}")
    return output_path


def generate_guidelines(output_dir):
    """Generate rating guidelines and error category reference files."""
    output_dir = Path(output_dir)

    # Description quality guidelines
    with open(output_dir / "GUIDELINES_description_quality.txt", "w", encoding="utf-8") as f:
        f.write("DESCRIPTION QUALITY EVALUATION GUIDELINES\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("TASK:\n")
        f.write("Rate each costume description on a scale of 1-5.\n\n")
        f.write("STEPS:\n")
        f.write("1. View the image at the given Image_Path\n")
        f.write("2. Read the model-generated Description\n")
        f.write("3. Rate overall quality (Rating_1to5): 1=poor, 5=excellent\n")
        f.write("4. Rate sub-aspects: Accuracy, Completeness, Cultural_Depth\n")
        f.write("5. Add any notes in the Notes column\n\n")
        f.write("RATING RUBRIC:\n")
        f.write("-" * 60 + "\n")
        f.write("Overall Quality (1-5):\n")
        f.write("  1 = Poor: Major errors, irrelevant content, or very superficial\n")
        f.write("  2 = Below Average: Several inaccuracies or missing key aspects\n")
        f.write("  3 = Average: Adequate description with some gaps\n")
        f.write("  4 = Good: Accurate and reasonably comprehensive\n")
        f.write("  5 = Excellent: Accurate, comprehensive, and culturally insightful\n\n")
        f.write("Accuracy (1-5):\n")
        f.write("  1 = Major factual errors  ->  5 = Completely accurate\n\n")
        f.write("Completeness (1-5):\n")
        f.write("  Aspects: style, patterns, materials, accessories, occasions\n")
        f.write("  1 = Covers 1 aspect  ->  5 = Covers all 5 aspects\n\n")
        f.write("Cultural Depth (1-5):\n")
        f.write("  1 = No cultural context  ->  5 = Expert-level cultural knowledge\n\n")
        f.write("IMPORTANT:\n")
        f.write("- Focus on factual accuracy about the costume\n")
        f.write("- Do not penalize for language style, only content quality\n")

    # Error categorization reference
    with open(output_dir / "GUIDELINES_error_categorization.txt", "w", encoding="utf-8") as f:
        f.write("ERROR CATEGORIZATION GUIDELINES\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("TASK:\n")
        f.write("Categorize why the model misclassified each costume image.\n\n")
        f.write("ERROR CATEGORIES:\n")
        f.write("-" * 60 + "\n")
        categories = [
            ("VS", "Visual Similarity",
             "The two ethnic groups have visually similar costume features"),
            ("RV", "Regional Variation",
             "Sub-regional variations within an ethnic group cause confusion"),
            ("AF", "Accessory Focus",
             "Model over-relies on accessories rather than overall costume"),
            ("CP", "Color/Pattern Error",
             "Misinterpretation of color schemes or pattern motifs"),
            ("IV", "Incomplete Visibility",
             "Key costume features not fully visible in image"),
            ("MA", "Modern Adaptation",
             "Modern or adapted costumes cause confusion"),
            ("KG", "Knowledge Gap",
             "Model lacks specific cultural knowledge to distinguish"),
            ("OT", "Other",
             "Does not fit above categories (explain in Notes)"),
        ]
        for code, name, definition in categories:
            f.write(f"  {code} - {name}\n")
            f.write(f"       {definition}\n\n")
        f.write("STEPS:\n")
        f.write("1. View the image at the given Image_Path\n")
        f.write("2. Note True_Label vs Predicted_Label\n")
        f.write("3. Select primary Error_Type code (VS/RV/AF/CP/IV/MA/KG/OT)\n")
        f.write("4. Optionally add Secondary_Type\n")
        f.write("5. Rate Confidence: High / Medium / Low\n")
        f.write("6. Add explanatory Notes\n")

    print(f"  -> Guidelines written to {output_dir}")


def collect_sample_images(selected_items, images_output_dir, data_dir="data"):
    """Copy sample images to the output directory for easy review."""
    images_output_dir = Path(images_output_dir)
    images_output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in selected_items:
        img_path = item.get("image_path", "")
        if not img_path:
            continue
        src = Path(img_path)
        if not src.exists():
            src = Path(data_dir) / img_path
        if src.exists():
            dst = images_output_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1

    print(f"  -> Copied {copied} unique images to {images_output_dir}")


def main():
    results_dir = "results/raw"
    output_dir = "results/human_eval"
    seed = 42

    print("=" * 60)
    print("Generating Human Evaluation Materials")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    print("\nLoading results...")
    results = load_results(results_dir)

    print("\n1. Description Quality Rating (150 samples = 30/model x 5 models)")
    desc_path = generate_description_quality_csv(
        results, f"{output_dir}/description_quality_rating.csv",
        n_samples=150, seed=seed
    )

    print("\n2. Error Categorization (30 samples)")
    error_path = generate_error_categorization_csv(
        results, f"{output_dir}/error_categorization.csv",
        n_samples=30, seed=seed
    )

    print("\n3. Guidelines")
    generate_guidelines(output_dir)

    # Collect images referenced in evaluation sheets
    print("\n4. Collecting sample images...")
    all_sample_items = []
    # Read back CSVs to get image paths
    for csv_file in [desc_path, error_path]:
        if csv_file and Path(csv_file).exists():
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_sample_items.append({"image_path": row.get("Image_Path", "")})

    collect_sample_images(all_sample_items, f"{output_dir}/images")

    # Summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "description_quality_samples": 150,
        "error_categorization_samples": 30,
        "models": ["qwen2.5-vl-7b", "qwen2-vl-7b", "llama-3.2-vision-11b",
                    "gpt-4o-mini", "claude-haiku-4.5"],
        "seed": seed,
    }
    with open(f"{output_dir}/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Done! Files in:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
