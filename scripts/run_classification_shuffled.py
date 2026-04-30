"""C3 sanity check — Task 1 classification with per-image shuffled MCQ options.

Peer-review concern (reviewers R2 @ IP&M and R3 @ ML):
    "The fixed MCQ option order (A:Miao, B:Dong, C:Yi, D:Li, E:Tibetan) means
    the 91% Miao predictions from LLaMA could be partially a position-A bias."

This script re-runs Task 1 with per-image shuffled option ordering.
For each image, a deterministic RNG seeded by the image ID produces a
random permutation of the 5 classes, constructs the MCQ prompt with
the new letter-to-class mapping, and queries the model. The
letter-to-class mapping is saved alongside the prediction so the
ground truth can be correctly scored.

Output format extends the standard Task 1 result JSON with
`letter_to_class` per image.

Usage:

    # Stratified 100 images across 5 groups × 2 languages (20 each)
    python scripts/run_classification_shuffled.py \\
        --model qwen2.5-vl-7b --language both --num-samples 100 \\
        --output results/raw/shuffled_mcq

See `slurm/run_shuffle_sanity_opensource.sh` for the Viking GPU job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Use the existing model registry
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.task1_classification import (  # noqa: E402
    MODEL_REGISTRY,
    create_model,
    load_model_config,
)
from src.utils import load_metadata, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

CLASSES = ["Miao", "Dong", "Yi", "Li", "Tibetan"]
LETTERS = ["A", "B", "C", "D", "E"]


def image_seed(image_id: str) -> int:
    """Stable per-image seed for reproducible shuffles."""
    return int(hashlib.sha256(image_id.encode("utf-8")).hexdigest(), 16) % (2 ** 32)


def build_shuffled_prompt(
    image_id: str,
    template: str,
    class_names_local: Dict[str, str],
    global_seed: int = 0,
) -> Tuple[str, Dict[str, str]]:
    """Build a shuffled-option prompt for a single image.

    Args:
        image_id: Stable per-image identifier (used to seed the RNG).
        template: Prompt template with `{options}` placeholder.
        class_names_local: Mapping from canonical English class name to
            its localised display name (e.g. "Miao" -> "苗族").
        global_seed: Optional extra seed offset (use 0 for reproducible
            shuffle across all runs; change to explore alternatives).

    Returns:
        (prompt_string, letter_to_class_dict). The letter_to_class_dict
        maps each letter ("A"..."E") to the canonical English class
        name chosen for that letter in THIS shuffle.
    """
    rng = random.Random(image_seed(image_id) ^ global_seed)
    shuffled = CLASSES.copy()
    rng.shuffle(shuffled)
    letter_to_class = dict(zip(LETTERS, shuffled))
    option_lines = "\n".join(
        f"{letter}) {class_names_local[letter_to_class[letter]]}"
        for letter in LETTERS
    )
    prompt = template.format(options=option_lines)
    return prompt, letter_to_class


def load_templates(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    t1 = config.get("task1_classification", {})
    return {
        "zh_template": t1["zh_template"].strip(),
        "en_template": t1["en_template"].strip(),
        "class_names": t1["class_names"],
    }


def stratified_sample(metadata: pd.DataFrame, num_samples: int,
                      seed: int) -> pd.DataFrame:
    groups = sorted(metadata["ethnic_group"].unique())
    per_group = max(1, num_samples // len(groups))
    parts = []
    for g in groups:
        sub = metadata[metadata["ethnic_group"] == g]
        n = min(per_group, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def run_shuffled_classification(
    model_name: str, language: str,
    data_path: str, output_dir: str,
    num_samples: int = 100,
    models_config_path: str = "configs/models.yaml",
    prompts_config_path: str = "configs/prompts.yaml",
    seed: int = 42,
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lang_code = "zh" if language in ("zh", "cn", "chinese") else "en"

    templates = load_templates(prompts_config_path)
    template = templates[f"{lang_code}_template"]
    class_names_local = templates["class_names"][lang_code]

    model_config = load_model_config(model_name, models_config_path)
    model = create_model(model_name, model_config)

    metadata = load_metadata(data_path)
    metadata = stratified_sample(metadata, num_samples, seed=seed)
    logger.info(
        f"Loaded {len(metadata)} stratified images "
        f"({num_samples // metadata['ethnic_group'].nunique()}/group)"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"task1_shuffled_{model_name}_{lang_code}_{timestamp}"

    results: Dict[str, dict] = {}
    n_correct = 0
    n_errors = 0
    start = time.time()

    for i, row in metadata.iterrows():
        img_id = row["image_id"]
        img_path = row["image_path"]
        true_class = row["ethnic_group"]

        # Per-image shuffled prompt
        prompt, letter_to_class = build_shuffled_prompt(
            img_id, template, class_names_local
        )

        try:
            pred_letter = model.classify(img_path, prompt)
        except Exception as e:
            logger.error(f"Inference failed on {img_id}: {e}")
            pred_letter = None
            n_errors += 1

        pred_class = (
            letter_to_class.get(pred_letter) if pred_letter else None
        )
        correct = (pred_class == true_class) if pred_class else False
        if correct:
            n_correct += 1

        results[img_id] = {
            "predicted_letter": pred_letter,
            "predicted_class": pred_class,
            "true_class": true_class,
            "true_letter": [
                lt for lt, cl in letter_to_class.items() if cl == true_class
            ][0] if pred_letter else None,
            "letter_to_class": letter_to_class,
            "correct": bool(correct),
            "image_path": img_path,
            "prompt": prompt,  # Kept for auditability
        }

        if (i + 1) % 20 == 0:
            logger.info(
                f"[{model_name} / {lang_code}] "
                f"{i + 1}/{len(metadata)}  acc_so_far={n_correct / (i + 1):.3f}"
            )

    elapsed = time.time() - start
    if hasattr(model, "unload"):
        model.unload()

    total = len(metadata)
    acc_overall = n_correct / total if total else 0.0

    # Per-group accuracy
    per_group = {}
    for g in sorted(metadata["ethnic_group"].unique()):
        ids = metadata[metadata["ethnic_group"] == g]["image_id"].tolist()
        corr = sum(1 for i_ in ids if results[i_]["correct"])
        per_group[g] = round(corr / len(ids), 3) if ids else 0.0

    # Position-marginal: does the model favour any letter regardless of content?
    letter_counts: Dict[str, int] = {lt: 0 for lt in LETTERS}
    for r in results.values():
        if r["predicted_letter"]:
            letter_counts[r["predicted_letter"]] = (
                letter_counts.get(r["predicted_letter"], 0) + 1
            )

    out = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "language": lang_code,
        "task": "classification_shuffled_mcq",
        "shuffle_seed_rule": "sha256(image_id)",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_images": total,
            "correct": n_correct,
            "errors": n_errors,
            "accuracy": round(acc_overall, 3),
            "per_group_accuracy": per_group,
            "predicted_letter_counts": letter_counts,
            "position_A_fraction": round(
                letter_counts["A"] / total if total else 0.0, 3
            ),
            "elapsed_time_seconds": elapsed,
        },
        "results": results,
    }

    out_path = output_dir / f"{experiment_id}_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info(f"Saved: {out_path}  (acc={acc_overall:.3f})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--language", default="both", choices=["zh", "en", "both"])
    ap.add_argument("--data", default="data/metadata.csv")
    ap.add_argument("--output", default="results/raw/shuffled_mcq")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--prompts-config", default="configs/prompts.yaml")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(level=args.log_level)

    langs = ["zh", "en"] if args.language == "both" else [args.language]
    for lang in langs:
        run_shuffled_classification(
            model_name=args.model,
            language=lang,
            data_path=args.data,
            output_dir=args.output,
            num_samples=args.num_samples,
            models_config_path=args.models_config,
            prompts_config_path=args.prompts_config,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
