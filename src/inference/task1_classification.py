"""Task 1: Classification inference pipeline."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from src.models import (
    BaseVLM,
    ClaudeAPI,
    GeminiAPI,
    GPT4VisionAPI,
    LLaMAVision,
    QwenVL,
)
from src.utils import (
    CheckpointManager,
    find_latest_checkpoint,
    get_image_paths_from_batch,
    get_labels_from_batch,
    get_task1_batches,
    load_metadata,
    log_inference_stats,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Model name to class mapping
MODEL_REGISTRY = {
    # Open-source models (Chinese-origin)
    "qwen2.5-vl-7b": QwenVL,
    "qwen2-vl-7b": QwenVL,
    # Open-source models (Western-origin)
    "llama-3.2-vision-11b": LLaMAVision,
    # Proprietary models (API)
    "gpt-4o-mini": GPT4VisionAPI,
    "gemini-2.5-flash": GeminiAPI,
    "claude-3.5-sonnet": ClaudeAPI,
    "claude-haiku-4.5": ClaudeAPI,
}


def load_model_config(model_name: str, config_path: str = "configs/models.yaml") -> dict:
    """Load model configuration from YAML file.

    Args:
        model_name: Model identifier.
        config_path: Path to models config file.

    Returns:
        Model configuration dictionary.

    Raises:
        ValueError: If model not found in config.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models_config = config.get("models", {})
    if model_name not in models_config:
        available = list(models_config.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")

    return models_config[model_name]


def load_prompt(language: str, config_path: str = "configs/prompts.yaml") -> str:
    """Load classification prompt for specified language.

    Args:
        language: Language code ('zh' or 'en').
        config_path: Path to prompts config file.

    Returns:
        Classification prompt string.

    Raises:
        ValueError: If language not found.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Prompts config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task1_config = config.get("task1_classification", {})

    # Map language codes
    lang_key = "zh" if language in ["zh", "cn", "chinese"] else "en"

    if lang_key not in task1_config:
        raise ValueError(f"Language '{language}' not found in prompts config")

    return task1_config[lang_key].strip()


def create_model(model_name: str, config: dict) -> BaseVLM:
    """Create model instance based on model name.

    Args:
        model_name: Model identifier.
        config: Model configuration dictionary.

    Returns:
        Initialized model wrapper.

    Raises:
        ValueError: If model type not supported.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Supported: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(config)


def calculate_running_accuracy(
    results: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, str],
) -> Tuple[float, int, int]:
    """Calculate running accuracy from results.

    Args:
        results: Dictionary of {image_id: {predicted: ..., ...}}.
        ground_truth: Dictionary of {image_id: label_letter}.

    Returns:
        Tuple of (accuracy, correct_count, total_count).
    """
    correct = 0
    total = 0

    for image_id, result in results.items():
        if image_id in ground_truth:
            predicted = result.get("predicted")
            actual = ground_truth[image_id]
            if predicted == actual:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def run_task1_classification(
    model_name: str,
    language: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 1,
    checkpoint_interval: int = 100,
    image_base_dir: Optional[str] = None,
    models_config_path: str = "configs/models.yaml",
    prompts_config_path: str = "configs/prompts.yaml",
    experiment_id: Optional[str] = None,
    limit: Optional[int] = None,
    num_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Task 1 classification on all images.

    Args:
        model_name: Model identifier (e.g., 'qwen2.5-vl-7b').
        language: Prompt language ('zh', 'cn', or 'en').
        data_path: Path to metadata CSV file.
        output_dir: Directory for saving results.
        batch_size: Number of images per batch (for local models).
        checkpoint_interval: Save checkpoint every N images.
        image_base_dir: Base directory for image paths.
        models_config_path: Path to models config.
        prompts_config_path: Path to prompts config.
        experiment_id: Unique experiment identifier.
        limit: Maximum number of images to process (for testing).
        num_samples: If set, use stratified sampling to select this many
            images total (equal per ethnic group). Useful for reducing
            API costs while maintaining balanced evaluation.
        seed: Random seed for stratified sampling.

    Returns:
        Dictionary with results and statistics.
    """
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize language code
    lang_code = "zh" if language in ["zh", "cn", "chinese"] else "en"

    if experiment_id is None:
        # Resume mode: search for latest matching checkpoint
        checkpoint_dir = Path(output_dir) / "checkpoints"
        found_id = find_latest_checkpoint(
            checkpoint_dir, model_name, "classification", lang_code
        )
        if found_id:
            experiment_id = found_id
            logger.info(f"Resuming experiment: {experiment_id}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"task1_{model_name}_{lang_code}_{timestamp}"
            logger.info(f"No checkpoint found, starting new experiment: {experiment_id}")

    logger.info(f"Starting Task 1 Classification")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Language: {lang_code}")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Experiment ID: {experiment_id}")

    # Load configurations
    logger.info("Loading configurations...")
    model_config = load_model_config(model_name, models_config_path)
    prompt = load_prompt(lang_code, prompts_config_path)
    logger.debug(f"Prompt:\n{prompt[:200]}...")

    # Load metadata
    logger.info("Loading metadata...")
    metadata = load_metadata(data_path, image_base_dir=image_base_dir)

    # Apply stratified sampling if num_samples specified
    if num_samples is not None and num_samples > 0 and num_samples < len(metadata):
        groups = metadata["ethnic_group"].unique()
        per_group = num_samples // len(groups)
        sampled_dfs = []
        for group in sorted(groups):
            group_df = metadata[metadata["ethnic_group"] == group]
            n = min(per_group, len(group_df))
            sampled_dfs.append(group_df.sample(n=n, random_state=seed))
        metadata = pd.concat(sampled_dfs).reset_index(drop=True)
        logger.info(
            f"Stratified sampling: {num_samples} total "
            f"({per_group} per group, {len(groups)} groups)"
        )

    # Apply limit if specified (for testing)
    if limit is not None and limit > 0:
        logger.info(f"Limiting to first {limit} images (test mode)")
        metadata = metadata.head(limit)

    total_images = len(metadata)
    logger.info(f"Loaded {total_images} images")

    # Create ground truth mapping
    ground_truth = dict(zip(metadata["image_id"], metadata["label_letter"]))

    # Initialize checkpoint manager
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        experiment_id=experiment_id,
        model_name=model_name,
        task="classification",
        language=lang_code,
        save_interval=checkpoint_interval,
    )

    # Try to resume from checkpoint
    checkpoint_manager.load()
    completed_ids = checkpoint_manager.completed_ids
    results = checkpoint_manager.results

    if len(completed_ids) > 0:
        logger.info(f"Resuming from checkpoint: {len(completed_ids)}/{total_images} completed")

    # Filter to unprocessed images
    remaining_metadata = metadata[~metadata["image_id"].isin(completed_ids)]
    remaining_count = len(remaining_metadata)

    if remaining_count == 0:
        logger.info("All images already processed!")
        return _finalize_results(
            results, ground_truth, model_name, lang_code, output_dir, experiment_id
        )

    logger.info(f"Processing {remaining_count} remaining images")

    # Create model
    logger.info(f"Loading model: {model_name}...")
    model = create_model(model_name, model_config)

    # Inference loop
    start_time = time.time()
    errors = 0
    processed = 0

    try:
        for batch in get_task1_batches(remaining_metadata, batch_size=batch_size):
            batch_start = time.time()

            image_paths = get_image_paths_from_batch(batch)
            image_ids = batch["image_id"].tolist()
            true_labels = get_labels_from_batch(batch)

            # Process batch
            if batch_size > 1 and hasattr(model, "batch_classify"):
                # Batch inference for local models
                try:
                    predictions = model.batch_classify(image_paths, prompt, batch_size=batch_size)
                except Exception as e:
                    logger.warning(f"Batch inference failed, falling back to single: {e}")
                    predictions = []
                    for path in image_paths:
                        try:
                            pred = model.classify(path, prompt)
                            predictions.append(pred)
                        except Exception as ex:
                            logger.error(f"Single inference failed for {path}: {ex}")
                            predictions.append(None)
            else:
                # Single image inference
                predictions = []
                for path in image_paths:
                    try:
                        pred = model.classify(path, prompt)
                        predictions.append(pred)
                    except Exception as e:
                        logger.error(f"Inference failed for {path}: {e}")
                        predictions.append(None)

            batch_time = time.time() - batch_start

            # Process results
            for i, (img_id, pred, true_label, img_path) in enumerate(
                zip(image_ids, predictions, true_labels, image_paths)
            ):
                time_ms = int((batch_time / len(image_ids)) * 1000)

                result = {
                    "predicted": pred,
                    "ground_truth": true_label,
                    "correct": pred == true_label if pred else False,
                    "time_ms": time_ms,
                    "image_path": img_path,
                }

                if pred is None:
                    errors += 1
                    result["error"] = True

                results[img_id] = result
                checkpoint_manager.add_result(img_id, result, auto_save=True)
                processed += 1

            # Log progress
            total_processed = len(completed_ids) + processed
            accuracy, correct, total = calculate_running_accuracy(results, ground_truth)

            if total_processed % 100 == 0 or total_processed == total_images:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (remaining_count - processed) / rate if rate > 0 else 0

                logger.info(
                    f"[{model_name}] Lang: {lang_code} | "
                    f"Progress: {total_processed}/{total_images} "
                    f"({100 * total_processed / total_images:.1f}%) | "
                    f"Accuracy so far: {100 * accuracy:.1f}% | "
                    f"Rate: {rate:.2f}/s | "
                    f"ETA: {eta_seconds / 60:.1f}min"
                )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user, saving checkpoint...")
        checkpoint_manager.save()
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        checkpoint_manager.save()
        raise

    finally:
        # Cleanup model
        if hasattr(model, "unload"):
            model.unload()

    # Finalize
    elapsed_time = time.time() - start_time
    checkpoint_manager.set_metadata("elapsed_time", elapsed_time)
    checkpoint_manager.set_metadata("errors", errors)

    return _finalize_results(
        results, ground_truth, model_name, lang_code, output_dir, experiment_id,
        elapsed_time=elapsed_time, errors=errors
    )


def _finalize_results(
    results: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, str],
    model_name: str,
    language: str,
    output_dir: Path,
    experiment_id: str,
    elapsed_time: float = 0,
    errors: int = 0,
) -> Dict[str, Any]:
    """Finalize and save results.

    Args:
        results: Dictionary of results.
        ground_truth: Ground truth labels.
        model_name: Model name.
        language: Language code.
        output_dir: Output directory.
        experiment_id: Experiment ID.
        elapsed_time: Total elapsed time.
        errors: Number of errors.

    Returns:
        Final results dictionary.
    """
    # Calculate final metrics
    accuracy, correct, total = calculate_running_accuracy(results, ground_truth)

    # Per-class accuracy
    class_correct = {label: 0 for label in "ABCDE"}
    class_total = {label: 0 for label in "ABCDE"}

    for img_id, result in results.items():
        true_label = ground_truth.get(img_id)
        if true_label:
            class_total[true_label] += 1
            if result.get("correct"):
                class_correct[true_label] += 1

    per_class_accuracy = {
        label: class_correct[label] / class_total[label] if class_total[label] > 0 else 0
        for label in "ABCDE"
    }

    # Build final output
    final_output = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "language": language,
        "task": "classification",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_images": total,
            "correct": correct,
            "errors": errors,
            "accuracy": accuracy,
            "per_class_accuracy": per_class_accuracy,
            "elapsed_time_seconds": elapsed_time,
        },
        "results": results,
    }

    # Save results
    output_file = output_dir / f"{experiment_id}_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Log summary
    log_inference_stats(
        model=model_name,
        total_images=total,
        elapsed_time=elapsed_time,
        errors=errors,
        task=f"Task 1 Classification ({language})",
        additional_stats={
            "accuracy": f"{100 * accuracy:.2f}%",
            "per_class_accuracy": {k: f"{100 * v:.2f}%" for k, v in per_class_accuracy.items()},
        },
    )

    return final_output


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Task 1: Classification on VLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to evaluate",
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        required=True,
        choices=["zh", "cn", "en"],
        help="Prompt language (zh/cn for Chinese, en for English)",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        required=True,
        help="Path to metadata CSV file",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/raw",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size for inference (local models only)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        "-c",
        type=int,
        default=100,
        help="Save checkpoint every N images",
    )
    parser.add_argument(
        "--image-base-dir",
        type=str,
        default=None,
        help="Base directory for image paths",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default="configs/models.yaml",
        help="Path to models configuration file",
    )
    parser.add_argument(
        "--prompts-config",
        type=str,
        default="configs/prompts.yaml",
        help="Path to prompts configuration file",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results/logs",
        help="Directory for log files",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(
        log_dir=args.log_dir,
        experiment_id=args.experiment_id,
        level=log_level,
        model_name=args.model,
    )

    # Run classification
    try:
        results = run_task1_classification(
            model_name=args.model,
            language=args.language,
            data_path=args.data,
            output_dir=args.output,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            image_base_dir=args.image_base_dir,
            models_config_path=args.models_config,
            prompts_config_path=args.prompts_config,
            experiment_id=args.experiment_id,
        )

        # Print final accuracy
        accuracy = results["summary"]["accuracy"]
        print(f"\n{'=' * 50}")
        print(f"Final Accuracy: {100 * accuracy:.2f}%")
        print(f"Results saved to: {args.output}")
        print(f"{'=' * 50}")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Checkpoint saved.")
        return 1

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
