"""Task 2: Cultural description generation pipeline."""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    load_metadata,
    log_inference_stats,
    sample_task2_images,
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


def load_description_prompt(
    language: str,
    config_path: str = "configs/prompts.yaml",
    variant: str = "default",
) -> str:
    """Load description prompt for specified language.

    Args:
        language: Language code ('zh' or 'en').
        config_path: Path to prompts config file.
        variant: Prompt variant. "default" loads the seeded-vocabulary
            prompt used in the main experiments. "neutral" loads the
            craft-technique-neutral prompt (C4 sanity check) which
            omits Miao-coded seed words ("embroidery/batik/brocade"
            etc.) and is used to verify that the Dong->Miao in-text
            confusion is not an artefact of prompt vocabulary.

    Returns:
        Description prompt string.

    Raises:
        ValueError: If language or variant not found.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Prompts config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    key = (
        "task2_description" if variant == "default"
        else f"task2_description_{variant}"
    )
    task2_config = config.get(key, {})
    if not task2_config:
        raise ValueError(
            f"Prompt variant '{variant}' not found under key '{key}' "
            f"in {config_path}"
        )

    # Map language codes
    lang_key = "zh" if language in ["zh", "cn", "chinese"] else "en"

    if lang_key not in task2_config:
        raise ValueError(
            f"Language '{language}' not found for variant '{variant}'"
        )

    return task2_config[lang_key].strip()


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


def calculate_description_stats(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics about generated descriptions.

    Args:
        results: Dictionary of results.

    Returns:
        Statistics dictionary.
    """
    lengths = []
    by_ethnic_group = {}

    for img_id, result in results.items():
        desc = result.get("description", "")
        if desc and desc != "ERROR":
            lengths.append(len(desc))

            ethnic_group = result.get("ethnic_group", "Unknown")
            if ethnic_group not in by_ethnic_group:
                by_ethnic_group[ethnic_group] = []
            by_ethnic_group[ethnic_group].append(len(desc))

    if not lengths:
        return {"avg_length": 0, "min_length": 0, "max_length": 0}

    return {
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "total_descriptions": len(lengths),
        "by_ethnic_group": {
            group: {
                "count": len(lens),
                "avg_length": sum(lens) / len(lens) if lens else 0,
            }
            for group, lens in by_ethnic_group.items()
        },
    }


def run_task2_description(
    model_name: str,
    language: str,
    sample_csv: str,
    output_dir: str,
    checkpoint_interval: int = 50,
    image_base_dir: Optional[str] = None,
    models_config_path: str = "configs/models.yaml",
    prompts_config_path: str = "configs/prompts.yaml",
    experiment_id: Optional[str] = None,
    num_samples: Optional[int] = None,
    seed: int = 42,
    prompt_variant: str = "default",
) -> Dict[str, Any]:
    """Run Task 2 description generation on sampled images.

    Args:
        model_name: Model identifier (e.g., 'qwen2.5-vl-7b').
        language: Prompt language ('zh', 'cn', or 'en').
        sample_csv: Path to sample metadata CSV or full metadata for sampling.
        output_dir: Directory for saving results.
        checkpoint_interval: Save checkpoint every N images.
        image_base_dir: Base directory for image paths.
        models_config_path: Path to models config.
        prompts_config_path: Path to prompts config.
        experiment_id: Unique experiment identifier.
        num_samples: Number of images to sample (if full metadata provided).
        seed: Random seed for sampling.

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
            checkpoint_dir, model_name, "description", lang_code
        )
        if found_id:
            experiment_id = found_id
            logger.info(f"Resuming experiment: {experiment_id}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            variant_suffix = (
                "" if prompt_variant == "default" else f"_{prompt_variant}"
            )
            experiment_id = (
                f"task2{variant_suffix}_{model_name}_{lang_code}_{timestamp}"
            )
            logger.info(f"No checkpoint found, starting new experiment: {experiment_id}")

    logger.info(f"Starting Task 2 Description Generation")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Language: {lang_code}")
    logger.info(f"  Sample CSV: {sample_csv}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Experiment ID: {experiment_id}")

    # Load configurations
    logger.info("Loading configurations...")
    model_config = load_model_config(model_name, models_config_path)
    prompt = load_description_prompt(
        lang_code, prompts_config_path, variant=prompt_variant
    )
    logger.info(f"Using prompt variant: {prompt_variant}")
    logger.debug(f"Prompt:\n{prompt[:200]}...")

    # Load metadata
    logger.info("Loading sample metadata...")
    metadata = load_metadata(sample_csv, image_base_dir=image_base_dir)

    # Sample if num_samples specified
    if num_samples and num_samples < len(metadata):
        logger.info(f"Sampling {num_samples} images from {len(metadata)} total...")
        metadata = sample_task2_images(metadata, n=num_samples, seed=seed, stratified=True)

    total_images = len(metadata)
    logger.info(f"Processing {total_images} images for description generation")

    # Initialize checkpoint manager
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        experiment_id=experiment_id,
        model_name=model_name,
        task="description",
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
            results, model_name, lang_code, output_dir, experiment_id
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
        for idx, row in remaining_metadata.iterrows():
            img_id = row["image_id"]
            img_path = row["image_path"]
            ethnic_group = row["ethnic_group"]

            inference_start = time.time()

            try:
                # Generate description
                description = model.describe(img_path, prompt)
                inference_time = time.time() - inference_start

                if description is None:
                    description = "ERROR"
                    errors += 1

                result = {
                    "description": description,
                    "ethnic_group": ethnic_group,
                    "time_ms": int(inference_time * 1000),
                    "image_path": img_path,
                    "language": lang_code,
                }

            except Exception as e:
                logger.error(f"Description failed for {img_path}: {e}")
                errors += 1
                result = {
                    "description": "ERROR",
                    "ethnic_group": ethnic_group,
                    "time_ms": 0,
                    "image_path": img_path,
                    "language": lang_code,
                    "error": str(e),
                }

            results[img_id] = result
            checkpoint_manager.add_result(img_id, result, auto_save=True)
            processed += 1

            # Log progress
            total_processed = len(completed_ids) + processed

            if total_processed % 10 == 0 or total_processed == total_images:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (remaining_count - processed) / rate if rate > 0 else 0

                # Calculate description length stats
                desc_lengths = [
                    len(r["description"]) for r in results.values()
                    if r.get("description") and r["description"] != "ERROR"
                ]
                avg_length = sum(desc_lengths) / len(desc_lengths) if desc_lengths else 0

                logger.info(
                    f"[{model_name}] Lang: {lang_code} | "
                    f"Progress: {total_processed}/{total_images} "
                    f"({100 * total_processed / total_images:.1f}%) | "
                    f"Avg desc length: {avg_length:.0f} chars | "
                    f"Errors: {errors} | "
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

        # Log cost summary for API models
        if hasattr(model, "get_cost_summary"):
            cost_summary = model.get_cost_summary()
            logger.info(f"API Cost Summary: {cost_summary}")
            checkpoint_manager.set_metadata("cost_summary", cost_summary)

    # Finalize
    elapsed_time = time.time() - start_time
    checkpoint_manager.set_metadata("elapsed_time", elapsed_time)
    checkpoint_manager.set_metadata("errors", errors)

    return _finalize_results(
        results, model_name, lang_code, output_dir, experiment_id,
        elapsed_time=elapsed_time, errors=errors
    )


def _finalize_results(
    results: Dict[str, Dict[str, Any]],
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
        model_name: Model name.
        language: Language code.
        output_dir: Output directory.
        experiment_id: Experiment ID.
        elapsed_time: Total elapsed time.
        errors: Number of errors.

    Returns:
        Final results dictionary.
    """
    # Calculate statistics
    stats = calculate_description_stats(results)

    # Build final output
    final_output = {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "language": language,
        "task": "description",
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_images": len(results),
            "successful": len(results) - errors,
            "errors": errors,
            "elapsed_time_seconds": elapsed_time,
            "description_stats": stats,
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
        total_images=len(results),
        elapsed_time=elapsed_time,
        errors=errors,
        task=f"Task 2 Description ({language})",
        additional_stats={
            "avg_description_length": f"{stats.get('avg_length', 0):.0f} chars",
            "min_length": stats.get("min_length", 0),
            "max_length": stats.get("max_length", 0),
        },
    )

    return final_output


def run_all_models_task2(
    sample_csv: str,
    output_dir: str,
    models: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Run Task 2 for all models and languages.

    Args:
        sample_csv: Path to sample metadata CSV.
        output_dir: Base output directory.
        models: List of model names (default: all).
        languages: List of languages (default: ['zh', 'en']).
        **kwargs: Additional arguments for run_task2_description.

    Returns:
        Dictionary mapping (model, language) to results.
    """
    if models is None:
        models = list(MODEL_REGISTRY.keys())
    if languages is None:
        languages = ["zh", "en"]

    all_results = {}

    total_runs = len(models) * len(languages)
    current_run = 0

    for model_name in models:
        for language in languages:
            current_run += 1
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Run {current_run}/{total_runs}: {model_name} - {language}")
            logger.info(f"{'=' * 60}\n")

            try:
                results = run_task2_description(
                    model_name=model_name,
                    language=language,
                    sample_csv=sample_csv,
                    output_dir=output_dir,
                    **kwargs,
                )
                all_results[(model_name, language)] = results

            except Exception as e:
                logger.error(f"Failed: {model_name} - {language}: {e}")
                all_results[(model_name, language)] = {"error": str(e)}

    return all_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Task 2: Description Generation on VLMs",
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
        help="Path to sample metadata CSV file",
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
        "--checkpoint-interval",
        "-c",
        type=int,
        default=50,
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
        "--num-samples",
        "-n",
        type=int,
        default=None,
        help="Number of images to sample (if using full metadata)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
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
    parser.add_argument(
        "--prompt-variant",
        type=str,
        default="default",
        choices=["default", "neutral"],
        help=("Prompt variant: 'default' uses the seeded-vocabulary prompt "
              "used in the main experiments; 'neutral' uses the C4-sanity "
              "prompt without Miao-coded seed terms"),
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

    # Run description generation
    try:
        results = run_task2_description(
            model_name=args.model,
            language=args.language,
            sample_csv=args.data,
            output_dir=args.output,
            checkpoint_interval=args.checkpoint_interval,
            image_base_dir=args.image_base_dir,
            models_config_path=args.models_config,
            prompts_config_path=args.prompts_config,
            experiment_id=args.experiment_id,
            num_samples=args.num_samples,
            seed=args.seed,
            prompt_variant=args.prompt_variant,
        )

        # Print summary
        stats = results["summary"]["description_stats"]
        print(f"\n{'=' * 50}")
        print(f"Description Generation Complete")
        print(f"Total: {results['summary']['total_images']}")
        print(f"Successful: {results['summary']['successful']}")
        print(f"Avg Length: {stats.get('avg_length', 0):.0f} characters")
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
