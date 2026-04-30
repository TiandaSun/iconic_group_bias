#!/usr/bin/env python3
"""Main inference script for running VLM evaluations.

Unified CLI for running Task 1 (classification) or Task 2 (description)
on any model with any language prompt.

Usage:
    # Run single model/language
    python scripts/run_inference.py --model qwen2.5-vl-7b --task 1 --language zh

    # Run both languages
    python scripts/run_inference.py --model gpt-4o-mini --task 1 --language both

    # Run Task 2 (description)
    python scripts/run_inference.py --model claude-haiku-4.5 --task 2 --language en

    # Resume from checkpoint
    python scripts/run_inference.py --model qwen2.5-vl-72b --task 1 --language zh --resume

    # Run all models for a task
    python scripts/run_inference.py --model all --task 1 --language both
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.task1_classification import run_task1_classification, MODEL_REGISTRY
from src.inference.task2_description import run_task2_description
from src.utils import setup_logging

logger = logging.getLogger(__name__)

# All available models
ALL_MODELS = list(MODEL_REGISTRY.keys())


def run_inference(
    model: str,
    task: int,
    language: str,
    data_path: str,
    output_dir: str,
    batch_size: int = 1,
    checkpoint_interval: int = 100,
    resume: bool = False,
    num_samples: int = None,
    seed: int = 42,
    limit: int = None,
) -> dict:
    """Run inference for a single model/task/language combination.

    Args:
        model: Model name.
        task: Task number (1 or 2).
        language: Language code ('zh', 'en').
        data_path: Path to data CSV.
        output_dir: Output directory.
        batch_size: Batch size for inference.
        checkpoint_interval: Save checkpoint every N images.
        resume: Whether to resume from checkpoint.
        num_samples: Number of samples for Task 2.
        seed: Random seed.
        limit: Maximum number of images to process (for testing).

    Returns:
        Results dictionary.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"task{task}_{model}_{language}_{timestamp}"

    if task == 1:
        return run_task1_classification(
            model_name=model,
            language=language,
            data_path=data_path,
            output_dir=output_dir,
            batch_size=batch_size,
            checkpoint_interval=checkpoint_interval,
            experiment_id=experiment_id if not resume else None,
            limit=limit,
            num_samples=num_samples,
            seed=seed,
        )
    elif task == 2:
        # Default to 500 samples for Task 2 if not specified
        task2_samples = num_samples if num_samples is not None else 500
        if limit is not None:
            task2_samples = min(task2_samples, limit)
        return run_task2_description(
            model_name=model,
            language=language,
            sample_csv=data_path,
            output_dir=output_dir,
            checkpoint_interval=checkpoint_interval,
            experiment_id=experiment_id if not resume else None,
            num_samples=task2_samples,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM inference for classification or description tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classification with Chinese prompts
  python scripts/run_inference.py --model qwen2.5-vl-7b --task 1 --language zh --data data/metadata.csv

  # Classification with both languages
  python scripts/run_inference.py --model gpt-4o-mini --task 1 --language both --data data/metadata.csv

  # Description generation
  python scripts/run_inference.py --model claude-haiku-4.5 --task 2 --language en --data data/metadata.csv --num-samples 500

  # Run all models
  python scripts/run_inference.py --model all --task 1 --language both --data data/metadata.csv

Available models: """ + ", ".join(ALL_MODELS)
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help=f"Model name or 'all'. Choices: {ALL_MODELS + ['all']}",
    )
    parser.add_argument(
        "--task", "-t",
        type=int,
        required=True,
        choices=[1, 2],
        help="Task: 1 (classification) or 2 (description)",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        required=True,
        choices=["zh", "cn", "en", "both"],
        help="Prompt language: zh/cn, en, or both",
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to metadata CSV file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/raw",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for local models",
    )
    parser.add_argument(
        "--checkpoint-interval", "-c",
        type=int,
        default=100,
        help="Save checkpoint every N images",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing checkpoint",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=None,
        help="Number of samples (stratified for Task 1, random for Task 2). "
             "If not set, Task 1 uses all images and Task 2 defaults to 500.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results/logs",
        help="Log directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images (for testing). Processes only first N images.",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_dir=args.log_dir, level=log_level)

    # Determine models to run
    if args.model.lower() == "all":
        models = ALL_MODELS
    elif args.model in ALL_MODELS:
        models = [args.model]
    else:
        # Try partial matching
        matches = [m for m in ALL_MODELS if args.model.lower() in m.lower()]
        if matches:
            models = matches
        else:
            logger.error(f"Unknown model: {args.model}")
            logger.info(f"Available models: {ALL_MODELS}")
            return 1

    # Determine languages
    if args.language.lower() == "both":
        languages = ["zh", "en"]
    else:
        languages = [args.language]

    # Run inference
    total_runs = len(models) * len(languages)
    current_run = 0
    all_results = {}

    logger.info(f"Starting {total_runs} inference run(s)")
    logger.info(f"Models: {models}")
    logger.info(f"Languages: {languages}")
    logger.info(f"Task: {args.task}")

    for model in models:
        for language in languages:
            current_run += 1
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Run {current_run}/{total_runs}: {model} - {language}")
            logger.info(f"{'=' * 60}\n")

            try:
                results = run_inference(
                    model=model,
                    task=args.task,
                    language=language,
                    data_path=args.data,
                    output_dir=args.output,
                    batch_size=args.batch_size,
                    checkpoint_interval=args.checkpoint_interval,
                    resume=args.resume,
                    num_samples=args.num_samples,
                    seed=args.seed,
                    limit=args.limit,
                )
                all_results[(model, language)] = results

                # Print summary
                if "summary" in results:
                    summary = results["summary"]
                    if args.task == 1:
                        logger.info(f"Accuracy: {summary.get('accuracy', 0):.2%}")
                    else:
                        logger.info(f"Descriptions: {summary.get('total_images', 0)}")

            except KeyboardInterrupt:
                logger.warning("Interrupted by user")
                return 1
            except Exception as e:
                logger.error(f"Failed: {model} - {language}: {e}")
                all_results[(model, language)] = {"error": str(e)}

    # Final summary
    print(f"\n{'=' * 60}")
    print("INFERENCE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Runs completed: {len(all_results)}/{total_runs}")
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
