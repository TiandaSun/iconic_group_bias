"""Logging utilities for VLM evaluation."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ColoredFormatter(logging.Formatter):
    """Formatter with colored output for console."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color codes for console output
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


class ModelContextFilter(logging.Filter):
    """Filter that adds model context to log records."""

    def __init__(self, model_name: str = ""):
        super().__init__()
        self.model_name = model_name

    def filter(self, record: logging.LogRecord) -> bool:
        """Add model name to record."""
        record.model_name = self.model_name
        return True

    def set_model(self, model_name: str) -> None:
        """Update the current model name."""
        self.model_name = model_name


class ProgressTracker:
    """Track and log inference progress."""

    def __init__(
        self,
        total: int,
        model_name: str,
        task: str,
        log_interval: int = 100,
    ) -> None:
        """Initialize progress tracker.

        Args:
            total: Total number of items to process.
            model_name: Name of the model.
            task: Task description.
            log_interval: Log progress every N items.
        """
        self.total = total
        self.model_name = model_name
        self.task = task
        self.log_interval = log_interval

        self.processed = 0
        self.errors = 0
        self.start_time = datetime.now()

        self.logger = logging.getLogger(__name__)

    def update(self, n: int = 1, errors: int = 0) -> None:
        """Update progress.

        Args:
            n: Number of items processed.
            errors: Number of errors encountered.
        """
        self.processed += n
        self.errors += errors

        if self.processed % self.log_interval == 0 or self.processed == self.total:
            self._log_progress()

    def _log_progress(self) -> None:
        """Log current progress."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.processed / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.processed) / rate if rate > 0 else 0

        self.logger.info(
            f"[{self.model_name}] {self.task}: "
            f"{self.processed}/{self.total} ({100 * self.processed / self.total:.1f}%) | "
            f"Rate: {rate:.2f}/s | "
            f"Errors: {self.errors} | "
            f"ETA: {remaining / 60:.1f}min"
        )

    def finish(self) -> Dict[str, Any]:
        """Finish tracking and return summary.

        Returns:
            Dictionary with progress statistics.
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.processed / elapsed if elapsed > 0 else 0

        summary = {
            "model_name": self.model_name,
            "task": self.task,
            "total": self.total,
            "processed": self.processed,
            "errors": self.errors,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
            "success_rate": (self.processed - self.errors) / self.processed if self.processed > 0 else 0,
        }

        self.logger.info(
            f"[{self.model_name}] {self.task} complete: "
            f"{self.processed} images in {elapsed / 60:.1f}min "
            f"({rate:.2f}/s, {self.errors} errors)"
        )

        return summary


def setup_logging(
    log_dir: Optional[Union[str, Path]] = None,
    experiment_id: Optional[str] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    model_name: str = "",
) -> logging.Logger:
    """Setup logging with file and console handlers.

    Args:
        log_dir: Directory for log files. If None, only console logging.
        experiment_id: Experiment identifier for log filename.
        level: Root logger level.
        console_level: Console handler level.
        file_level: File handler level.
        model_name: Current model name for context.

    Returns:
        Configured root logger.
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create formatters
    console_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = ColoredFormatter(console_format, datefmt=date_format)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler if log_dir provided
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_id:
            log_filename = f"{experiment_id}_{timestamp}.log"
        else:
            log_filename = f"vlm_eval_{timestamp}.log"

        log_path = log_dir / log_filename

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter(file_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        root_logger.info(f"Logging to {log_path}")

    # Add model context filter
    model_filter = ModelContextFilter(model_name)
    root_logger.addFilter(model_filter)

    return root_logger


def log_inference_stats(
    model: str,
    total_images: int,
    elapsed_time: float,
    errors: int,
    task: str = "inference",
    additional_stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Log inference statistics summary.

    Args:
        model: Model name.
        total_images: Total images processed.
        elapsed_time: Total elapsed time in seconds.
        errors: Number of errors.
        task: Task description.
        additional_stats: Additional statistics to log.
    """
    logger = logging.getLogger(__name__)

    rate = total_images / elapsed_time if elapsed_time > 0 else 0
    success_rate = (total_images - errors) / total_images * 100 if total_images > 0 else 0

    logger.info("=" * 60)
    logger.info(f"INFERENCE STATISTICS: {model}")
    logger.info("=" * 60)
    logger.info(f"Task:           {task}")
    logger.info(f"Total images:   {total_images}")
    logger.info(f"Successful:     {total_images - errors}")
    logger.info(f"Errors:         {errors}")
    logger.info(f"Success rate:   {success_rate:.1f}%")
    logger.info(f"Elapsed time:   {elapsed_time / 60:.2f} minutes")
    logger.info(f"Average rate:   {rate:.2f} images/second")
    logger.info(f"Avg per image:  {elapsed_time / total_images:.2f} seconds" if total_images > 0 else "N/A")

    if additional_stats:
        logger.info("-" * 40)
        for key, value in additional_stats.items():
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")

    logger.info("=" * 60)


def log_experiment_config(
    experiment_id: str,
    config: Dict[str, Any],
) -> None:
    """Log experiment configuration.

    Args:
        experiment_id: Experiment identifier.
        config: Configuration dictionary.
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"EXPERIMENT CONFIGURATION: {experiment_id}")
    logger.info("=" * 60)

    def log_dict(d: dict, indent: int = 0) -> None:
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                log_dict(value, indent + 1)
            elif isinstance(value, list) and len(value) > 5:
                logger.info(f"{prefix}{key}: [{len(value)} items]")
            else:
                logger.info(f"{prefix}{key}: {value}")

    log_dict(config)
    logger.info("=" * 60)


def log_model_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = "accuracy",
) -> None:
    """Log comparison of results across models.

    Args:
        results: Dictionary mapping model names to their metrics.
        metric: Primary metric to compare.
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"MODEL COMPARISON: {metric.upper()}")
    logger.info("=" * 60)

    # Sort by metric value
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get(metric, 0),
        reverse=True,
    )

    for rank, (model, metrics) in enumerate(sorted_models, 1):
        value = metrics.get(metric, "N/A")
        if isinstance(value, float):
            logger.info(f"{rank}. {model}: {value:.4f}")
        else:
            logger.info(f"{rank}. {model}: {value}")

    logger.info("=" * 60)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
