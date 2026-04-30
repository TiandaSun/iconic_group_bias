"""Checkpoint management for inference progress."""

import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage inference checkpoints for graceful interruption and resumption.

    This class handles saving and loading of inference progress, enabling
    experiments to be resumed after HPC job interruptions.

    Attributes:
        checkpoint_dir: Directory for storing checkpoint files.
        experiment_id: Unique identifier for the experiment.
        model_name: Name of the model being evaluated.
        task: Task type ('classification' or 'description').
        language: Prompt language ('zh' or 'en').
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        experiment_id: str,
        model_name: str,
        task: str,
        language: str,
        save_interval: int = 100,
    ) -> None:
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files.
            experiment_id: Unique experiment identifier.
            model_name: Name of the model.
            task: Task type ('classification' or 'description').
            language: Prompt language.
            save_interval: Save checkpoint every N images.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.task = task
        self.language = language
        self.save_interval = save_interval

        # State
        self._completed_ids: Set[str] = set()
        self._results: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        self._images_since_save = 0

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        logger.info(
            f"CheckpointManager initialized: {self.checkpoint_path} "
            f"(save every {save_interval} images)"
        )

    @property
    def checkpoint_filename(self) -> str:
        """Generate checkpoint filename."""
        safe_model = self.model_name.replace("/", "_").replace(" ", "_")
        return f"ckpt_{self.experiment_id}_{safe_model}_{self.task}_{self.language}.json"

    @property
    def checkpoint_path(self) -> Path:
        """Full path to checkpoint file."""
        return self.checkpoint_dir / self.checkpoint_filename

    @property
    def completed_ids(self) -> Set[str]:
        """Get set of completed image IDs."""
        return self._completed_ids.copy()

    @property
    def num_completed(self) -> int:
        """Get number of completed images."""
        return len(self._completed_ids)

    @property
    def results(self) -> Dict[str, Any]:
        """Get current results dictionary."""
        return self._results.copy()

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, saving checkpoint...")
            self.save()
            logger.info("Checkpoint saved, exiting gracefully")
            sys.exit(0)

        # Handle common termination signals
        signal.signal(signal.SIGTERM, signal_handler)  # HPC job termination
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C

        # SIGUSR1 for checkpoint without exit (HPC warning signal)
        def checkpoint_handler(signum, frame):
            logger.info("Received SIGUSR1, saving checkpoint...")
            self.save()
            logger.info("Checkpoint saved, continuing...")

        try:
            signal.signal(signal.SIGUSR1, checkpoint_handler)
        except (AttributeError, ValueError):
            pass  # Not available on all platforms

    def load(self) -> bool:
        """Load checkpoint if it exists.

        Returns:
            True if checkpoint was loaded, False otherwise.
        """
        if not self.checkpoint_path.exists():
            logger.info("No existing checkpoint found, starting fresh")
            return False

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Validate checkpoint matches current experiment
            if data.get("model_name") != self.model_name:
                logger.warning(
                    f"Checkpoint model mismatch: {data.get('model_name')} vs {self.model_name}"
                )
                return False

            if data.get("task") != self.task:
                logger.warning(
                    f"Checkpoint task mismatch: {data.get('task')} vs {self.task}"
                )
                return False

            if data.get("language") != self.language:
                logger.warning(
                    f"Checkpoint language mismatch: {data.get('language')} vs {self.language}"
                )
                return False

            # Load state
            self._completed_ids = set(data.get("completed_ids", []))
            self._results = data.get("results", {})
            self._metadata = data.get("metadata", {})

            logger.info(
                f"Loaded checkpoint: {len(self._completed_ids)} completed images, "
                f"last saved {data.get('last_saved', 'unknown')}"
            )
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def save(self) -> None:
        """Save current progress to checkpoint file."""
        data = {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "task": self.task,
            "language": self.language,
            "completed_ids": list(self._completed_ids),
            "results": self._results,
            "metadata": self._metadata,
            "num_completed": len(self._completed_ids),
            "last_saved": datetime.now().isoformat(),
        }

        # Write to temp file first for atomic save
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Atomic rename
            temp_path.replace(self.checkpoint_path)
            self._images_since_save = 0

            logger.debug(f"Checkpoint saved: {len(self._completed_ids)} images")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()

    def add_result(
        self,
        image_id: str,
        result: Any,
        auto_save: bool = True,
    ) -> None:
        """Add a result and mark image as completed.

        Args:
            image_id: Unique image identifier.
            result: Inference result (prediction, description, etc.).
            auto_save: Whether to auto-save based on interval.
        """
        self._completed_ids.add(image_id)
        self._results[image_id] = result
        self._images_since_save += 1

        if auto_save and self._images_since_save >= self.save_interval:
            self.save()

    def add_results_batch(
        self,
        image_ids: List[str],
        results: List[Any],
        auto_save: bool = True,
    ) -> None:
        """Add multiple results at once.

        Args:
            image_ids: List of image identifiers.
            results: List of corresponding results.
            auto_save: Whether to auto-save based on interval.
        """
        for image_id, result in zip(image_ids, results):
            self._completed_ids.add(image_id)
            self._results[image_id] = result

        self._images_since_save += len(image_ids)

        if auto_save and self._images_since_save >= self.save_interval:
            self.save()

    def is_completed(self, image_id: str) -> bool:
        """Check if an image has been processed.

        Args:
            image_id: Image identifier to check.

        Returns:
            True if image has been processed.
        """
        return image_id in self._completed_ids

    def get_result(self, image_id: str) -> Optional[Any]:
        """Get result for a specific image.

        Args:
            image_id: Image identifier.

        Returns:
            Result if available, None otherwise.
        """
        return self._results.get(image_id)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set experiment metadata.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get experiment metadata.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            Metadata value or default.
        """
        return self._metadata.get(key, default)

    def finalize(self) -> Dict[str, Any]:
        """Finalize and save checkpoint, return final results.

        Returns:
            Dictionary with all results and metadata.
        """
        self._metadata["finalized"] = True
        self._metadata["finalized_at"] = datetime.now().isoformat()
        self._metadata["total_completed"] = len(self._completed_ids)

        self.save()

        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "task": self.task,
            "language": self.language,
            "results": self._results,
            "metadata": self._metadata,
        }

    def clear(self) -> None:
        """Clear all checkpoint data (use with caution)."""
        self._completed_ids.clear()
        self._results.clear()
        self._metadata.clear()
        self._images_since_save = 0

        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

        logger.info("Checkpoint cleared")


def find_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
    model_name: str,
    task: str,
    language: str,
) -> Optional[str]:
    """Find the latest checkpoint experiment_id for a given model/task/language.

    Searches checkpoint files in the directory for one matching the specified
    model, task, and language, returning the experiment_id from the most recent.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        model_name: Model name to match.
        task: Task type ('classification' or 'description').
        language: Prompt language ('zh' or 'en').

    Returns:
        The experiment_id from the latest matching checkpoint, or None if not found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    best_ckpt = None
    best_time = ""

    for ckpt_file in checkpoint_dir.glob("ckpt_*.json"):
        try:
            with open(ckpt_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if (
                data.get("model_name") == model_name
                and data.get("task") == task
                and data.get("language") == language
                and not data.get("metadata", {}).get("finalized", False)
            ):
                last_saved = data.get("last_saved", "")
                if last_saved > best_time:
                    best_time = last_saved
                    best_ckpt = data.get("experiment_id")
        except Exception:
            continue

    if best_ckpt:
        logger.info(
            f"Found latest checkpoint for {model_name}/{task}/{language}: {best_ckpt}"
        )
    return best_ckpt


def resume_from_checkpoint(
    checkpoint_dir: Union[str, Path],
    experiment_id: str,
    model_name: str,
    task: str,
    language: str,
    image_ids: List[str],
) -> tuple:
    """Filter image IDs to only include unprocessed ones.

    Convenience function for resuming experiments.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        experiment_id: Experiment identifier.
        model_name: Model name.
        task: Task type.
        language: Prompt language.
        image_ids: Full list of image IDs to process.

    Returns:
        Tuple of (remaining_ids, checkpoint_manager, existing_results).
    """
    manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        experiment_id=experiment_id,
        model_name=model_name,
        task=task,
        language=language,
    )

    # Try to load existing checkpoint
    manager.load()

    # Filter to unprocessed images
    remaining_ids = [
        img_id for img_id in image_ids
        if not manager.is_completed(img_id)
    ]

    completed = len(image_ids) - len(remaining_ids)
    if completed > 0:
        logger.info(
            f"Resuming: {completed}/{len(image_ids)} already completed, "
            f"{len(remaining_ids)} remaining"
        )

    return remaining_ids, manager, manager.results


def get_all_checkpoints(checkpoint_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """List all checkpoints in a directory.

    Args:
        checkpoint_dir: Directory to search.

    Returns:
        List of checkpoint metadata dictionaries.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []

    for ckpt_file in checkpoint_dir.glob("ckpt_*.json"):
        try:
            with open(ckpt_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            checkpoints.append({
                "file": str(ckpt_file),
                "experiment_id": data.get("experiment_id"),
                "model_name": data.get("model_name"),
                "task": data.get("task"),
                "language": data.get("language"),
                "num_completed": data.get("num_completed", 0),
                "last_saved": data.get("last_saved"),
                "finalized": data.get("metadata", {}).get("finalized", False),
            })
        except Exception as e:
            logger.warning(f"Could not read checkpoint {ckpt_file}: {e}")

    return sorted(checkpoints, key=lambda x: x.get("last_saved", ""), reverse=True)
