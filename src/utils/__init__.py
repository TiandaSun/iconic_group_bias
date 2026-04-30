"""Utility functions for data loading, checkpointing, and logging."""

from src.utils.checkpoint import (
    CheckpointManager,
    find_latest_checkpoint,
    get_all_checkpoints,
    resume_from_checkpoint,
)
from src.utils.data_loader import (
    ETHNIC_GROUP_TO_LABEL,
    LABEL_TO_ETHNIC_GROUP,
    create_metadata_from_directory,
    get_image_paths_from_batch,
    get_labels_from_batch,
    get_task1_batches,
    load_metadata,
    sample_task2_images,
    split_by_language,
)
from src.utils.logging_utils import (
    ProgressTracker,
    get_logger,
    log_experiment_config,
    log_inference_stats,
    log_model_comparison,
    setup_logging,
)

__all__ = [
    # Data loading
    "load_metadata",
    "get_task1_batches",
    "get_image_paths_from_batch",
    "get_labels_from_batch",
    "sample_task2_images",
    "create_metadata_from_directory",
    "split_by_language",
    "ETHNIC_GROUP_TO_LABEL",
    "LABEL_TO_ETHNIC_GROUP",
    # Checkpointing
    "CheckpointManager",
    "find_latest_checkpoint",
    "resume_from_checkpoint",
    "get_all_checkpoints",
    # Logging
    "setup_logging",
    "get_logger",
    "log_inference_stats",
    "log_experiment_config",
    "log_model_comparison",
    "ProgressTracker",
]
