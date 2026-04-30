"""Inference pipelines for classification and description tasks."""

from src.inference.task1_classification import (
    MODEL_REGISTRY,
    create_model,
    load_model_config,
    load_prompt,
    run_task1_classification,
)
from src.inference.task2_description import (
    run_all_models_task2,
    run_task2_description,
)

__all__ = [
    # Task 1
    "run_task1_classification",
    # Task 2
    "run_task2_description",
    "run_all_models_task2",
    # Utilities
    "create_model",
    "load_model_config",
    "load_prompt",
    "MODEL_REGISTRY",
]
