"""Abstract base class for Vision-Language Model inference."""

import base64
import logging
import re
import time
from abc import ABC, abstractmethod
from functools import wraps
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

from PIL import Image

logger = logging.getLogger(__name__)

T = TypeVar("T")


class VLMError(Exception):
    """Base exception for VLM-related errors."""

    pass


class VLMAPIError(VLMError):
    """Exception for API-related errors (rate limits, timeouts, etc.)."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class VLMInferenceError(VLMError):
    """Exception for inference-related errors."""

    pass


def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (VLMAPIError,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        exponential_base: Base for exponential backoff calculation.
        retryable_exceptions: Tuple of exception types that trigger retry.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    # Check if exception is marked as non-retryable
                    if isinstance(e, VLMAPIError) and not e.retryable:
                        raise

                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * exponential_base, max_delay)

            # Should not reach here, but satisfy type checker
            raise last_exception or VLMError("Unexpected retry loop exit")

        return wrapper

    return decorator


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Model inference.

    This class defines the interface for all VLM implementations, whether
    they are local open-source models or proprietary API-based models.

    Attributes:
        config: Model configuration dictionary.
        _name: Model name.
        _origin: Model origin ('chinese' or 'western').
        _model_type: Model type ('open-source' or 'proprietary').
    """

    # Valid classification outputs
    VALID_LABELS = {"A", "B", "C", "D", "E"}

    # Label to ethnic group mapping
    LABEL_MAPPING = {
        "A": "Miao",
        "B": "Dong",
        "C": "Yi",
        "D": "Li",
        "E": "Tibetan",
    }

    def __init__(self, config: dict) -> None:
        """Initialize the VLM with configuration.

        Args:
            config: Model configuration dictionary containing:
                - name: Model display name
                - origin: 'chinese' or 'western'
                - type: 'open-source' or 'proprietary'
                - model_path_or_id: HuggingFace ID or API model name
                - max_tokens: Maximum tokens for generation
        """
        self.config = config
        self._name = config.get("name", "Unknown")
        self._origin = config.get("origin", "unknown")
        self._model_type = config.get("type", "unknown")
        self._model_path_or_id = config.get("model_path_or_id", "")
        self._max_tokens = config.get("max_tokens", 512)

        logger.info(f"Initialized {self._name} (origin: {self._origin})")

    @property
    def name(self) -> str:
        """Get the model name.

        Returns:
            Model display name.
        """
        return self._name

    @property
    def origin(self) -> str:
        """Get the model origin.

        Returns:
            Model origin: 'chinese' or 'western'.
        """
        return self._origin

    @property
    def model_type(self) -> str:
        """Get the model type.

        Returns:
            Model type: 'open-source' or 'proprietary'.
        """
        return self._model_type

    @property
    def model_id(self) -> str:
        """Get the model path or ID.

        Returns:
            HuggingFace model ID or API model name.
        """
        return self._model_path_or_id

    @abstractmethod
    def classify(self, image_path: str, prompt: str) -> str:
        """Classify the ethnic group from a costume image.

        Args:
            image_path: Path to the input image file.
            prompt: Classification prompt (Chinese or English).

        Returns:
            Predicted label letter (A, B, C, D, or E).

        Raises:
            VLMInferenceError: If classification fails.
        """
        pass

    @abstractmethod
    def describe(self, image_path: str, prompt: str) -> str:
        """Generate a cultural description of the costume.

        Args:
            image_path: Path to the input image file.
            prompt: Description prompt (Chinese or English).

        Returns:
            Generated cultural description text.

        Raises:
            VLMInferenceError: If description generation fails.
        """
        pass

    def load_image(self, image_path: str) -> Image.Image:
        """Load and validate an image from file path.

        Args:
            image_path: Path to the image file.

        Returns:
            PIL Image object.

        Raises:
            VLMError: If image cannot be loaded.
        """
        path = Path(image_path)
        if not path.exists():
            raise VLMError(f"Image file not found: {image_path}")

        try:
            image = Image.open(path)
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            raise VLMError(f"Failed to load image {image_path}: {e}")

    def image_to_base64(
        self, image: Union[str, Image.Image], format: str = "JPEG"
    ) -> str:
        """Convert an image to base64 string.

        Args:
            image: Either a file path string or PIL Image object.
            format: Output image format (JPEG, PNG, etc.).

        Returns:
            Base64-encoded image string.

        Raises:
            VLMError: If conversion fails.
        """
        try:
            if isinstance(image, str):
                image = self.load_image(image)

            buffer = BytesIO()
            image.save(buffer, format=format)
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode("utf-8")
        except VLMError:
            raise
        except Exception as e:
            raise VLMError(f"Failed to convert image to base64: {e}")

    def resize_image(
        self,
        image: Image.Image,
        max_size: int = 1024,
        maintain_aspect: bool = True,
    ) -> Image.Image:
        """Resize image to fit within max dimensions.

        Args:
            image: PIL Image object.
            max_size: Maximum dimension (width or height).
            maintain_aspect: Whether to maintain aspect ratio.

        Returns:
            Resized PIL Image object.
        """
        width, height = image.size

        if width <= max_size and height <= max_size:
            return image

        if maintain_aspect:
            ratio = min(max_size / width, max_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
        else:
            new_width = new_height = max_size

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def extract_classification_label(self, response: str) -> str:
        """Extract classification label from model response.

        This method implements robust parsing to handle various response
        formats from different models.

        Args:
            response: Raw model response text.

        Returns:
            Extracted label (A, B, C, D, or E).

        Raises:
            VLMInferenceError: If no valid label can be extracted.
        """
        if not response:
            raise VLMInferenceError("Empty response from model")

        # Clean the response
        cleaned = response.strip().upper()

        # Strategy 1: Check if response is exactly a single letter
        if cleaned in self.VALID_LABELS:
            return cleaned

        # Strategy 2: Check if response starts with a valid letter
        if cleaned and cleaned[0] in self.VALID_LABELS:
            # Verify it's followed by non-letter or end of string
            if len(cleaned) == 1 or not cleaned[1].isalpha():
                return cleaned[0]

        # Strategy 3: Look for patterns like "A)", "A.", "A:", "(A)", "[A]"
        patterns = [
            r"^([A-E])\s*[\)\.\:\]>]",  # A), A., A:, A], A>
            r"^\(([A-E])\)",  # (A)
            r"^\[([A-E])\]",  # [A]
            r"^答案[是为：:\s]*([A-E])",  # Chinese: 答案是A, 答案为A
            r"^answer[:\s]*([A-E])",  # English: answer: A
            r"^选择?\s*([A-E])",  # Chinese: 选A, 选择A
            r"^choice[:\s]*([A-E])",  # English: choice: A
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Strategy 4: Find first standalone letter A-E in the response
        # Look for letter surrounded by word boundaries or punctuation
        match = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", cleaned)
        if match:
            return match.group(1)

        # Strategy 5: Look for ethnic group names and map back to letters
        reverse_mapping = {v.upper(): k for k, v in self.LABEL_MAPPING.items()}
        # Also add Chinese names
        chinese_names = {
            "苗": "A",
            "苗族": "A",
            "侗": "B",
            "侗族": "B",
            "彝": "C",
            "彝族": "C",
            "黎": "D",
            "黎族": "D",
            "藏": "E",
            "藏族": "E",
        }
        reverse_mapping.update(chinese_names)

        for name, label in reverse_mapping.items():
            if name in cleaned or name in response:
                logger.debug(f"Extracted label {label} from ethnic group name: {name}")
                return label

        # If all strategies fail, raise an error
        raise VLMInferenceError(
            f"Could not extract valid label from response: {response[:100]}..."
        )

    def validate_response(
        self, response: str, task: str = "classification"
    ) -> bool:
        """Validate that the response is appropriate for the task.

        Args:
            response: Model response text.
            task: Task type ('classification' or 'description').

        Returns:
            True if response is valid, False otherwise.
        """
        if not response or not response.strip():
            return False

        if task == "classification":
            try:
                self.extract_classification_label(response)
                return True
            except VLMInferenceError:
                return False
        elif task == "description":
            # Description should have minimum length
            return len(response.strip()) >= 50
        else:
            return True

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(name={self._name}, origin={self._origin})"
