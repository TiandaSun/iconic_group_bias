"""Google Gemini API wrapper."""

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

from src.models.base import (
    BaseVLM,
    VLMAPIError,
    VLMInferenceError,
    retry_with_exponential_backoff,
)

logger = logging.getLogger(__name__)


@dataclass
class CostTracker:
    """Track API usage costs."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    cost_per_1k_input: float = 0.00015
    cost_per_1k_output: float = 0.0006

    @property
    def total_cost(self) -> float:
        """Calculate total cost in USD."""
        input_cost = (self.total_input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (self.total_output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost

    def add_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from a request."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1

    def add_failure(self) -> None:
        """Record a failed request."""
        self.failed_requests += 1

    def summary(self) -> dict:
        """Get usage summary."""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
        }


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_minute: int = 1000):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute.
        """
        self.rpm = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()
        elapsed = now - self.last_request_time

        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class GeminiAPI(BaseVLM):
    """Wrapper for Google Gemini API (gemini-2.5-flash).

    This class handles API-based inference using the google-generativeai SDK,
    with built-in retry logic and cost tracking.

    Attributes:
        model: Gemini GenerativeModel instance.
        cost_tracker: Tracks API usage and costs.
        rate_limiter: Enforces rate limits.
    """

    def __init__(self, config: dict) -> None:
        """Initialize Gemini API client.

        Args:
            config: Model configuration dictionary.

        Raises:
            VLMAPIError: If API key is not set.
        """
        super().__init__(config)

        # Get API key from environment
        self.api_key = os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise VLMAPIError("GOOGLE_API_KEY environment variable not set", retryable=False)

        # Configuration
        self._timeout = config.get("timeout", 60)
        self._max_retries = config.get("max_retries", 5)
        self._rate_limit_rpm = config.get("rate_limit_rpm", 1000)

        # Initialize model lazily
        self._model = None
        self._configured = False

        # Cost tracking
        cost_per_1k_input = config.get("cost_per_1k_input", 0.00015)
        cost_per_1k_output = config.get("cost_per_1k_output", 0.0006)
        self.cost_tracker = CostTracker(
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )

        # Rate limiting
        self.rate_limiter = RateLimiter(requests_per_minute=self._rate_limit_rpm)

        logger.info(f"Initialized {self._name} API client (RPM limit: {self._rate_limit_rpm})")

    def _configure_api(self) -> None:
        """Configure the Gemini API."""
        if self._configured:
            return

        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        self._configured = True

    @property
    def model(self):
        """Lazy initialization of Gemini model."""
        if self._model is None:
            import google.generativeai as genai

            self._configure_api()

            # Configure generation settings
            generation_config = genai.GenerationConfig(
                max_output_tokens=self._max_tokens,
                temperature=0,
            )

            # Configure safety settings to be permissive for research
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            self._model = genai.GenerativeModel(
                model_name=self._model_path_or_id,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

        return self._model

    def _load_image_for_gemini(self, image_path: str, detail: str = "high") -> Image.Image:
        """Load image in format suitable for Gemini API.

        Args:
            image_path: Path to the image file.
            detail: Image detail level ('low' or 'high'). When 'low',
                resizes image to reduce token count.

        Returns:
            PIL Image object.
        """
        image = self.load_image(image_path)
        if detail == "low":
            image = self.resize_image(image, max_size=512)
        return image

    @retry_with_exponential_backoff(max_retries=5, initial_delay=1.0, max_delay=60.0)
    def _call_api(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int,
        detail: str = "high",
    ) -> Optional[str]:
        """Make API call with retry logic.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt.
            max_tokens: Maximum tokens to generate.
            detail: Image detail level ('low' or 'high').

        Returns:
            Generated text response.

        Raises:
            VLMAPIError: On retryable API errors.
        """
        import google.generativeai as genai
        from google.api_core import exceptions as google_exceptions

        self.rate_limiter.wait_if_needed()

        try:
            start_time = time.time()

            # Load image
            image = self._load_image_for_gemini(image_path, detail=detail)

            # Prepare content
            content = [image, prompt]

            # Generate response
            response = self.model.generate_content(
                content,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0,
                ),
            )

            # Track usage if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self.cost_tracker.add_usage(
                    input_tokens=response.usage_metadata.prompt_token_count or 0,
                    output_tokens=response.usage_metadata.candidates_token_count or 0,
                )
            else:
                # Estimate tokens if not provided
                self.cost_tracker.total_requests += 1

            inference_time = time.time() - start_time
            logger.debug(f"API call completed in {inference_time:.2f}s")

            # Handle blocked responses
            if not response.candidates:
                logger.warning("Response blocked by safety filters")
                return None

            return response.text.strip()

        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limit/quota exceeded: {e}")
            raise VLMAPIError(f"Rate limit exceeded: {e}", retryable=True)

        except google_exceptions.DeadlineExceeded as e:
            logger.warning(f"Request timeout: {e}")
            raise VLMAPIError(f"Request timeout: {e}", retryable=True)

        except google_exceptions.ServiceUnavailable as e:
            logger.warning(f"Service unavailable: {e}")
            raise VLMAPIError(f"Service unavailable: {e}", retryable=True)

        except google_exceptions.InternalServerError as e:
            logger.warning(f"Internal server error: {e}")
            raise VLMAPIError(f"Server error: {e}", retryable=True)

        except google_exceptions.InvalidArgument as e:
            logger.error(f"Invalid argument: {e}")
            raise VLMAPIError(f"Invalid argument: {e}", retryable=False)

        except Exception as e:
            logger.error(f"Unexpected Gemini API error: {e}")
            # Assume retryable for unknown errors
            raise VLMAPIError(f"API error: {e}", retryable=True)

    def _generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        detail: str = "high",
    ) -> Optional[str]:
        """Generate response for a single image-prompt pair.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt.
            max_tokens: Maximum tokens to generate.
            detail: Image detail level ('low' or 'high').

        Returns:
            Generated text response or None on persistent failure.
        """
        if max_tokens is None:
            max_tokens = self._max_tokens

        start_time = time.time()

        try:
            response = self._call_api(image_path, prompt, max_tokens, detail=detail)
            inference_time = time.time() - start_time
            logger.debug(f"Total inference time: {inference_time:.2f}s")
            return response

        except VLMAPIError as e:
            self.cost_tracker.add_failure()
            logger.error(f"API call failed after retries: {e}")
            return None

        except Exception as e:
            self.cost_tracker.add_failure()
            logger.error(f"Unexpected error: {e}")
            return None

    def classify(self, image_path: str, prompt: str) -> Optional[str]:
        """Classify the ethnic group from a costume image.

        Uses detail='low' (resized image) to reduce token cost.

        Args:
            image_path: Path to the input image file.
            prompt: Classification prompt.

        Returns:
            Predicted label letter (A-E) or None on failure.
        """
        start_time = time.time()

        response = self._generate(image_path, prompt, max_tokens=32, detail="low")

        if response is None:
            logger.warning(f"[{self._name}] Classification failed for {image_path}")
            return None

        try:
            label = self.extract_classification_label(response)
            inference_time = time.time() - start_time
            logger.info(
                f"[{self._name}] Classification: {label} "
                f"(time: {inference_time:.2f}s)"
            )
            return label

        except VLMInferenceError as e:
            logger.warning(f"[{self._name}] Could not extract label: {e}")
            return None

    def describe(self, image_path: str, prompt: str) -> Optional[str]:
        """Generate a cultural description of the costume.

        Uses detail='high' (full resolution) for rich visual descriptions.

        Args:
            image_path: Path to the input image file.
            prompt: Description prompt.

        Returns:
            Generated cultural description or None on failure.
        """
        start_time = time.time()

        response = self._generate(image_path, prompt, max_tokens=self._max_tokens, detail="high")

        if response is None:
            logger.warning(f"[{self._name}] Description failed for {image_path}")
            return None

        inference_time = time.time() - start_time
        logger.info(
            f"[{self._name}] Description generated "
            f"({len(response)} chars, time: {inference_time:.2f}s)"
        )

        return response

    def batch_classify(
        self,
        image_paths: List[str],
        prompt: str,
        batch_size: int = 1,
    ) -> List[Optional[str]]:
        """Batch classification for multiple images.

        Args:
            image_paths: List of image file paths.
            prompt: Classification prompt.
            batch_size: Ignored for API (sequential processing).

        Returns:
            List of predicted labels (None for failures).
        """
        results = []
        total_start = time.time()

        for i, path in enumerate(image_paths):
            label = self.classify(path, prompt)
            results.append(label)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - total_start
                logger.info(
                    f"Progress: {i + 1}/{len(image_paths)} "
                    f"({elapsed:.1f}s elapsed, "
                    f"cost: ${self.cost_tracker.total_cost:.4f})"
                )

        total_time = time.time() - total_start
        logger.info(
            f"Batch classification complete: {len(image_paths)} images "
            f"in {total_time:.2f}s (cost: ${self.cost_tracker.total_cost:.4f})"
        )

        return results

    def batch_describe(
        self,
        image_paths: List[str],
        prompt: str,
        batch_size: int = 1,
    ) -> List[Optional[str]]:
        """Batch description generation for multiple images.

        Args:
            image_paths: List of image file paths.
            prompt: Description prompt.
            batch_size: Ignored for API (sequential processing).

        Returns:
            List of generated descriptions (None for failures).
        """
        results = []
        total_start = time.time()

        for i, path in enumerate(image_paths):
            desc = self.describe(path, prompt)
            results.append(desc)

            if (i + 1) % 5 == 0:
                elapsed = time.time() - total_start
                logger.info(
                    f"Description progress: {i + 1}/{len(image_paths)} "
                    f"(cost: ${self.cost_tracker.total_cost:.4f})"
                )

        total_time = time.time() - total_start
        logger.info(
            f"Batch description complete: {len(image_paths)} images "
            f"in {total_time:.2f}s (cost: ${self.cost_tracker.total_cost:.4f})"
        )

        return results

    def get_cost_summary(self) -> dict:
        """Get API usage and cost summary.

        Returns:
            Dictionary with usage statistics and costs.
        """
        return self.cost_tracker.summary()

    def unload(self) -> None:
        """Clean up API client."""
        self._model = None
        self._configured = False
        logger.info(f"Unloaded {self._name} (Final cost: ${self.cost_tracker.total_cost:.4f})")
