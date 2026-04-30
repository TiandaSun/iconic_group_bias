"""Anthropic Claude API wrapper."""

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

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
    cost_per_1k_input: float = 0.003
    cost_per_1k_output: float = 0.015

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

    def __init__(self, requests_per_minute: int = 50):
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


class ClaudeAPI(BaseVLM):
    """Wrapper for Anthropic Claude API (claude-haiku-4.5).

    This class handles API-based inference using the Anthropic Python SDK,
    with built-in rate limiting, retry logic, and cost tracking.

    Attributes:
        client: Anthropic API client.
        cost_tracker: Tracks API usage and costs.
        rate_limiter: Enforces rate limits.
    """

    def __init__(self, config: dict) -> None:
        """Initialize Claude API client.

        Args:
            config: Model configuration dictionary.

        Raises:
            VLMAPIError: If API key is not set.
        """
        super().__init__(config)

        # Get API key from environment
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise VLMAPIError("ANTHROPIC_API_KEY environment variable not set", retryable=False)

        # Configuration
        self._timeout = config.get("timeout", 60)
        self._max_retries = config.get("max_retries", 5)
        self._rate_limit_rpm = config.get("rate_limit_rpm", 50)

        # Initialize client lazily
        self._client = None

        # Cost tracking (Claude 3.5 Sonnet pricing)
        cost_per_1k_input = config.get("cost_per_1k_input", 0.003)
        cost_per_1k_output = config.get("cost_per_1k_output", 0.015)
        self.cost_tracker = CostTracker(
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        )

        # Rate limiting (Claude has stricter limits)
        self.rate_limiter = RateLimiter(requests_per_minute=self._rate_limit_rpm)

        logger.info(f"Initialized {self._name} API client (RPM limit: {self._rate_limit_rpm})")

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(
                api_key=self.api_key,
                timeout=self._timeout,
            )
        return self._client

    def _create_image_content(self, image_path: str, detail: str = "high") -> dict:
        """Create image content block for Claude API.

        Args:
            image_path: Path to the image file.
            detail: Image detail level ('low' or 'high'). When 'low',
                resizes image to reduce token count.

        Returns:
            Image content block with base64 data.
        """
        if detail == "low":
            # Resize to small image to reduce token cost for classification
            image = self.load_image(image_path)
            image = self.resize_image(image, max_size=512)
            base64_image = self.image_to_base64(image)
        else:
            base64_image = self.image_to_base64(image_path)

        # Determine media type from file extension
        extension = image_path.lower().split(".")[-1]
        media_type_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }
        media_type = media_type_map.get(extension, "image/jpeg")

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_image,
            },
        }

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
        from anthropic import (
            APIError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

        self.rate_limiter.wait_if_needed()

        try:
            start_time = time.time()

            # Prepare message content
            content = [
                self._create_image_content(image_path, detail=detail),
                {"type": "text", "text": prompt},
            ]

            # Make API call
            response = self.client.messages.create(
                model=self._model_path_or_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )

            # Track usage
            self.cost_tracker.add_usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            inference_time = time.time() - start_time
            logger.debug(
                f"API call completed in {inference_time:.2f}s "
                f"(in: {response.usage.input_tokens}, out: {response.usage.output_tokens})"
            )

            # Extract text from response
            if response.content and len(response.content) > 0:
                return response.content[0].text.strip()

            logger.warning("Empty response from Claude API")
            return None

        except RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise VLMAPIError(f"Rate limit exceeded: {e}", retryable=True)

        except APITimeoutError as e:
            logger.warning(f"API timeout: {e}")
            raise VLMAPIError(f"API timeout: {e}", retryable=True)

        except InternalServerError as e:
            logger.warning(f"Internal server error: {e}")
            raise VLMAPIError(f"Server error: {e}", retryable=True)

        except APIError as e:
            logger.error(f"API error: {e}")
            # Check if error is likely transient
            if hasattr(e, "status_code") and e.status_code >= 500:
                raise VLMAPIError(f"Server error: {e}", retryable=True)
            raise VLMAPIError(f"API error: {e}", retryable=False)

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

        Note: Claude API processes sequentially due to rate limits.

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
        self._client = None
        logger.info(f"Unloaded {self._name} (Final cost: ${self.cost_tracker.total_cost:.4f})")
