"""OpenAI GPT-4 Vision API wrapper."""

import logging
import os
import time
from dataclasses import dataclass, field
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

    def __init__(self, requests_per_minute: int = 500):
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


class GPT4VisionAPI(BaseVLM):
    """Wrapper for OpenAI GPT-4 Vision API (gpt-4o-mini).

    This class handles API-based inference using the OpenAI Python SDK,
    with built-in rate limiting and cost tracking.

    Attributes:
        client: OpenAI API client.
        cost_tracker: Tracks API usage and costs.
        rate_limiter: Enforces rate limits.
    """

    def __init__(self, config: dict) -> None:
        """Initialize GPT-4 Vision API client.

        Args:
            config: Model configuration dictionary.

        Raises:
            VLMAPIError: If API key is not set.
        """
        super().__init__(config)

        # Get API key from environment
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise VLMAPIError("OPENAI_API_KEY environment variable not set", retryable=False)

        # Configuration
        self._timeout = config.get("timeout", 60)
        self._max_retries = config.get("max_retries", 5)
        self._rate_limit_rpm = config.get("rate_limit_rpm", 500)

        # Initialize client lazily
        self._client = None

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

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=self._timeout,
            )
        return self._client

    def _create_image_content(self, image_path: str, detail: str = "high") -> dict:
        """Create image content for API request.

        Args:
            image_path: Path to the image file.
            detail: Image detail level ('low' or 'high'). Low uses fixed
                85 tokens per image; high tiles the image at ~170 tokens/tile.

        Returns:
            Image content dictionary with base64 data.
        """
        base64_image = self.image_to_base64(image_path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": detail,
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
            Generated text response or None on failure.

        Raises:
            VLMAPIError: On retryable API errors.
        """
        from openai import APIError, APITimeoutError, RateLimitError

        self.rate_limiter.wait_if_needed()

        try:
            start_time = time.time()

            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        self._create_image_content(image_path, detail=detail),
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Make API call
            response = self.client.chat.completions.create(
                model=self._model_path_or_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )

            # Track usage
            if response.usage:
                self.cost_tracker.add_usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                )

            inference_time = time.time() - start_time
            logger.debug(f"API call completed in {inference_time:.2f}s")

            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise VLMAPIError(f"Rate limit exceeded: {e}", retryable=True)

        except APITimeoutError as e:
            logger.warning(f"API timeout: {e}")
            raise VLMAPIError(f"API timeout: {e}", retryable=True)

        except APIError as e:
            logger.error(f"API error: {e}")
            if e.status_code and e.status_code >= 500:
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

        Uses detail='low' to minimize image token cost (85 tokens/image
        instead of hundreds with high detail tiling).

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

        Uses detail='high' for rich visual information needed for descriptions.

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

        Note: API calls are sequential due to rate limiting.

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
