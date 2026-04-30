"""LLaMA 3.2 Vision model wrapper for 11B variant."""

import logging
import time
from typing import List, Optional, Union

import torch
from PIL import Image

from src.models.base import BaseVLM, VLMInferenceError

logger = logging.getLogger(__name__)


class LLaMAVision(BaseVLM):
    """Wrapper for LLaMA 3.2 Vision 11B model.

    This class handles loading and inference for the LLaMA 3.2 Vision
    Instruct model using transformers MllamaForConditionalGeneration.

    Attributes:
        model: The loaded LLaMA Vision model.
        processor: The processor for input preparation.
    """

    def __init__(self, config: dict) -> None:
        """Initialize LLaMA Vision model.

        Args:
            config: Model configuration dictionary.
        """
        super().__init__(config)

        self.model = None
        self.processor = None
        self._device = None

        self._loaded = False

    def _load_model(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            return

        try:
            from transformers import AutoProcessor, MllamaForConditionalGeneration

            logger.info(f"Loading {self._name} from {self._model_path_or_id}...")
            start_time = time.time()

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
                dtype = torch.bfloat16
            else:
                self._device = "cpu"
                dtype = torch.float32
                logger.warning("No GPU available, using CPU (slow)")

            # Load model - use explicit device to avoid meta tensor issues
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self._model_path_or_id,
                torch_dtype=dtype,
                device_map=self._device,
                trust_remote_code=True,
            )

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self._model_path_or_id,
                trust_remote_code=True,
            )

            self.model.eval()
            self._loaded = True

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s on device: {self._device}")

        except Exception as e:
            raise VLMInferenceError(f"Failed to load LLaMA Vision model: {e}")

    def _prepare_messages(
        self, image: Union[str, Image.Image], prompt: str
    ) -> list:
        """Prepare messages in LLaMA Vision chat format.

        Args:
            image: Image path or PIL Image object.
            prompt: Text prompt.

        Returns:
            List of messages in chat format.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages

    def _generate(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate response for a single image-prompt pair.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.
        """
        self._load_model()

        if max_new_tokens is None:
            max_new_tokens = self._max_tokens

        start_time = time.time()

        try:
            # Load image
            image = self.load_image(image_path)

            # Prepare messages
            messages = self._prepare_messages(image, prompt)

            # Apply chat template
            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                images=image,
                text=input_text,
                return_tensors="pt",
            ).to(self._device)

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            # Decode response (remove input tokens)
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )

            inference_time = time.time() - start_time
            logger.debug(f"Inference time: {inference_time:.2f}s")

            return response.strip()

        except Exception as e:
            raise VLMInferenceError(f"Generation failed: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def classify(self, image_path: str, prompt: str) -> str:
        """Classify the ethnic group from a costume image.

        Args:
            image_path: Path to the input image file.
            prompt: Classification prompt.

        Returns:
            Predicted label letter (A, B, C, D, or E).
        """
        start_time = time.time()

        response = self._generate(image_path, prompt, max_new_tokens=32)
        label = self.extract_classification_label(response)

        inference_time = time.time() - start_time
        logger.info(
            f"[{self._name}] Classification: {label} "
            f"(time: {inference_time:.2f}s)"
        )

        return label

    def describe(self, image_path: str, prompt: str) -> str:
        """Generate a cultural description of the costume.

        Args:
            image_path: Path to the input image file.
            prompt: Description prompt.

        Returns:
            Generated cultural description text.
        """
        start_time = time.time()

        response = self._generate(image_path, prompt, max_new_tokens=self._max_tokens)

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
        batch_size: int = 4,
    ) -> List[str]:
        """Batch classification for multiple images.

        Args:
            image_paths: List of image file paths.
            prompt: Classification prompt.
            batch_size: Number of images to process at once.

        Returns:
            List of predicted labels.
        """
        self._load_model()

        results = []
        total_start = time.time()

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_start = time.time()

            try:
                batch_results = self._batch_generate(
                    batch_paths, prompt, max_new_tokens=32
                )

                for response in batch_results:
                    try:
                        label = self.extract_classification_label(response)
                    except VLMInferenceError:
                        label = "INVALID"
                        logger.warning(f"Could not extract label from: {response[:50]}")
                    results.append(label)

                batch_time = time.time() - batch_start
                logger.info(
                    f"Batch {i // batch_size + 1}: {len(batch_paths)} images "
                    f"in {batch_time:.2f}s ({batch_time / len(batch_paths):.2f}s/image)"
                )

            except Exception as e:
                logger.error(f"Batch processing failed, falling back: {e}")
                for path in batch_paths:
                    try:
                        label = self.classify(path, prompt)
                    except Exception:
                        label = "ERROR"
                    results.append(label)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.time() - total_start
        logger.info(
            f"Batch classification complete: {len(image_paths)} images "
            f"in {total_time:.2f}s ({total_time / len(image_paths):.2f}s/image avg)"
        )

        return results

    def _batch_generate(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 32,
    ) -> List[str]:
        """Generate responses for a batch of images.

        Args:
            image_paths: List of image file paths.
            prompt: Text prompt (same for all images).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            List of generated responses.
        """
        # Load images
        images = [self.load_image(path) for path in image_paths]

        # Prepare inputs for batch
        messages = self._prepare_messages(images[0], prompt)
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # Process batch
        inputs = self.processor(
            images=images,
            text=[input_text] * len(images),
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode responses
        responses = []
        for i in range(len(image_paths)):
            generated = output_ids[i, inputs.input_ids.shape[1]:]
            response = self.processor.decode(generated, skip_special_tokens=True)
            responses.append(response.strip())

        return responses

    def batch_describe(
        self,
        image_paths: List[str],
        prompt: str,
        batch_size: int = 2,
    ) -> List[str]:
        """Batch description generation for multiple images.

        Args:
            image_paths: List of image file paths.
            prompt: Description prompt.
            batch_size: Number of images to process at once.

        Returns:
            List of generated descriptions.
        """
        self._load_model()

        results = []
        total_start = time.time()

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_start = time.time()

            try:
                batch_results = self._batch_generate(
                    batch_paths, prompt, max_new_tokens=self._max_tokens
                )
                results.extend(batch_results)

                batch_time = time.time() - batch_start
                logger.info(
                    f"Description batch {i // batch_size + 1}: {len(batch_paths)} images "
                    f"in {batch_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Batch description failed: {e}")
                for path in batch_paths:
                    try:
                        desc = self.describe(path, prompt)
                    except Exception:
                        desc = "ERROR"
                    results.append(desc)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_time = time.time() - total_start
        logger.info(
            f"Batch description complete: {len(image_paths)} images "
            f"in {total_time:.2f}s"
        )

        return results

    def unload(self) -> None:
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Unloaded {self._name}")
