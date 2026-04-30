"""Qwen2.5-VL model wrapper for 7B and 72B variants."""

import logging
import time
from typing import List, Optional, Union

import torch
from PIL import Image

from src.models.base import BaseVLM, VLMInferenceError

logger = logging.getLogger(__name__)


class QwenVL(BaseVLM):
    """Wrapper for Qwen2.5-VL models (7B and 72B variants).

    This class handles loading and inference for Qwen2.5-VL-Instruct models
    using the transformers library with the Qwen2-VL processor.

    Attributes:
        model: The loaded Qwen2-VL model.
        processor: The Qwen2-VL processor for input preparation.
        device: Device(s) the model is loaded on.
    """

    def __init__(self, config: dict) -> None:
        """Initialize Qwen2.5-VL model.

        Args:
            config: Model configuration dictionary.
        """
        super().__init__(config)

        self.model = None
        self.processor = None
        self._device = None
        self._is_multi_gpu = config.get("tensor_parallel", False)
        self._gpu_required = config.get("gpu_required", 1)

        # Lazy loading - model loads on first inference
        self._loaded = False

    def _is_qwen2_5_model(self) -> bool:
        """Check if the model is a Qwen2.5-VL variant.

        Returns:
            True if Qwen2.5-VL, False if Qwen2-VL.
        """
        model_id_lower = self._model_path_or_id.lower()
        return "qwen2.5" in model_id_lower or "qwen2_5" in model_id_lower

    def _load_model(self) -> None:
        """Load the model and processor."""
        if self._loaded:
            return

        try:
            from transformers import AutoProcessor

            # Import the correct model class based on model variant
            if self._is_qwen2_5_model():
                from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
                logger.info(f"Using Qwen2.5-VL model class for {self._name}")
            else:
                from transformers import Qwen2VLForConditionalGeneration as ModelClass
                logger.info(f"Using Qwen2-VL model class for {self._name}")

            logger.info(f"Loading {self._name} from {self._model_path_or_id}...")
            start_time = time.time()

            # Determine device configuration
            if self._is_multi_gpu and torch.cuda.device_count() >= self._gpu_required:
                # Multi-GPU loading for 72B model
                logger.info(
                    f"Loading with device_map='auto' across {self._gpu_required} GPUs"
                )
                self.model = ModelClass.from_pretrained(
                    self._model_path_or_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                self._device = "auto"
            else:
                # Single GPU loading
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = ModelClass.from_pretrained(
                    self._model_path_or_id,
                    torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
                    device_map=self._device,
                    trust_remote_code=True,
                )

            self.processor = AutoProcessor.from_pretrained(
                self._model_path_or_id,
                trust_remote_code=True,
            )

            self.model.eval()
            self._loaded = True

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s on device: {self._device}")

        except Exception as e:
            raise VLMInferenceError(f"Failed to load Qwen2.5-VL model: {e}")

    def _prepare_messages(
        self, image: Union[str, Image.Image], prompt: str
    ) -> list:
        """Prepare messages in Qwen2-VL chat format.

        Args:
            image: Image path or PIL Image object.
            prompt: Text prompt.

        Returns:
            List of messages in chat format.
        """
        if isinstance(image, str):
            image_content = {"type": "image", "image": image}
        else:
            image_content = {"type": "image", "image": image}

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return messages

    def _generate(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate response for a single image-prompt pair.

        Args:
            image: Image path or PIL Image object.
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
            # Prepare input
            messages = self._prepare_messages(image, prompt)

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            if isinstance(image, str):
                image_obj = self.load_image(image)
            else:
                image_obj = image

            inputs = self.processor(
                text=[text],
                images=[image_obj],
                padding=True,
                return_tensors="pt",
            )

            # Move inputs to device
            if self._device != "auto":
                inputs = inputs.to(self._device)
            else:
                inputs = inputs.to(self.model.device)

            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            # Decode response
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            response = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            inference_time = time.time() - start_time
            logger.debug(f"Inference time: {inference_time:.2f}s")

            return response.strip()

        except Exception as e:
            raise VLMInferenceError(f"Generation failed: {e}")
        finally:
            # Memory management
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
            prompt: Classification prompt (same for all images).
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
                batch_results = self._batch_generate(batch_paths, prompt, max_new_tokens=32)

                for response in batch_results:
                    try:
                        label = self.extract_classification_label(response)
                    except VLMInferenceError:
                        label = "INVALID"
                        logger.warning(f"Could not extract label from: {response[:50]}")
                    results.append(label)

                batch_time = time.time() - batch_start
                logger.info(
                    f"Batch {i // batch_size + 1}: processed {len(batch_paths)} images "
                    f"in {batch_time:.2f}s ({batch_time / len(batch_paths):.2f}s/image)"
                )

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Fallback to individual processing
                for path in batch_paths:
                    try:
                        label = self.classify(path, prompt)
                    except Exception:
                        label = "ERROR"
                    results.append(label)

            # Memory cleanup between batches
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
        # Prepare all inputs
        texts = []
        images = []

        for path in image_paths:
            messages = self._prepare_messages(path, prompt)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)
            images.append(self.load_image(path))

        # Process batch
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        if self._device != "auto":
            inputs = inputs.to(self._device)
        else:
            inputs = inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # Decode responses
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        responses = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return [r.strip() for r in responses]

    def batch_describe(
        self,
        image_paths: List[str],
        prompt: str,
        batch_size: int = 2,
    ) -> List[str]:
        """Batch description generation for multiple images.

        Args:
            image_paths: List of image file paths.
            prompt: Description prompt (same for all images).
            batch_size: Number of images to process at once (smaller for descriptions).

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
                # Fallback to individual processing
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
