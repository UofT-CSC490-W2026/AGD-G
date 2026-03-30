"""
LLaVA-backed surrogate model for text-targeted attacks.

This keeps the existing AttackVLMText attack class and swaps the surrogate
objective from CLIP similarity to LLaVA answer likelihood.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .base import TextTargetModel


class LlavaTextTargetModel(TextTargetModel):
    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "llava-hf/llava-1.5-7b-hf",
    ) -> None:
        super().__init__()
        self.device = device
        self.model_id = model_id

        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.dtype = torch.float16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.requires_grad_(False)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
        self.processor.num_additional_image_tokens = 1

        crop_size = getattr(self.processor.image_processor, "crop_size", {})
        self.image_size = int(crop_size.get("height", self.model.config.vision_config.image_size))
        self.image_mean = torch.tensor(self.processor.image_processor.image_mean, dtype=torch.float32)
        self.image_std = torch.tensor(self.processor.image_processor.image_std, dtype=torch.float32)
        self._dummy_pil = Image.new("RGB", self.get_image_size(), color=(0, 0, 0))

    def get_image_size(self) -> Tuple[int, int]:
        return (self.image_size, self.image_size)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        px = F.interpolate(images, size=self.get_image_size(), mode="bicubic", align_corners=False)
        mean = self.image_mean.to(px.device, px.dtype).view(1, 3, 1, 1)
        std = self.image_std.to(px.device, px.dtype).view(1, 3, 1, 1)
        return (px - mean) / std

    @staticmethod
    def _parse_question_answer(text: str) -> tuple[str, str]:
        if text.startswith("Question:") and "\nAnswer:" in text:
            question, answer = text.split("\nAnswer:", 1)
            question = question.replace("Question:", "", 1).strip()
            return question, answer.strip()
        return "What is the graph about?", text.strip()

    def _build_target_example(self, text: str) -> dict[str, torch.Tensor]:
        question, answer = self._parse_question_answer(text)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        answer_suffix = f" {answer}" if answer else ""
        full_text = f"{prompt}{answer_suffix}"

        prompt_tokens = self.processor(
            images=self._dummy_pil,
            text=prompt,
            return_tensors="pt",
        )
        full_tokens = self.processor(
            images=self._dummy_pil,
            text=full_text,
            return_tensors="pt",
        )
        labels = full_tokens["input_ids"].clone()
        prompt_token_count = prompt_tokens["input_ids"].shape[1]
        labels[:, :prompt_token_count] = -100

        return {
            "input_ids": full_tokens["input_ids"],
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }

    def embed_text(self, text: str | Sequence[str], detach: bool = True) -> Any:
        texts = [text] if isinstance(text, str) else list(text)
        examples = [self._build_target_example(item) for item in texts]
        return examples[0] if isinstance(text, str) else examples

    def embed_image(self, image: torch.Tensor, detach: bool = False) -> Any:
        return image.detach() if detach else image

    def __call__(self, image: torch.Tensor, text: Any) -> torch.Tensor:
        if torch.is_tensor(text):
            flat_image = image.reshape(image.shape[0], -1)
            flat_text = text.reshape(text.shape[0], -1)
            return F.cosine_similarity(flat_image, flat_text, dim=-1)

        examples = text if isinstance(text, list) else [text]
        losses = []
        for idx in range(image.shape[0]):
            example = examples[idx if len(examples) > 1 else 0]
            pixel_values = self._preprocess(image[idx:idx + 1]).to(self.device, self.dtype)
            outputs = self.model(
                input_ids=example["input_ids"].to(self.device),
                attention_mask=example["attention_mask"].to(self.device),
                labels=example["labels"].to(self.device),
                pixel_values=pixel_values,
            )
            losses.append(-outputs.loss)
        return torch.stack(losses)
