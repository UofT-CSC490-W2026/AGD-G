import re
from typing import Optional, List

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

from .base import RawResult, TargetingModel

SYSTEM_INSTRUCTION = (
    "You are an expert in data visualization and visual association.\n\n"
    "Your task is to analyze a provided chart image and generate a brief caption "
    "for a new chart. This new chart must share the same visual structure (chart type) "
    "as the original image but feature a different, highly iconic, and tangible subject.\n\n"
    "Rules:\n"
    "1. Identify the chart type from the image (e.g., Bar chart, Heatmap, Pie chart, Line graph).\n"
    "2. Identify the main subject of the original chart.\n"
    "3. Find the single most iconic, concrete, visual item that an average person "
    "immediately associates with that subject "
    "(e.g., Europe -> vacation spots, Italy -> pizza toppings, Depression -> illness rates).\n"
    "4. The new subject must be a photographable, physical object or real scene -- not abstract.\n"
    "5. Your final caption must be exactly 3 to 6 words long.\n"
    "6. Output your step-by-step reasoning inside <thinking> tags, "
    "then your final caption inside <output> tags."
)

FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "A bar chart showing European export volumes by country.",
        "<thinking>\n"
        "Chart type: Bar chart.\n"
        "Original subject: European export volumes.\n"
        "Core concept: Europe.\n"
        "Tangible association: Vacation spots.\n"
        "Drafting caption: Bar chart of European vacation spots.\n"
        "Word count check: 6 words. This is acceptable.\n"
        "</thinking>\n"
        "<output>Bar chart of European vacation spots</output>",
    ),
    (
        "A heatmap of depression rates by region.",
        "<thinking>\n"
        "Chart type: Heatmap.\n"
        "Original subject: Depression rates by region.\n"
        "Core concept: Depression.\n"
        "Tangible association: Illness.\n"
        "Drafting caption: Heatmap of regional illness rates.\n"
        "Word count check: 5 words. This is acceptable.\n"
        "</thinking>\n"
        "<output>Heatmap of regional illness rates</output>",
    ),
    (
        "A pie chart of Italian economic output.",
        "<thinking>\n"
        "Chart type: Pie chart.\n"
        "Original subject: Italian economic output.\n"
        "Core concept: Italy.\n"
        "Tangible association: Pizza toppings.\n"
        "Drafting caption: Pie chart of Italian pizza toppings.\n"
        "Word count check: 6 words. This is acceptable.\n"
        "</thinking>\n"
        "<output>Pie chart of Italian pizza toppings</output>",
    ),
]

_OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.DOTALL)
_THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)


def _parse_output(raw: str) -> str:
    """Extract the caption from ``<output>`` tags, falling back to *raw*."""
    match = _OUTPUT_RE.search(raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


def _parse_thinking(raw: str) -> str:
    """Extract the reasoning trace from ``<thinking>`` tags, returning empty string if absent."""
    match = _THINKING_RE.search(raw)
    return match.group(1).strip() if match else ""


class QwenTargetingModel(TargetingModel):
    def __init__(self, model_tag: Optional[str] = "Qwen/Qwen2.5-VL-7B-Instruct", device="cuda") -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_tag)
        self.model = AutoModelForImageTextToText.from_pretrained(model_tag).to(self.device)

        self.model.eval()
        self.model.requires_grad_(False)

    def __call__(self, images: List[Image.Image], clean_texts: List[str]) -> List[str]:
        """
        Convert image/clean_text pairs to target texts which differ from the clean texts.
        Returns parsed target captions only (thinking trace is discarded).
        """
        return [r["target"] for r in self._generate(images, clean_texts, max_new_tokens=40)]

    def generate_raw(self, images: List[Image.Image], clean_texts: List[str]) -> List[RawResult]:
        """
        Like ``__call__`` but returns the full model output including the thinking trace.
        Uses a larger token budget (200) to ensure the complete trace is captured.
        """
        return self._generate(images, clean_texts, max_new_tokens=200)

    def _generate(
        self,
        images: List[Image.Image],
        clean_texts: List[str],
        max_new_tokens: int,
    ) -> List[RawResult]:
        """Shared inference logic used by both ``__call__`` and ``generate_raw``."""
        messages = [self.get_message(img, txt) for img, txt in zip(images, clean_texts)]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        results: List[RawResult] = []
        for output in outputs:
            generated = output[input_len:]
            raw = self.processor.decode(generated, skip_special_tokens=True)
            results.append({"target": _parse_output(raw), "thinking": _parse_thinking(raw)})
        return results

    def get_message(self, image: Image.Image, clean_text: str) -> list[dict]:
        messages: list[dict] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}],
            },
        ]

        for user_text, assistant_text in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            })

        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": clean_text},
            ],
        })

        return messages
