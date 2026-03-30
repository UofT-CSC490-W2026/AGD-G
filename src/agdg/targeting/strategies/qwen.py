import re
from typing import Optional, List

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

from .base import RawResult, TargetingModel

SYSTEM_INSTRUCTION = (
    "You are an expert in data visualization.\n\n"
    "Your task is to analyze a provided chart image and generate a brief caption "
    "for a completely unrelated chart. This new chart must share the same visual "
    "structure (chart type) as the original image but describe a subject from an "
    "entirely different domain.\n\n"
    "Rules:\n"
    "1. Identify the chart type from the image (e.g., Bar chart, Heatmap, Pie chart, Line graph).\n"
    "2. Identify the domain of the original chart (e.g., finance, healthcare, sports).\n"
    "3. Choose a subject from a completely different domain -- it must share NO words, "
    "concepts, or associations with the original subject. "
    "(e.g., retail sales -> animal migration patterns, "
    "election results -> ocean temperature changes, "
    "hospital patient counts -> recipe ingredient ratios).\n"
    "4. The new subject must be a concrete, real-world topic -- not abstract.\n"
    "5. Your final caption must be exactly 3 to 6 words long.\n"
    "6. Output your step-by-step reasoning inside <thinking> tags, "
    "then your final caption inside <output> tags."
)

FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "A bar chart showing European export volumes by country.",
        "<thinking>\n"
        "Chart type: Bar chart.\n"
        "Original domain: International trade / economics.\n"
        "I must pick a completely different domain with no overlap.\n"
        "Different domain: Wildlife biology.\n"
        "New subject: Bird species population by continent.\n"
        "Drafting caption: Bar chart of bird species populations.\n"
        "Word count check: 6 words. Acceptable.\n"
        "</thinking>\n"
        "<output>Bar chart of bird species populations</output>",
    ),
    (
        "A heatmap of depression rates by region.",
        "<thinking>\n"
        "Chart type: Heatmap.\n"
        "Original domain: Mental health / epidemiology.\n"
        "I must pick a completely different domain with no overlap.\n"
        "Different domain: Culinary arts.\n"
        "New subject: Spice usage intensity across cuisines.\n"
        "Drafting caption: Heatmap of global spice usage.\n"
        "Word count check: 5 words. Acceptable.\n"
        "</thinking>\n"
        "<output>Heatmap of global spice usage</output>",
    ),
    (
        "A line graph of quarterly software sales revenue.",
        "<thinking>\n"
        "Chart type: Line graph.\n"
        "Original domain: Technology / business revenue.\n"
        "I must pick a completely different domain with no overlap.\n"
        "Different domain: Marine biology.\n"
        "New subject: Ocean temperature over the seasons.\n"
        "Drafting caption: Line graph of ocean temperatures.\n"
        "Word count check: 5 words. Acceptable.\n"
        "</thinking>\n"
        "<output>Line graph of ocean temperatures</output>",
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
        return [r["target"] for r in self._generate(images, clean_texts, max_new_tokens=200)]

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
