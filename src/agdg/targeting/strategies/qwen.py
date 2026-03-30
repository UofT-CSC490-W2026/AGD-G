from .base import TargetingModel
from typing import Optional, List

from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

SYSTEM_PROMPT = """
<system>
You are an expert in data visualization and visual association.

Your task is to analyze a provided chart image and generate a brief caption for a new chart. This new chart must share the same visual structure (chart type) as the original image but feature a different, highly iconic, and tangible subject.

**Rules for Generation:**
1. **Identify the Chart Type:** Look at the provided image and accurately determine the chart type (e.g., Bar chart, Heatmap, Pie chart, Line graph).
2. **Identify the Main Subject:** Determine the core noun or topic represented in the original chart.
3. **Determine the Visual Association:** Find the single most iconic, concrete, visual item that an average person immediately associates with that subject.
   * Association logic: Europe -> vacation spots, Italy -> pizza toppings, Depression -> illness rates, Math -> calculators.
4. **Tangibility Requirement:** The new subject must be a photographable, physical object or real scene. It cannot be an abstract concept.
5. **Length Constraint:** Your final caption must be exactly 3 to 6 words long.
6. **Output Format:** You must first output your step-by-step reasoning inside <thinking> tags. Then, provide your final 3 to 6 word caption inside <output> tags.

**Examples:**

User Input:
Assistant:
<thinking>
Chart type: Bar chart.
Original subject: European export volumes.
Core concept: Europe.
Tangible association: Vacation spots.
Drafting caption: Bar chart of European vacation spots.
Word count check: 6 words. This is acceptable.
</thinking>
<output>Bar chart of European vacation spots</output>

User Input:
Assistant:
<thinking>
Chart type: Heatmap.
Original subject: Depression rates by region.
Core concept: Depression.
Tangible association: Illness.
Drafting caption: Heatmap of regional illness rates.
Word count check: 5 words. This is acceptable.
</thinking>
<output>Heatmap of regional illness rates</output>

User Input:
Assistant:
<thinking>
Chart type: Pie chart.
Original subject: Italian economic output.
Core concept: Italy.
Tangible association: Pizza toppings.
Drafting caption: Pie chart of Italian pizza toppings.
Word count check: 6 words. This is acceptable.
</thinking>
<output>Pie chart of Italian pizza toppings</output>
</system>
"""

class QwenTargetingModel(TargetingModel):
    def __init__(self, model_tag: Optional[str] = "Qwen/Qwen2.5-VL-7B-Instruct", device="cuda") -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_tag)
        self.model = AutoModelForImageTextToText.from_pretrained(model_tag).to(self.device)
        self.system_prompt = SYSTEM_PROMPT

        self.model.eval()
        self.model.requires_grad_(False)

    def __call__(self, images: List[Image.Image], clean_texts: List[str]) -> List[str]:
        """
        Convert image/clean_text pairs to target texts which differ from the clean texts
        """
        messages = [self.get_message(i, c) for i, c in zip(images, clean_texts)]
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
            outputs = self.model.generate(**inputs, max_new_tokens=40)
        results = []
        for output in outputs:
            generated=output[input_len:]
            results.append(self.processor.decode(generated, skip_special_tokens=True))
        return results

    def get_message(self, image: Image.Image, clean_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": clean_text}
                ],
            },
        ]
