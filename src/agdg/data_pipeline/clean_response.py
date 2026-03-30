"""
Generate clean answers for preprocessed charts and store them in clean_answers.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from PIL import Image

from agdg.data_pipeline.aws import rds, s3

if TYPE_CHECKING:
    import torch

BATCH_SIZE = 100
DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_PROMPT = "What is the graph about?"


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_vlm(model_id: str = DEFAULT_MODEL_ID) -> tuple[object, object, str, object]:
    import torch
    from transformers import (
        AutoModelForVisualQuestionAnswering,
        AutoProcessor,
        LlavaForConditionalGeneration,
    )

    device = _get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(model_id)
    if "llava" in model_id.lower():
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForVisualQuestionAnswering.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)

    model.eval()
    return processor, model, device, dtype


def _prepare_inputs(
    processor: object,
    model_id: str,
    raw_image: Image.Image,
    prompt: str,
    device: str,
    dtype: object,
):
    import torch

    if "llava" in model_id.lower():
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        rendered_prompt = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(images=raw_image, text=rendered_prompt, return_tensors="pt").to(device)
    else:
        rendered_prompt = prompt
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)

    if dtype == torch.float16 and "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    return inputs, rendered_prompt


def generate_image_response(
    *,
    processor: object,
    model: object,
    model_id: str,
    image_bytes: bytes,
    prompt: str = DEFAULT_PROMPT,
    device: str,
    dtype: object,
    max_new_tokens: int = 256,
) -> str:
    import torch

    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs, _ = _prepare_inputs(processor, model_id, raw_image, prompt, device, dtype)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    if "llava" in model_id.lower():
        prompt_len = inputs["input_ids"].shape[1]
        output_ids = output_ids[:, prompt_len:]

    decoded = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return decoded.strip()


def generate_clean_responses(max_rows: int = 0, model_id: str = DEFAULT_MODEL_ID):
    """Caption every preprocessed chart that lacks a clean answer for *model_id*."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("clean_response")

    processor, model, device, dtype = load_vlm(model_id)
    log.info("Loaded %s on %s", model_id, device)

    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            limit_clause = "LIMIT %s" if max_rows > 0 else ""
            params: Sequence[object] = (model_id, max_rows) if max_rows > 0 else (model_id,)
            cur.execute(
                f"""
                SELECT s.id, s.clean_chart
                FROM samples s
                LEFT JOIN clean_answers ca
                  ON ca.sample_id = s.id
                 AND ca.clean_answer_model = %s
                WHERE s.clean_chart IS NOT NULL
                  AND ca.id IS NULL
                ORDER BY s.id
                {limit_clause}
                """,
                params,
            )
            rows = cur.fetchall()

    log.info("Found %s samples to process", len(rows))
    if not rows:
        return {"processed": 0}

    processed = 0
    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            for sample_id, clean_chart in rows:
                try:
                    answer = generate_image_response(
                        processor=processor,
                        model=model,
                        model_id=model_id,
                        image_bytes=s3.get_image(clean_chart),
                        prompt=DEFAULT_PROMPT,
                        device=device,
                        dtype=dtype,
                    )
                    cur.execute(
                        """
                        INSERT INTO clean_answers (sample_id, clean_answer_model, clean_answer)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (sample_id, clean_answer_model) DO NOTHING
                        """,
                        (sample_id, model_id, answer),
                    )
                    processed += cur.rowcount
                    if processed and processed % BATCH_SIZE == 0:
                        conn.commit()
                        log.info("  Processed %s/%s", processed, len(rows))
                except Exception as exc:
                    log.error("Error processing sample %s: %s", sample_id, exc)

    log.info("Done: %s samples processed", processed)
    return {"processed": processed}
