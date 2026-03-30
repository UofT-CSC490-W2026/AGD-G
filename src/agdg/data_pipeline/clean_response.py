"""
Generates clean responses (captions) for original charts using a VLM.
Stores the results in the hidden_answer column.
"""
import logging
import io
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForVisualQuestionAnswering

from agdg.data_pipeline.aws import get_db_connection, get_image

BATCH_SIZE = 100
DEFAULT_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def generate_clean_responses(max_rows: int = 0, model_id: str = DEFAULT_MODEL_ID):
    """
    Pull images from AWS and generate generic captions using the specified VLM.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("clean_response")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    log.info(f"Loading model {model_id} on {device}...")
    
    # Simple factory logic based on model_id
    # We can expand this as more models are added
    processor = AutoProcessor.from_pretrained(model_id)
    
    if "llava" in model_id.lower():
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        # Fallback to generic VQA model loader
        model = AutoModelForVisualQuestionAnswering.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        
    model.eval()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            limit = f"LIMIT {max_rows}" if max_rows > 0 else ""
            cur.execute(f"""
                SELECT id, good_graph
                FROM samples
                WHERE good_graph IS NOT NULL
                  AND hidden_answer IS NULL
                ORDER BY id
                {limit}
            """)
            rows = cur.fetchall()

    log.info(f"Found {len(rows)} samples to process")

    if not rows:
        return {"processed": 0}

    processed = 0
    prompt_question = "What is the graph about?"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for sid, good_uuid in rows:
                try:
                    img_bytes = get_image(good_uuid)
                    raw_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    # Handle different prompt structures
                    if "llava" in model_id.lower():
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": prompt_question},
                                ],
                            },
                        ]
                        prompt = processor.apply_chat_template(
                            conversation,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
                    else:
                        # Generic fallback for other models
                        inputs = processor(images=raw_image, text=prompt_question, return_tensors="pt").to(device)

                    if dtype == torch.float16 and "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                        )

                    # Extract only the generated part if it's a causal model like LLaVA
                    if "llava" in model_id.lower():
                        prompt_len = inputs["input_ids"].shape[1]
                        generated_ids = output_ids[:, prompt_len:]
                        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    else:
                        description = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                    cur.execute(
                        "UPDATE samples SET hidden_answer = %s WHERE id = %s",
                        (description, sid),
                    )
                    processed += 1

                    if processed % BATCH_SIZE == 0:
                        conn.commit()
                        log.info(f"  Processed {processed}/{len(rows)}")

                except Exception as e:
                    log.error(f"Error processing sample {sid}: {e}")
                    continue

    log.info(f"Done: {processed} samples processed")
    return {"processed": processed}
