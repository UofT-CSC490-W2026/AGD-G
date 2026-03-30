"""
Evaluate adversarial charts by querying a VLM and measuring whether the answer
is closer to the attack target than to the clean answer.
"""

from __future__ import annotations

import io
import logging

from PIL import Image
from sentence_transformers import SentenceTransformer
from psycopg2.extras import Json

from agdg.data_pipeline.aws import rds, s3
from agdg.data_pipeline.clean_response import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT,
    INFERENCE_BATCH_SIZE,
    generate_batch_responses,
    load_vlm,
)
from agdg.scoring.similarity import determine_winner, get_device

DB_COMMIT_SIZE = 100
SIMILARITY_MODEL_ID = "all-MiniLM-L6-v2"


def _similarity_scores(model, output_text: str, text_a: str, text_b: str) -> tuple[float, float]:
    import torch.nn.functional as F

    embeddings = model.encode([output_text, text_a, text_b], convert_to_tensor=True)
    output_vec = embeddings[0].unsqueeze(0)
    vec_a = embeddings[1].unsqueeze(0)
    vec_b = embeddings[2].unsqueeze(0)
    score_a = F.cosine_similarity(output_vec, vec_a).item()
    score_b = F.cosine_similarity(output_vec, vec_b).item()
    return score_a, score_b


def evaluate_all(
    model_name: str = DEFAULT_MODEL_ID,
    max_rows: int = 0,
    target_strategy: str | None = None,
    source: str | None = None,
    batch_size: int = INFERENCE_BATCH_SIZE,
):
    """Query a VLM on each unevaluated adversarial chart and record whether the attack succeeded."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("evaluate")

    device = get_device()
    log.info("Loading similarity model %s on %s...", SIMILARITY_MODEL_ID, device)
    sim_model = SentenceTransformer(SIMILARITY_MODEL_ID, device=str(device))
    processor, vlm, vlm_device, vlm_dtype = load_vlm(model_name)

    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            limit_clause = "LIMIT %s" if max_rows > 0 else ""
            strategy_clause = "AND ta.target_strategy = %s" if target_strategy else ""
            source_clause = "AND s.chart_source = %s" if source else ""
            params = [model_name]
            if target_strategy:
                params.append(target_strategy)
            if source:
                params.append(source)
            if max_rows > 0:
                params.append(max_rows)
            cur.execute(
                f"""
                SELECT
                    ac.id,
                    ca.clean_answer,
                    ta.target_answer,
                    ac.adversarial_chart
                FROM adversarial_charts ac
                JOIN target_answers ta ON ta.id = ac.target_answer_id
                JOIN clean_answers ca ON ca.id = ta.clean_answer_id
                JOIN samples s ON s.id = ca.sample_id
                LEFT JOIN adversarial_answers aa
                  ON aa.adversarial_chart_id = ac.id
                 AND aa.adversarial_answer_model = %s
                WHERE aa.id IS NULL
                  {strategy_clause}
                  {source_clause}
                ORDER BY ac.id
                {limit_clause}
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    log.info("Found %s adversarial charts to evaluate", len(rows))
    if not rows:
        return {"evaluated": 0}

    evaluated = 0
    succeeded = 0
    failed = 0
    winners = {"A": 0, "B": 0, "Tie": 0, "Neither": 0}

    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            for batch_start in range(0, len(rows), batch_size):
                batch = rows[batch_start:batch_start + batch_size]
                images, valid = [], []
                for row in batch:
                    try:
                        images.append(Image.open(io.BytesIO(s3.get_image(row[3]))).convert("RGB"))
                        valid.append(row)
                    except Exception as exc:
                        log.error("Error fetching adversarial_chart_id=%s: %s", row[0], exc)
                if not images:
                    continue
                try:
                    output_answers = generate_batch_responses(
                        processor=processor,
                        model=vlm,
                        model_id=model_name,
                        images=images,
                        prompt=DEFAULT_PROMPT,
                        device=vlm_device,
                        dtype=vlm_dtype,
                    )
                except Exception as exc:
                    log.error("Error in batch inference %s-%s: %s", batch_start, batch_start + batch_size, exc)
                    continue

                for (adversarial_chart_id, clean_answer, target_answer, _), output_answer in zip(valid, output_answers):
                    try:
                        score_a, score_b = _similarity_scores(sim_model, output_answer, clean_answer, target_answer)
                        winner = determine_winner(score_a, score_b)
                        attack_succeeded = winner == "B"
                        cur.execute(
                            """
                            INSERT INTO adversarial_answers (
                                adversarial_chart_id,
                                adversarial_answer_model,
                                answer_text,
                                attack_succeeded,
                                eval_meta
                            )
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (adversarial_chart_id, adversarial_answer_model) DO NOTHING
                            """,
                            (
                                adversarial_chart_id,
                                model_name,
                                output_answer,
                                attack_succeeded,
                                Json({"winner": winner, "score_clean": score_a, "score_target": score_b}),
                            ),
                        )
                        winners[winner] = winners.get(winner, 0) + 1
                        succeeded += int(attack_succeeded)
                        failed += int(not attack_succeeded)
                        evaluated += 1
                    except Exception as exc:
                        log.error("Error evaluating adversarial_chart_id=%s: %s", adversarial_chart_id, exc)

                conn.commit()
                log.info("  Evaluated %s/%s", min(batch_start + batch_size, len(rows)), len(rows))

    asr_pct = (succeeded / evaluated * 100) if evaluated else 0.0
    log.info("Done: %s evaluated, ASR %.1f%%", evaluated, asr_pct)
    return {
        "evaluated": evaluated,
        "succeeded": succeeded,
        "failed": failed,
        "winners": winners,
        "asr_pct": round(asr_pct, 1),
    }
