"""
Evaluate adversarial images by querying a VLM and computing similarity scores.
Compares VLM response with clean caption (hidden_answer) and target text (attack_target_text).
"""
import logging
import torch
from sentence_transformers import SentenceTransformer

from agdg.data_pipeline.aws import get_db_connection, get_image
from agdg.scoring.similarity import evaluate_similarity, get_device
from agdg.data_pipeline.aws import rds, s3

BATCH_SIZE = 100
SIMILARITY_MODEL_ID = "all-MiniLM-L6-v2"


def evaluate_all(model_name: str, max_rows: int = 0):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("evaluate")

    device = get_device()
    log.info(f"Loading similarity model {SIMILARITY_MODEL_ID} on {device}...")
    sim_model = SentenceTransformer(SIMILARITY_MODEL_ID, device=str(device))

    # TODO: Load VLM model for output_answer generation
    # For now, we assume output_answer might already be filled or we stub it.
    # The user asked to pull image data from aws clean image caption, target image caption 
    # and model generated response and compute similarity.

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            limit = f"LIMIT {max_rows}" if max_rows > 0 else ""
            # We need hidden_answer (clean), attack_target_text (target), and output_answer (response)
            cur.execute(f"""
                SELECT id, hidden_answer, attack_target_text, output_answer
                FROM samples
                WHERE adversarial_graph IS NOT NULL
                  AND output_answer IS NOT NULL
                  AND similarity_winner IS NULL
                ORDER BY id
                {limit}
            """)
            rows = cur.fetchall()

    log.info(f"Found {len(rows)} samples to score similarity")

    if not rows:
        return {"evaluated": 0}

    evaluated = 0
    winners = {"A": 0, "B": 0, "Tie": 0, "Neither": 0}

    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            for sid, clean_caption, target_text, output_answer in rows:
                if not output_answer or not clean_caption or not target_text:
                    log.warning(f"Sample {sid} missing required text for similarity scoring")
                    continue

                # evaluate_similarity(model, output_text, text_a, text_b)
                # A = clean_caption, B = target_text
                winner = evaluate_similarity(
                    sim_model, 
                    output_answer, 
                    clean_caption, 
                    target_text
                )

                cur.execute(
                    """
                    UPDATE samples
                    SET similarity_winner = %s
                    WHERE id = %s
                    """,
                    (winner, sid),
                )
                
                winners[winner] = winners.get(winner, 0) + 1
                evaluated += 1

                if evaluated % BATCH_SIZE == 0:
                    conn.commit()
                    log.info(f"  Scored {evaluated}/{len(rows)}")

    log.info(f"Done: {evaluated} evaluated")
    for k, v in winners.items():
        pct = (v / evaluated * 100) if evaluated > 0 else 0
        log.info(f"  Winner {k}: {v} ({pct:.1f}%)")

    # Overall Success Rate: Winner B (Target)
    success_rate = (winners["B"] / evaluated * 100) if evaluated > 0 else 0
    log.info(f"Overall Attack Success Rate (Winner B): {success_rate:.1f}%")

    return {"evaluated": evaluated, "winners": winners, "success_rate_pct": round(success_rate, 1)}
