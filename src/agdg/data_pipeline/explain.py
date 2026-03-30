"""
For a target VLM, find its intrinsic accuracy in explaining charts.
TODO: Replace stub with actual VLM API calls.
Currently writes a placeholder string to hidden_answer.
"""
import logging

from agdg.data_pipeline.aws import get_db_connection, get_image

BATCH_SIZE = 100


def explain_all(max_rows: int = 0):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("explain")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            limit = f"LIMIT {max_rows}" if max_rows > 0 else ""
            cur.execute(f"""
                SELECT id, good_graph, question
                FROM samples
                WHERE good_graph IS NOT NULL
                  AND hidden_answer IS NULL
                ORDER BY id
                {limit}
            """)
            rows = cur.fetchall()

    log.info(f"Found {len(rows)} samples to explain")

    if not rows:
        return {"explained": 0}

    explained = 0

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for sid, good_uuid, question in rows:
                img_bytes = get_image(good_uuid)

                # TODO: Send image + question to VLM API (e.g. GPT-4V, Claude)
                description = f"[PLACEHOLDER] Description of chart {good_uuid} for: {question}"

                cur.execute(
                    "UPDATE samples SET hidden_answer = %s WHERE id = %s",
                    (description, sid),
                )
                explained += 1

                if explained % BATCH_SIZE == 0:
                    conn.commit()
                    log.info(f"  Explained {explained}/{len(rows)}")

    log.info(f"Done: {explained} samples explained")
    return {"explained": explained}
