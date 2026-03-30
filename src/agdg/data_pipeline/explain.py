"""
For a target VLM, find its intrinsic accuracy in explaining charts.
TODO: Replace stub with actual VLM API calls.
Currently writes a placeholder string to hidden_answer.
"""
import logging

from agdg.data_pipeline.aws import rds, s3

BATCH_SIZE = 100

def explain_all(model_name: str, max_rows: int = 0):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("explain")

    explained = 0

    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            for row in rds.iter_clean_answer_inputs(conn, model_name):
                clean_chart = row["clean_chart"]
                clean_bytes = s3.get_image(clean_chart)

                # TODO: Send image + question to VLM API (e.g. GPT-4V, Claude)
                description = f"[PLACEHOLDER] Description of chart {clean_chart}"

                rds.insert_clean_answer(
                    cur,
                    row["sample_id"],
                    description,
                    model_name,
                )

                explained += 1

                if explained % BATCH_SIZE == 0:
                    conn.commit()
                    log.info(f"  Explained {explained}")

    log.info(f"Done: {explained} samples explained")
    return {"explained": explained}
