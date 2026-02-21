"""
For a target VLM, find its intrinsic accuracy in explaining charts.
TODO: Replace stub with actual VLM API calls.
Currently writes a placeholder string to hidden_answer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

from aws import get_db_connection, get_image

app = modal.App("agd-explain-charts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("boto3", "psycopg2-binary", "Pillow")
    .add_local_file("../aws.py", "/root/aws.py")
    .add_local_file("../config.py", "/root/config.py")
)

BATCH_SIZE = 100


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("aws"), modal.Secret.from_name("aws-rds")],
    timeout=3600, memory=4096,
)
def explain_all(max_rows: int = 0):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("explain")

    # Get rows that have been preprocessed but not yet explained
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
                # Download preprocessed image
                img_bytes = get_image(good_uuid)

                # TODO: Send image + question to VLM API (e.g. GPT-4V, Claude)
                # and get a natural language description of the chart.
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


@app.local_entrypoint()
def main(l: int = 0):
    r = explain_all.remote(max_rows=l)
    for k, v in r.items():
        print(f"  {k}: {v}")