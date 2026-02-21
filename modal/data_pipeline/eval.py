"""
TODO: Replace stub with actual VLM inference.
Currently compares good_answer with a placeholder output and
randomly assigns attack_succeeded.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

from aws import get_db_connection, get_image

app = modal.App("agd-evaluate")

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
def evaluate_all(max_rows: int = 0):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("evaluate")

    # Get rows with adversarial images that haven't been evaluated
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            limit = f"LIMIT {max_rows}" if max_rows > 0 else ""
            cur.execute(f"""
                SELECT id, adversarial_graph, question, good_answer
                FROM samples
                WHERE adversarial_graph IS NOT NULL
                  AND output_answer IS NULL
                ORDER BY id
                {limit}
            """)
            rows = cur.fetchall()

    log.info(f"Found {len(rows)} samples to evaluate")

    if not rows:
        return {"evaluated": 0, "succeeded": 0, "failed": 0}

    evaluated = 0
    succeeded = 0
    failed = 0

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for sid, adv_uuid, question, good_answer in rows:
                # Download adversarial image
                adv_bytes = get_image(adv_uuid)

                # TODO: Send adversarial image + question to target VQA model
                # and capture its response.
                output_answer = good_answer

                # Compare
                attack_succeeded = output_answer.strip().lower() != good_answer.strip().lower()

                cur.execute(
                    """
                    UPDATE samples
                    SET output_answer = %s, attack_succeeded = %s
                    WHERE id = %s
                    """,
                    (output_answer, attack_succeeded, sid),
                )
                evaluated += 1
                if attack_succeeded:
                    succeeded += 1
                else:
                    failed += 1

                if evaluated % BATCH_SIZE == 0:
                    conn.commit()
                    log.info(f"  Evaluated {evaluated}/{len(rows)} "
                             f"(succeeded: {succeeded}, failed: {failed})")

    log.info(f"Done: {evaluated} evaluated, "
             f"{succeeded} attacks succeeded, {failed} attacks failed")
    asr = (succeeded / evaluated * 100) if evaluated > 0 else 0
    log.info(f"Attack Success Rate: {asr:.1f}%")

    return {"evaluated": evaluated, "succeeded": succeeded, "failed": failed, "asr_pct": round(asr, 1)}


@app.local_entrypoint()
def main(l: int = 0):
    r = evaluate_all.remote(max_rows=l)
    for k, v in r.items():
        print(f"  {k}: {v}")