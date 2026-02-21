"""
TODO: Replace stub with actual SD 1.5 adversarial perturbation.
Currently loads good_graph and saves it as adversarial_graph
(identity attack — no perturbation).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

from aws import get_db_connection, get_image, put_image

app = modal.App("agd-attack")

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
    timeout=7200, memory=8192,
)
def attack_all(max_rows: int = 0):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("attack")

    # Get distinct good_graphs ready for attack
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            limit = f"LIMIT {max_rows}" if max_rows > 0 else ""
            cur.execute(f"""
                SELECT DISTINCT good_graph
                FROM samples
                WHERE good_graph IS NOT NULL
                  AND adversarial_graph IS NULL
                ORDER BY good_graph
                {limit}
            """)
            good_uuids = [r[0] for r in cur.fetchall()]

    log.info(f"Found {len(good_uuids)} unique images to attack")

    if not good_uuids:
        return {"attacked": 0, "rows_updated": 0}

    attacked = 0
    rows_updated = 0

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for good_uuid in good_uuids:
                # Download preprocessed image
                good_bytes = get_image(good_uuid)

                log.info(f"  good={good_uuid} ({len(good_bytes)} bytes)")

                # TODO: Run AGD attack using Stable Diffusion 1.5
                #   1. Encode good_graph with SD VAE
                #   2. Apply adversarial perturbation in latent space
                #   3. Decode back to pixel space
                adversarial_bytes = good_bytes

                # Upload adversarial image → new UUID
                adversarial_uuid = put_image(adversarial_bytes)

                # Update all rows sharing this good_graph
                cur.execute(
                    """
                    UPDATE samples
                    SET adversarial_graph = %s
                    WHERE good_graph = %s
                      AND adversarial_graph IS NULL
                    """,
                    (str(adversarial_uuid), str(good_uuid)),
                )
                rows_updated += cur.rowcount
                attacked += 1

                if attacked % BATCH_SIZE == 0:
                    conn.commit()
                    log.info(f"  Attacked {attacked}/{len(good_uuids)} ({rows_updated} rows)")

    log.info(f"Done: {attacked} images, {rows_updated} rows updated")
    return {"attacked": attacked, "rows_updated": rows_updated}


@app.local_entrypoint()
def main(l: int = 0):
    r = attack_all.remote(max_rows=l)
    for k, v in r.items():
        print(f"  {k}: {v}")