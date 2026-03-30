"""
TODO: Replace stub with actual SD 1.5 adversarial perturbation.
Currently loads good_graph and saves it as adversarial_graph
(identity attack -- no perturbation).
"""
import logging

from agdg.data_pipeline.aws import get_db_connection, get_image, put_image

BATCH_SIZE = 100


def attack_all(max_rows: int = 0):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("attack")

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
                good_bytes = get_image(good_uuid)

                log.info(f"  good={good_uuid} ({len(good_bytes)} bytes)")

                # TODO: Run AGD attack using Stable Diffusion 1.5
                adversarial_bytes = good_bytes

                adversarial_uuid = put_image(adversarial_bytes)

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
