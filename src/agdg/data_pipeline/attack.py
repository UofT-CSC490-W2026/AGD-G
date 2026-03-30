"""
TODO: Replace stub with actual SD 1.5 adversarial perturbation.
Currently loads good_graph and saves it as adversarial_graph
(identity attack -- no perturbation).
"""
import logging

from agdg.data_pipeline.aws import rds, s3

BATCH_SIZE = 100

def attack_all(method: str, surrogate: str, max_rows: int = 0):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("attack")

    attacked = 0
    rows_updated = 0

    with rds.get_db_connection() as conn:
        for row in rds.iter_attack_inputs(conn, method, surrogate):
            with conn.cursor() as cur:
                clean_answer = row["clean_answer"]
                clean_chart = row["clean_chart"]
                target_answer = row["target_answer"]
                clean_bytes = s3.get_image(clean_chart)

                log.info(f"  clean={clean_chart} ({len(clean_bytes)} bytes)")

                # TODO: Run Attack
                adversarial_bytes = clean_bytes

                adversarial_chart = s3.put_image(adversarial_bytes)

                rds.insert_adversarial_chart(
                    cur,
                    row["target_answer_id"],
                    adversarial_chart,
                    method,
                    surrogate,
                )

                rows_updated += cur.rowcount
                attacked += 1

                if attacked % BATCH_SIZE == 0:
                    conn.commit()
                    log.info(f"  Attacked {attacked} ({rows_updated} rows)")

    log.info(f"Done: {attacked} images, {rows_updated} rows updated")
    return {"attacked": attacked, "rows_updated": rows_updated}
