"""
TODO: Replace stub with actual VLM inference.
Currently compares good_answer with a placeholder output and
randomly assigns attack_succeeded.
"""
import logging

from agdg.data_pipeline.aws import rds, s3

BATCH_SIZE = 100


def evaluate_all(model_name: str, max_rows: int = 0):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("evaluate")

    evaluated = 0
    succeeded = 0
    failed = 0

    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            for row in rds.iter_eval_inputs(conn, model_name):
                target_answer = row["target_answer"]
                clean_answer = row["clean_answer"]
                clean_chart = row["clean_chart"]
                adversarial_chart = row["adversarial_chart"]
                clean_bytes = s3.get_image(clean_chart)
                adversarial_bytes = s3.get_image(adversarial_chart)

                # TODO: Send adversarial image + question to target VQA model
                output_answer = clean_answer

                attack_succeeded = output_answer.strip().lower() != clean_answer.strip().lower()

                rds.insert_adversarial_answer(
                    cur,
                    row["adversarial_chart_id"],
                    output_answer,
                    model_name,
                    attack_succeeded,
                )

                evaluated += 1
                if attack_succeeded:
                    succeeded += 1
                else:
                    failed += 1

                if evaluated % BATCH_SIZE == 0:
                    conn.commit()
                    log.info(f"  Evaluated {evaluated} "
                             f"(succeeded: {succeeded}, failed: {failed})")

    log.info(f"Done: {evaluated} evaluated, "
             f"{succeeded} attacks succeeded, {failed} attacks failed")
    asr = (succeeded / evaluated * 100) if evaluated > 0 else 0
    log.info(f"Attack Success Rate: {asr:.1f}%")

    return {"evaluated": evaluated, "succeeded": succeeded, "failed": failed, "asr_pct": round(asr, 1)}
