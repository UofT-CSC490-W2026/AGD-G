"""
Generate adversarial charts from clean charts using AttackVLM.
"""

from __future__ import annotations

import io
import logging

from PIL import Image

from agdg.attack.attack import build_attack_method
from agdg.data_pipeline.aws import rds, s3

BATCH_SIZE = 100
DEFAULT_ATTACKER = "targeted_text"
DEFAULT_SURROGATE = "llava"


def attack_all(
    method: str = DEFAULT_ATTACKER,
    surrogate: str = DEFAULT_SURROGATE,
    max_rows: int = 0,
    steps: int = 300,
    target_strategy: str | None = None,
):
    """Generate adversarial charts for all target answers that lack one for *method*/*surrogate*."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("attack")

    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            limit_clause = "LIMIT %s" if max_rows > 0 else ""
            strategy_clause = "AND ta.target_strategy = %s" if target_strategy else ""
            params = [method, surrogate]
            if target_strategy:
                params.append(target_strategy)
            if max_rows > 0:
                params.append(max_rows)
            cur.execute(
                f"""
                SELECT
                    ta.id,
                    ca.clean_answer,
                    s.clean_chart,
                    ta.target_answer
                FROM target_answers ta
                JOIN clean_answers ca ON ca.id = ta.clean_answer_id
                JOIN samples s ON s.id = ca.sample_id
                LEFT JOIN adversarial_charts ac
                  ON ac.target_answer_id = ta.id
                 AND ac.attack_method = %s
                 AND ac.attack_surrogate = %s
                WHERE s.clean_chart IS NOT NULL
                  {strategy_clause}
                  AND ac.id IS NULL
                ORDER BY ta.id
                {limit_clause}
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    log.info("Found %s charts to attack", len(rows))
    if not rows:
        return {"attacked": 0, "rows_updated": 0}

    attacked = 0
    rows_updated = 0
    attack_method = build_attack_method(attacker=method, model=surrogate)

    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            for target_answer_id, clean_answer, clean_chart, target_answer in rows:
                try:
                    clean_image = Image.open(io.BytesIO(s3.get_image(clean_chart))).convert("RGB")
                    adversarial = attack_method.attack(
                        clean=clean_image,
                        target=target_answer,
                        strength=1.0,
                        hyperparameters={
                            "source_text": clean_answer,
                            "steps": steps,
                        },
                    )
                    output = io.BytesIO()
                    adversarial.save(output, format="PNG")
                    adversarial_chart = s3.put_image(output.getvalue())
                    cur.execute(
                        """
                        INSERT INTO adversarial_charts (
                            target_answer_id,
                            adversarial_chart,
                            attack_method,
                            attack_surrogate,
                            attack_meta
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (target_answer_id, attack_method, attack_surrogate) DO NOTHING
                        """,
                        (target_answer_id, str(adversarial_chart), method, surrogate, None),
                    )
                    rows_updated += cur.rowcount
                    attacked += 1
                    if attacked % BATCH_SIZE == 0:
                        conn.commit()
                        log.info("  Attacked %s/%s", attacked, len(rows))
                except Exception as exc:
                    log.error("Error attacking target_answer_id=%s: %s", target_answer_id, exc)

    log.info("Done: %s images, %s rows updated", attacked, rows_updated)
    return {"attacked": attacked, "rows_updated": rows_updated}
