"""
Generate target captions for preprocessed charts and store them in target_answers.
"""

from __future__ import annotations

import io
import logging
from typing import Dict, List, Optional

from PIL import Image

from agdg.data_pipeline.aws import rds, s3
from agdg.targeting.targeting import build_targeting_strategy

BATCH_SIZE = 100


def generate_target_responses(
    targeting_strategy: str,
    max_rows: int = 0,
    source: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
) -> Dict:
    """
    Generate and persist target captions for all clean charts that do not yet
    have one for *targeting_strategy*.

    Args:
        targeting_strategy: Strategy key, e.g. ``"qwen"``.
        max_rows: Stop after this many rows (0 = no limit).
        source: Restrict to a single dataset source, e.g. ``"ChartBench"``.
            ``None`` processes all sources.
        batch_size: Number of DB rows to commit per batch.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("target_response")

    targeter = build_targeting_strategy(targeting_strategy)

    processed = 0
    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            for row in rds.iter_target_inputs(conn, targeting_strategy, source=source):
                if max_rows > 0 and processed >= max_rows:
                    break
                clean_answer_id = row["clean_answer_id"]
                clean_answer = row["clean_answer"]
                clean_chart = row["clean_chart"]
                clean_image = Image.open(io.BytesIO(s3.get_image(clean_chart))).convert("RGB")
                try:
                    targets = targeter([clean_image], [clean_answer])
                    rds.insert_target_answer(
                        cur,
                        clean_answer_id,
                        targets[0],
                        targeting_strategy,
                    )
                    processed += cur.rowcount
                    if processed and processed % batch_size == 0:
                        conn.commit()
                        log.info("  Processed %s", processed)
                except Exception as exc:
                    log.error("Error processing sample %s: %s", clean_answer_id, exc)

    log.info("Done: %s samples processed", processed)
    return {"processed": processed}


def preview_target_responses(
    targeting_strategy: str,
    per_source: int = 10,
) -> List[Dict]:
    """
    Generate target captions for a small sample from each dataset source and
    return the full results including thinking traces.  Nothing is written to
    the database.

    Args:
        targeting_strategy: Strategy key, e.g. ``"qwen"``.
        per_source: Number of images to sample from each dataset source.

    Returns:
        List of dicts with keys ``source``, ``clean_answer``, ``thinking``, ``target``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("target_response.preview")

    targeter = build_targeting_strategy(targeting_strategy)
    results = []

    with rds.get_db_connection() as conn:
        for row in rds.iter_target_inputs_sampled(conn, targeting_strategy, per_source=per_source):
            clean_answer = row["clean_answer"]
            clean_chart = row["clean_chart"]
            chart_source = row["chart_source"]
            clean_image = Image.open(io.BytesIO(s3.get_image(clean_chart))).convert("RGB")
            try:
                raw_results = targeter.generate_raw([clean_image], [clean_answer])
                r = raw_results[0]
                results.append({
                    "source": chart_source,
                    "clean_answer": clean_answer,
                    "thinking": r["thinking"],
                    "target": r["target"],
                })
                log.info("[%s] %s -> %s", chart_source, clean_answer, r["target"])
            except Exception as exc:
                log.error("Error previewing sample from %s: %s", chart_source, exc)

    return results
