"""
Generate clean answers for preprocessed charts and store them in clean_answers.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Dict, Optional

from PIL import Image

from agdg.data_pipeline.aws import rds, s3
from agdg.targeting.targeting import build_targeting_strategy

if TYPE_CHECKING:
    import torch

BATCH_SIZE = 100

def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_target_responses(
    targeting_strategy: str,
    model_id: Optional[str] = None,
    max_rows: int = 0,
) -> Dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("clean_response")

    targeter = build_targeting_strategy(targeting_strategy)

    processed = 0
    with rds.get_db_connection() as conn:
        with conn.cursor() as cur:
            # TODO: get multiple rows at once for batching
            for row in rds.iter_target_inputs(conn, targeting_strategy):
                clean_answer_id = row["clean_answer_id"]
                clean_answer = row["clean_answer"]
                clean_chart = row["clean_chart"]
                clean_image = Image.open(io.BytesIO(s3.get_image(clean_chart))).convert("RGB")
                try:
                    target_response = targeter([clean_image], [clean_answer])
                    rds.insert_target_answer(
                        cur,
                        clean_answer_id,
                        target_response,
                        targeting_strategy,
                    )
                    processed += cur.rowcount
                    if processed and processed % BATCH_SIZE == 0:
                        conn.commit()
                        log.info("  Processed %s", processed)
                except Exception as exc:
                    log.error("Error processing sample %s: %s", clean_answer_id, exc)

    log.info("Done: %s samples processed", processed)
    return {"processed": processed}
