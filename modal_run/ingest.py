"""
Modal wrapper for the data-ingestion pipeline: clean, import datasets, preprocess.
Replaces the old per-script Modal apps and the subprocess-based main.py orchestrator.

    modal run modal_run/ingest.py
    modal run modal_run/ingest.py -- --clean --limit 100
    modal run modal_run/ingest.py -- --skip-import
"""
import modal
from modal_run.image import build_data_pipeline_image

from agdg.data_pipeline.clean import clean as do_clean
from agdg.data_pipeline.import_chartbench import import_chartbench
from agdg.data_pipeline.import_chartx import import_chartx
from agdg.data_pipeline.import_chartqax import import_chartqax
from agdg.data_pipeline.preprocess_charts import preprocess_all

app = modal.App(
    "agd-ingest",
    image=build_data_pipeline_image(),
    secrets=[
        modal.Secret.from_name("aws"),
        modal.Secret.from_name("huggingface"),
    ],
)


@app.function(timeout=7200, memory=4096)
def ingest(
    max_rows: int = 0,
    clean: bool = False,
    skip_import: bool = False,
    skip_preprocess: bool = False,
    source: str | None = None,
):
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("ingest")

    if clean:
        log.info("=== Clean ===")
        do_clean()

    if not skip_import:
        if source is None or source == "ChartBench":
            log.info("=== Import ChartBench ===")
            import_chartbench(max_rows=max_rows)

        if source is None or source == "ChartX":
            log.info("=== Import ChartX ===")
            import_chartx(max_rows=max_rows or None)

        if source is None or source == "ChartQA-X":
            log.info("=== Import ChartQA-X ===")
            import_chartqax(max_rows=max_rows or None)

    if not skip_preprocess:
        log.info("=== Preprocess ===")
        result = preprocess_all()
        log.info(f"Preprocess result: {result}")

    log.info("=== Ingest complete ===")


@app.local_entrypoint()
def main(
    limit: int = 0,
    clean: bool = False,
    skip_import: bool = False,
    skip_preprocess: bool = False,
    source: str | None = None,
):
    ingest.remote(
        max_rows=limit,
        clean=clean,
        skip_import=skip_import,
        skip_preprocess=skip_preprocess,
        source=source,
    )
