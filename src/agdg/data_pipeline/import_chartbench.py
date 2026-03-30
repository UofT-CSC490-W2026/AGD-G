import os
import logging
import shutil
import zipfile

from agdg.data_pipeline.aws import s3, rds
from agdg.data_pipeline.chart_type import ChartType

HF_DATASET_ID = "SincereX/ChartBench"
HF_SUBSET = "chart_bench"
HF_SPLIT = "test_data"
HF_IMAGE_ZIP = "data/test.zip"

BATCH_SIZE = 100

CHARTBENCH_TYPE_MAP: dict[str, ChartType] = {
    "area":        ChartType.AREA,
    "bar":         ChartType.BAR,
    "box":         ChartType.BOX,
    "combination": ChartType.OTHER,
    "line":        ChartType.LINE,
    "node_link":   ChartType.NODE,
    "pie":         ChartType.PIE,
    "radar":       ChartType.RADAR,
    "scatter":     ChartType.SCATTER,
}


def import_chartbench(max_rows: int = 0, clean: bool = False):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("import_chartbench")

    if clean:
        log.info("CLEANING...")
        s3.wipe_s3(logger=log)
        rds.wipe_rds()
        log.info("Clean done\n")

    rds.create_table_if_not_exists()
    log.info("Schema ready")

    log.info(f"Downloading {HF_IMAGE_ZIP} from {HF_DATASET_ID} ...")
    zip_path = hf_hub_download(
        repo_id=HF_DATASET_ID, repo_type="dataset", filename=HF_IMAGE_ZIP,
    )
    extract_dir = "/tmp/chartbench_extracted"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    log.info(f"Extracted to {extract_dir}")

    log.info(f"Loading {HF_DATASET_ID} split={HF_SPLIT} ...")
    ds = load_dataset(HF_DATASET_ID, HF_SUBSET, split=HF_SPLIT)
    total = min(max_rows, len(ds)) if max_rows is not None else len(ds)
    log.info(f"Processing {total} of {len(ds)} rows")

    uploaded: dict[str, tuple[str, int, int]] = {}
    inserted = 0
    skipped = 0
    unmapped_types: set[str] = set()

    with rds.get_db_connection() as conn:
        with conn.cursor() as cursor:
            for i, row in enumerate(ds):
                if i >= total:
                    break

                img_path = row["image"]
                chart_type_raw = row["type"]["chart"]

                graph_type = CHARTBENCH_TYPE_MAP.get(chart_type_raw)
                if graph_type is None:
                    if chart_type_raw not in unmapped_types:
                        log.warning(f"  Unknown chart type '{chart_type_raw}' → OTHER")
                        unmapped_types.add(chart_type_raw)
                    graph_type = ChartType.OTHER

                if img_path not in uploaded:
                    relative_img_path = img_path.replace("./data/", "")
                    local_src = os.path.join(extract_dir, relative_img_path)

                    if not os.path.exists(local_src):
                        log.warning(f"  Missing: {local_src}")
                        skipped += 1
                        continue

                    with open(local_src, "rb") as f:
                        image_uuid = s3.put_image(f.read())

                    uploaded[img_path] = str(image_uuid)

                image_uuid_str = uploaded[img_path]

                rds.insert_sample(
                    cursor,
                    "ChartBench",
                    str(graph_type),
                    row["conversation"][0]["query"],
                    row["conversation"][0]["label"],
                    image_uuid_str,
                )
                inserted += 1

                if inserted % BATCH_SIZE == 0:
                    conn.commit()
                    log.info(f"  Inserted {inserted}/{total} ({len(uploaded)} images)")

    log.info(f"Done: {inserted} inserted, {skipped} skipped, "
             f"{len(uploaded)} unique images")
    if unmapped_types:
        log.warning(f"Unmapped types defaulted to OTHER: {unmapped_types}")

    shutil.rmtree(extract_dir, ignore_errors=True)

    return {
        "total_rows": total,
        "rows_inserted": inserted,
        "rows_skipped": skipped,
        "unique_images": len(uploaded),
    }
