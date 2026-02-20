import modal
import os
import logging
import shutil
import zipfile

from aws import (
    get_db_connection, put_image, create_schema_if_not_exists,
    GraphType, BUCKET, IMAGE_PREFIX,
)

# Modal setup
app = modal.App("agd-chartbench-pipeline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets",
        "huggingface_hub",
        "boto3",
        "psycopg2-binary",
        "Pillow",
    )
    .add_local_file("aws.py", "/root/aws.py")
)

# Configuration

HF_DATASET_ID = "SincereX/ChartBench"
HF_SUBSET = "chart_bench"
HF_SPLIT = "test_data"
HF_IMAGE_ZIP = "data/test.zip"

BATCH_SIZE = 100

# Map chartbench dataset type to RDS schema
CHARTBENCH_TYPE_MAP: dict[str, GraphType] = {
    "area":        GraphType.AREA,
    "bar":         GraphType.BAR,
    "box":         GraphType.BOX,
    "combination": GraphType.OTHER,
    "line":        GraphType.LINE,
    "node_link":   GraphType.NODE,
    "pie":         GraphType.PIE,
    "radar":       GraphType.RADAR,
    "scatter":     GraphType.SCATTER,
}

# Core import logic

@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("aws"),
        modal.Secret.from_name("aws-rds"),
    ],
    timeout=3600,
    memory=4096,
)
def import_chartbench(max_rows: int = 0, clean: bool = False):
    import boto3
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from PIL import Image

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("import_chartbench")

    # ── Clean (optional) ─────────────────────────────────────────────
    if clean:
        log.info("CLEANING...")
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS samples")
                cur.execute("DROP TYPE IF EXISTS graph_type")

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=IMAGE_PREFIX + "/"):
            objects = page.get("Contents", [])
            if objects:
                s3.delete_objects(
                    Bucket=BUCKET,
                    Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
                )
                log.info(f"  Deleted {len(objects)} S3 objects")
        log.info("Clean done\n")

    # Create schema
    create_schema_if_not_exists()
    log.info("Schema ready")

    # Download and extract images
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

    # Load dataset
    log.info(f"Loading {HF_DATASET_ID} split={HF_SPLIT} ...")
    ds = load_dataset(HF_DATASET_ID, HF_SUBSET, split=HF_SPLIT)
    total = min(max_rows, len(ds)) if max_rows > 0 else len(ds)
    log.info(f"Processing {total} of {len(ds)} rows")

    # Process rows
    INSERT_SQL = """
        INSERT INTO samples (
            source, graph_type, question, good_answer,
            raw_graph, original_width, original_height
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    # hf_image_path → (uuid_str, width, height)
    uploaded: dict[str, tuple[str, int, int]] = {}
    inserted = 0
    skipped = 0
    unmapped_types: set[str] = set()

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for i, row in enumerate(ds):
                if i >= total:
                    break

                img_path = row["image"]
                chart_type_raw = row["type"]["chart"]

                # Map ChartBench type → GraphType enum
                graph_type = CHARTBENCH_TYPE_MAP.get(chart_type_raw)
                if graph_type is None:
                    if chart_type_raw not in unmapped_types:
                        log.warning(f"  Unknown chart type '{chart_type_raw}' → OTHER")
                        unmapped_types.add(chart_type_raw)
                    graph_type = GraphType.OTHER

                # Upload image via put_image (once per unique)
                if img_path not in uploaded:
                    relative_img_path = img_path.replace("./data/", "")
                    local_src = os.path.join(extract_dir, relative_img_path)

                    if not os.path.exists(local_src):
                        log.warning(f"  Missing: {local_src}")
                        skipped += 1
                        continue

                    with Image.open(local_src) as img:
                        width, height = img.size

                    with open(local_src, "rb") as f:
                        image_uuid = put_image(f.read())

                    uploaded[img_path] = (str(image_uuid), width, height)

                # Insert QA row
                image_uuid_str, width, height = uploaded[img_path]

                cur.execute(INSERT_SQL, (
                    "ChartBench",
                    str(graph_type),
                    row["conversation"][0]["query"],
                    row["conversation"][0]["label"],
                    image_uuid_str,
                    width,
                    height,
                ))
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

# Entrypoint
@app.local_entrypoint()
def main(max_rows: int = 0, clean: bool = False):
    result = import_chartbench.remote(max_rows=max_rows, clean=clean)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")