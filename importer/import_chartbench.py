"""
Modal importer: ChartBench → S3 + RDS

Dataset: https://huggingface.co/datasets/SincereX/ChartBench
Schema per row:
  - id (int)
  - image (str): relative path e.g. "./data/train/area/area/chart_0/image.png"
  - type (dict): {"chart": "area", "image": "area", "task": "CR", "QA": "Acc+"}
  - conversation (list[2]): [{"label": "Yes", "query": "..."}, {"label": "No", "query": "..."}]

Import strategy:
  1. Download image zip from HF repo via hf_hub_download → extract locally
  2. Load the HF parquet data (test_data split: 10.5k rows)
  3. For each QA row, read image from extracted zip → upload to S3
  4. INSERT row into RDS with S3 path + question + answer

Mapping to RDS schema:
  - source       → "ChartBench"
  - graph_type   → type["chart"]  (area, bar, line, pie, scatter, ...)
  - question     → conversation[0]["query"]  (the "Yes" version of the question)
  - good_graph   → s3://{bucket}/good_graphs/ChartBench/{image_hash}.png
  - good_answer  → conversation[0]["label"]  (always "Yes")
"""

import modal
import os
import logging
import shutil
import zipfile
from pathlib import PurePosixPath

# Modal Setup
app = modal.App("import-chartbench")

# Create a container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets",
        "huggingface_hub",
        "boto3",
        "psycopg2-binary",
        "Pillow",
    )
)

# AWS and HuggingFace Configuration
S3_BUCKET = "agd-dev-tyson"
S3_PREFIX = "good_graphs/ChartBench"
AWS_REGION = "ca-central-1"

HF_DATASET_ID = "SincereX/ChartBench"
HF_SUBSET = "chart_bench"
HF_SPLIT = "test_data"

HF_IMAGE_ZIP = "data/test.zip"
BATCH_SIZE = 100


def image_path_to_s3_key(image_path: str) -> str:
    """
    Convert a HF-relative image path to a deterministic S3 key.

    Example:
      "./data/train/area/area/chart_0/image.png"
      → "good_graphs/ChartBench/train/area/area/chart_0.png"

    We strip the leading "./data/" and trailing "/image.png" to get a clean
    hierarchy, then prepend the S3_PREFIX.
    """
    clean = image_path.replace("./data/", "").replace(".\\data\\", "")

    p = PurePosixPath(clean)
    chart_id = str(p.parent)

    return f"{S3_PREFIX}/{chart_id}.png"


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("aws"),
        modal.Secret.from_name("aws-rds"),
    ],
    timeout=3600,
    memory=2048,
)


def import_chartbench():
    """Main import function: HuggingFace → S3 + RDS."""
    import os
    import boto3
    import psycopg2
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("import_chartbench")

    # Connect to AWS S3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    log.info(f"Connected to S3 bucket: {S3_BUCKET}")

    # Connect to RDS PostgreSQL
    conn = psycopg2.connect(
        host="agd-dev-postgres.cdsyi46ammw7.ca-central-1.rds.amazonaws.com",
        port=5432,
        database="postgres",
        user="postgres",
        password=os.environ["DB_PASSWORD"],
        sslmode="require",
    )
    conn.autocommit = False
    cur = conn.cursor()
    log.info("Connected to RDS")

    # Ensure table exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id                SERIAL PRIMARY KEY,
            source            TEXT NOT NULL,
            graph_type        TEXT NOT NULL,
            question          TEXT NOT NULL,
            good_graph        TEXT NOT NULL,
            good_answer       TEXT NOT NULL,
            hidden_graph      TEXT,
            hidden_answer     TEXT,
            adversarial_graph TEXT,
            output_answer     TEXT,
            attack_success    BOOLEAN
        );
    """)
    conn.commit()
    log.info("Table 'samples' ready")

    # Download and extract image zip from HuggingFace
    log.info(f"Downloading {HF_IMAGE_ZIP} from {HF_DATASET_ID} ...")
    zip_path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        repo_type="dataset",
        filename=HF_IMAGE_ZIP,
    )
    log.info(f"Zip downloaded: {zip_path}")

    extract_dir = "/tmp/chartbench_extracted"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    log.info(f"Extracted to {extract_dir}")

    # Load dataset from HuggingFace
    log.info(f"Loading {HF_DATASET_ID} split={HF_SPLIT} ...")
    ds = load_dataset(HF_DATASET_ID, HF_SUBSET, split=HF_SPLIT)
    total = len(ds)
    log.info(f"Loaded {total} rows")


    INSERT_SQL = """
        INSERT INTO samples (source, graph_type, question, good_graph, good_answer)
        VALUES (%s, %s, %s, %s, %s)
    """
    uploaded: set[str] = set()
    inserted = 0
    skipped = 0

    for i, row in enumerate(ds):
        img_path = row["image"]
        s3_key = image_path_to_s3_key(img_path)

        if s3_key not in uploaded:
            relative_img_path = img_path.replace("./data/", "")
            local_src = os.path.join(extract_dir, relative_img_path)

            if not os.path.exists(local_src):
                log.warning(f"  Image not found in zip: {local_src}")
                skipped += 1
                continue

            with open(local_src, "rb") as f:
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key=s3_key,
                    Body=f.read(),
                    ContentType="image/png",
                )
            uploaded.add(s3_key)

        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        chart_type = row["type"]["chart"]
        question = row["conversation"][0]["query"]
        answer = row["conversation"][0]["label"]

        cur.execute(INSERT_SQL, (
            "ChartBench",
            chart_type,
            question,
            s3_uri,
            answer,
        ))
        inserted += 1

        # Commit in batches
        if inserted % BATCH_SIZE == 0:
            conn.commit()
            log.info(f"  Inserted {inserted}/{total} rows "
                     f"({len(uploaded)} unique images uploaded)")

    # Final commit
    conn.commit()
    log.info(f"Import complete: {inserted} inserted, {skipped} skipped, "
             f"{len(uploaded)} unique images")

    # Cleanup
    cur.close()
    conn.close()
    shutil.rmtree(extract_dir, ignore_errors=True)

    return {
        "total_rows": total,
        "unique_images": len(uploaded),
        "rows_inserted": inserted,
        "rows_skipped": skipped,
    }

# Entrypoint

@app.local_entrypoint()
def main():
    result = import_chartbench.remote()
    for k, v in result.items():
        print(f"  {k:20s}: {v}")