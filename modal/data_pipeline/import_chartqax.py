"""
ChartQAx data pipeline: loads ChartQA-X from Hugging Face, processes chart images
and QA pairs, and (TODO) writes metadata to RDS and images to S3.
"""

# python import is executed before modal ships the code
# the local file config is in the parent directory thus missing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal
from config import GraphType

data_pipeline_image = (
    modal.Image.debian_slim(python_version="3.13.5")
    .uv_pip_install(
        "datasets==4.5.0",
        "huggingface_hub==1.4.1",
        "datasets==4.5.0",
        "pillow==12.1.1",
        "boto3",
        "psycopg2-binary"
    )
    .add_local_file("../aws.py", "/root/aws.py")
    .add_local_file("../config.py", "/root/config.py")
)

app = modal.App(
        "data-pipeline",
        image=data_pipeline_image,
        secrets=[
            modal.Secret.from_name("aws"),
            modal.Secret.from_name("huggingface"),
        ],
    )

@app.function(
    timeout=3600,
    memory=4096,
)
def main(max_rows : int | None = None):
    """Load ChartQA-X dataset, extract chart images and QA pairs, and process each row."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import zipfile
    import aws
    import os

    aws.create_table_if_not_exists()

    # Load ChartQA-X dataset metadata (splits and row references)
    repo_id = "shamanthakhegde/ChartQA-X"
    ds = load_dataset(repo_id)

    # Optional zip-based image bundle fallback.
    # If dataset rows already include an "image" field, this is not needed.
    zip_filename = os.environ.get("HF_IMAGE_ZIP", "")
    zip_root = os.environ.get("HF_IMAGE_ROOT", "ChartQA-X")
    extract_path = "/tmp/chartqa_x"
    os.makedirs(extract_path, exist_ok=True)

    if zip_filename:
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=zip_filename,
            repo_type="dataset",
        )
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_path)

    with aws.get_db_connection() as conn:
        with conn.cursor() as cursor:
            import_dataset(ds, cursor, extract_path, zip_root, max_rows)

def import_dataset(ds, cursor, extract_path, zip_root, max_rows : int | None) -> None:
    import re
    import aws

    num_imported = 0
    for split_name, split in ds.items():
        printed_schema_preview = False
        for row in split:
            graph_type = row.get("chart_type") or row.get("type") or row.get("chart_id", "unknown")

            qa_pair = row.get("QA", {})
            question = qa_pair.get("input", "")
            answer = qa_pair.get("output", "")
            if not question:
                question = row.get(
                    "question",
                    row.get("Question", row.get("query", row.get("input", ""))),
                )
            if not answer:
                answer = row.get(
                    "answer",
                    row.get("Answer", row.get("label", row.get("output", ""))),
                )

            if not printed_schema_preview:
                keys = sorted(list(row.keys()))
                print(f"[schema preview] split={split_name} keys={keys}")
                for candidate in [
                    "image",
                    "image_path",
                    "img",
                    "imgname",
                    "image_name",
                    "file_name",
                    "filename",
                    "Question",
                    "question",
                    "Answer",
                    "query",
                    "answer",
                    "label",
                ]:
                    if candidate in row:
                        print(f"[schema preview] {candidate}={row[candidate]}")
                printed_schema_preview = True

            try:
                image = get_image(row, extract_path, zip_root)
            except ValueError as e:
                print(f"Skipping row: {str(e)}")
                continue
            image = image.convert("RGB")
            image_bytes = image_to_bytes(image)
            graph_type = re.sub(r'_[0-9]+', '', graph_type)
            graph_type = map_graph_type(graph_type)

            print(f'[SAMPLE {num_imported+1}] {graph_type} GRAPH ({len(image_bytes)} bytes): "{question}" "{answer}"')
            image_key = aws.put_image(image_bytes)
            aws.add_sample(cursor, "ChartQA-X", graph_type, question, answer, image_key)

            num_imported += 1
            if max_rows is not None and num_imported >= max_rows:
                return

def get_image(row, extract_path, zip_root) -> bytes:
    from PIL import Image
    from io import BytesIO
    import os
    # Preferred: direct image column from HF dataset.
    image = row.get("image") or row.get("image_path")
    if isinstance(image, dict):
        # datasets Image feature may provide {"bytes": ..., "path": ...}
        if image.get("bytes"):
            return Image.open(BytesIO(image["bytes"]))
        elif image.get("path"):
            return Image.open(image["path"])
    if isinstance(image, str):
        maybe_local_path = image.lstrip("./")
        fallback_path = os.path.join(extract_path, zip_root, maybe_local_path)
        if os.path.exists(maybe_local_path):
            return Image.open(maybe_local_path)
        elif os.path.exists(fallback_path):
            return Image.open(fallback_path)
    if image is not None:
        return image

    image_path = str(
        row.get("img")
        or row.get("imgname")
        or row.get("image_name")
        or row.get("file_name")
        or row.get("filename")
        or ""
    ).lstrip("./")
    final_image_path = os.path.join(extract_path, zip_root, image_path)
    if not image_path or not os.path.exists(final_image_path):
        raise ValueError(f"Row missing image: {image_path}")
    return Image.open(final_image_path)

def image_to_bytes(image) -> bytes:
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

def map_graph_type(graph_type: str) -> GraphType:
    mapping = {
        "two_col": GraphType.BAR,
        "multi_col": GraphType.BAR,
    }
    return mapping.get(graph_type, GraphType.OTHER)

@app.local_entrypoint()
def local_entrypoint(*arglist):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int, default=None, help="Maximum number of samples to import")
    args = parser.parse_args(args=arglist)
    main.remote(max_rows=args.limit)
