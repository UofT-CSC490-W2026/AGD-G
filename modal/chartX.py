"""
ChartX data pipeline: loads the ChartX dataset from Hugging Face, processes chart images
and QA pairs, and (TODO) writes metadata to RDS and images to S3.
"""
import modal

app = modal.App("data-pipeline")

# Container image with Python 3.13 and dependencies for dataset loading and image handling
dataPipelineImage = (
    modal.Image.debian_slim(python_version="3.13.5")
    .uv_pip_install(
        "datasets==4.5.0",
        "huggingface_hub==1.4.1",
        "pillow==12.1.1",
        "boto3",
        "psycopg2-binary"
    )
    .add_local_file("aws.py", "/root/aws.py")
)

@app.function(
    image=dataPipelineImage, 
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("aws"),
        modal.Secret.from_name("aws-rds"),
    ]
)
def main():
    """Load ChartX dataset, extract chart images and QA pairs, and process each row."""
    from datasets import load_dataset
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from aws import put_image, get_db_connection
    import zipfile
    import os
    import io

    CHARTX_SOURCE = "ChartX"

    # Load ChartX dataset metadata (splits and row references)
    ds = load_dataset("InternScience/ChartX")

    # Download the zip containing chart PNGs from the Hugging Face dataset repo
    zip_path = hf_hub_download(
        repo_id="InternScience/ChartX",
        filename="ChartX_png.zip",
        repo_type="dataset",
    )

    with get_db_connection() as conn:

        extract_path = "/tmp"
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_path)

        MAX_FAILURES = 50
        failures = 0

        with conn.cursor() as cursor:

            for split in ds:        
                for row in range(5):
                    if failures >= MAX_FAILURES:
                        break
                    try:
                        chart_type = row["chart_type"]
                        imagePath = row["img"]
                        qaPair = row["QA"]
                        question = qaPair["input"]
                        answer = qaPair["output"]

                        # Resolve image path and load as RGB
                        image_path = imagePath.lstrip("./")
                        final_image_path = os.path.join(extract_path, "ChartX_png", image_path)
                        image = Image.open(final_image_path)

                        # Convert PIL Image to bytes for S3 upload
                        image_bytes = io.BytesIO()
                        image.save(image_bytes, format='PNG')
                        uuid = put_image(image_bytes.getvalue())

                        cursor.execute(
                            """
                            INSERT INTO samples (source, graph_type, question, good_answer, raw_graph)
                            VALUES (%s, %s, %s, %s, %s);
                            """,
                            (CHARTX_SOURCE, chart_type, question, answer, uuid),
                        )
                        conn.commit()
                    except Exception:
                        failures += 1
                        if failures >= MAX_FAILURES:
                            break
                        continue

            if failures >= MAX_FAILURES:
                raise RuntimeError(f"Stopping after {MAX_FAILURES} failures.")
