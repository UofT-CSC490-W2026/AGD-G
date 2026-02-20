"""
ChartX data pipeline: loads the ChartX dataset from Hugging Face, processes chart images
and QA pairs, and (TODO) writes metadata to RDS and images to S3.
"""
import modal
from config import GraphType

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
    .add_local_file("config.py", "/root/config.py")
)

# Modal application config
app = modal.App(
    "data-pipeline",
    image=dataPipelineImage,
    secrets=[
        modal.Secret.from_name("aws"),
        modal.Secret.from_name("aws-rds"),
        modal.Secret.from_name("huggingface"),
    ],
)

# ChartX dataset chart_type string -> GraphType (multiple chart types map to one GraphType)
CHART_TYPE_TO_GRAPH_TYPE = {
    "3D-Bar": GraphType.THREE_D,
    "bar_chart_num": GraphType.BAR,
    "bar_chart": GraphType.BAR,
    "histogram": GraphType.BAR,
    "candlestick": GraphType.CANDLE,
    "multi-axes": GraphType.OTHER,
    "rings": GraphType.PIE,
    "area_chart": GraphType.AREA,
    "box": GraphType.BOX,
    "funnel": GraphType.BAR,
    "line_chart": GraphType.LINE,
    "line_chart_num": GraphType.LINE,
    "pie_chart": GraphType.PIE,
    "rose": GraphType.RADAR,
    "bubble": GraphType.SCATTER,
    "heatmap": GraphType.HEATMAP,
    "radar": GraphType.RADAR,
    "treemap": GraphType.TREEMAP,
}


def chart_type_to_graph_type(chart_type: str) -> GraphType:
    """Map a ChartX chart_type string to GraphType. Returns OTHER for unknown types."""
    return CHART_TYPE_TO_GRAPH_TYPE.get(chart_type, GraphType.OTHER)


@app.function(
    timeout=3600,
    memory=4096,
)
def main(max_rows : int | None = None):
    """Load ChartX dataset, extract chart images and QA pairs, and process each row."""
    from datasets import load_dataset
    from PIL import Image
    from huggingface_hub import hf_hub_download
    import zipfile
    import os
    import io
    import aws

    CHARTX_SOURCE = "ChartX"

    aws.create_table_if_not_exists()

    # Load ChartX dataset metadata (splits and row references)
    ds = load_dataset("InternScience/ChartX")

    # Download the zip containing chart PNGs from the Hugging Face dataset repo
    zip_path = hf_hub_download(
        repo_id="InternScience/ChartX",
        filename="ChartX_png.zip",
        repo_type="dataset",
    )

    with aws.get_db_connection() as conn:
        extract_path = "/tmp"
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_path)

        MAX_FAILURES = 50
        failures = 0
        processed = 0

        with conn.cursor() as cursor:

            for split_name in ds:
                for row in ds[split_name]:
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
                        image_bytes = image_bytes.getvalue()
                        uuid = aws.put_image(image_bytes)

                        graph_type = chart_type_to_graph_type(chart_type)

                        print(f'[SAMPLE {processed+1}] {graph_type} GRAPH ({len(image_bytes)} bytes): "{question}" "{answer}"')

                        aws.add_sample(
                                cursor,
                                CHARTX_SOURCE,
                                str(graph_type),
                                question,
                                answer,
                                str(uuid)
                            )
                        conn.commit()
                        processed += 1
                    except Exception as e:
                        failures += 1
                        import traceback

                        print(f"Row failed (failure {failures}/{MAX_FAILURES}): {e!r}")
                        traceback.print_exc()
                        if failures >= MAX_FAILURES:
                            break
                        continue

                    if max_rows is not None and processed >= max_rows:
                        print(f"Processed {processed} rows.")
                        return

            if failures >= MAX_FAILURES:
                raise RuntimeError(f"Stopping after {MAX_FAILURES} failures.")


    print(f"Processed {processed} rows.")

@app.local_entrypoint()
def local_entrypoint(*arglist):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--limit", type=int, default=None, help="Maximum number of samples to import")
    args = parser.parse_args(args=arglist)
    main.remote(max_rows=args.limit)
