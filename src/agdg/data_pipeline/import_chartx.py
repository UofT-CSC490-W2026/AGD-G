"""
ChartX data pipeline: loads the ChartX dataset from Hugging Face, processes chart images
and QA pairs, and writes metadata to RDS and images to S3.
"""
from agdg.data_pipeline import aws
from agdg.data_pipeline.schema import GraphType

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


def import_chartx(max_rows: int | None = None):
    """Load ChartX dataset, extract chart images and QA pairs, and process each row."""
    from datasets import load_dataset
    from PIL import Image
    from huggingface_hub import hf_hub_download
    import zipfile
    import os
    import io

    CHARTX_SOURCE = "ChartX"

    aws.create_table_if_not_exists()

    ds = load_dataset("InternScience/ChartX")

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

                        image_path = imagePath.lstrip("./")
                        final_image_path = os.path.join(extract_path, "ChartX_png", image_path)
                        image = Image.open(final_image_path)

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
                            str(uuid),
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
