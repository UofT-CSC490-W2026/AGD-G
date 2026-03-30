"""
ChartX data pipeline: loads the ChartX dataset from Hugging Face, processes chart images
and QA pairs, and writes metadata to RDS and images to S3.
"""
from agdg.data_pipeline.aws import rds, s3
from agdg.data_pipeline.chart_type import ChartType

CHART_TYPE_TO_GRAPH_TYPE = {
    "3D-Bar": ChartType.THREE_D,
    "bar_chart_num": ChartType.BAR,
    "bar_chart": ChartType.BAR,
    "histogram": ChartType.BAR,
    "candlestick": ChartType.CANDLE,
    "multi-axes": ChartType.OTHER,
    "rings": ChartType.PIE,
    "area_chart": ChartType.AREA,
    "box": ChartType.BOX,
    "funnel": ChartType.BAR,
    "line_chart": ChartType.LINE,
    "line_chart_num": ChartType.LINE,
    "pie_chart": ChartType.PIE,
    "rose": ChartType.RADAR,
    "bubble": ChartType.SCATTER,
    "heatmap": ChartType.HEATMAP,
    "radar": ChartType.RADAR,
    "treemap": ChartType.TREEMAP,
}


def chart_type_to_graph_type(chart_type: str) -> ChartType:
    """Map a ChartX chart_type string to GraphType. Returns OTHER for unknown types."""
    return CHART_TYPE_TO_GRAPH_TYPE.get(chart_type, ChartType.OTHER)


def import_chartx(max_rows: int | None = None):
    """Load ChartX dataset, extract chart images and QA pairs, and process each row."""
    from datasets import load_dataset
    from PIL import Image
    from huggingface_hub import hf_hub_download
    import zipfile
    import os
    import io

    CHARTX_SOURCE = "ChartX"

    rds.create_table_if_not_exists()

    ds = load_dataset("InternScience/ChartX")

    zip_path = hf_hub_download(
        repo_id="InternScience/ChartX",
        filename="ChartX_png.zip",
        repo_type="dataset",
    )

    with rds.get_db_connection() as conn:
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
                        uuid = s3.put_image(image_bytes)

                        graph_type = chart_type_to_graph_type(chart_type)

                        print(f'[SAMPLE {processed+1}] {graph_type} GRAPH ({len(image_bytes)} bytes): "{question}" "{answer}"')

                        rds.insert_sample(
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
