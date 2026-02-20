"""
ChartQAx data pipeline: loads ChartQA-X from Hugging Face, processes chart images
and QA pairs, and (TODO) writes metadata to RDS and images to S3.
"""
import modal

app = modal.App("data-pipeline")

dataPipelineImage = (
    modal.Image.debian_slim(python_version="3.13.5")
    .uv_pip_install(
        "datasets==4.5.0",
        "huggingface_hub==1.4.1",
        "pillow==12.1.1"
    )
)

@app.function(
    image=dataPipelineImage, 
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def main():
    """Load ChartQA-X dataset, extract chart images and QA pairs, and process each row."""
    from datasets import load_dataset
    from PIL import Image
    from huggingface_hub import hf_hub_download
    from io import BytesIO
    import zipfile
    import os

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

    for split_name, split in ds.items():
        printed_schema_preview = False
        for row in split:
            chart_type = row.get("chart_type", row.get("type", "unknown"))
            if chart_type == "unknown":
                chart_type = row.get("chart_id", "unknown")

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

            # Preferred: direct image column from HF dataset.
            image = row.get("image") or row.get("image_path")
            if isinstance(image, dict):
                # datasets Image feature may provide {"bytes": ..., "path": ...}
                if image.get("bytes"):
                    image = Image.open(BytesIO(image["bytes"]))
                elif image.get("path"):
                    image = Image.open(image["path"])
            elif isinstance(image, str):
                maybe_local_path = image.lstrip("./")
                fallback_path = os.path.join(extract_path, zip_root, maybe_local_path)
                if os.path.exists(maybe_local_path):
                    image = Image.open(maybe_local_path)
                elif os.path.exists(fallback_path):
                    image = Image.open(fallback_path)
                else:
                    image = None

            if image is None:
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
                    print(f"Skipping row with missing image: {image_path}")
                    continue
                image = Image.open(final_image_path)

            image = image.convert("RGB")

            print(image, chart_type, question, answer)
