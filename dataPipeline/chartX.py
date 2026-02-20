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
        "pillow==12.1.1"
    )
)

@app.function(
    image=dataPipelineImage, 
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def main():
    """Load ChartX dataset, extract chart images and QA pairs, and process each row."""
    from datasets import load_dataset
    from PIL import Image
    from huggingface_hub import hf_hub_download
    import zipfile
    import os

    # Load ChartX dataset metadata (splits and row references)
    ds = load_dataset("InternScience/ChartX")

    # Download the zip containing chart PNGs from the Hugging Face dataset repo
    zip_path = hf_hub_download(
        repo_id="InternScience/ChartX",
        filename="ChartX_png.zip",
        repo_type="dataset",
    )

    for split in ds:
        # Extract the zip once per split (images are referenced by path in the dataset)
        extract_path = "/tmp"
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_path)

        for row in split:
            chart_type = row["chart_type"]
            imagePath = row["img"]
            qaPair = row["QA"]
            question = qaPair["input"]
            answer = qaPair["output"]

            # Resolve image path and load as RGB
            image_path = imagePath.lstrip("./")
            final_image_path = os.path.join(extract_path, "ChartX_png", image_path)
            image = Image.open(final_image_path).convert("RGB")

            print(image, chart_type, question, answer)

        # TODO:
        # write data to RDS table
        # get image
        # save image to S3 bucket
        # update RDS table with image URL
