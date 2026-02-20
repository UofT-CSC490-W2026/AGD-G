"""
Wipe S3 images and RDS tables.
Place in data_pipeline/ alongside the import scripts.

    modal run clean.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("boto3", "psycopg2-binary")
    .add_local_file("../aws.py", "/root/aws.py")
    .add_local_file("../config.py", "/root/config.py")
)

app = modal.App(
    "agd-clean",
    image=image,
    secrets=[
        modal.Secret.from_name("aws"),
        modal.Secret.from_name("aws-rds"),
    ],
)


@app.function()
def clean():
    import aws
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("clean")

    aws.wipe_s3(logger=log)
    aws.wipe_rds()
    aws.create_table_if_not_exists()
    log.info("Clean done — schema recreated")


@app.local_entrypoint()
def main():
    clean.remote()