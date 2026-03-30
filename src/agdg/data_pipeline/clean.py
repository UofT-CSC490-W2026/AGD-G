"""
Wipe S3 images and RDS tables.
Place in data_pipeline/ alongside the import scripts.

    modal run clean.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(".", "/root")
    .pip_install(".")
    .add_local_python_source("modal_run", copy=False)
)

app = modal.App(
    "agd-clean",
    image=image,
    secrets=[
        modal.Secret.from_name("aws"),
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
