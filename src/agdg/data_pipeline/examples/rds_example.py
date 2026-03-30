import modal

# python import is executed before modal ships the code
# the local file config is in the parent directory thus missing
import sys
from pathlib import Path
from agdg.data_pipeline.aws import get_db_connection

aws_image = (
    modal.Image.debian_slim()
    .add_local_dir(".", "/root")
    .pip_install(".")
    .add_local_python_source("modal_run", copy=False)
)

app = modal.App(
    "rds-example",
    image=aws_image,
    secrets=[
        modal.Secret.from_name("aws"),
    ],
)


@app.function()
def get_db_version():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute('SELECT version();')
        print(cur.fetchone()[0])


@app.local_entrypoint()
def main():
    get_db_version.remote()
