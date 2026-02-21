import modal

# python import is executed before modal ships the code
# the local file config is in the parent directory thus missing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aws import get_db_connection

aws_image = (
        modal.Image.debian_slim()
        .uv_pip_install(["boto3", "psycopg2-binary"])
        .add_local_file("aws.py", "/root/aws.py")
        .add_local_file("config.py", "/root/config.py")
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
