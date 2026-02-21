import modal
import random
import string
from uuid import UUID

# python import is executed before modal ships the code
# the local file config is in the parent directory thus missing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aws import get_image, put_image


aws_image = (
        modal.Image.debian_slim()
        .uv_pip_install(["boto3", "psycopg2-binary"])
        .add_local_file("aws.py", "/root/aws.py")
        .add_local_file("config.py", "/root/config.py")
    )

app = modal.App(
        "s3-example",
        image=aws_image,
        secrets=[
            modal.Secret.from_name("aws"),
        ],
    )


def random_bytes(length: int) -> str:
    return ''.join(random.choices(string.ascii_letters, k=length))


@app.function()
def upload(body: str) -> UUID:
    key = put_image(body.encode('utf-8'))
    print(f'Stored    "{body}" at   {str(key)}')
    return key


@app.function()
def download(key: UUID) -> None:
    body = get_image(key).decode("utf-8")
    print(f'Retrieved "{body}" from {str(key)}')


@app.local_entrypoint()
def main():
    body = random_bytes(20)
    key = upload.remote(body)
    download.remote(key)
