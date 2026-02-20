import modal
import random
import string
from aws import get_object, put_object, BUCKET


aws_image = (
        modal.Image.debian_slim()
        .uv_pip_install(["boto3", "psycopg2-binary"])
        .add_local_file("aws.py", "/root/aws.py")
    )

app = modal.App(
        "s3-example",
        image=aws_image,
        secrets=[
            modal.Secret.from_name("aws"),
        ],
    )


def random_string(length: int) -> str:
    return ''.join(random.choices(string.ascii_letters, k=length))


@app.function()
def upload(key: str, body: str) -> None:
    put_object(key, body)
    print(f'Stored    "{body}" at   s3://{BUCKET}/{key}')


@app.function()
def download(key: str) -> None:
    body = get_object(key).decode("utf-8")
    print(f'Retrieved "{body}" from s3://{BUCKET}/{key}')


@app.local_entrypoint()
def main():
    key = random_string(12)
    body = ' '.join(random_string(5) for _ in range(3))
    upload.remote(key, body)
    download.remote(key)
