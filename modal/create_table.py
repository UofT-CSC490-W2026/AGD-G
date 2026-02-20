import modal
from aws import create_schema_if_not_exists


aws_image = (
        modal.Image.debian_slim()
        .uv_pip_install(["boto3", "psycopg2-binary"])
        .add_local_file("aws.py", "/root/aws.py")
    )

app = modal.App(
        "create-rds-table",
        image=aws_image,
        secrets=[
            modal.Secret.from_name("aws"),
            modal.Secret.from_name("aws-rds"),
        ],
    )


@app.function()
def create_table() -> None:
    create_schema_if_not_exists()


@app.local_entrypoint()
def main():
    create_table.remote()
