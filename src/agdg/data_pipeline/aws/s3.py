import os
from uuid import UUID, uuid4

import boto3
from botocore.exceptions import ClientError

IMAGE_PREFIX = 'samples/'
IMAGE_POSTFIX = '.png'
BUCKET = os.environ.get("AGDG_S3_BUCKET", "agd-dev-tyson")
S3_ENDPOINT = os.environ.get("AGDG_S3_ENDPOINT")  # e.g. http://localhost:9000 for MinIO
S3_REGION = os.environ.get("AGDG_S3_REGION", "ca-central-1")


def _s3_client():
    kwargs: dict = {"region_name": S3_REGION}
    if S3_ENDPOINT:
        kwargs["endpoint_url"] = S3_ENDPOINT
    return boto3.client("s3", **kwargs)


def put_image(image: bytes) -> UUID:
    """Upload an image to the S3 bucket with a UUID key and return the key."""
    key = uuid4()
    s3 = _s3_client()
    s3.put_object(
        Bucket=BUCKET,
        Key=IMAGE_PREFIX + str(key) + IMAGE_POSTFIX,
        Body=image,
    )
    return key


def get_image(key: UUID | str) -> bytes:
    """
    Get an image from S3 with the given key and return its contents as bytes.
    Raise KeyError if no such image exists.
    """
    s3 = _s3_client()
    try:
        response = s3.get_object(
            Bucket=BUCKET,
            Key=IMAGE_PREFIX + str(key) + IMAGE_POSTFIX,
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise KeyError from e
        else:
            raise
    return response["Body"].read()


def wipe_s3(logger=None) -> None:
    """Delete all the images in the S3 bucket."""
    s3 = _s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=IMAGE_PREFIX):
        objects = page.get("Contents", [])
        if objects:
            s3.delete_objects(
                Bucket=BUCKET,
                Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
            )
            if logger:
                logger.info(f"  Deleted {len(objects)} S3 objects")
