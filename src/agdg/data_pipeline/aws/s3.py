from uuid import UUID, uuid4
import boto3

IMAGE_PREFIX='samples/'
IMAGE_POSTFIX='.png'
BUCKET='agd-dev-tyson'

def put_image(image: bytes) -> UUID:
    """
    Upload an image to the S3 bucket with a UUID key and return the key
    """
    key = uuid4()
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=BUCKET,
        Key=IMAGE_PREFIX + str(key) + IMAGE_POSTFIX,
        Body=image
    )
    return key


def get_image(key: UUID | str) -> bytes:
    """
    Get an image from S3 with the given key and return its contents as bytes.
    Raise KeyError if no such image exists.
    """
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(
            Bucket=BUCKET,
            Key=IMAGE_PREFIX + str(key) + IMAGE_POSTFIX,
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise KeyError from e
        else:
            raise  # re-raise unexpected errors
    return response["Body"].read()


def wipe_s3(logger=None) -> None:
    """
    Delete all the images in the S3 bucket
    """
    s3 = boto3.client("s3")
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
