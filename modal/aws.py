import os
import modal
import boto3
import psycopg2
from contextlib import contextmanager

BUCKET='agd-dev-tyson'
RDS_HOST='agd-dev-postgres.cdsyi46ammw7.ca-central-1.rds.amazonaws.com'

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(
            host=RDS_HOST,
            port=5432,
            database='postgres',
            user='postgres',
            password=os.environ["DB_PASSWORD"],
            sslmode='require'
        )
        yield conn
    finally:
        if conn:
            conn.close()


def put_object(key: str, body: str) -> None:
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=body
    )


def get_object(key: str) -> bytes:
    s3 = boto3.client("s3")
    response = s3.get_object(
        Bucket=BUCKET,
        Key=key
    )
    body = response["Body"].read()
    return body
