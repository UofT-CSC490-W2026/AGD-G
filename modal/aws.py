import os
from uuid import UUID, uuid4
import boto3
from botocore.exceptions import ClientError
import psycopg2
from contextlib import contextmanager
from enum import Enum

BUCKET='agd-dev-tyson'
RDS_HOST='agd-dev-postgres.cdsyi46ammw7.ca-central-1.rds.amazonaws.com'
IMAGE_PREFIX='samples/'
IMAGE_POSTFIX='.png'

class GraphType(Enum):
    THREE_D = 1
    AREA = 2
    BAR = 3
    BOX = 4
    CANDLE = 5
    HEATMAP = 6
    LINE = 7
    NODE = 8
    OTHER = 9
    PIE = 10
    RADAR = 11
    SCATTER = 12
    TREEMAP = 13

    def __str__(self):
        return self.name

    @staticmethod
    def get_names():
        """
        Get names in the enum format PostgreSQL expects, like
        'FOO', 'BAR', 'BAZ'
        """
        return ", ".join(f"'{member.name}'" for member in GraphType)


@contextmanager
def get_db_connection():
    """
    Make a connection to the RDS PostgreSQL database.
    Use `with get_db_connection() as conn:` and it will automatically
    close when the `with` block ends.
    """
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
        conn.commit()
    finally:
        if conn:
            conn.close()


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


def create_table_if_not_exists() -> None:
    """
    Create the SQL table for this project's test data, and do nothing
    if it already exists.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                DO $$
                BEGIN
                    CREATE TYPE GRAPH_TYPE AS ENUM ({GraphType.get_names()});
                EXCEPTION
                    WHEN duplicate_object THEN NULL;
                END $$;
                """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id                SERIAL PRIMARY KEY,
                    source            TEXT NOT NULL,
                    graph_type        GRAPH_TYPE NOT NULL,
                    question          TEXT NOT NULL,
                    good_answer       TEXT NOT NULL,
                    raw_graph         UUID NOT NULL,
                    original_width    INTEGER,
                    original_height   INTEGER,
                    preprocess_meta   JSONB,
                    good_graph        UUID,
                    hidden_graph      UUID,
                    hidden_answer     TEXT,
                    adversarial_graph UUID,
                    output_answer     TEXT,
                    attack_succeeded  BOOLEAN,
                    created_at        TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)

def add_sample_row(
        cursor,
        source: str,
		graph_type: GraphType,
		question: str,
		answer: str,
        graph: UUID) -> None:
    cursor.execute(
            """
            INSERT INTO samples (source, graph_type, question, good_answer, raw_graph)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (source, str(graph_type), question, answer, str(graph))
        )

def wipe_rds() -> None:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS samples")
            cursor.execute("DROP TYPE IF EXISTS graph_type")

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
