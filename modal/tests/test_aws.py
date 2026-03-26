import sys
import types
from uuid import UUID

import pytest

from conftest import ensure_modal_root, import_fresh


class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeCursor:
    def __init__(self):
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self):
        self.closed = False
        self.committed = False
        self.cursor_obj = FakeCursor()

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


def load_aws(monkeypatch, *, s3_client=None, rds_client=None, connection=None):
    ensure_modal_root()

    fake_boto3 = types.ModuleType("boto3")
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_botocore = types.ModuleType("botocore")
    fake_botocore_exceptions = types.ModuleType("botocore.exceptions")
    fake_botocore_exceptions.ClientError = FakeClientError
    fake_botocore.exceptions = fake_botocore_exceptions

    client_map = {
        "s3": s3_client,
        "rds": rds_client,
    }
    fake_boto3.client = lambda service, region_name=None: client_map[service]
    fake_psycopg2.connect = lambda **kwargs: connection

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_botocore_exceptions)
    return import_fresh("aws")


def test_get_db_connection_commits_and_closes(monkeypatch):
    class FakeRdsClient:
        def generate_db_auth_token(self, **kwargs):
            self.kwargs = kwargs
            return "token"

    connection = FakeConnection()
    rds_client = FakeRdsClient()
    aws = load_aws(monkeypatch, rds_client=rds_client, connection=connection)

    with aws.get_db_connection() as conn:
        assert conn is connection

    assert connection.committed is True
    assert connection.closed is True
    assert rds_client.kwargs["DBUsername"] == aws.RDS_USER


def test_get_image_returns_bytes(monkeypatch):
    class FakeBody:
        def read(self):
            return b"image-bytes"

    class FakeS3Client:
        def get_object(self, **kwargs):
            self.kwargs = kwargs
            return {"Body": FakeBody()}

    s3_client = FakeS3Client()
    aws = load_aws(monkeypatch, s3_client=s3_client, connection=FakeConnection())

    assert aws.get_image("abc") == b"image-bytes"
    assert s3_client.kwargs["Bucket"] == aws.BUCKET


def test_get_image_raises_key_error_for_missing_object(monkeypatch):
    class FakeS3Client:
        def get_object(self, **kwargs):
            raise FakeClientError("NoSuchKey")

    aws = load_aws(monkeypatch, s3_client=FakeS3Client(), connection=FakeConnection())

    with pytest.raises(KeyError):
        aws.get_image("missing")


def test_get_image_reraises_unexpected_s3_error(monkeypatch):
    class FakeS3Client:
        def get_object(self, **kwargs):
            raise FakeClientError("AccessDenied")

    aws = load_aws(monkeypatch, s3_client=FakeS3Client(), connection=FakeConnection())

    with pytest.raises(FakeClientError):
        aws.get_image("missing")


def test_put_image_uploads_bytes_and_returns_uuid(monkeypatch):
    class FakeS3Client:
        def put_object(self, **kwargs):
            self.kwargs = kwargs

    s3_client = FakeS3Client()
    aws = load_aws(monkeypatch, s3_client=s3_client, connection=FakeConnection())

    key = aws.put_image(b"png-bytes")

    assert isinstance(key, UUID)
    assert s3_client.kwargs["Bucket"] == aws.BUCKET
    assert s3_client.kwargs["Body"] == b"png-bytes"
    assert s3_client.kwargs["Key"].startswith(aws.IMAGE_PREFIX)


def test_create_table_if_not_exists_executes_schema_statements(monkeypatch):
    class FakeRdsClient:
        def generate_db_auth_token(self, **kwargs):
            return "token"

    connection = FakeConnection()
    aws = load_aws(
        monkeypatch,
        s3_client=object(),
        rds_client=FakeRdsClient(),
        connection=connection,
    )

    aws.create_table_if_not_exists()

    queries = "\n".join(query for query, _ in connection.cursor_obj.executed)
    assert "CREATE TYPE GRAPH_TYPE AS ENUM" in queries
    assert "CREATE TABLE IF NOT EXISTS samples" in queries


def test_add_sample_inserts_expected_values(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    cursor = FakeCursor()

    aws.add_sample(
        cursor,
        source="chartx",
        graph_type=aws.GraphType.BAR,
        question="q",
        answer="a",
        graph="1234",
    )

    query, params = cursor.executed[0]
    assert "INSERT INTO samples" in query
    assert params == ("chartx", "BAR", "q", "a", "1234")


def test_wipe_rds_drops_table_and_type(monkeypatch):
    class FakeRdsClient:
        def generate_db_auth_token(self, **kwargs):
            return "token"

    connection = FakeConnection()
    aws = load_aws(
        monkeypatch,
        s3_client=object(),
        rds_client=FakeRdsClient(),
        connection=connection,
    )

    aws.wipe_rds()

    queries = [query for query, _ in connection.cursor_obj.executed]
    assert queries == ["DROP TABLE IF EXISTS samples", "DROP TYPE IF EXISTS graph_type"]


def test_wipe_s3_deletes_paginated_objects(monkeypatch):
    class FakePaginator:
        def paginate(self, **kwargs):
            return [
                {"Contents": [{"Key": "samples/1.png"}, {"Key": "samples/2.png"}]},
                {"Contents": []},
            ]

    class FakeS3Client:
        def __init__(self):
            self.deleted = []

        def get_paginator(self, name):
            assert name == "list_objects_v2"
            return FakePaginator()

        def delete_objects(self, **kwargs):
            self.deleted.append(kwargs)

    s3_client = FakeS3Client()
    aws = load_aws(monkeypatch, s3_client=s3_client, connection=FakeConnection())

    aws.wipe_s3()

    assert len(s3_client.deleted) == 1
    assert s3_client.deleted[0]["Delete"]["Objects"][0]["Key"] == "samples/1.png"


def test_wipe_s3_logs_deleted_count(monkeypatch):
    class FakePaginator:
        def paginate(self, **kwargs):
            return [{"Contents": [{"Key": "samples/1.png"}]}]

    class FakeS3Client:
        def get_paginator(self, name):
            return FakePaginator()

        def delete_objects(self, **kwargs):
            return None

    class FakeLogger:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(message)

    logger = FakeLogger()
    aws = load_aws(monkeypatch, s3_client=FakeS3Client(), connection=FakeConnection())

    aws.wipe_s3(logger=logger)

    assert logger.messages == ["  Deleted 1 S3 objects"]
