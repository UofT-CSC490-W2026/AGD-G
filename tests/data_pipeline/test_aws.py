import importlib
import sys
import types
from uuid import UUID

import pytest


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


class BatchCursor:
    def __init__(self, batches):
        self.executed = []
        self._batches = list(batches)

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchmany(self, size):
        if self._batches:
            return self._batches.pop(0)
        return []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class BatchConnection:
    def __init__(self, batches):
        self.cursor_obj = BatchCursor(batches)

    def cursor(self):
        return self.cursor_obj


def load_aws(monkeypatch, *, s3_client=None, rds_client=None, connection=None):
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

    importlib.invalidate_caches()
    sys.modules.pop("agdg.data_pipeline.aws", None)
    sys.modules.pop("agdg.data_pipeline.aws.rds", None)
    sys.modules.pop("agdg.data_pipeline.aws.s3", None)
    return importlib.import_module("agdg.data_pipeline.aws")


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
    assert "CREATE TABLE IF NOT EXISTS samples" in queries
    assert "CREATE TABLE IF NOT EXISTS clean_answers" in queries


def test_add_sample_inserts_expected_values(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    cursor = FakeCursor()

    aws.add_sample(
        cursor,
        source="chartx",
        graph_type=aws.GraphType.BAR,
        question="q",
        answer="a",
        chart="1234",
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
    assert queries == ["DROP SCHEMA public CASCADE;", "CREATE SCHEMA public;"]


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


def test_insert_preprocessing_wraps_meta_and_updates_sample(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    cursor = FakeCursor()

    aws.rds.insert_preprocessing(cursor, 7, "chart-uuid", 800, 600, {"crop_box": [1, 2, 3, 4]})

    query, params = cursor.executed[0]
    assert "UPDATE samples" in query
    assert params[0] == 800
    assert params[1] == 600
    assert getattr(params[2], "adapted", None) == {"crop_box": [1, 2, 3, 4]}
    assert params[3:] == ("chart-uuid", 7)


def test_insert_clean_target_and_adversarial_rows(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    cursor = FakeCursor()

    aws.rds.insert_clean_answer(cursor, 3, "clean text", "llava")
    aws.rds.insert_target_answer(cursor, 5, "target text", "smoke")
    aws.rds.insert_adversarial_chart(cursor, 6, "adv-chart", "AttackVLM", "llava", {"steps": 20})
    aws.rds.insert_adversarial_answer(cursor, 8, "answer", "llava", True, {"winner": "A"})

    assert "INSERT INTO clean_answers" in cursor.executed[0][0]
    assert cursor.executed[0][1] == (3, "llava", "clean text")
    assert "INSERT INTO target_answers" in cursor.executed[1][0]
    assert cursor.executed[1][1] == (5, "target text", "smoke")
    assert "INSERT INTO adversarial_charts" in cursor.executed[2][0]
    assert cursor.executed[2][1][:4] == (6, "adv-chart", "AttackVLM", "llava")
    assert getattr(cursor.executed[2][1][4], "adapted", None) == {"steps": 20}
    assert "INSERT INTO adversarial_answers" in cursor.executed[3][0]
    assert cursor.executed[3][1] == (8, "llava", "answer", True, {"winner": "A"})


def test_iter_preprocessor_inputs_batches_rows(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    conn = BatchConnection([[(1, "raw-1")], [(2, "raw-2")], []])

    rows = list(aws.rds.iter_preprocessor_inputs(conn, batch_size=1))

    assert rows == [
        {"sample_id": 1, "raw_chart": "raw-1"},
        {"sample_id": 2, "raw_chart": "raw-2"},
    ]
    assert "SELECT id, raw_chart" in conn.cursor_obj.executed[0][0]


def test_iter_clean_answer_inputs_filters_by_model(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    conn = BatchConnection([[(3, "clean-uuid")], []])

    rows = list(aws.rds.iter_clean_answer_inputs(conn, "llava", batch_size=10))

    assert rows == [{"sample_id": 3, "clean_chart": "clean-uuid"}]
    assert conn.cursor_obj.executed[0][1] == ("llava",)


def test_iter_target_inputs_yields_expected_payload(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    conn = BatchConnection([[(4, "clean answer", "clean-uuid")], []])

    rows = list(aws.rds.iter_target_inputs(conn, "smoke", batch_size=5))

    assert rows == [
        {
            "clean_answer_id": 4,
            "clean_answer": "clean answer",
            "clean_chart": "clean-uuid",
        }
    ]
    assert conn.cursor_obj.executed[0][1] == ("smoke",)


def test_iter_target_inputs_with_source(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    conn = BatchConnection([[(4, "clean answer", "clean-uuid")], []])

    rows = list(aws.rds.iter_target_inputs(conn, "smoke", source="ChartBench", batch_size=5))

    assert rows == [
        {
            "clean_answer_id": 4,
            "clean_answer": "clean answer",
            "clean_chart": "clean-uuid",
        }
    ]
    assert conn.cursor_obj.executed[0][1] == ("smoke", "ChartBench")


def test_iter_target_inputs_sampled(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())

    class SampledCursor:
        def __init__(self):
            self.executed = []

        def execute(self, query, params=None):
            self.executed.append((query, params))

        def fetchall(self):
            return [
                (1, "answer1", "chart-uuid1", "ChartBench"),
                (2, "answer2", "chart-uuid2", "ChartX"),
            ]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class SampledConn:
        def __init__(self):
            self.cursor_obj = SampledCursor()

        def cursor(self):
            return self.cursor_obj

    conn = SampledConn()
    rows = list(aws.rds.iter_target_inputs_sampled(conn, "qwen", per_source=5))

    assert len(rows) == 2
    assert rows[0]["chart_source"] == "ChartBench"
    assert rows[1]["clean_answer"] == "answer2"
    assert "UNION ALL" in conn.cursor_obj.executed[0][0]


def test_iter_attack_inputs_yields_expected_payload(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    conn = BatchConnection([[(9, "clean answer", "clean-uuid", "target answer")], []])

    rows = list(aws.rds.iter_attack_inputs(conn, "AttackVLM", "llava", batch_size=2))

    assert rows == [
        {
            "target_answer_id": 9,
            "clean_answer": "clean answer",
            "clean_chart": "clean-uuid",
            "target_answer": "target answer",
        }
    ]
    assert conn.cursor_obj.executed[0][1] == ("AttackVLM", "llava")


def test_iter_eval_inputs_yields_expected_payload(monkeypatch):
    aws = load_aws(monkeypatch, s3_client=object(), connection=FakeConnection())
    conn = BatchConnection([[(10, "clean answer", "clean-uuid", "target answer", "adv-uuid")], []])

    rows = list(aws.rds.iter_eval_inputs(conn, "llava", batch_size=2))

    assert rows == [
        {
            "adversarial_chart_id": 10,
            "clean_answer": "clean answer",
            "clean_chart": "clean-uuid",
            "target_answer": "target answer",
            "adversarial_chart": "adv-uuid",
        }
    ]
    assert conn.cursor_obj.executed[0][1] == ("llava",)
