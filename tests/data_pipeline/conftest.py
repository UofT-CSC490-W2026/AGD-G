"""Shared fixtures for data_pipeline tests."""
import io
import sys
import types
from unittest.mock import MagicMock

import pytest
from PIL import Image


def make_png(size=(32, 16), color=(255, 0, 0)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def fake_boto(monkeypatch):
    """Inject fake boto3/psycopg2/botocore so aws.py can be imported."""
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = MagicMock()
    fake_psycopg2 = types.ModuleType("psycopg2")
    fake_psycopg2.connect = MagicMock()
    fake_botocore = types.ModuleType("botocore")
    fake_botocore_exc = types.ModuleType("botocore.exceptions")
    fake_botocore_exc.ClientError = type("ClientError", (Exception,), {})
    fake_botocore.exceptions = fake_botocore_exc

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.exceptions", fake_botocore_exc)


class FakeCursor:
    def __init__(self, inserts=None):
        self.inserts = inserts if inserts is not None else []

    def execute(self, query, params=None):
        if query.strip().upper().startswith("INSERT"):
            self.inserts.append(params)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConnection:
    def __init__(self, cursor=None):
        self._cursor = cursor or FakeCursor()
        self.commits = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def build_fake_aws(*, inserts=None, put_returns="fake-uuid"):
    """Return (fake_aws_module, put_calls) for use with monkeypatch.setattr(module, 'aws', ...)."""
    inserts = inserts if inserts is not None else []
    put_calls = []
    conn = FakeConnection(FakeCursor(inserts))

    fake = types.ModuleType("agdg.data_pipeline.aws")
    fake.wipe_s3 = MagicMock()
    fake.wipe_rds = MagicMock()
    fake.create_table_if_not_exists = MagicMock()
    fake.put_image = lambda data: (put_calls.append(data), put_returns)[1]
    fake.add_sample = MagicMock()
    fake.get_db_connection = lambda: conn
    return fake, put_calls, conn
