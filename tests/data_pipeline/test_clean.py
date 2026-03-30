import sys
import types
from unittest.mock import MagicMock


def _install_fake_boto(monkeypatch):
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


def _load_module(monkeypatch):
    import importlib
    _install_fake_boto(monkeypatch)
    sys.modules.pop("agdg.data_pipeline.aws", None)
    sys.modules.pop("agdg.data_pipeline.clean", None)
    return importlib.import_module("agdg.data_pipeline.clean")


def test_clean_runs_expected_aws_steps(monkeypatch):
    module = _load_module(monkeypatch)
    calls = []

    fake_aws = types.ModuleType("agdg.data_pipeline.aws")
    fake_aws.wipe_s3 = lambda logger=None: calls.append("wipe_s3")
    fake_aws.wipe_rds = lambda: calls.append("wipe_rds")
    fake_aws.create_table_if_not_exists = lambda: calls.append("create_table")
    monkeypatch.setattr(module, "aws", fake_aws)

    module.clean()

    assert calls == ["wipe_s3", "wipe_rds", "create_table"]
