"""Tests that modal_run/attack.py wires up Modal correctly."""
import sys
import types
from unittest.mock import MagicMock

from .helpers import install_fake_modal, import_fresh


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


def _load(monkeypatch):
    install_fake_modal()
    _install_fake_boto(monkeypatch)
    for mod in list(sys.modules):
        if mod.startswith("modal_run.") or mod.startswith("agdg.data_pipeline."):
            sys.modules.pop(mod, None)
    return import_fresh("modal_run.attack")


def test_attack_app_name(monkeypatch):
    module = _load(monkeypatch)
    assert module.app.args[0] == "agd-attack-pipeline"


def test_attack_calls_pipeline(monkeypatch):
    module = _load(monkeypatch)

    monkeypatch.setattr(
        module,
        "attack_all",
        lambda max_rows=0: {"attacked": 5, "rows_updated": 10},
    )

    result = module.attack(max_rows=0)
    assert result == {"attacked": 5, "rows_updated": 10}
