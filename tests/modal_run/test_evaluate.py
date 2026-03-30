"""Tests that modal_run/evaluate.py wires up Modal correctly."""
import sys
import types
from unittest.mock import MagicMock

import pytest

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
    return import_fresh("modal_run.evaluate")


def test_evaluate_app_name(monkeypatch):
    module = _load(monkeypatch)
    assert module.app.args[0] == "agd-evaluate"


def test_evaluate_calls_pipeline(monkeypatch):
    module = _load(monkeypatch)

    monkeypatch.setattr(
        module,
        "evaluate_all",
        lambda max_rows=0: {"evaluated": 3, "succeeded": 1, "failed": 2, "asr_pct": 33.3},
    )

    result = module.evaluate(max_rows=0)
    assert result["evaluated"] == 3


def test_explain_calls_pipeline(monkeypatch):
    module = _load(monkeypatch)

    monkeypatch.setattr(
        module,
        "explain_all",
        lambda max_rows=0: {"explained": 7},
    )

    result = module.explain(max_rows=0)
    assert result == {"explained": 7}


def test_main_dispatches_evaluate(monkeypatch, capsys):
    module = _load(monkeypatch)

    monkeypatch.setattr(
        module,
        "evaluate_all",
        lambda max_rows=0: {"evaluated": 1, "succeeded": 0, "failed": 1, "asr_pct": 0.0},
    )

    module.main(mode="evaluate", limit=0)
    out = capsys.readouterr().out
    assert "evaluated" in out


def test_main_dispatches_explain(monkeypatch, capsys):
    module = _load(monkeypatch)

    monkeypatch.setattr(
        module,
        "explain_all",
        lambda max_rows=0: {"explained": 2},
    )

    module.main(mode="explain", limit=0)
    out = capsys.readouterr().out
    assert "explained" in out


def test_main_rejects_unknown_mode(monkeypatch):
    module = _load(monkeypatch)

    with pytest.raises(ValueError, match="Unknown mode"):
        module.main(mode="bogus", limit=0)
