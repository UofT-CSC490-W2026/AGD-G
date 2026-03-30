"""Tests that modal_run/ingest.py wires up Modal correctly."""
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
    return import_fresh("modal_run.ingest")


def test_ingest_app_name(monkeypatch):
    module = _load(monkeypatch)
    assert module.app.args[0] == "agd-ingest"


def test_ingest_calls_pipeline_steps_in_order(monkeypatch):
    module = _load(monkeypatch)
    calls = []

    monkeypatch.setattr(module, "do_clean", lambda: calls.append("clean"))
    monkeypatch.setattr(
        module, "import_chartbench", lambda max_rows=0: calls.append("chartbench")
    )
    monkeypatch.setattr(
        module, "import_chartx", lambda max_rows=None: calls.append("chartx")
    )
    monkeypatch.setattr(
        module, "import_chartqax", lambda max_rows=None: calls.append("chartqax")
    )
    monkeypatch.setattr(
        module,
        "preprocess_all",
        lambda: (
            calls.append("preprocess"),
            {"unique_images": 0, "rows_updated": 0, "skipped": 0},
        )[1],
    )

    module.ingest(max_rows=5, clean=True, skip_import=False, skip_preprocess=False)

    assert calls == ["clean", "chartbench", "chartx", "chartqax", "preprocess"]


def test_ingest_skips_import_when_requested(monkeypatch):
    module = _load(monkeypatch)
    calls = []

    monkeypatch.setattr(
        module,
        "preprocess_all",
        lambda: (
            calls.append("preprocess"),
            {"unique_images": 0, "rows_updated": 0, "skipped": 0},
        )[1],
    )

    module.ingest(max_rows=0, clean=False, skip_import=True, skip_preprocess=False)

    assert calls == ["preprocess"]


def test_ingest_skips_preprocess_when_requested(monkeypatch):
    module = _load(monkeypatch)
    calls = []

    monkeypatch.setattr(
        module, "import_chartbench", lambda max_rows=0: calls.append("chartbench")
    )
    monkeypatch.setattr(
        module, "import_chartx", lambda max_rows=None: calls.append("chartx")
    )
    monkeypatch.setattr(
        module, "import_chartqax", lambda max_rows=None: calls.append("chartqax")
    )

    module.ingest(max_rows=0, clean=False, skip_import=False, skip_preprocess=True)

    assert calls == ["chartbench", "chartx", "chartqax"]


def test_main_delegates_to_ingest_remote(monkeypatch):
    module = _load(monkeypatch)
    captured = {}

    def fake_remote(**kwargs):
        captured.update(kwargs)

    module.ingest.remote = fake_remote

    module.main(limit=10, clean=True, skip_import=False, skip_preprocess=True)

    assert captured == {
        "max_rows": 10,
        "clean": True,
        "skip_import": False,
        "skip_preprocess": True,
    }
