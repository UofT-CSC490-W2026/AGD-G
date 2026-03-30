"""Tests that modal_run/target.py wires up Modal correctly."""
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
        if mod.startswith("modal_run.") or mod.startswith("agdg.data_pipeline.") or mod.startswith("agdg.targeting"):
            sys.modules.pop(mod, None)
    return import_fresh("modal_run.target")


def test_target_app_name(monkeypatch):
    module = _load(monkeypatch)
    assert module.app.args[0] == "agd-target-pipeline"


def test_generate_targets_calls_pipeline(monkeypatch):
    module = _load(monkeypatch)

    monkeypatch.setattr(
        module,
        "generate_target_responses",
        lambda **kwargs: {"processed": 12},
    )

    result = module.generate_targets(max_rows=0, strategy="qwen")
    assert result == {"processed": 12}


def test_preview_targets_calls_pipeline(monkeypatch):
    module = _load(monkeypatch)

    monkeypatch.setattr(
        module,
        "preview_target_responses",
        lambda **kwargs: [{"source": "X", "clean_answer": "c", "thinking": "t", "target": "tgt"}],
    )

    result = module.preview_targets(per_source=5, strategy="qwen")
    assert len(result) == 1


def test_main_generate_prints_remote_results(monkeypatch, capsys):
    module = _load(monkeypatch)

    def fake_remote(**kwargs):
        return {"processed": 5}

    module.generate_targets.remote = fake_remote

    module.main(max_rows=5, strategy="qwen")

    out = capsys.readouterr().out
    assert "processed" in out


def test_main_preview_prints_results(monkeypatch, capsys):
    module = _load(monkeypatch)

    def fake_remote(**kwargs):
        return [{"source": "CB", "clean_answer": "clean", "thinking": "hmm", "target": "tgt"}]

    module.preview_targets.remote = fake_remote

    module.main(preview=True, per_source=3, strategy="qwen")

    out = capsys.readouterr().out
    assert "CB" in out
    assert "tgt" in out
