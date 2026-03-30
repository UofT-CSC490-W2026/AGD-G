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


class FakeCursor:
    def __init__(self, rows, updates):
        self.rows = rows
        self.updates = updates
        self.rowcount = 1

    def execute(self, query, params=None):
        if query.lstrip().startswith("UPDATE"):
            self.updates.append((query, params))

    def fetchall(self):
        return list(self.rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self, rows, updates):
        self.rows = rows
        self.updates = updates
        self.commits = 0

    def cursor(self):
        return FakeCursor(self.rows, self.updates)

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _load_module(monkeypatch, rows):
    import importlib
    _install_fake_boto(monkeypatch)
    updates = []

    class FakeConnectionFactory:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return FakeConnection(rows if self.calls == 1 else [], updates)

    factory = FakeConnectionFactory()

    sys.modules.pop("agdg.data_pipeline.aws", None)
    sys.modules.pop("agdg.data_pipeline.eval", None)
    module = importlib.import_module("agdg.data_pipeline.eval")

    monkeypatch.setattr(module, "get_db_connection", factory)
    monkeypatch.setattr(module, "get_image", lambda key: f"image:{key}".encode())

    return module, updates, factory


def test_evaluate_all_returns_zero_when_no_rows(monkeypatch):
    module, updates, _ = _load_module(monkeypatch, [])
    result = module.evaluate_all()
    assert result == {"evaluated": 0, "succeeded": 0, "failed": 0}
    assert updates == []


def test_evaluate_all_updates_rows_and_reports_counts(monkeypatch):
    rows = [(1, "adv-1", "What happened?", "Correct answer")]
    module, updates, _ = _load_module(monkeypatch, rows)

    result = module.evaluate_all()

    assert result["evaluated"] == 1
    assert result["succeeded"] == 0
    assert result["failed"] == 1
    assert result["asr_pct"] == 0.0
    assert updates[0][1] == ("Correct answer", False, 1)


def test_evaluate_all_counts_success_and_commits_batches(monkeypatch):
    class ToggleAnswer:
        def __init__(self):
            self.calls = 0

        def strip(self):
            return self

        def lower(self):
            self.calls += 1
            return "first" if self.calls == 1 else "second"

    rows = [(1, "adv-1", "What happened?", ToggleAnswer())]
    module, updates, factory = _load_module(monkeypatch, rows)
    module.BATCH_SIZE = 1

    result = module.evaluate_all()

    assert result["evaluated"] == 1
    assert result["succeeded"] == 1
    assert result["failed"] == 0
    assert result["asr_pct"] == 100.0
    assert factory.calls == 2
