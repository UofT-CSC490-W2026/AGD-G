import sys
import types

from tests.helpers import ensure_modal_root, import_fresh, install_fake_modal


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


def load_module(rows):
    ensure_modal_root()
    install_fake_modal()
    updates = []

    class FakeConnectionFactory:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return FakeConnection(rows if self.calls == 1 else [], updates)

    factory = FakeConnectionFactory()
    fake_aws = types.ModuleType("aws")
    fake_aws.get_db_connection = factory
    fake_aws.get_image = lambda key: f"image:{key}".encode()
    sys.modules["aws"] = fake_aws
    module = import_fresh("data_pipeline.eval")
    return module, updates, factory


def test_evaluate_all_returns_zero_when_no_rows():
    module, updates, _ = load_module([])

    result = module.evaluate_all()

    assert result == {"evaluated": 0, "succeeded": 0, "failed": 0}
    assert updates == []


def test_evaluate_all_updates_rows_and_reports_counts():
    rows = [(1, "adv-1", "What happened?", "Correct answer")]
    module, updates, _ = load_module(rows)

    result = module.evaluate_all()

    assert result["evaluated"] == 1
    assert result["succeeded"] == 0
    assert result["failed"] == 1
    assert result["asr_pct"] == 0.0
    assert updates[0][1] == ("Correct answer", False, 1)


def test_evaluate_all_counts_success_and_commits_batches():
    class ToggleAnswer:
        def __init__(self):
            self.calls = 0

        def strip(self):
            return self

        def lower(self):
            self.calls += 1
            return "first" if self.calls == 1 else "second"

    rows = [(1, "adv-1", "What happened?", ToggleAnswer())]
    module, updates, factory = load_module(rows)
    module.BATCH_SIZE = 1

    result = module.evaluate_all()

    assert result["evaluated"] == 1
    assert result["succeeded"] == 1
    assert result["failed"] == 0
    assert result["asr_pct"] == 100.0
    assert factory.calls == 2


def test_eval_main_prints_remote_result(capsys):
    module, _, _ = load_module([])
    module.evaluate_all.remote = lambda max_rows: {"evaluated": max_rows, "failed": 0}

    module.main(l=3)

    out = capsys.readouterr().out
    assert "evaluated" in out
    assert "3" in out
