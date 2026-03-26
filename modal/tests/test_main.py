import argparse
import subprocess

import pytest

from conftest import ensure_modal_root, import_fresh


def load_module():
    ensure_modal_root()
    return import_fresh("data_pipeline.main")


def test_run_returns_true_on_success(monkeypatch):
    module = load_module()

    monkeypatch.setattr(subprocess, "run", lambda cmd: type("Result", (), {"returncode": 0})())

    assert module.run(["modal", "run", "clean.py"], "Clean") is True


def test_run_returns_false_on_failure(monkeypatch):
    module = load_module()

    monkeypatch.setattr(subprocess, "run", lambda cmd: type("Result", (), {"returncode": 1})())

    assert module.run(["modal", "run", "clean.py"], "Clean") is False


def test_main_runs_requested_steps(monkeypatch):
    module = load_module()
    args = argparse.Namespace(limit=5, clean=True, skip_import=False, skip_preprocess=False)
    popen_calls = []
    run_calls = []

    class FakeProcess:
        def __init__(self, cmd):
            self.cmd = cmd
            self.returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(module, "run", lambda cmd, label: run_calls.append((cmd, label)) or True)
    monkeypatch.setattr(module.subprocess, "Popen", lambda cmd: popen_calls.append(cmd) or FakeProcess(cmd))
    monkeypatch.setattr(module.sys, "exit", lambda code: (_ for _ in ()).throw(AssertionError(f"unexpected exit {code}")))

    module.main()

    assert run_calls[0][0] == ["modal", "run", "clean.py"]
    assert run_calls[1][0] == ["modal", "run", module.PREPROCESSOR]
    assert len(popen_calls) == len(module.IMPORTERS)
    assert popen_calls[0][-2:] == ["-l", "5"]


def test_main_exits_when_clean_fails(monkeypatch):
    module = load_module()
    args = argparse.Namespace(limit=None, clean=True, skip_import=False, skip_preprocess=False)

    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(module, "run", lambda cmd, label: False)

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1


def test_main_exits_when_any_step_fails(monkeypatch):
    module = load_module()
    args = argparse.Namespace(limit=None, clean=False, skip_import=False, skip_preprocess=True)

    class FakeProcess:
        def __init__(self, returncode):
            self.returncode = returncode

        def wait(self):
            return 0

    codes = iter([0, 1, 0])

    monkeypatch.setattr(module.argparse.ArgumentParser, "parse_args", lambda self: args)
    monkeypatch.setattr(module.subprocess, "Popen", lambda cmd: FakeProcess(next(codes)))

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
