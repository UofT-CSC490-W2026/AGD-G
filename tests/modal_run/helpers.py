import importlib
import os
import sys
import types
from pathlib import Path

MODAL_ROOT = Path("modal_run")


class _FakeImageBuilder:
    def pip_install(self, *args, **kwargs):
        return self

    def pip_install_from_pyproject(self, path, *args, **kwargs):
        assert os.path.isfile(path), f"pip_install_from_pyproject: file not found: {path}"
        return self

    def uv_pip_install(self, *args, **kwargs):
        return self

    def add_local_file(self, src, dest=None, **kwargs):
        assert os.path.exists(src), f"add_local_file: source not found: {src}"
        return self

    def add_local_dir(self, src, dest=None, **kwargs):
        assert os.path.isdir(src), f"add_local_dir: directory not found: {src}"
        return self

    def add_local_python_source(self, name, **kwargs):
        found = (
            os.path.isdir(name)
            or os.path.isfile(name + ".py")
            or os.path.isdir(os.path.join("src", name))
        )
        assert found, f"add_local_python_source: package/module not found: {name}"
        return self

    def env(self, *args, **kwargs):
        return self


class _FakeSecret:
    @staticmethod
    def from_name(name):
        return {"secret": name}


class _FakeVolume:
    @staticmethod
    def from_name(name, **kwargs):
        return {"volume": name}


class _FakeApp:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def function(self, *args, **kwargs):
        def decorator(func):
            func.remote = func
            return func

        return decorator

    def local_entrypoint(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator


def install_fake_modal():
    fake_modal = types.ModuleType("modal")
    fake_modal.Image = types.SimpleNamespace(
        debian_slim=lambda python_version=None: _FakeImageBuilder()
    )
    fake_modal.Secret = _FakeSecret
    fake_modal.Volume = _FakeVolume
    fake_modal.App = _FakeApp
    fake_modal.asgi_app = lambda *args, **kwargs: (lambda func: func)
    sys.modules["modal"] = fake_modal
    return fake_modal


def ensure_modal_root():
    modal_root = str(MODAL_ROOT)
    if modal_root not in sys.path:
        sys.path.insert(0, modal_root)


def import_fresh(module_name):
    importlib.invalidate_caches()
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)
