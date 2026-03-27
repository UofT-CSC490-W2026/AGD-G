import importlib
import sys
import types
from pathlib import Path


MODAL_ROOT = Path(__file__).resolve().parents[1]


class _FakeImageBuilder:
    def pip_install(self, *args, **kwargs):
        return self

    def uv_pip_install(self, *args, **kwargs):
        return self

    def add_local_file(self, *args, **kwargs):
        return self


class _FakeSecret:
    @staticmethod
    def from_name(name):
        return {"secret": name}


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
    fake_modal.App = _FakeApp
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
