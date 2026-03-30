import importlib
import subprocess
import sys

import pytest


def require_torch():
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("torch is not importable in this environment", allow_module_level=True)
    return importlib.import_module("torch")
