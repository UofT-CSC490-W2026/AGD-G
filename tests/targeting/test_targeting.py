"""Tests for the build_targeting_strategy factory."""
import sys
import types
from unittest.mock import MagicMock

import pytest


def test_unknown_strategy_raises(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_transformers(monkeypatch)
    _clear_targeting_modules()

    from agdg.targeting.targeting import build_targeting_strategy

    with pytest.raises(ValueError, match="Unknown targeting strategy"):
        build_targeting_strategy("nonexistent")


def test_qwen_strategy_returns_targeting_model(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_transformers(monkeypatch)
    _clear_targeting_modules()

    from agdg.targeting.targeting import build_targeting_strategy
    from agdg.targeting.strategies.base import TargetingModel

    model = build_targeting_strategy("qwen", device="cpu")
    assert isinstance(model, TargetingModel)


def _install_fake_torch(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _install_fake_transformers(monkeypatch):
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = None
    fake_model.requires_grad_.return_value = None

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = MagicMock()
    fake_transformers.AutoProcessor.from_pretrained.return_value = MagicMock()
    fake_transformers.AutoModelForImageTextToText = MagicMock()
    fake_transformers.AutoModelForImageTextToText.from_pretrained.return_value = fake_model
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_get_device_cuda(monkeypatch):
    _install_fake_torch(monkeypatch)
    _clear_targeting_modules()
    fake_torch = sys.modules["torch"]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)

    from agdg.targeting.targeting import _get_device
    assert _get_device() == "cuda"


def test_get_device_mps(monkeypatch):
    _install_fake_torch(monkeypatch)
    _clear_targeting_modules()
    fake_torch = sys.modules["torch"]
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: True)
    )

    from agdg.targeting.targeting import _get_device
    assert _get_device() == "mps"


def test_strategy_type_error_fallback(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_transformers(monkeypatch)
    _clear_targeting_modules()

    from agdg.targeting import targeting as targeting_module

    class NoDeviceStrategy:
        def __init__(self):
            self.built = True
        def __call__(self, images, clean_texts):
            return []

    monkeypatch.setattr(targeting_module, "QwenTargetingModel", NoDeviceStrategy)
    result = targeting_module.build_targeting_strategy("qwen", device="cpu")
    assert result.built is True


def _clear_targeting_modules():
    for mod in list(sys.modules):
        if mod.startswith("agdg.targeting"):
            sys.modules.pop(mod, None)
