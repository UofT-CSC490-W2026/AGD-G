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


def _clear_targeting_modules():
    for mod in list(sys.modules):
        if mod.startswith("agdg.targeting"):
            sys.modules.pop(mod, None)
