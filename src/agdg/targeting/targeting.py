"""
Reusable AttackVLM entry point for single-image attacks.
"""

from __future__ import annotations

from PIL import Image

from .strategies.base import TargetingModel

QwenTargetingModel = None


def _get_device(device: str | None = None) -> str:
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_dependencies_loaded() -> None:
    global QwenTargetingModel

    if QwenTargetingModel is None:
        from agdg.targeting.strategies.qwen import (
            QwenTargetingModel as _QwenTargetingModel,
        )

        QwenTargetingModel = _QwenTargetingModel


def build_targeting_strategy(model: str, device: str | None = None) -> TargetingModel:
    _ensure_dependencies_loaded()
    resolved_device = _get_device(device)
    strategies = {
        "qwen": QwenTargetingModel,
    }
    try:
        strategy = strategies[model]
    except KeyError as exc:
        raise ValueError(f"Unknown targeting strategy: {model}") from exc
    try:
        return strategy(device=resolved_device)
    except TypeError:
        return strategy()
