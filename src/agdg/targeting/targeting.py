"""
Factory for targeting strategies.

A *targeting strategy* takes a clean chart image and its caption, then
produces a new target caption that preserves the chart type but changes
the subject (e.g., "Bar chart of European exports" -> "Bar chart of
European vacation spots").

Usage::

    from agdg.targeting.targeting import build_targeting_strategy

    targeter = build_targeting_strategy("qwen")          # auto-detect device
    targeter = build_targeting_strategy("qwen", "cuda")  # explicit device

    targets = targeter([image], ["A pie chart of GDP"])
    # -> ["Pie chart of Italian pizza toppings"]

To add a new strategy, create a subclass of
:class:`~agdg.targeting.strategies.base.TargetingModel` in the
``strategies/`` package and register it in the ``strategies`` dict
inside :func:`build_targeting_strategy`.
"""

from __future__ import annotations

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
