from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from PIL import Image


class TargetModel(Protocol):
    def objective(
        self, images: torch.Tensor, targets: Sequence[str | Image.Image]
    ) -> torch.Tensor:
        """
        Return a differentiable per-sample objective tensor with shape [B].
        Lower values mean the attack is doing better.
        """


@dataclass
class AttackResult:
    adversarial: list[Image.Image]
    success: list[bool]
    scores: list[float] | None = None


class TextAttackMethod(Protocol):
    def attack(self, clean: Sequence[Image.Image], target: Sequence[str]) -> AttackResult:
        """
        Run attack on clean images with text targets and return results.
        """


class ImageAttackMethod(Protocol):
    def attack(
        self, clean: Sequence[Image.Image], target: Sequence[Image.Image]
    ) -> AttackResult:
        """
        Run attack on clean images with image targets and return results.
        """
