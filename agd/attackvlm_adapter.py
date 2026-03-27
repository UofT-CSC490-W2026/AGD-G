from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from PIL import Image

from agd.core.attacks.attackvlm import AttackVLMText
from agd.core.models.clip_target import PatchTextCLIPModel, TextCLIPModel


@dataclass
class AttackResult:
    adversarial: list[Image.Image]
    success: list[bool]
    scores: list[float] | None = None


@dataclass
class AttackVLMTextAdapter:
    """
    Thin adapter over the refactored agd.core AttackVLM implementation.
    """

    device: str = "cuda"
    strength: float = 1.0
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    model_name: str = "clip_text_patch"

    def __post_init__(self) -> None:
        model_classes = {
            "clip_text": TextCLIPModel,
            "clip_text_patch": PatchTextCLIPModel,
        }
        model_class = model_classes[self.model_name]
        self.model = model_class(device=self.device)
        self.method = AttackVLMText(self.model, device=self.device)

    def attack(self, clean: Sequence[Image.Image], target: Sequence[str]) -> AttackResult:
        adversarial = []
        success = []
        for image, target_text in zip(clean, target):
            adv = self.method.attack(
                clean=image,
                target=target_text,
                strength=self.strength,
                hyperparameters=self.hyperparameters,
            )
            adversarial.append(adv)
            success.append(False)
        return AttackResult(adversarial=adversarial, success=success, scores=None)
