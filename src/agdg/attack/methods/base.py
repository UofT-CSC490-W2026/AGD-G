"""
Base classes for attacks.

There are three interfaces for attack methods: untargeted, text-targeted,
and image-targeted, which correspond to the different types of targets which
may be provided. One attack method can probably implement multiple or all
of these interfaces.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Tuple, List, Optional, Dict, Any
import torch
from PIL import Image
from agdg.attack.surrogates.base import (
    ImageTargetModel,
    TextTargetModel,
)

class UntargetedAttackMethod:
    """
    Abstract base class for a method that guides an image's semantics away
    from its original state.
    """
    def __init__(self, model: ImageTargetModel) -> None:  # pragma: no cover
        self.model = model

    @abstractmethod
    def attack(
        self,
		clean: Image.Image | Sequence[Image.Image],
		strength: float,
		hyperparameters: Optional[Dict] = None
    ) -> Image.Image | Sequence[Image.Image]:
        """
        Run the attack on the clean image (or batch). Return the adversarial image (or images).
        Use the given attack strength, which will map to some hyperparameter inside the method.
        Other hyperparameters can be method-specific.
        """
        pass  # pragma: no cover


class TextAttackMethod:
    """
    Abstract base class for a method that guides an image's caption towards target text.
    """
    def __init__(self, model: TextTargetModel) -> None:  # pragma: no cover
        self.model = model

    @abstractmethod
    def attack(
        self,
		clean: Image.Image | Sequence[Image.Image],
		target: str | Sequence[str],
		strength: float,
		hyperparameters: Optional[Dict] = None
    ) -> Image.Image | Sequence[Image.Image]:
        """
        Run the attack on the clean image and target description (or batches of each).
        Return the adversarial image.
        Use the given attack strength, which will map to some hyperparameter inside the method.
        Other hyperparameters can be method-specific.
        """
        pass  # pragma: no cover


class ImageAttackMethod:
    """
    Abstract base class for a method that guides an image's caption towards a
    target image.
    """
    def __init__(self, model: ImageTargetModel) -> None:  # pragma: no cover
        self.model = model

    @abstractmethod
    def attack(
        self,
		clean: Image.Image | Sequence[Image.Image],
		target: Image.Image | Sequence[Image.Image],
		strength: float,
		hyperparameters: Optional[Dict] = None
    ) -> Image.Image | Sequence[Image.Image]:
        """
        Run the attack on the clean image and target image (or batches of each).
        Return the adversarial image.
        Use the given attack strength, which will map to some hyperparameter inside the method.
        Other hyperparameters can be method-specific
        """
        pass  # pragma: no cover
