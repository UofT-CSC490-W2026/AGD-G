"""
Interfaces for the core of this project.
Implemented as abstract base classes with no implemented methods, since
Python doesn't support real interfaces.

There are two interfaces for target models, which act as differentiable
judges to guide attack methods.

There are three interfaces for attack methods: untargeted, text-targeted,
and image-targeted, which correspond to the different types of targets which
may be provided. One attack method can probably implement multiple or all
of these interfaces.

Any model can ideally be matched with any method. The only limitation is
that text-targeted methods require a judge model that understands both text
and images, and that image-targeted and un-targeted methods require a model
that can compare two images.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict, Any
import torch
from PIL import Image

# ====================================================================================================
# Judge/target model interfaces

class ImageTargetModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Given two image embeddings (from embed_image), return a similarity score.
        """
        pass

    @abstractmethod
    def embed_image(self, image: torch.Tensor, detach: bool = False) -> Any:
        """
        Return features / embedding for image.
        Format is irrelevant as it should only be passed to __call__.
        """
        pass

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        pass


class TextTargetModel(ImageTargetModel, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Given an image embedding (from embed_image) and text embedding
        (from embed_text), return similarity score.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str, detach: bool = True) -> Any:
        """
        Return features / embedding for text.
        Format is irrelevant as it should only be passed to __call__.
        """
        pass

    @abstractmethod
    def embed_image(self, image: torch.Tensor, detach: bool = False) -> Any:
        """
        Return features / embedding for image.
        Format is irrelevant as it should only be passed to __call__.
        """
        pass

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        pass


# ====================================================================================================
# Attack method interfaces

class UntargetedAttackMethod:
    """
    Abstract base class for a method that guides an image's semantics away
    from its original state.
    """
    def __init__(self, model: ImageTargetModel) -> None:
        self.model = model

    @abstractmethod
    def attack(
        self,
		clean: Image.Image,
		strength: float,
		hyperparameters: Optional[Dict] = None
    ) -> Image.Image:
        """
        Run the attack on the clean image. Return the adversarial image.
        Use the given attack strength, which will map to some hyperparameter inside the method.
        Other hyperparameters can be method-specific.
        """
        pass


class TextAttackMethod:
    """
    Abstract base class for a method that guides an image's caption towards target text.
    """
    def __init__(self, model: TextTargetModel) -> None:
        self.model = model

    @abstractmethod
    def attack(
        self,
		clean: Image.Image,
		target: str,
		strength: float,
		hyperparameters: Optional[Dict] = None
    ) -> Image.Image:
        """
        Run the attack on the clean image and target description. Return the adversarial image.
        Use the given attack strength, which will map to some hyperparameter inside the method.
        Other hyperparameters can be method-specific.
        """
        pass


class ImageAttackMethod:
    """
    Abstract base class for a method that guides an image's caption towards a
    target image.
    """
    def __init__(self, model: ImageTargetModel) -> None:
        self.model = model

    @abstractmethod
    def attack(
        self,
		clean: Image.Image,
		target: Image.Image,
		strength: float,
		hyperparameters: Optional[Dict] = None
    ) -> Image.Image:
        """
        Run the attack on the clean image and target image. Return the adversarial image.
        Use the given attack strength, which will map to some hyperparameter inside the method.
        Other hyperparameters can be method-specific
        """
        pass
