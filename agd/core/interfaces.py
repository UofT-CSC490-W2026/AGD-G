from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
import torch
from PIL import Image

class TextTargetModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of images and target text embeddings,
        return similarity scores between each pair.
        """
        pass

    @abstractmethod
    def embed(self, texts: List[str]) -> torch.Tensor:
        """
        Pre-compute embeddings for target texts
        """
        pass

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        pass

class ImageTargetModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of images and target image embeddings,
        return similarity scores between each pair.
        """
        pass

    @abstractmethod
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """
        Pre-compute embeddings for target images
        """
        pass

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        pass

class TextAttackMethod:
    def __init__(self, model: TextTargetModel) -> None:
        self.model = model

    @abstractmethod
    def text_attack(
        self,
		clean: List[Image.Image],
		target: List[str],
		strength: float,
		hyperparameters: Optional[Dict] = None
    ) -> Tuple[List[Image.Image], List[bool]]:
        """
        Run the attack on the list of clean images and corresponding target descriptions.
        Return the list of adversarial images and whether each attack succeeded or not.
        Use the given attack strength, which will map to some hyperparameter inside the method.
        Other hyperparameters can be method-specific
        """
        pass

class ImageAttackMethod:
    def __init__(self, model: ImageTargetModel) -> None:
        self.model = model

    @abstractmethod
    def image_attack(
        self,
		clean: List[Image.Image],
		target: List[Image.Image],
		strength: float,
		hyperparameters: Optional[Dict] = None
    ) -> Tuple[List[Image.Image], List[bool]]:
        """
        Run the attack on the list of clean images and corresponding target images.
        Return the list of adversarial images and whether each attack succeeded or not.
        Use the given attack strength, which will map to some hyperparameter inside the method.
        Other hyperparameters can be method-specific
        """
        pass
