"""
Base classes for the core of this project.

There are two interfaces for target models, which act as differentiable
judges to guide attack methods.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Tuple, List, Optional, Dict, Any
import torch
from PIL import Image

class ImageTargetModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Given two image embeddings (from embed_image), return a similarity score.
        Works with either a single image and text or a batch of each.
        """
        pass

    @abstractmethod
    def embed_image(self, image: torch.Tensor, detach: bool = False) -> Any:
        """
        Return features / embedding for image.
        Takes single image or a batch.
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
        Given image embeddings (from embed_image) and text embeddings
        (from embed_text), return similarity scores as a tensor.
        Works with either a single image and text or a batch of each.
        """
        pass

    @abstractmethod
    def embed_text(self, text: str | Sequence[str], detach: bool = True) -> Any:
        """
        Return features / embedding for text.
        Takes single text or a batch.
        Format is irrelevant as it should only be passed to __call__.
        """
        pass

    @abstractmethod
    def embed_image(self, image: torch.Tensor, detach: bool = False) -> Any:
        """
        Return features / embedding for image.
        Takes single image or a batch.
        Format is irrelevant as it should only be passed to __call__.
        """
        pass

    @abstractmethod
    def get_image_size(self) -> Tuple[int, int]:
        pass
