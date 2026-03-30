"""
Base class for models that generate a target caption
for a clean caption.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from PIL import Image

class TargetingModel(ABC):
    @abstractmethod
    def __init__(self, model_tag: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def __call__(self, images: List[Image.Image], clean_texts: List[str]) -> List[str]:
        """
        Convert image/clean_text pairs to target texts which differ from the clean texts
        """
        pass

