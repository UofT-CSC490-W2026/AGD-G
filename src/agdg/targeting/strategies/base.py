"""
Base class for models that generate a target caption
for a clean caption.
"""
from abc import ABC, abstractmethod
from typing import Optional, List, TypedDict
from PIL import Image


class RawResult(TypedDict):
    """Full model output for a single image/caption pair, including the thinking trace."""
    target: str
    thinking: str


class TargetingModel(ABC):
    @abstractmethod
    def __init__(self, model_tag: Optional[str] = None) -> None:
        pass  # pragma: no cover

    @abstractmethod
    def __call__(self, images: List[Image.Image], clean_texts: List[str]) -> List[str]:
        """
        Convert image/clean_text pairs to target texts which differ from the clean texts.
        Returns parsed target captions only (no thinking trace).
        """
        pass  # pragma: no cover

    def generate_raw(self, images: List[Image.Image], clean_texts: List[str]) -> List[RawResult]:
        """
        Like ``__call__`` but returns the full model output including the thinking trace.

        The default implementation delegates to ``__call__`` and returns an empty thinking
        string. Subclasses that support chain-of-thought output should override this.
        """
        return [{"target": t, "thinking": ""} for t in self(images, clean_texts)]
