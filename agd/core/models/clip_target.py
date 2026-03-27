"""
The CLIP model embeds both images and text in the same latent space so they
can be compared. This also allows text-text and image-image comparisons,
the latter of which is used here.

There is only one base CLIP model used; the three implementations
are wrappers that fit the model into interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ..interfaces import (
    TextTargetModel,
    ImageTargetModel,
)

# ====================================================================================================
# Base CLIP model

class CLIPModel(ABC):
    def __init__(self, device="cuda", model_id="openai/clip-vit-large-patch14-336") -> None:
        super().__init__()
        self.device = device
        from transformers import CLIPModel, CLIPProcessor

        self.clip_model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_id)
        self.clip_model.requires_grad_(False)

        self.image_size = int(self.clip_model.config.vision_config.image_size)
        self.vision_model = self.clip_model.vision_model

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        px = F.interpolate(images, size=self.get_image_size(), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=px.device, dtype=px.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=px.device, dtype=px.dtype).view(1, 3, 1, 1)
        return (px - mean) / std

    def get_image_size(self) -> Tuple[int, int]:
        image_size = int(self.clip_model.config.vision_config.image_size)
        # CLIP images are square
        return (image_size, image_size)


# ====================================================================================================
# Child classes that implement interfaces

class TextCLIPModel(CLIPModel, TextTargetModel):
    """
    CLIP model that assesses similarity between an image and text. Text and images must be
    converted to values in embedding space by the `embed_image` and `embed_text` methods
    before being compared. These methods are separated from the comparison to allow
    the user to embed one constant side of the comparison just once instead of recomputing
    both embeddings on every comparison.
    """
    def __init__(self, device="cuda", model_id="openai/clip-vit-large-patch14-336") -> None:
        super().__init__(device, model_id)

    def __call__(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Given an image and target text embedding, return a similarity score.
        """
        return F.cosine_similarity(image, text).mean()

    def embed_image(self, image: Image.Image, detach: bool = False) -> Any:
        """
        Get projected embedding for image, which can be compared to text.
        """
        px = self._preprocess(image)
        out = self.vision_model(pixel_values=px, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states

        projected = self.clip_model.visual_projection(out.last_hidden_state[:, 0, :])
        features = projected / projected.norm(p=2, dim=-1, keepdim=True)
        return features.detach() if detach else features

    def embed_text(self, text: str, detach: bool = True) -> Any:
        """
        Get projected embedding for text, which can be compared to image.
        """
        tokenized = self.clip_processor.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model.text_model(**tokenized, return_dict=True)
            text_embed = self.clip_model.text_projection(out.pooler_output)
        text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
        return text_embed.detach() if detach else text_embed

class PatchTextCLIPModel(TextCLIPModel, TextTargetModel):
    """
    This is an extension to the TextCLIPModel. Instead of simply comparing the
    text with the summary vectors that CLIP generates for the image, it compares
    the text with every "patch" of the image. The similarity with the most
    similar patch is returned.
    """
    #override
    def __call__(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Given an image and target text embedding, return a similarity score.
        """
        sim_matrix = torch.matmul(image, text.transpose(-1, -2)).squeeze(-1)
        return torch.max(sim_matrix, dim=-1)[0].mean()

    #override
    def embed_image(self, image: Image.Image, detach: bool = False) -> Any:
        """
        Get projected embedding for image, which can be compared to text.
        """
        px = self._preprocess(image)
        out = self.vision_model(pixel_values=px, return_dict=True)
        patches = out.last_hidden_state[:, 1:, :]
        projected_patches = self.clip_model.visual_projection(patches)
        projected_patches = projected_patches / projected_patches.norm(p=2, dim=-1, keepdim=True)
        return projected_patches.detach() if detach else projected_patches

class ImageCLIPModel(CLIPModel, ImageTargetModel):
    """
    CLIP model that assesses similarity between two images. Images must be
    converted to values in embedding space by the `embed_image` method
    before being compared. This method is separated from the comparison to allow
    the user to embed one constant side of the comparison just once instead of recomputing
    both embeddings on every comparison.

    For the ImageCLIPModel (the image-image comparator) only, we don't just use the final
    projections to embedding space, but actually select the states from multiple hidden layers,
    which encode lower-level details about the images.
    """
    def __init__(self, device="cuda", model_id="openai/clip-vit-large-patch14-336") -> None:
        super().__init__(device, model_id)

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """
        Given two image embeddings (from embed_image), return a similarity score.
        """
        sim = 0.0
        for fet1, fet2 in zip(image1, image2):
            sim += F.cosine_similarity(fet1, fet2).mean()
        return sim / len(image1)

    def embed_image(self, image: torch.Tensor, detach: bool = False) -> List[torch.Tensor]:
        """
        Return list of hidden features with final projection at the end.
        """
        px = self._preprocess(image)
        out = self.vision_model(pixel_values=px, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states

        features = []
        for idx in [-1, -6, -12, -18, -24]:
            h = hidden_states[idx]
            patch_mean = h[:, 1:, :].mean(dim=1)
            patch_mean = patch_mean / patch_mean.norm(p=2, dim=-1, keepdim=True)
            features.append(patch_mean)

        projected = self.clip_model.visual_projection(out.last_hidden_state[:, 0, :])
        projected = projected / projected.norm(p=2, dim=-1, keepdim=True)
        if detach:
            projected = projected.detach()
        features.append(projected)
        return features
