"""
Attack script. Not polished; just runs a test attack with AttackVLM right now.
Used by the Modal script.
"""

from .attacks.attackvlm import (
    AttackVLMUntargeted,
    AttackVLMImage,
    AttackVLMText,
    AttackVLMOCR,
)

from .models.clip_target import (
    ImageCLIPModel,
    TextCLIPModel,
    PatchTextCLIPModel
)

import os
import shutil
from PIL import Image

def attack(
    eps: float = 16 / 255,
    alpha: float = 2 / 255,
    steps: int = 300,
    repel_clean: float = 0.35,
    attacker: str = "targeted_text",
    model: str = "clip_text_patch",
    clean_image_path: str = "/root/data_viz.png",
    target_question: str = "What is the exact value for Category B?",
    target_response: str = "42",
    source_response: str = "",
    repel_source_text: float = 0.5,
    eot_samples: int = 1,
    eot_noise_std: float = 0.0,
    eot_brightness: float = 0.0,
    eot_shift_px: int = 0,
    ocr_edge_quantile: float = 0.88,
    ocr_std_quantile: float = 0.70,
    ocr_dilate_kernel: int = 7,
    ocr_max_area_ratio: float = 0.22,
    profile: str = "simple",
):
    models = {
        "clip_image": ImageCLIPModel,
        "clip_text": TextCLIPModel,
        "clip_text_patch": PatchTextCLIPModel,
    }
    model = models[model]()
    attackers = {
        "untargeted": AttackVLMUntargeted,
        "targeted_image": AttackVLMImage,
        "targeted_text": AttackVLMText,
        "targeted_text_ocr": AttackVLMOCR,
    }
    attacker = attackers[attacker](model)
    target_path = "/root/target.jpg" if isinstance(attacker, AttackVLMImage) else None
    target_text = None
    source_text = None

    if isinstance(attacker, AttackVLMText):
        target_text = f"Question: {target_question}\nAnswer: {target_response}" if target_question else target_response
        if source_response:
            source_text = f"Question: {target_question}\nAnswer: {source_response}" if target_question else source_response

    adv_img = attacker.attack(
        clean=Image.open(clean_image_path),
        target_image_path=target_path,
        target_text=target_text,
        source_text=source_text,
        eps=eps,
        alpha=alpha,
        steps=steps,
        repel_clean=repel_clean,
        repel_source_text=repel_source_text,
        eot_samples=eot_samples,
        eot_noise_std=eot_noise_std,
        eot_brightness=eot_brightness,
        eot_shift_px=eot_shift_px,
        ocr_edge_quantile=ocr_edge_quantile,
        ocr_std_quantile=ocr_std_quantile,
        ocr_dilate_kernel=ocr_dilate_kernel,
        ocr_max_area_ratio=ocr_max_area_ratio,
    )
    adv_img.save("/root/images/adversarial_output.png")

