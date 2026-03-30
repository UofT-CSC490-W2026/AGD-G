"""
Attack script. Not polished; just runs a test attack with AttackVLM right now.
Used by the Modal script.
"""

from agdg.attack.methods.attackvlm import (
    AttackVLMUntargeted,
    AttackVLMImage,
    AttackVLMText,
    AttackVLMOCR,
)

from agdg.attack.surrogates.clip import (
    ImageCLIPModel,
    TextCLIPModel,
    PatchTextCLIPModel
)
from .models.llava_target import LlavaTextTargetModel

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
    model_classes = {
        "clip_image": ImageCLIPModel,
        "clip_text": TextCLIPModel,
        "clip_text_patch": PatchTextCLIPModel,
        "llava15_text": LlavaTextTargetModel,
    }
    target_model = model_classes[model]()
    attack_classes = {
        "untargeted": AttackVLMUntargeted,
        "targeted_image": AttackVLMImage,
        "targeted_text": AttackVLMText,
        "targeted_text_ocr": AttackVLMOCR,
    }
    attack_method = attack_classes[attacker](target_model)

    clean_image = Image.open(clean_image_path)
    hyperparameters = {
        "alpha": alpha,
        "steps": steps,
        "max_eps": eps,
        "repel_clean": repel_clean,
        "repel_source_text": repel_source_text,
        "eot_samples": eot_samples,
        "eot_noise_std": eot_noise_std,
        "eot_brightness": eot_brightness,
        "eot_shift_px": eot_shift_px,
        "ocr_edge_quantile": ocr_edge_quantile,
        "ocr_std_quantile": ocr_std_quantile,
        "ocr_dilate_kernel": ocr_dilate_kernel,
        "ocr_max_area_ratio": ocr_max_area_ratio,
        "profile": profile,
    }

    if isinstance(attack_method, AttackVLMImage):
        target_image = Image.open("/root/target.jpg")
        adv_img = attack_method.attack(
            clean=clean_image,
            target=target_image,
            strength=1.0,
            hyperparameters=hyperparameters,
        )
    elif isinstance(attack_method, AttackVLMText):
        target_text = (
            f"Question: {target_question}\nAnswer: {target_response}"
            if target_question else target_response
        )
        if source_response:
            hyperparameters["source_text"] = (
                f"Question: {target_question}\nAnswer: {source_response}"
                if target_question else source_response
            )
        adv_img = attack_method.attack(
            clean=clean_image,
            target=target_text,
            strength=1.0,
            hyperparameters=hyperparameters,
        )
    else:
        adv_img = attack_method.attack(
            clean=clean_image,
            strength=1.0,
            hyperparameters=hyperparameters,
        )

    if isinstance(adv_img, Image.Image):
        adv_img.save("/root/images/adversarial_output.png")
    else:
        for i, image in enumerate(adv_img):
            image.save(f"/root/images/adversarial_output_{i+1}.png")
