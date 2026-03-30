"""
Reusable AttackVLM entry point for single-image attacks.
"""

from __future__ import annotations

from PIL import Image

AttackVLMImage = None
AttackVLMOCR = None
AttackVLMText = None
AttackVLMUntargeted = None
ImageCLIPModel = None
PatchTextCLIPModel = None
TextCLIPModel = None
LlavaTextTargetModel = None


def _get_device(device: str | None = None) -> str:
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_dependencies_loaded() -> None:
    global AttackVLMImage
    global AttackVLMOCR
    global AttackVLMText
    global AttackVLMUntargeted
    global ImageCLIPModel
    global PatchTextCLIPModel
    global TextCLIPModel
    global LlavaTextTargetModel

    if AttackVLMImage is None:
        from agdg.attack.methods.attackvlm import (
            AttackVLMImage as _AttackVLMImage,
            AttackVLMOCR as _AttackVLMOCR,
            AttackVLMText as _AttackVLMText,
            AttackVLMUntargeted as _AttackVLMUntargeted,
        )
        from agdg.attack.surrogates.clip import (
            ImageCLIPModel as _ImageCLIPModel,
            PatchTextCLIPModel as _PatchTextCLIPModel,
            TextCLIPModel as _TextCLIPModel,
        )
        from .surrogates.llava_target import LlavaTextTargetModel as _LlavaTextTargetModel

        AttackVLMImage = _AttackVLMImage
        AttackVLMOCR = _AttackVLMOCR
        AttackVLMText = _AttackVLMText
        AttackVLMUntargeted = _AttackVLMUntargeted
        ImageCLIPModel = _ImageCLIPModel
        PatchTextCLIPModel = _PatchTextCLIPModel
        TextCLIPModel = _TextCLIPModel
        LlavaTextTargetModel = _LlavaTextTargetModel


def build_target_model(model: str, device: str | None = None):
    _ensure_dependencies_loaded()
    resolved_device = _get_device(device)
    model_classes = {
        "clip_image": ImageCLIPModel,
        "clip_text": TextCLIPModel,
        "clip_text_patch": PatchTextCLIPModel,
        "llava15_text": LlavaTextTargetModel,
        "llava": LlavaTextTargetModel,
    }
    try:
        model_class = model_classes[model]
    except KeyError as exc:
        raise ValueError(f"Unknown surrogate model: {model}") from exc
    try:
        return model_class(device=resolved_device)
    except TypeError:
        return model_class()


def build_attack_method(attacker: str, model: str, device: str | None = None):
    _ensure_dependencies_loaded()
    attack_classes = {
        "untargeted": AttackVLMUntargeted,
        "targeted_image": AttackVLMImage,
        "targeted_text": AttackVLMText,
        "targeted_text_ocr": AttackVLMOCR,
    }
    try:
        attack_class = attack_classes[attacker]
    except KeyError as exc:
        raise ValueError(f"Unknown attacker: {attacker}") from exc
    target_model = build_target_model(model, device=device)
    try:
        return attack_class(target_model, device=_get_device(device))
    except TypeError:
        return attack_class(target_model)


def _format_question_answer(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}" if question else answer


def generate_adversarial_image(
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
    device: str | None = None,
):
    attack_method = build_attack_method(attacker=attacker, model=model, device=device)
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
        return attack_method.attack(
            clean=clean_image,
            target=target_image,
            strength=1.0,
            hyperparameters=hyperparameters,
        )

    if isinstance(attack_method, AttackVLMText):
        if source_response:
            hyperparameters["source_text"] = _format_question_answer(target_question, source_response)
        return attack_method.attack(
            clean=clean_image,
            target=_format_question_answer(target_question, target_response),
            strength=1.0,
            hyperparameters=hyperparameters,
        )

    return attack_method.attack(
        clean=clean_image,
        strength=1.0,
        hyperparameters=hyperparameters,
    )


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
    device: str | None = None,
):
    adv_img = generate_adversarial_image(
        eps=eps,
        alpha=alpha,
        steps=steps,
        repel_clean=repel_clean,
        attacker=attacker,
        model=model,
        clean_image_path=clean_image_path,
        target_question=target_question,
        target_response=target_response,
        source_response=source_response,
        repel_source_text=repel_source_text,
        eot_samples=eot_samples,
        eot_noise_std=eot_noise_std,
        eot_brightness=eot_brightness,
        eot_shift_px=eot_shift_px,
        ocr_edge_quantile=ocr_edge_quantile,
        ocr_std_quantile=ocr_std_quantile,
        ocr_dilate_kernel=ocr_dilate_kernel,
        ocr_max_area_ratio=ocr_max_area_ratio,
        profile=profile,
        device=device,
    )

    if isinstance(adv_img, Image.Image):
        adv_img.save("/root/images/adversarial_output.png")
        return

    for i, image in enumerate(adv_img):
        image.save(f"/root/images/adversarial_output_{i + 1}.png")
