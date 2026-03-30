"""
Modal script to run the test attack. Not set up for AWS integration or production at all.
"""
import modal
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agd.core.attack import attack
except (ImportError, ModuleNotFoundError):
    from core.attack import attack

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "transformers>=4.48", "pillow", "tqdm", "accelerate")
    .add_local_file("xray-fish-profile.png", "/root/clean_image.png", copy=True)
    .add_local_file("data_viz.png", "/root/data_viz.png", copy=True)
    .add_local_file("target.jpg", "/root/target.jpg", copy=True)
    .add_local_python_source("core", copy=False)
)

hf_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
image_volume = modal.Volume.from_name("agd_images", create_if_missing=True)

app = modal.App(
    "pgd",
    secrets=[modal.Secret.from_name("huggingface")],
    image=modal_image,
    volumes={
        "/root/.cache/huggingface": hf_volume,
        "/root/images": image_volume,
    },
    include_source=True,
)

@app.function(gpu="A10G:1", timeout=900)
def attack_test(
    epsilon: float = 0.0627,
    step_size: float = 0.0078,
    steps: int = 300,
    repel_clean: float = 0.0,
    attacker: str = "targeted_text",
    model: str = "llava15_text",
    clean_image_path: str = "/root/data_viz.png",
    target_question: str = "Which country was the leading pharmaceutical supplier to Germany in 2019?",
    target_response: str = "Ireland",
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
    attack(
        eps=epsilon,
        alpha=step_size,
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
    )
