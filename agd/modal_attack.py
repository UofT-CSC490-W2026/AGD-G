import modal
from core.attack import attack

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "transformers", "pillow", "tqdm", "accelerate")
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
    repel_clean: float = 0.35,
    mode: str = "targeted_text_dense",
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
    attack(
        eps=epsilon,
        alpha=step_size,
        steps=steps,
        repel_clean=repel_clean,
        mode=mode,
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
