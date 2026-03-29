"""
Run the Pytorch profiler on Modal.

Afterwards, you can pull the profile with:

$ modal volume get profiling <profile_name>_<date_time>.json

ensure the whole file was pulled, because Modal sometimes silently fails to complete the transfer:

$ tail <profile_name>_<date_time>.json

There may be some corrupted function names which will cause Perfetto (https://ui.perfetto.dev)
to fail while loading the file. Fix these by tracking them down and replacing with the
stream editor `sed` (not a text editor or Vim, as memory constraints and bugs will corrupt the
file further):

$ sed -e 's/.#O9...0./corrupted/' -e 's/0"#@+/corrupted/' < cat attackvlm_2026-03-28_04:11.json > trace.json
"""
import sys
import os
import datetime
import pickle
from typing import Callable, List
from PIL import Image
from torch.profiler import profile, ProfilerActivity
import torch
import numpy as np
import modal
from sentence_transformers import SentenceTransformer

file_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(file_path) + '/..')
from core.interfaces import TextTargetModel
from core.attacks.attackvlm import AttackVLMText
from core.models.clip_target import (
    ImageCLIPModel,
    TextCLIPModel,
    PatchTextCLIPModel,
)
from scoring.eval import evaluate_similarity

# GPU on which to run profiles
GPU="A10G:1"

# ================================================================================
# Modal setup

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "transformers", "pillow", "tqdm", "accelerate", "sentence_transformers")
    .add_local_dir("../core", "/root/core", copy=False)
    .add_local_dir("../scoring", "/root/scoring", copy=False)
    .add_local_dir("test_images", "/root/test_images", copy=False)
)

hf_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
profiles_volume = modal.Volume.from_name("profiling", create_if_missing=True)

app = modal.App(
    "profiling",
    secrets=[modal.Secret.from_name("huggingface")],
    image=modal_image,
    volumes={
        "/root/.cache/huggingface": hf_volume,
        "/root/profiles": profiles_volume,
    },
    include_source=True,
)

# ================================================================================
# Main profiling function

def profile_with_pytorch(function: Callable, *args, warmup: int = 10, **kwargs):
    # Run function without profiling a few times to warm up CUDA
    print(f"[{function.__name__}] Warming up CUDA for {warmup} iterations")
    for _ in range(warmup):
        function(*args, **kwargs)
    # Run function with profiling
    print(f"[{function.__name__}] Starting profiler")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, with_stack=True) as prof:
        function(*args, **kwargs)
    print(f"[{function.__name__}] Finished profiler")
    print(f"[{function.__name__}] Exporting profile")
    averages = prof.key_averages()
    time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    save_file_name = f"profiles/{function.__name__}_{time}"
    with open(save_file_name + ".pkl", "wb") as f:
        pickle.dump(averages, f)
    print(f"[{function.__name__}] Saved raw profile data to {save_file_name}.pkl")
    prof.export_chrome_trace(save_file_name + ".json")
    print(f"[{function.__name__}] Saved Perfetto-compatible profile to {save_file_name}.json")

# ================================================================================
# Profiling targets

def run_attackvlm_text(model: TextTargetModel, images: List[Image.Image], targets: List[str], batch: bool = True):
    attackvlm = AttackVLMText(model)
    if batch:
        attackvlm.attack(clean=images, target=targets, strength=1.0)
    else:
        for image, target in zip(images, targets):
            attackvlm.attack(clean=image, target=target, strength=1.0)

def run_clip_text(image: Image.Image):
    model = TextCLIPModel()
    embed_image = model.embed_image(image)
    embed_text = model.embed_text("a baseball")
    model(embed_image, embed_text)

def run_clip_text_patch(image: Image.Image):
    model = PatchTextCLIPModel()
    embed_image = model.embed_image(image)
    embed_text = model.embed_text("a baseball")
    model(embed_image, embed_text)

def run_clip_image(image1: Image.Image, image2: Image.Image):
    model = ImageCLIPModel()
    embed1 = model.embed_image(image1)
    embed2 = model.embed_image(image2)
    model(embed1, embed2)

def run_eval_similarity(model: SentenceTransformer, output_text: str, text_a: str, text_b: str):
    # Run repeatedly so we can differentiate first-time overhead from repeated computation
    for _ in range(10):
        evaluate_similarity(model, output_text, text_a, text_b)

# ================================================================================
# Helpers

def load_image_tensor(path: str) -> torch.Tensor:
    """
    Load an image as a Tensor after converting to 512x512 float [0-1] format
    """
    image = Image.open(path)
    image = image.convert("RGB").resize((512, 512))
    clean_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0).to("cuda")

# ================================================================================
# Modal functions

@app.function(gpu=GPU, timeout=1200)
def profile_attackvlm(warmup: int, num_images: int = 10, batch: bool = True):
    image = Image.open("test_images/test-image1.png")
    images = [image] * num_images
    targets = ["a baseball"] * num_images
    model = PatchTextCLIPModel()
    profile_with_pytorch(run_attackvlm_text, model, images, targets, warmup=warmup, batch=batch)

@app.function(gpu=GPU, timeout=1200)
def profile_clip_text(warmup: int):
    image = load_image_tensor("test_images/test-image1.png")
    profile_with_pytorch(run_clip_text, image, warmup=warmup)

@app.function(gpu=GPU, timeout=1200)
def profile_clip_text_patch(warmup: int):
    image = load_image_tensor("test_images/test-image1.png")
    profile_with_pytorch(run_clip_text_patch, image, warmup=warmup)

@app.function(gpu=GPU, timeout=1200)
def profile_clip_image(warmup: int):
    image1 = load_image_tensor("test_images/test-image1.png")
    image2 = load_image_tensor("test_images/test-image2.png")
    profile_with_pytorch(run_clip_image, image1, image2, warmup=warmup)

@app.function(gpu=GPU, timeout=1200)
def profile_eval_similarity(warmup: int):
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v5-text-small-text-matching",
        device="cuda",
        trust_remote_code=True
    )
    profile_with_pytorch(
        run_eval_similarity,
        model,
		output_text="Pigs cannot fly",
		text_a="Swine are never airborne",
		text_b="Horses have hooves",
        warmup=warmup,
    )

@app.local_entrypoint()
def main(
    attackvlm:       bool = False,
    clip_text:       bool = False,
    clip_text_patch: bool = False,
    clip_image:      bool = False,
    eval_similarity: bool = False,
    warmup: int = 10,
) -> None:
    """
    Run the selected profilers. If none are selected, run all.
    Profilers will be run in parallel as different Modal functions.
    """
    # Choose profiling functions from command-line arguments
    functions = []
    profile_all = not any([attackvlm, clip_text, clip_text_patch, clip_image, eval_similarity])
    if profile_all or attackvlm:       functions.append(profile_attackvlm)
    if profile_all or clip_text:       functions.append(profile_clip_text)
    if profile_all or clip_text_patch: functions.append(profile_clip_text_patch)
    if profile_all or clip_image:      functions.append(profile_clip_image)
    if profile_all or eval_similarity: functions.append(profile_eval_similarity)

    # Spawn functions in parallel
    futures = [function.spawn(warmup=warmup) for function in functions]

    # Wait for functions to exit
    for f in futures:
        f.get()
