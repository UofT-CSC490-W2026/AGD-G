import modal
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import os
from io import BytesIO

# ---------------- Modal setup ----------------

hf_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App(
    "target-gen",
    secrets=[modal.Secret.from_name("huggingface")],
    image=modal.Image.debian_slim().pip_install(
        "torch",
        "transformers",
        "accelerate",
        "sentence-transformers",
        "pillow",
        "einops",
    ),
    volumes={
        "/root/.cache/huggingface": hf_volume,
    },
    include_source=True,
)

# ---------------- Models ----------------

LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
LLAMA_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


# ---------------- Helper functions ----------------

def load_image_from_bytes(image_bytes):
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def caption_image(image, processor, model, device):
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    caption = processor.decode(output[0], skip_special_tokens=True)

    # Extract assistant response only
    if "ASSISTANT:" in caption:
        caption = caption.split("ASSISTANT:")[-1].strip()

    return caption


def generate_targets(clean_caption, tokenizer, model, device, temperature):
    """
    Generate 64 candidates via 8 prompts, each generating 8 outputs.
    """

    system_prompt = (
        "You rewrite image captions by replacing key nouns with closely related alternatives "
        "while preserving overall structure and topic.\n\n"
        "Here are examples:\n"
        "Input: A red car racing on a track\n"
        "Output: A red motorcycle racing on a track\n"
        "\n"
        "Input: People playing basketball in a gym\n"
        "Output: Athletes playing volleyball in an indoor court\n"
        "\n"
        "Now rewrite the following caption into 8 diverse variants.\n"
        "Keep them semantically similar but meaningfully different.\n"
        "Return exactly 8 lines.\n\n"
        f"Caption: {clean_caption}\n\n"
        "Variants:\n"
    )

    inputs = tokenizer(system_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract lines after "Variants:"
    variants = []
    for line in text.split("\n"):
        line = line.strip("- ").strip()
        if line and line.lower() != "variants:" and len(line) > 10:
            variants.append(line)

    # Deduplicate and trim to 64
    variants = list(dict.fromkeys(variants))
    return variants[:64]


def embed_texts(texts, embed_model, device):
    embeddings = embed_model.encode(texts, convert_to_tensor=True, device=device)
    return embeddings


def compute_similarities(clean_emb, target_embs):
    sims = F.cosine_similarity(clean_emb.unsqueeze(0), target_embs)
    return sims


# ---------------- Modal function ----------------

@app.function(
    gpu="A100",
    timeout=60 * 60,
)
def run_analysis(
    image_bytes: bytes,
    temperature: float = 0.8,
    pairwise: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = load_image_from_bytes(image_bytes)

    # -------- Load LLaVA --------
    llava_processor = AutoProcessor.from_pretrained(LLAVA_MODEL_ID)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # -------- Load LLaMA --------
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    llama_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # -------- Load embedding model --------
    embed_model = SentenceTransformer(EMBED_MODEL_ID, device=device)

    # -------- Caption image --------
    clean_caption = caption_image(image, llava_processor, llava_model, device)
    print("\nCLEAN CAPTION:\n", clean_caption)

    # -------- Generate targets --------
    all_targets = []
    for _ in range(8):
        batch = generate_targets(
            clean_caption,
            llama_tokenizer,
            llama_model,
            device,
            temperature,
        )
        all_targets.extend(batch)
    target_captions = all_targets[:64]

    print(f"\nGENERATED {len(target_captions)} TARGETS\n")

    for i, t in enumerate(target_captions):
        print(f"{i+1}: {t}")

    # -------- Embeddings --------
    all_texts = [clean_caption] + target_captions
    embeddings = embed_texts(all_texts, embed_model, device)

    clean_emb = embeddings[0]
    target_embs = embeddings[1:]

    sims = compute_similarities(clean_emb, target_embs)

    sims_list = sims.cpu().tolist()

    # -------- Analysis --------
    min_idx = sims.argmin().item()
    max_idx = sims.argmax().item()

    print("\nSIMILARITY ANALYSIS:")
    print(f"Min similarity: {sims[min_idx].item():.4f}")
    print(f"Max similarity: {sims[max_idx].item():.4f}")
    print(f"Mean similarity: {sims.mean().item():.4f}")

    print("\nMost distant target:")
    print(target_captions[min_idx])

    print("\nClosest target:")
    print(target_captions[max_idx])

    if pairwise:
        print("\nPAIRWISE TARGET SIMILARITIES:")
        pair_embs = target_embs
        pair_sims = F.cosine_similarity(
            pair_embs.unsqueeze(1), pair_embs.unsqueeze(0), dim=-1
        )
        print(pair_sims.cpu())


# ---------------- CLI entry ----------------

@app.local_entrypoint()
def main(
    image_path: str = "target.png",
    temperature: float = 0.8,
    pairwise: bool = False,
):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    run_analysis.remote(
        image_bytes=image_bytes,
        temperature=temperature,
        pairwise=pairwise,
    )
