"""
Modal script for Stage 2: RGF refinement of adversarial images.

Loads LLaVA-LLaMA-3-8B and a CLIP text model, then refines a Stage 1
adversarial image by querying LLaVA with Rademacher noise perturbations
to estimate pseudo-gradients toward the target text.

Reads the Stage 1 image from the agd_images volume and writes the
refined image back to the same volume.

Usage:
    # First generate Stage 1 adversarial image
    modal run modal_attack.py

    # Then refine with RGF
    modal run modal_rgf.py --question "..." --target-response "..."

    # Then evaluate
    modal run eval_vlm.py --question "..." --target-response "..."
"""
import modal

try:
    from agd.core.attacks.attackvlm import QueryRefinement
    from agd.core.models.clip_target import TextCLIPModel
except (ImportError, ModuleNotFoundError):
    from core.attacks.attackvlm import QueryRefinement
    from core.models.clip_target import TextCLIPModel

LLAVA_MODEL_ID = "llava-hf/llama3-llava-next-8b-hf"

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "transformers==4.44.2", "pillow", "tqdm", "accelerate",
        "sentencepiece", "protobuf",
    )
    .add_local_file("data_viz.png", "/root/data_viz.png", copy=True)
    .add_local_python_source("core", copy=False)
)

hf_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
image_volume = modal.Volume.from_name("agd_images", create_if_missing=True)

app = modal.App(
    "rgf-refine",
    secrets=[modal.Secret.from_name("huggingface")],
    image=modal_image,
    volumes={
        "/root/.cache/huggingface": hf_volume,
        "/root/images": image_volume,
    },
    include_source=True,
)


@app.cls(gpu="A10G:1", timeout=3600)
class RGFRefine:
    """Loads LLaVA + CLIP, then runs RGF refinement on a Stage 1 adversarial image."""

    @modal.enter()
    def load_models(self):
        import torch
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
        )

        # LLaVA for black-box queries
        self.processor = LlavaNextProcessor.from_pretrained(LLAVA_MODEL_ID)
        self.llava = LlavaNextForConditionalGeneration.from_pretrained(
            LLAVA_MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.llava.to("cuda")
        self.llava_input_device = torch.device("cuda:0")

        # CLIP for text embedding in RGF
        self.clip_model = TextCLIPModel(device="cuda")

    def vlm_infer(self, image, question):
        """Query LLaVA with an image and question."""
        import torch

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.llava_input_device)

        with torch.no_grad():
            output = self.llava.generate(
                **inputs,
                max_new_tokens=64,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        parts = decoded.split("assistant")
        return parts[-1].strip() if len(parts) > 1 else decoded.strip()

    @modal.method()
    def refine(
        self,
        question: str,
        target_response: str,
        source_response: str = "",
        clean_image_path: str = "/root/data_viz.png",
        adv_image_path: str = "/root/images/adversarial_output.png",
        output_path: str = "/root/images/adversarial_output_rgf.png",
        eps: float = 8 / 255,
        sigma: float = 4 / 255,
        alpha: float = 0.5 / 255,
        steps: int = 8,
        num_query: int = 50,
        num_sub_query: int = 10,
        repel_source_text: float = 0.5,
        image_l2_penalty: float = 0.05,
        smooth_kernel: int = 5,
        ocr_edge_quantile: float = 0.88,
        ocr_std_quantile: float = 0.70,
        ocr_dilate_kernel: int = 7,
        ocr_max_area_ratio: float = 0.22,
        score_on_answer_only: bool = True,
    ):
        """Run RGF refinement on a Stage 1 adversarial image."""
        from PIL import Image

        clean_img = Image.open(clean_image_path)
        adv_img = Image.open(adv_image_path)

        if score_on_answer_only:
            target_text = target_response
            source_text = source_response or None
        else:
            target_text = (
                f"Question: {question}\nAnswer: {target_response}"
                if question
                else target_response
            )
            source_text = (
                f"Question: {question}\nAnswer: {source_response}"
                if source_response and question
                else source_response
            ) or None

        refiner = QueryRefinement(self.clip_model, device="cuda")
        refined_img = refiner.refine(
            clean=clean_img,
            adv=adv_img,
            target_text=target_text,
            vlm_fn=self.vlm_infer,
            question=question,
            source_text=source_text,
            target_answer=target_response,
            source_answer=source_response or None,
            eps=eps,
            sigma=sigma,
            alpha=alpha,
            steps=steps,
            num_query=num_query,
            num_sub_query=num_sub_query,
            repel_source_text=repel_source_text,
            image_l2_penalty=image_l2_penalty,
            smooth_kernel=smooth_kernel,
            ocr_edge_quantile=ocr_edge_quantile,
            ocr_std_quantile=ocr_std_quantile,
            ocr_dilate_kernel=ocr_dilate_kernel,
            ocr_max_area_ratio=ocr_max_area_ratio,
        )
        refined_img.save(output_path)
        print(f"Refined image saved to {output_path}")

        # Quick sanity check: query LLaVA with the refined image
        response = self.vlm_infer(refined_img, question)
        print(f"Question: {question}")
        print(f"Target: {target_response}")
        print(f"VLM response: {response}")

        return {"target": target_response, "vlm_response": response}


@app.local_entrypoint()
def main(
    question: str = "Which country was the leading pharmaceutical supplier to Germany in 2019?",
    target_response: str = "Ireland",
    source_response: str = "",
    clean_image_path: str = "/root/data_viz.png",
    adv_image_path: str = "/root/images/adversarial_output.png",
    output_path: str = "/root/images/adversarial_output_rgf.png",
    steps: int = 8,
    num_query: int = 50,
):
    rgf = RGFRefine()
    results = rgf.refine.remote(
        question=question,
        target_response=target_response,
        source_response=source_response,
        clean_image_path=clean_image_path,
        adv_image_path=adv_image_path,
        output_path=output_path,
        steps=steps,
        num_query=num_query,
    )
    print()
    for k, v in results.items():
        print(f"{k}: {v}")
