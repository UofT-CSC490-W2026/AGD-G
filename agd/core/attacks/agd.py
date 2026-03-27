"""
Unsuccessful AGD implementation. Kept here as evidence that we could not
reproduce the method described in the paper.
"""

import logging
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import modal

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

class AGDAttack:
    def __init__(
            self,
            model_id="runwayml/stable-diffusion-v1-5",
            sd_device="cuda",
            vae_device="cuda",
            clip_device="cuda",
            small_vram=False,
		):
        """
        Initializes the Adversarial-Guided Diffusion (AGD) attacker.

        Args:
            model_id (str): HuggingFace model ID for Stable Diffusion.
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.sd_device = sd_device
        self.vae_device = vae_device
        self.clip_device = clip_device

        # Load Stable Diffusion
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(self.sd_device)

        # Disable safety checker for adversarial research purposes
        self.pipe.safety_checker = None

        # Use DDIM or DDPM scheduler as per paper logic (Algorithm 1 implies standard denoising)
        # The paper mentions standard reverse sampling (Eq 6, 7)[cite: 138].
        self.scheduler = self.pipe.scheduler
        self.vae = self.pipe.vae.to(self.vae_device)
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        # Load CLIP for guidance (Eq 168: "compute this similarity using CLIP features")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.clip_device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # Use attention slicing to save on VRAM if needed
        if small_vram:
            self.pipe.enable_attention_slicing()

        # Freeze models
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.clip_model.requires_grad_(False)

    def get_image_embedding_from_tensor(self, pixel_values):
        """
        Computes CLIP embeddings directly from a torch tensor.
        pixel_values: Tensor of shape [B, 3, H, W], range [0, 1]
        """
        # 1. Resize to CLIP input size (224x224)
        # Using bicubic interpolation to match CLIP's original preprocessing
        pixels_resized = torch.nn.functional.interpolate(
            pixel_values,
            size=(224, 224),
            mode="bicubic",
            align_corners=False
        )

        # 2. Normalize (Hardcoded constants for OpenAI CLIP)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(pixel_values.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(pixel_values.device).view(1, 3, 1, 1)
        pixels_norm = (pixels_resized - mean) / std

        # 3. Pass through CLIP Image Encoder
        # We use the model directly, not the processor
        embed = self.clip_model.get_image_features(pixel_values=pixels_norm).pooler_output

        # 4. Final Normalization (Unit Vector)
        return embed / embed.norm(p=2, dim=-1, keepdim=True)

    def get_image_embedding_from_pil(self, image):
        """Computes normalized CLIP image embedding."""
        # Preprocess image for CLIP
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.clip_device)
        embed = self.clip_model.get_image_features(**inputs).pooler_output
        return (embed / embed.mean(dim=-1, keepdim=True)).to(self.sd_device)

    def latent_to_image(self, latents):
        """Decodes VAE latents to PIL image for CLIP loss calculation."""
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # Convert tensor to PIL for export
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        return [Image.fromarray(img) for img in image]

    def attack(self,
               clean_image_path,
               clean_prompt,
               target_prompt,
               diffusion_steps=100,
               delta=5,
               gamma=6.0,
               start_step=0,
               inject_steps=50,
               momentum_lambda=0.9):
        """
        Executes Algorithm 1: Adversarial image generation.

        Args:
            clean_image_path (str): Path to x_0_cle.
            target_prompt (str): The target text a^tar.
            steps (int): Total diffusion steps T (Paper uses T=100)[cite: 231].
            delta (int): The truncation step Delta (Paper uses 5)[cite: 231].
            gamma (float): Scale parameter (Paper uses 6.0)[cite: 231].
            N (int): Inner iterations (Paper uses 50)[cite: 231].
            momentum_lambda (float): Momentum factor (Paper uses 0.9)[cite: 231].
        """
        # Preprocess Clean Image (x_0_cle)
        clean_pil = Image.open(clean_image_path).convert("RGB").resize((512, 512))
        clean_latents = self.encode_image(clean_pil).to(self.sd_device)

        # Algorithm 1, Line 1: Generate Target Image x_0_tar = T2I(a_tar)
        # We generate a synthetic target image to guide the adversarial features.
        try:
            target_image_pil = Image.open("images/target.png")
        except FileNotFoundError:
            logger.info("Generating target image")
            target_image_pil = self.pipe(target_prompt, num_inference_steps=100).images[0]
            target_image_pil.save("images/target.png")

        # Algorithm 1, Line 2: Obtain CLIP features z_0_tar = CLIP(x_0_tar)
        logger.debug("Generating target image embedding")
        z_target = self.get_image_embedding_from_pil(target_image_pil).detach()

        logger.debug("Running forward diffusion (adding noise)")
        # Algorithm 1, Line 3: Add noise to clean image to get x_T_cle
        # x_T = sqrt(alpha_T)*x_0 + sqrt(1-alpha_T)*epsilon
        self.scheduler.set_timesteps(diffusion_steps)
        timesteps = self.scheduler.timesteps[start_step:]

        noise = torch.randn_like(clean_latents)
        # Add noise to the clean latent to verify the starting point T
        x_t = self.scheduler.add_noise(clean_latents, noise, timesteps[0:1])

        logger.debug("Embedding prompt for Stable Diffusion")
        text_inputs = self.tokenizer(
            clean_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.sd_device)

        # Get the embeddings directly from the Text Encoder
        with torch.no_grad():
            clean_prompt_embeds = self.text_encoder(text_inputs.input_ids)[0]

        logger.info("Starting Reverse Diffusion...")

        # Algorithm 1, Line 4: Loop t from T down to 1
        iterator = tqdm(timesteps)
        for i, t in enumerate(iterator):
            # Current time step index
            # Check condition: if t > Delta (Algorithm 1, Line 5)
            # Note: Diffusers timesteps go 999 -> 0. We map this to T -> 1 logic.
            # Assuming 'steps' corresponds to the paper's T.
            current_step_num = diffusion_steps - i

            # Standard Denoising (Reconstruction Phase)
            if current_step_num > delta:
                # Algorithm 1, Line 5: x_{t-1} = denoise(x_t, epsilon_theta(x_t))
                with torch.no_grad():
                    noise_pred = self.unet(
                        x_t, t, encoder_hidden_states=clean_prompt_embeds
                    ).sample
                    x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample

            # Adversarial Guided Diffusion (Attack Phase)
            else: # if t <= Delta (Algorithm 1, Line 7)

                # Algorithm 1, Line 10: x_tilde = x_t; epsilon_bar = 0
                x_tilde = x_t.detach().clone()
                epsilon_bar = torch.zeros_like(x_t)

                # Need initial value so that line 14 has something to work with
                # on the first loop iteration.
                # Tihs is not detailed in the pseudocode in the paper.
                epsilon_tar = epsilon_bar

                # Get initial noise prediction for the current step
                # (epsilon_theta(x_t))
                with torch.no_grad():
                    epsilon_initial = self.unet(
                        x_t, t, encoder_hidden_states=clean_prompt_embeds
                    ).sample

                guided_noise = epsilon_initial.clone()

                # Algorithm 1, Line 11: Inner loop tau from N down to 1
                for tau in range(inject_steps):
                    # Enable gradient calculation on x_tilde for guidance
                    x_tilde = x_tilde.detach().requires_grad_(True)

                    # Algorithm 1, Line 13: epsilon = epsilon + gamma * sign(epsilon_bar)
                    # The perturbation is applied to the noise used for denoising
                    guided_noise = guided_noise + gamma * torch.sign(epsilon_bar)

                    # Algorithm 1, Line 14: epsilon_bar = lambda*epsilon_bar + (1-lambda)*epsilon_tar
                    epsilon_bar = momentum_lambda * epsilon_bar + (1 - momentum_lambda) * epsilon_tar

                    # Algorithm 1, Line 15: x_tilde = denoise(x_tilde, epsilon)
                    # We simulate a step with the guided noise to see where it goes
                    # We don't actually step 'forward' in time, we refine the current step's output
                    # Ideally, we project x_tilde using the scheduler's transfer function
                    # For simplicity in this script, we perform a scheduler step
                    # to get the proposed x_{t-1} candidate
                    x_tilde = self.scheduler.step(guided_noise, t, x_t).prev_sample.requires_grad_(True)

                    # Algorithm 1, Line 16: z = CLIP(x_tilde)
                    latents = 1 / 0.18215 * x_tilde
                    pixels = self.vae.decode(latents).sample
                    pixels = (pixels / 2 + 0.5).clamp(0, 1).to(self.clip_device)
                    z_current = self.get_image_embedding_from_tensor(pixels).to(self.sd_device)
                    loss = -torch.cosine_similarity(z_current, z_target).mean()

                    # Algorithm 1, Line 17: epsilon^tar calculation
                    # Eq (10): epsilon^tar approx -sqrt(1-alpha)*grad_log_p
                    # We compute gradients of loss w.r.t x_tilde
                    grad = torch.autograd.grad(loss, x_tilde)[0]

                    # Normalize gradient to act as noise component
                    epsilon_tar = -grad

                # Algorithm 1, Line 18: Final epsilon for this step
                # tilde_epsilon = epsilon_theta(x_t) + gamma * sign(epsilon_bar)
                final_noise_pred = epsilon_initial + gamma * torch.sign(epsilon_bar)

                # Algorithm 1, Line 19: x_{t-1} = denoise(x_t, tilde_epsilon)
                x_t = self.scheduler.step(final_noise_pred, t, x_t).prev_sample

        # 6. Decode final adversarial image
        final_image = self.latent_to_image(x_t)[0]
        return final_image

    def encode_image(self, image):
        """Encodes PIL image to latents."""
        logger.debug("Converting image with size %d x %d", image.size[0], image.size[1])
        img_tensor = torch.tensor(np.array(image)).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(device=self.vae_device, dtype=torch.float16)
        img_tensor = 2.0 * img_tensor - 1.0
        logger.debug("Encoding image tensor with size %s", img_tensor.shape)
        # Trying mean instead of sample for better determinism
        #latents = self.pipe.vae.encode(img_tensor).latent_dist.sample()
        latents = self.pipe.vae.encode(img_tensor).latent_dist.mean
        logger.debug("Image encoding done")
        return latents * 0.18215

modal_image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "torch",
            "diffusers",
            "transformers",
            "pillow",
            "tqdm",
            "accelerate"
        )
        .add_local_file("xray-fish-profile.png", "/root/clean_image.png")
    )

hf_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
image_volume = modal.Volume.from_name("agd_images", create_if_missing=True)

app = modal.App(
        "agd",
        secrets=[modal.Secret.from_name("huggingface")],
        image=modal_image,
        volumes={
            "/root/.cache/huggingface": hf_volume,
            "/root/images": image_volume,
        },
    )

@app.function(
    gpu="T4:1",
    timeout=3600,
)
def agd_test(
    diffusion_steps: int = 100,
    delta: int = 5,
    inject_steps: int = 50,
    clean_prompt: str = "A fish",
    target_prompt: str = "A baseball",
    start_step: int = 0,
    gamma: float = 6.0,
):
    # Ensure you have an image named 'clean_input.png'
    logger.info("AGD Class loaded")
    attacker = AGDAttack()
    for delta in [10, 20, 30]:
        for actual_steps in [1]:
            for gamma in [6, 12, 18]:
                for inject_steps in [50, 200, 400]:
                    start_step = 100 - delta - actual_steps
                    logger.info(f"Running AGD with delta={delta} actual_steps={actual_steps} gamma={gamma} N={inject_steps}")
                    adv_img = attacker.attack(
                        clean_image_path="clean_image.png",
                        clean_prompt=clean_prompt,
                        target_prompt=target_prompt,
                        diffusion_steps=diffusion_steps,
                        delta=delta,
                        gamma=gamma,
                        start_step=start_step,
                        inject_steps=inject_steps,
                    )
                    adv_img.save(f"images/delta={delta}_actual_steps={actual_steps}_gamma={gamma}_N={inject_steps}.png")

# Usage Example
if __name__ == "__main__":
    agd_test.remote()
