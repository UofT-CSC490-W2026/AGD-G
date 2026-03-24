import logging
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import modal

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AGDAttack:
    def __init__(
        self,
        model_id="runwayml/stable-diffusion-v1-5",
        sd_device="cuda",
        vae_device="cuda",
        clip_device="cuda",
        small_vram=False,
    ):
        self.sd_device = sd_device
        self.vae_device = vae_device
        self.clip_device = clip_device

        # Load Stable Diffusion
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(self.sd_device)
        self.pipe.safety_checker = None

        # Switch to DDPMScheduler — the paper uses DDPM reverse sampling
        # (Eqs 6-7) and edit-friendly DDPM noise space (ref [19]).
        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.scheduler = self.scheduler

        self.vae = self.pipe.vae          # already on sd_device
        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        # CLIP for adversarial guidance (Eq 9-10 / line 16-17)
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.clip_device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        if small_vram:
            self.pipe.enable_attention_slicing()

        # Freeze everything — gradients only flow through x_tilde in AGD
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.clip_model.requires_grad_(False)

    # ------------------------------------------------------------------ #
    #  CLIP helpers                                                        #
    # ------------------------------------------------------------------ #

    def get_image_embedding_from_tensor(self, pixel_values):
        """
        Differentiable CLIP image embedding.
        pixel_values: [B, 3, H, W] in [0, 1], any dtype.
        """
        px = torch.nn.functional.interpolate(
            pixel_values, size=(224, 224), mode="bicubic", align_corners=False
        )
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=pixel_values.device, dtype=pixel_values.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=pixel_values.device, dtype=pixel_values.dtype,
        ).view(1, 3, 1, 1)
        px = (px - mean) / std
        out = self.clip_model.get_image_features(pixel_values=px)
        embed = out.pooler_output if hasattr(out, "pooler_output") else out
        return embed / embed.norm(p=2, dim=-1, keepdim=True)

    def get_image_embedding_from_pil(self, image):
        """Non-differentiable CLIP embedding from a PIL image."""
        inputs = self.clip_processor(images=image, return_tensors="pt").to(
            self.clip_device
        )
        out = self.clip_model.get_image_features(**inputs)
        embed = out.pooler_output if hasattr(out, "pooler_output") else out
        # FIX: L2-normalize (was .mean before, inconsistent with tensor path)
        return (embed / embed.norm(p=2, dim=-1, keepdim=True)).to(self.sd_device)

    # ------------------------------------------------------------------ #
    #  VAE helpers                                                         #
    # ------------------------------------------------------------------ #

    def encode_image(self, image):
        """PIL → scaled VAE latent."""
        img = torch.tensor(np.array(image)).float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = img.to(device=self.vae_device, dtype=torch.float16)
        img = 2.0 * img - 1.0
        return self.vae.encode(img).latent_dist.mean * 0.18215

    def decode_latents_to_pixels(self, latents):
        """Scaled latent → pixel tensor [B,3,H,W] in [0,1].  Differentiable."""
        latents = latents / 0.18215
        # VAE is fp16 — cast input, keep output in input's original dtype
        pixels = self.vae.decode(latents.half()).sample.to(latents.dtype)
        return (pixels / 2 + 0.5).clamp(0, 1)

    def latent_to_pil(self, latents):
        """Scaled latent → list of PIL images."""
        with torch.no_grad():
            px = self.decode_latents_to_pixels(latents)
        px = px.cpu().permute(0, 2, 3, 1).float().numpy()
        px = (px * 255).round().astype("uint8")
        return [Image.fromarray(p) for p in px]

    # ------------------------------------------------------------------ #
    #  Manual DDPM reverse step                                            #
    # ------------------------------------------------------------------ #

    def ddpm_step(self, x_t, noise_pred, t_val, t_prev_val, added_noise=None):
        """
        One reverse DDPM step x_t  →  x_{t-1}.

        Implements Eqs (6)-(7) for *non-consecutive* timesteps (as produced
        by set_timesteps with fewer steps than training).

        If added_noise is None the step is deterministic (μ only).
        Otherwise x_{t-1} = μ + σ · added_noise.
        """
        device = x_t.device
        acp = self.scheduler.alphas_cumprod.to(device)

        ab_t = acp[t_val].float()
        ab_prev = (
            acp[t_prev_val].float()
            if t_prev_val >= 0
            else torch.tensor(1.0, device=device)
        )

        # Work in float32 for numerical stability
        eps_f = noise_pred.float()
        x_f = x_t.float()

        # Predicted x_0 (Eq 4)
        # NOTE: Do NOT clamp here — in latent diffusion, latents range well
        # beyond [-1, 1].  SD v1.5 config has clip_sample=false.
        pred_x0 = (x_f - (1 - ab_t).sqrt() * eps_f) / ab_t.sqrt()

        # Coefficients for the posterior mean μ
        alpha_ratio = ab_t / ab_prev          # α_t  (step-local)
        beta_t = 1 - alpha_ratio              # β_t  (step-local)
        beta_bar_t = 1 - ab_t                 # 1 - ᾱ_t
        beta_bar_prev = (1 - ab_prev).clamp(min=1e-20)

        coeff_x0 = ab_prev.sqrt() * beta_t / beta_bar_t
        coeff_xt = alpha_ratio.sqrt() * beta_bar_prev / beta_bar_t
        mu = coeff_x0 * pred_x0 + coeff_xt * x_f

        if added_noise is not None:
            sigma_sq = (beta_bar_prev / beta_bar_t * beta_t).clamp(min=0)
            mu = mu + sigma_sq.sqrt() * added_noise.float()

        return mu.to(x_t.dtype)

    # ------------------------------------------------------------------ #
    #  DDPM Inversion  (edit-friendly noise space, ref [19])               #
    # ------------------------------------------------------------------ #

    def ddpm_inversion(self, clean_latents, prompt_embeds, noise, timesteps):
        """
        Compute the noise maps {e_t} such that the DDPM reverse process
        with these maps perfectly reconstructs x_0^cle from x_T^cle.

        Reference: Huberman-Spiegelglas et al., "An Edit Friendly DDPM
        Noise Space: Inversion and Manipulations", CVPR 2024.

        Algorithm
        ---------
        1.  Build the forward trajectory using q(x_t | x_0):
                x_t = sqrt(ᾱ_t) · x_0 + sqrt(1-ᾱ_t) · ε
            where ε is the *same* noise for every t.

        2.  Walk the reverse process from x_T.  At each step t → t-1:
            a.  Predict noise with the UNet:  ε_θ(x_t, t)
            b.  Compute the posterior mean μ_t  (Eq 7)
            c.  Solve for e_t:
                    x_{t-1}^fwd = μ_t + σ_t · e_t
                    ⟹  e_t = (x_{t-1}^fwd − μ_t) / σ_t

        Returns
        -------
        x_T         :  noised latent at the starting timestep
        noise_maps  :  dict  {timestep_value  →  e_t  tensor}
        """
        device = clean_latents.device
        acp = self.scheduler.alphas_cumprod.to(device)

        # x_T from the forward marginal
        t_T = timesteps[0].item()
        ab_T = acp[t_T]
        x_T = ab_T.sqrt() * clean_latents + (1 - ab_T).sqrt() * noise

        noise_maps = {}
        x_t = x_T.clone()

        for i, t in enumerate(tqdm(timesteps, desc="DDPM inversion")):
            t_val = t.item()
            t_prev_val = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1

            # Forward-marginal target at t_prev
            if t_prev_val >= 0:
                ab_prev = acp[t_prev_val].float()
                x_prev_target = (
                    ab_prev.sqrt() * clean_latents + (1 - ab_prev).sqrt() * noise
                )
            else:
                ab_prev = torch.tensor(1.0, device=device)
                x_prev_target = clean_latents

            # UNet prediction (model is fp16, latents are fp32)
            with torch.no_grad():
                noise_pred = self.unet(
                    x_t.half(), t, encoder_hidden_states=prompt_embeds
                ).sample.float()

            # Deterministic μ
            mu = self.ddpm_step(x_t, noise_pred, t_val, t_prev_val, added_noise=None)

            # σ for this step
            ab_t = acp[t_val].float()
            ab_prev_f = ab_prev.float()
            alpha_ratio = ab_t / ab_prev_f
            beta_t = 1 - alpha_ratio
            beta_bar_t = 1 - ab_t
            beta_bar_prev = (1 - ab_prev_f).clamp(min=1e-20)
            sigma = ((beta_bar_prev / beta_bar_t * beta_t).clamp(min=0)).sqrt()

            # Solve for e_t
            if sigma > 1e-8:
                e_t = ((x_prev_target.float() - mu.float()) / sigma).to(x_t.dtype)
            else:
                e_t = torch.zeros_like(x_t)

            noise_maps[t_val] = e_t.detach()

            # Advance — equals x_prev_target by construction
            x_t = x_prev_target.clone()

        return x_T, noise_maps

    # ------------------------------------------------------------------ #
    #  Main attack  (Algorithm 1)                                          #
    # ------------------------------------------------------------------ #

    def attack(
        self,
        clean_image_path,
        clean_prompt,
        target_prompt,
        diffusion_steps=100,
        delta=5,
        gamma=6.0,
        inject_steps=50,
        momentum_lambda=0.9,
    ):
        """
        Algorithm 1: Adversarial image generation.

        Parameters match the paper's defaults:
            T = 100,  Δ = 5,  γ = 6.0,  N = 50,  λ = 0.9
        """

        # --- Encode clean image (x_0^cle) ---
        clean_pil = Image.open(clean_image_path).convert("RGB").resize((512, 512))
        clean_latents = self.encode_image(clean_pil).to(self.sd_device)

        # --- Alg 1, Line 1: x_0^tar = T2I(a^tar) ---
        logger.info("Generating target image via T2I")
        target_pil = self.pipe(
            target_prompt, num_inference_steps=diffusion_steps
        ).images[0]
        target_pil.save("images/target.png")

        # --- Alg 1, Line 2: z_0^tar = CLIP(x_0^tar) ---
        logger.info("Computing target CLIP embedding")
        z_target = self.get_image_embedding_from_pil(target_pil).detach()

        # --- Encode clean prompt ---
        tok = self.tokenizer(
            clean_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.sd_device)
        with torch.no_grad():
            prompt_embeds = self.text_encoder(tok.input_ids)[0]

        # --- Setup scheduler timesteps ---
        self.scheduler.set_timesteps(diffusion_steps)
        timesteps = self.scheduler.timesteps  # descending

        # --- Alg 1, Line 3: Forward diffusion + DDPM inversion ---
        # Work in float32 throughout inversion + reconstruction to prevent
        # precision loss compounding over ~95 reverse steps.
        logger.info("Running DDPM inversion to compute noise maps")
        clean_latents_f32 = clean_latents.float()
        noise = torch.randn_like(clean_latents_f32)
        x_T, noise_maps = self.ddpm_inversion(
            clean_latents_f32, prompt_embeds, noise, timesteps
        )

        # --- Reverse diffusion  (Lines 4-22) ---
        logger.info("Starting reverse diffusion")
        x_t = x_T.clone()  # already float32

        for i, t in enumerate(tqdm(timesteps, desc="Reverse diffusion")):
            t_val = t.item()
            t_prev_val = timesteps[i + 1].item() if i + 1 < len(timesteps) else -1
            step_num = diffusion_steps - i   # counts T, T-1, ..., 1

            # ============================================================
            # Reconstruction phase  (t > Δ)  — Lines 5-6
            # Uses edit-friendly noise maps for high-fidelity reconstruction
            # ============================================================
            if step_num > delta:
                with torch.no_grad():
                    # UNet is fp16 — cast input, then back to float32
                    eps = self.unet(
                        x_t.half(), t, encoder_hidden_states=prompt_embeds
                    ).sample.float()
                    e_t = noise_maps.get(t_val, torch.zeros_like(x_t))
                    x_t = self.ddpm_step(x_t, eps, t_val, t_prev_val, added_noise=e_t)

            # ============================================================
            # AGD phase  (t ≤ Δ)  — Lines 7-21
            # ============================================================
            else:
                # --- DEBUG: save reconstructed image before AGD starts ---
                if step_num == delta:
                    logger.info(
                        "Entering AGD phase at step_num=%d, timestep=%d",
                        step_num, t_val,
                    )
                    recon_img = self.latent_to_pil(x_t)[0]
                    recon_img.save("images/debug_reconstructed.png")
                    logger.info("Saved reconstruction to images/debug_reconstructed.png")
                # Line 11:  ε = ε_θ(x_t);  ε̄ = 0
                epsilon_bar = torch.zeros_like(x_t)
                epsilon_tar = torch.zeros_like(x_t)

                with torch.no_grad():
                    epsilon_theta = self.unet(
                        x_t.half(), t, encoder_hidden_states=prompt_embeds
                    ).sample.float()

                guided_noise = epsilon_theta.clone()

                # Line 12:  inner loop  τ = N … 1
                for tau in range(inject_steps):
                    # Line 13:  ε = ε + γ · sign(ε̄)
                    guided_noise = guided_noise + gamma * torch.sign(epsilon_bar)

                    # Line 14:  ε̄ = λ · ε̄ + (1-λ) · ε^tar
                    epsilon_bar = (
                        momentum_lambda * epsilon_bar
                        + (1 - momentum_lambda) * epsilon_tar
                    )

                    # Line 15:  x̃ = denoise(x_t, ε)   — deterministic probe
                    x_tilde = self.ddpm_step(
                        x_t, guided_noise, t_val, t_prev_val
                    ).requires_grad_(True)

                    # Line 16:  z = CLIP(x̃)
                    pixels = self.decode_latents_to_pixels(x_tilde).to(
                        device=self.clip_device, dtype=torch.float32
                    )
                    z_cur = self.get_image_embedding_from_tensor(pixels).to(
                        self.sd_device
                    )
                    loss = -torch.cosine_similarity(z_cur, z_target).mean()

                    # Line 17:  ε^tar from gradient
                    # Paper Eq (17): the Gaussian log-likelihood formulation
                    # produces a 1/√(1-ᾱ) amplification, not √(1-ᾱ) damping.
                    # With cosine similarity loss, sign() discards magnitude,
                    # so we just need a meaningful direction — use raw gradient.
                    grad = torch.autograd.grad(loss, x_tilde)[0]
                    epsilon_tar = -grad

                # Lines 19-20:  final denoising step for this timestep
                # Include noise maps (e_t) to pull result back toward clean image
                final_eps = epsilon_theta + gamma * torch.sign(epsilon_bar)
                e_t = noise_maps.get(t_val, torch.zeros_like(x_t))
                x_t = self.ddpm_step(x_t, final_eps, t_val, t_prev_val, added_noise=e_t)

        # --- Decode final adversarial image ---
        return self.latent_to_pil(x_t)[0]


# ================================================================== #
#  Modal infrastructure                                                #
# ================================================================== #

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch", "diffusers", "transformers", "pillow", "tqdm", "accelerate"
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


@app.function(gpu="T4:1", timeout=600)
def agd_test(
    diffusion_steps: int = 100,
    delta: int = 5,
    inject_steps: int = 50,
    clean_prompt: str = "A fish",
    target_prompt: str = "A baseball",
    gamma: float = 1.0,
    momentum_lambda: float = 0.9,
    recon_only: bool = False,
):
    logger.info("Initializing AGD attacker")
    attacker = AGDAttack(small_vram=True)

    # recon_only: skip AGD entirely by setting delta=0
    effective_delta = 0 if recon_only else delta

    logger.info(
        "Running %s (delta=%d)",
        "reconstruction only" if recon_only else "full attack",
        effective_delta,
    )
    adv_img = attacker.attack(
        clean_image_path="clean_image.png",
        clean_prompt=clean_prompt,
        target_prompt=target_prompt,
        diffusion_steps=diffusion_steps,
        delta=effective_delta,
        gamma=gamma,
        inject_steps=inject_steps,
        momentum_lambda=momentum_lambda,
    )
    adv_img.save("images/adversarial_output.png")
    logger.info("Saved output to images/adversarial_output.png")


@app.local_entrypoint()
def main(recon_only: bool = False):
    """
    modal run agd.py                  # full attack
    modal run agd.py --recon-only     # reconstruction only
    """
    agd_test.remote(recon_only=recon_only)