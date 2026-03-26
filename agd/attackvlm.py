"""
MF-ii PGD attack on CLIP ViT-L/14-336 — LLaVA-1.5's vision encoder.

LLaVA-1.5 uses CLIP ViT-L/14-336px as its vision encoder, so we attack
the matching CLIP backbone to improve transfer.

Usage:
    modal run attackvlm.py
    modal run attackvlm.py --epsilon 0.125 --steps 500
"""
import logging
import tempfile
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import modal

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TargetModel(torch.nn.Module):
    def __init__(self) -> None:
        """Download model, initialize whatever."""
        super().__init__()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable model outputs on a batch of images.
        """
        raise NotImplementedError


class TextAttackMethod:
    def __init__(self, model: TargetModel) -> None:
        """Initialize whatever needs initialization."""
        self.model = model

    def attack(
        self,
        clean: List[Image.Image],
        target: List[str],
        strength: float,
        hyperparameters: Optional[Dict] = None,
    ) -> Tuple[List[Image.Image], List[bool]]:
        assert len(clean) == len(target)
        raise NotImplementedError


class ImageAttackMethod:
    def __init__(self, model: TargetModel) -> None:
        """Initialize whatever needs initialization."""
        self.model = model

    def attack(
        self,
        clean: List[Image.Image],
        target: List[Image.Image],
        strength: float,
        hyperparameters: Optional[Dict] = None,
    ) -> Tuple[List[Image.Image], List[bool]]:
        assert len(clean) == len(target)
        raise NotImplementedError


class CLIPAttack:
    def __init__(self, device="cuda", model_id="openai/clip-vit-large-patch14-336"):
        self.device = device

        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP surrogate: %s", model_id)

        # Float32, single device for clean gradient flow
        self.clip_model = CLIPModel.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_id)
        self.clip_model.requires_grad_(False)

        # Keep preprocessing aligned with the model's expected resolution.
        self.image_size = int(self.clip_model.config.vision_config.image_size)
        # ViT-L/14 has 24 transformer layers.
        self.vision_model = self.clip_model.vision_model

    def _preprocess(self, pixel_tensor):
        """[B, 3, H, W] in [0,1] -> CLIP-normalized SxS."""
        px = F.interpolate(
            pixel_tensor,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=px.device, dtype=px.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=px.device, dtype=px.dtype,
        ).view(1, 3, 1, 1)
        return (px - mean) / std

    def get_features(self, pixel_tensor):
        """
        Multi-layer feature extraction from CLIP vision encoder.
        Returns list of normalized features from multiple layers.
        """
        px = self._preprocess(pixel_tensor)

        out = self.vision_model(
            pixel_values=px,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = out.hidden_states  # 25 tensors (embed + 24 layers)

        features = []
        # Sample layers across the full depth
        for idx in [-1, -6, -12, -18, -24]:
            h = hidden_states[idx]
            # Use ALL patch tokens, not just CLS — richer signal
            patch_mean = h[:, 1:, :].mean(dim=1)  # skip CLS, mean over patches
            patch_mean = patch_mean / patch_mean.norm(p=2, dim=-1, keepdim=True)
            features.append(patch_mean)

        # Also match the final projected embedding
        projected = self.clip_model.visual_projection(out.last_hidden_state[:, 0, :])
        projected = projected / projected.norm(p=2, dim=-1, keepdim=True)
        features.append(projected)

        return features

    def get_target_features(self, image_path):
        target_pil = Image.open(image_path).convert("RGB")
        target_np = np.array(target_pil.resize((512, 512))).astype(np.float32) / 255.0
        target_t = (
            torch.from_numpy(target_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            feats = self.get_features(target_t)
        return [f.detach() for f in feats]

    def get_text_feature(self, text: str):
        """
        CLIP text embedding projected into the shared image-text space.
        """
        tokenized = self.clip_processor.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            out = self.clip_model.text_model(
                **tokenized,
                return_dict=True,
            )
            text_embed = self.clip_model.text_projection(out.pooler_output)
        text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
        return text_embed.detach()

    @staticmethod
    def _eot_augment(
        x: torch.Tensor,
        noise_std: float = 0.0,
        brightness: float = 0.0,
        shift_px: int = 0,
    ) -> torch.Tensor:
        """
        Lightweight differentiable augmentations for transfer (EOT).
        """
        y = x
        if noise_std > 0:
            y = y + noise_std * torch.randn_like(y)
        if brightness > 0:
            scale = 1.0 + (2.0 * torch.rand((x.shape[0], 1, 1, 1), device=x.device) - 1.0) * brightness
            y = y * scale
        if shift_px > 0:
            dy = int(torch.randint(-shift_px, shift_px + 1, (1,), device=x.device).item())
            dx = int(torch.randint(-shift_px, shift_px + 1, (1,), device=x.device).item())
            y = torch.roll(y, shifts=(dy, dx), dims=(2, 3))
        return y.clamp(0, 1)

    @staticmethod
    def _build_text_like_mask(
        clean_t: torch.Tensor,
        edge_quantile: float = 0.88,
        std_quantile: float = 0.70,
        dilate_kernel: int = 7,
        max_area_ratio: float = 0.22,
    ) -> torch.Tensor:
        """
        OCR-free heuristic mask that emphasizes text/tick/legend-like regions.
        Returns [B,1,H,W] binary float mask.
        """
        gray = (
            0.299 * clean_t[:, 0:1, :, :]
            + 0.587 * clean_t[:, 1:2, :, :]
            + 0.114 * clean_t[:, 2:3, :, :]
        )

        sobel_x = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
            dtype=gray.dtype,
            device=gray.device,
        ).unsqueeze(1)
        sobel_y = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
            dtype=gray.dtype,
            device=gray.device,
        ).unsqueeze(1)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-12)

        gthr = torch.quantile(grad.flatten(1), edge_quantile, dim=1).view(-1, 1, 1, 1)
        edge_mask = (grad >= gthr).float()

        mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        mean2 = F.avg_pool2d(gray * gray, kernel_size=9, stride=1, padding=4)
        std = torch.sqrt((mean2 - mean * mean).clamp(min=0.0) + 1e-12)
        sthr = torch.quantile(std.flatten(1), std_quantile, dim=1).view(-1, 1, 1, 1)
        std_mask = (std >= sthr).float()

        mask = edge_mask * std_mask
        k = int(max(3, dilate_kernel))
        if k % 2 == 0:
            k += 1
        for _ in range(2):
            mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)

        if mask.mean().item() > max_area_ratio:
            cgrad = grad * mask
            cthr = torch.quantile(cgrad.flatten(1), 1.0 - max_area_ratio, dim=1).view(-1, 1, 1, 1)
            mask = (cgrad >= cthr).float()
            mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)

        return mask.clamp(0, 1)

    def attack(
        self,
        clean_image_path,
        target_image_path=None,
        target_text: Optional[str] = None,
        source_text: Optional[str] = None,
        eps=16 / 255,
        alpha=2 / 255,
        steps=300,
        repel_clean: float = 0.35,
        repel_source_text: float = 0.5,
        eot_samples: int = 1,
        eot_noise_std: float = 0.0,
        eot_brightness: float = 0.0,
        eot_shift_px: int = 0,
        ocr_edge_quantile: float = 0.88,
        ocr_std_quantile: float = 0.70,
        ocr_dilate_kernel: int = 7,
        ocr_max_area_ratio: float = 0.22,
        mode: str = "targeted_image",
    ):
        clean_pil = Image.open(clean_image_path).convert("RGB").resize((512, 512))
        clean_np = np.array(clean_pil).astype(np.float32) / 255.0
        clean_t = (
            torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        clean_feats = self.get_features(clean_t)
        perturbation_mask = None
        target_feats = None
        target_text_feat = None
        source_text_feat = None
        if mode == "targeted_image":
            if not target_image_path:
                raise ValueError("target_image_path is required for mode=targeted_image")
            target_feats = self.get_target_features(target_image_path)
        elif mode == "targeted_text":
            if not target_text:
                raise ValueError("target_text is required for mode=targeted_text")
            target_text_feat = self.get_text_feature(target_text)
            if source_text:
                source_text_feat = self.get_text_feature(source_text)
        elif mode == "targeted_text_ocr":
            if not target_text:
                raise ValueError("target_text is required for mode=targeted_text_ocr")
            target_text_feat = self.get_text_feature(target_text)
            if source_text:
                source_text_feat = self.get_text_feature(source_text)
            perturbation_mask = self._build_text_like_mask(
                clean_t,
                edge_quantile=ocr_edge_quantile,
                std_quantile=ocr_std_quantile,
                dilate_kernel=ocr_dilate_kernel,
                max_area_ratio=ocr_max_area_ratio,
            )
            logger.info("OCR-like mask coverage: %.3f", perturbation_mask.mean().item())
        elif mode != "untargeted":
            raise ValueError(
                "Unknown mode: "
                f"{mode}. Use 'targeted_image', 'targeted_text', "
                "'targeted_text_ocr', or 'untargeted'."
            )

        # Random start within eps-ball
        delta = torch.empty_like(clean_t).uniform_(-eps, eps)
        delta = (clean_t + delta).clamp(0, 1) - clean_t
        delta.requires_grad_(True)

        best_loss = float("inf")
        best_adv = clean_t.clone()

        logger.info(
            "CLIP PGD: mode=%s eps=%.4f alpha=%.4f steps=%d feature_layers=6",
            mode, eps, alpha, steps,
        )

        for i in tqdm(range(steps), desc="CLIP PGD"):
            adv = (clean_t + delta).clamp(0, 1)
            loss = 0.0
            sim_clean = 0.0
            sim_target = 0.0
            sim_text = 0.0
            sim_source_text = 0.0
            n = max(1, int(eot_samples))
            for _ in range(n):
                adv_aug = self._eot_augment(
                    adv,
                    noise_std=eot_noise_std,
                    brightness=eot_brightness,
                    shift_px=eot_shift_px,
                )
                adv_feats = self.get_features(adv_aug)

                sc = 0.0
                for f_adv, f_cln in zip(adv_feats, clean_feats):
                    sc += F.cosine_similarity(f_adv, f_cln).mean()
                sc = sc / len(adv_feats)
                sim_clean += sc

                if mode == "targeted_image":
                    st = 0.0
                    for f_adv, f_tar in zip(adv_feats, target_feats):
                        st += F.cosine_similarity(f_adv, f_tar).mean()
                    st = st / len(adv_feats)
                    sim_target += st
                    loss = loss + (-st + repel_clean * sc)
                elif mode in {"targeted_text", "targeted_text_ocr"}:
                    stx = F.cosine_similarity(adv_feats[-1], target_text_feat).mean()
                    sim_text += stx
                    step_loss = -stx + repel_clean * sc
                    if source_text_feat is not None:
                        ssrc = F.cosine_similarity(adv_feats[-1], source_text_feat).mean()
                        sim_source_text += ssrc
                        step_loss = step_loss + repel_source_text * ssrc
                    loss = loss + step_loss
                else:
                    # Untargeted: only push away from clean semantics.
                    loss = loss + sc

            loss = loss / n
            sim_clean = sim_clean / n
            sim_target = sim_target / n
            sim_text = sim_text / n
            sim_source_text = sim_source_text / n

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_adv = adv.detach().clone()

            loss.backward()

            with torch.no_grad():
                delta.data = delta.data - alpha * delta.grad.sign()
                delta.data = delta.data.clamp(-eps, eps)
                if perturbation_mask is not None:
                    delta.data = delta.data * perturbation_mask
                delta.data = (clean_t + delta.data).clamp(0, 1) - clean_t

            delta.grad.zero_()

            if (i + 1) % 50 == 0:
                if mode == "targeted_image":
                    logger.info(
                        "  Step %d/%d  sim_target=%.4f sim_clean=%.4f loss=%.4f",
                        i + 1, steps, sim_target.item(), sim_clean.item(), loss.item()
                    )
                elif mode in {"targeted_text", "targeted_text_ocr"}:
                    if source_text_feat is not None:
                        logger.info(
                            "  Step %d/%d  sim_text=%.4f sim_source_text=%.4f sim_clean=%.4f loss=%.4f",
                            i + 1, steps, sim_text.item(), sim_source_text.item(), sim_clean.item(), loss.item()
                        )
                    else:
                        logger.info(
                            "  Step %d/%d  sim_text=%.4f sim_clean=%.4f loss=%.4f",
                            i + 1, steps, sim_text.item(), sim_clean.item(), loss.item()
                        )
                else:
                    logger.info(
                        "  Step %d/%d  sim_clean=%.4f loss=%.4f",
                        i + 1, steps, sim_clean.item(), loss.item()
                    )

        logger.info("Best avg similarity: %.4f", -best_loss)

        adv_np = best_adv[0].cpu().permute(1, 2, 0).numpy()
        adv_np = (adv_np * 255).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(adv_np)


class CLIPTargetModel(TargetModel):
    """
    Thin wrapper over CLIPAttack that satisfies the TargetModel interface.
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.backend = CLIPAttack(device=device)
        self.device = device

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Return final projected CLIP embedding [B, D] as differentiable output.
        feats = self.backend.get_features(images)
        return feats[-1]

    def get_multilayer_features(self, images: torch.Tensor):
        return self.backend.get_features(images)


class CLIPTextAttackMethod(TextAttackMethod):
    """
    Text-target interface over current PGD implementation.

    Supports:
      - untargeted mode (push away from clean semantics)
      - targeted_text mode (pull image toward target QA text semantics)
    """

    def __init__(self, model: TargetModel) -> None:
        super().__init__(model)
        if not isinstance(model, CLIPTargetModel):
            raise TypeError("CLIPTextAttackMethod expects a CLIPTargetModel")

    @staticmethod
    def _strength_to_eps(strength: float, hyperparameters: Optional[Dict]) -> float:
        if strength < 0:
            raise ValueError("strength must be non-negative")
        hp = hyperparameters or {}
        max_eps = float(hp.get("max_eps", 32 / 255))
        if strength > 1.0:
            return min(strength / 255.0, max_eps)
        return min(strength * max_eps, max_eps)

    @staticmethod
    def _pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
        arr = np.array(image.convert("RGB").resize((512, 512))).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    @staticmethod
    def _avg_sim(feats_a, feats_b) -> torch.Tensor:
        sim = 0.0
        for fa, fb in zip(feats_a, feats_b):
            sim += F.cosine_similarity(fa, fb).mean()
        return sim / len(feats_a)

    def attack(
        self,
        clean: List[Image.Image],
        target: List[str],
        strength: float,
        hyperparameters: Optional[Dict] = None,
    ) -> Tuple[List[Image.Image], List[bool]]:
        assert len(clean) == len(target)
        hp = hyperparameters or {}
        profile = str(hp.get("profile", "simple"))

        eps = self._strength_to_eps(strength, hp)
        alpha = float(hp.get("alpha", 2 / 255))
        steps = int(hp.get("steps", 300))
        mode = str(hp.get("mode", "targeted_text"))
        repel_clean = float(hp.get("repel_clean", 0.35))
        question = hp.get("question")
        questions = hp.get("questions")
        source_response = hp.get("source_response")
        source_responses = hp.get("source_responses")
        success_sim_threshold = float(hp.get("success_sim_threshold", 0.20))
        success_text_threshold = float(hp.get("success_text_threshold", 0.25))
        repel_source_text = float(hp.get("repel_source_text", 0.5))
        # Simple profile: full-image targeted text, no OCR mask, no EOT.
        if profile == "simple":
            if mode == "targeted_text_ocr":
                mode = "targeted_text"
            eot_samples = 1
            eot_noise_std = 0.0
            eot_brightness = 0.0
            eot_shift_px = 0
            ocr_edge_quantile = 0.88
            ocr_std_quantile = 0.70
            ocr_dilate_kernel = 7
            ocr_max_area_ratio = 0.22
        else:
            eot_samples = int(hp.get("eot_samples", 1))
            eot_noise_std = float(hp.get("eot_noise_std", 0.0))
            eot_brightness = float(hp.get("eot_brightness", 0.0))
            eot_shift_px = int(hp.get("eot_shift_px", 0))
            ocr_edge_quantile = float(hp.get("ocr_edge_quantile", 0.88))
            ocr_std_quantile = float(hp.get("ocr_std_quantile", 0.70))
            ocr_dilate_kernel = int(hp.get("ocr_dilate_kernel", 7))
            ocr_max_area_ratio = float(hp.get("ocr_max_area_ratio", 0.22))

        adv_images: List[Image.Image] = []
        success_flags: List[bool] = []

        for idx, (clean_img, target_text) in enumerate(zip(clean, target)):
            q = question
            if questions is not None:
                q = questions[idx]
            qa_text = (
                f"Question: {q}\nAnswer: {target_text}" if q else target_text
            )
            src_resp = source_response
            if source_responses is not None:
                src_resp = source_responses[idx]
            source_qa_text = None
            if src_resp:
                source_qa_text = (
                    f"Question: {q}\nAnswer: {src_resp}" if q else str(src_resp)
                )
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp_clean:
                clean_img.convert("RGB").save(tmp_clean.name)
                adv_img = self.model.backend.attack(
                    clean_image_path=tmp_clean.name,
                    target_image_path=None,
                    target_text=qa_text if mode in {"targeted_text", "targeted_text_ocr"} else None,
                    source_text=source_qa_text if mode in {"targeted_text", "targeted_text_ocr"} else None,
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
                    mode=mode,
                )

            adv_images.append(adv_img)

            clean_t = self._pil_to_tensor(clean_img, self.model.device)
            adv_t = self._pil_to_tensor(adv_img, self.model.device)
            with torch.no_grad():
                if mode in {"targeted_text", "targeted_text_ocr"}:
                    target_text_feat = self.model.backend.get_text_feature(qa_text)
                    adv_embed = self.model.forward(adv_t)
                    sim_text = F.cosine_similarity(adv_embed, target_text_feat).mean().item()
                    success_flags.append(sim_text > success_text_threshold)
                else:
                    sim_clean = self._avg_sim(
                        self.model.get_multilayer_features(adv_t),
                        self.model.get_multilayer_features(clean_t),
                    ).item()
                    success_flags.append(sim_clean < success_sim_threshold)

        return adv_images, success_flags


class CLIPImageAttackMethod(ImageAttackMethod):
    """
    Image-targeted attack method that matches the requested interface.
    """

    def __init__(self, model: TargetModel) -> None:
        super().__init__(model)
        if not isinstance(model, CLIPTargetModel):
            raise TypeError("CLIPImageAttackMethod expects a CLIPTargetModel")

    @staticmethod
    def _strength_to_eps(strength: float, hyperparameters: Optional[Dict]) -> float:
        if strength < 0:
            raise ValueError("strength must be non-negative")
        hp = hyperparameters or {}
        max_eps = float(hp.get("max_eps", 32 / 255))
        if strength > 1.0:
            return min(strength / 255.0, max_eps)
        return min(strength * max_eps, max_eps)

    @staticmethod
    def _pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
        arr = np.array(image.convert("RGB").resize((512, 512))).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    @staticmethod
    def _avg_sim(feats_a, feats_b) -> torch.Tensor:
        sim = 0.0
        for fa, fb in zip(feats_a, feats_b):
            sim += F.cosine_similarity(fa, fb).mean()
        return sim / len(feats_a)

    def attack(
        self,
        clean: List[Image.Image],
        target: List[Image.Image],
        strength: float,
        hyperparameters: Optional[Dict] = None,
    ) -> Tuple[List[Image.Image], List[bool]]:
        assert len(clean) == len(target)
        hp = hyperparameters or {}

        eps = self._strength_to_eps(strength, hp)
        alpha = float(hp.get("alpha", 2 / 255))
        steps = int(hp.get("steps", 300))
        repel_clean = float(hp.get("repel_clean", 0.35))
        success_target_threshold = float(hp.get("success_target_threshold", 0.80))

        adv_images: List[Image.Image] = []
        success_flags: List[bool] = []

        for clean_img, target_img in zip(clean, target):
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp_clean:
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp_target:
                    clean_img.convert("RGB").save(tmp_clean.name)
                    target_img.convert("RGB").save(tmp_target.name)
                    adv_img = self.model.backend.attack(
                        clean_image_path=tmp_clean.name,
                        target_image_path=tmp_target.name,
                        eps=eps,
                        alpha=alpha,
                        steps=steps,
                        repel_clean=repel_clean,
                        mode="targeted_image",
                    )

            adv_images.append(adv_img)

            adv_t = self._pil_to_tensor(adv_img, self.model.device)
            target_t = self._pil_to_tensor(target_img, self.model.device)
            with torch.no_grad():
                sim_target = self._avg_sim(
                    self.model.get_multilayer_features(adv_t),
                    self.model.get_multilayer_features(target_t),
                ).item()
            success_flags.append(sim_target > success_target_threshold)

        return adv_images, success_flags


# ================================================================== #
#  Modal                                                               #
# ================================================================== #

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "transformers", "pillow", "tqdm", "accelerate")
    .add_local_file("xray-fish-profile.png", "/root/clean_image.png")
    .add_local_file("data_viz.png", "/root/data_viz.png")
    .add_local_file("target.jpg", "/root/target.jpg")
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
)


@app.function(gpu="A10G:1", timeout=900)
def attack_test(
    eps: float = 16 / 255,
    alpha: float = 2 / 255,
    steps: int = 300,
    repel_clean: float = 0.35,
    mode: str = "targeted_image",
    clean_image_path: str = "/root/clean_image.png",
    target_question: str = "",
    target_response: str = "",
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
    import os
    import shutil

    attacker = CLIPAttack()
    target_path = "/root/target.jpg" if mode == "targeted_image" else None
    target_text = None
    source_text = None
    if profile == "simple":
        # Minimal full-image setup: no OCR mask and no EOT.
        if mode == "targeted_text_ocr":
            mode = "targeted_text"
        eot_samples = 1
        eot_noise_std = 0.0
        eot_brightness = 0.0
        eot_shift_px = 0

    if mode in {"targeted_text", "targeted_text_ocr"}:
        if target_response:
            target_text = (
                f"Question: {target_question}\nAnswer: {target_response}"
                if target_question
                else target_response
            )
            if source_response:
                source_text = (
                    f"Question: {target_question}\nAnswer: {source_response}"
                    if target_question
                    else source_response
                )
        else:
            raise ValueError("target_response is required for targeted text modes")
    adv_img = attacker.attack(
        clean_image_path=clean_image_path,
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
        mode=mode,
    )
    adv_img.save("/root/images/adversarial_output.png")
    if mode == "targeted_image":
        shutil.copy("/root/target.jpg", "/root/images/target.png")
    else:
        # Ensure eval won't accidentally read a stale target image.
        if os.path.exists("/root/images/target.png"):
            os.remove("/root/images/target.png")
    logger.info("Done.")


@app.local_entrypoint()
def main(
    epsilon: float = 0.0627,
    step_size: float = 0.0078,
    steps: int = 300,
    repel_clean: float = 0.35,
    mode: str = "targeted_image",
    clean_image_path: str = "/root/clean_image.png",
    target_question: str = "",
    target_response: str = "",
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
    attack_test.remote(
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
