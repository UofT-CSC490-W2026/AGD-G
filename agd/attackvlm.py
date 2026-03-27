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
        super().__init__()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class TextAttackMethod:
    def __init__(self, model: TargetModel) -> None:
        self.model = model

    def attack(self, clean: List[Image.Image], target: List[str], strength: float, hyperparameters: Optional[Dict] = None) -> Tuple[List[Image.Image], List[bool]]:
        assert len(clean) == len(target)
        raise NotImplementedError

class ImageAttackMethod:
    def __init__(self, model: TargetModel) -> None:
        self.model = model

    def attack(self, clean: List[Image.Image], target: List[Image.Image], strength: float, hyperparameters: Optional[Dict] = None) -> Tuple[List[Image.Image], List[bool]]:
        assert len(clean) == len(target)
        raise NotImplementedError

class CLIPAttack:
    def __init__(self, device="cuda", model_id="openai/clip-vit-large-patch14-336"):
        self.device = device
        from transformers import CLIPModel, CLIPProcessor
        
        self.clip_model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_id)
        self.clip_model.requires_grad_(False)

        self.image_size = int(self.clip_model.config.vision_config.image_size)
        self.vision_model = self.clip_model.vision_model

    def _preprocess(self, pixel_tensor):
        px = F.interpolate(pixel_tensor, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=px.device, dtype=px.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=px.device, dtype=px.dtype).view(1, 3, 1, 1)
        return (px - mean) / std

    def get_features(self, pixel_tensor):
        px = self._preprocess(pixel_tensor)
        out = self.vision_model(pixel_values=px, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states

        features = []
        for idx in [-1, -6, -12, -18, -24]:
            h = hidden_states[idx]
            patch_mean = h[:, 1:, :].mean(dim=1)
            patch_mean = patch_mean / patch_mean.norm(p=2, dim=-1, keepdim=True)
            features.append(patch_mean)

        projected = self.clip_model.visual_projection(out.last_hidden_state[:, 0, :])
        projected = projected / projected.norm(p=2, dim=-1, keepdim=True)
        features.append(projected)
        return features

    def get_dense_projected_features(self, pixel_tensor):
        px = self._preprocess(pixel_tensor)
        out = self.vision_model(pixel_values=px, return_dict=True)
        patches = out.last_hidden_state[:, 1:, :]
        projected_patches = self.clip_model.visual_projection(patches)
        projected_patches = projected_patches / projected_patches.norm(p=2, dim=-1, keepdim=True)
        return projected_patches

    def get_target_features(self, image_path):
        target_pil = Image.open(image_path).convert("RGB")
        target_np = np.array(target_pil.resize((512, 512))).astype(np.float32) / 255.0
        target_t = torch.from_numpy(target_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.get_features(target_t)
        return [f.detach() for f in feats]

    def get_text_feature(self, text: str):
        tokenized = self.clip_processor.tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out = self.clip_model.text_model(**tokenized, return_dict=True)
            text_embed = self.clip_model.text_projection(out.pooler_output)
        text_embed = text_embed / text_embed.norm(p=2, dim=-1, keepdim=True)
        return text_embed.detach()

    @staticmethod
    def _eot_augment(x: torch.Tensor, noise_std: float = 0.0, brightness: float = 0.0, shift_px: int = 0) -> torch.Tensor:
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
    def _build_text_like_mask(clean_t: torch.Tensor, edge_quantile: float = 0.88, std_quantile: float = 0.70, dilate_kernel: int = 7, max_area_ratio: float = 0.22) -> torch.Tensor:
        gray = 0.299 * clean_t[:, 0:1, :, :] + 0.587 * clean_t[:, 1:2, :, :] + 0.114 * clean_t[:, 2:3, :, :]
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=gray.dtype, device=gray.device).unsqueeze(1)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=gray.dtype, device=gray.device).unsqueeze(1)
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

    def attack(self, clean_image_path, target_image_path=None, target_text: Optional[str] = None, source_text: Optional[str] = None, eps=16 / 255, alpha=2 / 255, steps=300, repel_clean: float = 0.35, repel_source_text: float = 0.5, eot_samples: int = 1, eot_noise_std: float = 0.0, eot_brightness: float = 0.0, eot_shift_px: int = 0, ocr_edge_quantile: float = 0.88, ocr_std_quantile: float = 0.70, ocr_dilate_kernel: int = 7, ocr_max_area_ratio: float = 0.22, mode: str = "targeted_image"):
        clean_pil = Image.open(clean_image_path).convert("RGB").resize((512, 512))
        clean_np = np.array(clean_pil).astype(np.float32) / 255.0
        clean_t = torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        clean_feats = self.get_features(clean_t)
        perturbation_mask = None
        target_feats = None
        target_text_feat = None
        source_text_feat = None
        
        if mode == "targeted_image":
            if not target_image_path:
                raise ValueError("target_image_path is required for mode=targeted_image")
            target_feats = self.get_target_features(target_image_path)
        elif mode in {"targeted_text", "targeted_text_ocr", "targeted_text_dense"}:
            if not target_text:
                raise ValueError(f"target_text is required for mode={mode}")
            target_text_feat = self.get_text_feature(target_text)
            if source_text:
                source_text_feat = self.get_text_feature(source_text)
            if mode == "targeted_text_ocr":
                perturbation_mask = self._build_text_like_mask(
                    clean_t, edge_quantile=ocr_edge_quantile, std_quantile=ocr_std_quantile, dilate_kernel=ocr_dilate_kernel, max_area_ratio=ocr_max_area_ratio
                )
        elif mode != "untargeted":
            raise ValueError(f"Unknown mode: {mode}")

        delta = torch.empty_like(clean_t).uniform_(-eps, eps)
        delta = (clean_t + delta).clamp(0, 1) - clean_t
        delta.requires_grad_(True)

        best_loss = float("inf")
        best_adv = clean_t.clone()

        for i in tqdm(range(steps), desc="CLIP PGD"):
            adv = (clean_t + delta).clamp(0, 1)
            loss = 0.0
            sim_clean = 0.0
            sim_target = 0.0
            sim_text = 0.0
            sim_source_text = 0.0
            n = max(1, int(eot_samples))
            
            for _ in range(n):
                adv_aug = self._eot_augment(adv, noise_std=eot_noise_std, brightness=eot_brightness, shift_px=eot_shift_px)
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
                elif mode == "targeted_text_dense":
                    adv_dense = self.get_dense_projected_features(adv_aug)
                    sim_matrix = torch.matmul(adv_dense, target_text_feat.transpose(-1, -2)).squeeze(-1)
                    stx = torch.max(sim_matrix, dim=-1)[0].mean()
                    sim_text += stx
                    step_loss = -stx + repel_clean * sc
                    if source_text_feat is not None:
                        source_sim_matrix = torch.matmul(adv_dense, source_text_feat.transpose(-1, -2)).squeeze(-1)
                        ssrc = torch.max(source_sim_matrix, dim=-1)[0].mean()
                        sim_source_text += ssrc
                        step_loss = step_loss + repel_source_text * ssrc
                    loss = loss + step_loss
                else:
                    loss = loss + sc

            loss = loss / n

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

        adv_np = best_adv[0].cpu().permute(1, 2, 0).numpy()
        adv_np = (adv_np * 255).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(adv_np)

modal_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "transformers", "pillow", "tqdm", "accelerate")
    .add_local_file("xray-fish-profile.png", "/root/clean_image.png", copy=True)
    .add_local_file("data_viz.png", "/root/data_viz.png", copy=True)
    .add_local_file("target.jpg", "/root/target.jpg", copy=True)
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
    import os
    import shutil

    attacker = CLIPAttack()
    target_path = "/root/target.jpg" if mode == "targeted_image" else None
    target_text = None
    source_text = None

    if mode in {"targeted_text", "targeted_text_ocr", "targeted_text_dense"}:
        target_text = f"Question: {target_question}\nAnswer: {target_response}" if target_question else target_response
        if source_response:
            source_text = f"Question: {target_question}\nAnswer: {source_response}" if target_question else source_response

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

@app.local_entrypoint()
def main(
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