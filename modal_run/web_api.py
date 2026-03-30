"""
Browser-facing Modal API for chart attack demos.

This app exposes:
- GET /health
- POST /attack
- GET /docs

`/attack` accepts an uploaded chart image plus attack parameters, generates an
adversarial chart, runs the clean and adversarial charts through a VLM, and
returns both answers along with renderable image data URLs.
"""

import base64
import io
import os
import tempfile
from typing import Any

import modal

from modal_run.image import build_web_image

APP_NAME = "agdg-web-api"
GPU_CONFIG = os.environ.get("AGDG_MODAL_GPU", "A10G")
DEFAULT_MODEL_ID = os.environ.get("AGDG_VLM_MODEL_ID", "llava-hf/llava-1.5-7b-hf")
DEFAULT_ATTACKER = os.environ.get("AGDG_ATTACK_METHOD", "targeted_text")
DEFAULT_SURROGATE = os.environ.get("AGDG_ATTACK_SURROGATE", "clip_text_patch")
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5500,http://127.0.0.1:5500,http://[::1]:5500,https://avantitandon.github.io,https://uoft-csc490-w2026.github.io"
)
DEFAULT_WEB_ENV = {
    "AGDG_ALLOWED_ORIGINS": DEFAULT_ALLOWED_ORIGINS,
    "AGDG_FORCE_MOCK": "0",
}

hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
app = modal.App(APP_NAME)

_vlm_bundle: tuple[object, object, str, object] | None = None


def get_allowed_origins(value: str | None = None) -> list[str]:
    raw_value = value if value is not None else os.environ.get("AGDG_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


def _image_to_png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _mock_attack_response(prompt: str, target_response: str, image_bytes: bytes) -> dict[str, Any]:
    original_url = _image_bytes_to_data_url(image_bytes)
    return {
        "mode": "mock",
        "attack_method": DEFAULT_ATTACKER,
        "surrogate": DEFAULT_SURROGATE,
        "evaluation_model": DEFAULT_MODEL_ID,
        "clean_answer": f"[mock-clean] {prompt}",
        "adversarial_answer": f"[mock-adv] {target_response or prompt}",
        "adversarial_image": original_url,
        "original_image": original_url,
    }


def _get_vlm_bundle() -> tuple[object, object, str, object]:
    global _vlm_bundle
    if _vlm_bundle is None:
        from agdg.data_pipeline.clean_response import load_vlm

        _vlm_bundle = load_vlm(DEFAULT_MODEL_ID)
    return _vlm_bundle


def run_attack_pipeline(
    *,
    image_bytes: bytes,
    prompt: str,
    target_response: str,
    source_response: str = "",
    epsilon: float = 16 / 255,
    step_size: float = 2 / 255,
    steps: int = 300,
    attacker: str = DEFAULT_ATTACKER,
    surrogate_model: str = DEFAULT_SURROGATE,
) -> dict[str, Any]:
    if os.environ.get("AGDG_FORCE_MOCK", "0") == "1":
        return _mock_attack_response(prompt, target_response, image_bytes)

    from agdg.attack.attack import generate_adversarial_image
    from agdg.data_pipeline.clean_response import generate_image_response

    processor, model, device, dtype = _get_vlm_bundle()

    with tempfile.NamedTemporaryFile(suffix=".png") as temp_input:
        temp_input.write(image_bytes)
        temp_input.flush()
        adversarial_image = generate_adversarial_image(
            eps=epsilon,
            alpha=step_size,
            steps=steps,
            attacker=attacker,
            model=surrogate_model,
            clean_image_path=temp_input.name,
            target_question=prompt,
            target_response=target_response,
            source_response=source_response,
            device=device,
        )

    adversarial_bytes = _image_to_png_bytes(adversarial_image)
    clean_answer = generate_image_response(
        processor=processor,
        model=model,
        model_id=DEFAULT_MODEL_ID,
        image_bytes=image_bytes,
        prompt=prompt,
        device=device,
        dtype=dtype,
    )
    adversarial_answer = generate_image_response(
        processor=processor,
        model=model,
        model_id=DEFAULT_MODEL_ID,
        image_bytes=adversarial_bytes,
        prompt=prompt,
        device=device,
        dtype=dtype,
    )

    return {
        "mode": "live",
        "attack_method": attacker,
        "surrogate": surrogate_model,
        "evaluation_model": DEFAULT_MODEL_ID,
        "clean_answer": clean_answer,
        "adversarial_answer": adversarial_answer,
        "adversarial_image": _image_bytes_to_data_url(adversarial_bytes),
        "original_image": _image_bytes_to_data_url(image_bytes),
    }


@app.function(
    image=build_web_image(),
    gpu=GPU_CONFIG,
    timeout=3600,
    env=DEFAULT_WEB_ENV,
    volumes={
        "/root/.cache/huggingface": hf_cache_volume,
    },
)
@modal.asgi_app(label="agdg-api")
def web():
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="AGD-G API", version="0.2.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health() -> dict[str, Any]:
        mode = "mock" if os.environ.get("AGDG_FORCE_MOCK", "0") == "1" else "live"
        return {
            "ok": True,
            "mode": mode,
            "allowed_origins": get_allowed_origins(),
            "attack_method": DEFAULT_ATTACKER,
            "surrogate": DEFAULT_SURROGATE,
            "evaluation_model": DEFAULT_MODEL_ID,
        }

    @api.post("/attack")
    async def attack(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        target_response: str = Form(...),
        source_response: str = Form(default=""),
        epsilon: float = Form(default=16 / 255),
        step_size: float = Form(default=2 / 255),
        steps: int = Form(default=300),
        attacker: str = Form(default=DEFAULT_ATTACKER),
        surrogate_model: str = Form(default=DEFAULT_SURROGATE),
    ) -> dict[str, Any]:
        try:
            image_bytes = await image.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Uploaded image is empty.")
            return run_attack_pipeline(
                image_bytes=image_bytes,
                prompt=prompt,
                target_response=target_response,
                source_response=source_response,
                epsilon=epsilon,
                step_size=step_size,
                steps=steps,
                attacker=attacker,
                surrogate_model=surrogate_model,
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - safety net around remote infra
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return api
