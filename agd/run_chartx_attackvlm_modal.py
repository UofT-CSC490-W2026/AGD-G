from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import modal


app = modal.App("attackvlm-chartx")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "huggingface_hub",
        "pillow",
        "numpy",
        "tqdm",
        "accelerate",
        "sentencepiece",
    )
    .add_local_dir("agd", "/root/project/agd")
)

hf_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("attackvlm-chartx-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G:1",
    timeout=7200,
    memory=32768,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        "/root/.cache/huggingface": hf_volume,
        "/root/outputs": output_volume,
    },
)
def generate_chartx_attackvlm(
    limit: int = 5,
    device: str = "cuda",
    mode: str = "untargeted",
    target_answer: str = "",
    strength: float = 1.0,
    steps: int = 300,
    alpha: float = 2 / 255,
    max_eps: float = 32 / 255,
    profile: str = "simple",
    output_subdir: str = "attackvlm_chartx",
):
    import sys

    sys.path.insert(0, "/root/project")

    from agd.run_chartx_attackvlm import (
        _download_chartx_zip,
        _load_chartx_rows,
        _load_image_from_zip,
        _resolve_targets,
        _save_image,
    )
    from agd.attackvlm_adapter import AttackVLMTextAdapter

    output_dir = Path("/root/outputs") / output_subdir
    originals_dir = output_dir / "clean"
    adversarial_dir = output_dir / "adversarial"
    metadata_path = output_dir / "metadata.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    hyperparameters = {
        "mode": mode,
        "steps": steps,
        "alpha": alpha,
        "max_eps": max_eps,
        "profile": profile,
    }
    if mode == "targeted_text":
        hyperparameters["question"] = None

    attack = AttackVLMTextAdapter(
        device=device,
        strength=strength,
        hyperparameters=hyperparameters,
    )

    written = 0
    zip_path = _download_chartx_zip()

    with zipfile.ZipFile(zip_path, "r") as archive, metadata_path.open("w", encoding="utf-8") as metadata_file:
        for index, (split_name, row) in enumerate(_load_chartx_rows(limit)):
            question = row["QA"]["input"]
            answer = row["QA"]["output"]
            image = _load_image_from_zip(archive, row["img"])
            target = _resolve_targets(
                mode=mode,
                question=question,
                answer=answer,
                untargeted_placeholder="",
                target_answer=target_answer or None,
            )

            result = attack.attack([image], [target])
            adversarial = result.adversarial[0]
            success = result.success[0]

            stem = f"{index:05d}"
            clean_path = originals_dir / f"{stem}.png"
            adv_path = adversarial_dir / f"{stem}.png"
            _save_image(image, clean_path)
            _save_image(adversarial, adv_path)

            record = {
                "id": stem,
                "split": split_name,
                "chart_type": row["chart_type"],
                "question": question,
                "answer": answer,
                "target": target,
                "mode": mode,
                "success": success,
                "clean_path": str(clean_path),
                "adversarial_path": str(adv_path),
            }
            metadata_file.write(json.dumps(record) + "\n")
            metadata_file.flush()
            written += 1
            print(f"[{stem}] success={success} question={question!r}")

    output_volume.commit()
    return {
        "written": written,
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
    }


@app.local_entrypoint()
def main(
    limit: int = 5,
    mode: str = "untargeted",
    target_answer: str = "",
    strength: float = 1.0,
    steps: int = 300,
    alpha: float = 2 / 255,
    max_eps: float = 32 / 255,
    profile: str = "simple",
    output_subdir: str = "attackvlm_chartx",
):
    result = generate_chartx_attackvlm.remote(
        limit=limit,
        device="cuda",
        mode=mode,
        target_answer=target_answer,
        strength=strength,
        steps=steps,
        alpha=alpha,
        max_eps=max_eps,
        profile=profile,
        output_subdir=output_subdir,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
