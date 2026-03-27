from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

from PIL import Image

from agd.attackvlm_adapter import AttackVLMTextAdapter


def _load_chartx_rows(limit: int | None):
    from datasets import load_dataset

    ds = load_dataset("InternScience/ChartX")
    count = 0
    for split_name in ds:
        for row in ds[split_name]:
            yield split_name, row
            count += 1
            if limit is not None and count >= limit:
                return


def _download_chartx_zip() -> Path:
    from huggingface_hub import hf_hub_download

    zip_path = hf_hub_download(
        repo_id="InternScience/ChartX",
        filename="ChartX_png.zip",
        repo_type="dataset",
    )
    return Path(zip_path)


def _load_image_from_zip(zip_file: zipfile.ZipFile, image_path: str) -> Image.Image:
    normalized = image_path.lstrip("./")
    archive_path = f"ChartX_png/{normalized}"
    with zip_file.open(archive_path) as handle:
        image_bytes = handle.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _resolve_targets(
    *,
    mode: str,
    question: str,
    answer: str,
    untargeted_placeholder: str,
    target_answer: str | None,
) -> str:
    if mode == "untargeted":
        return untargeted_placeholder
    if target_answer is None:
        raise ValueError("--target-answer is required for targeted_text mode")
    return target_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/attackvlm_chartx"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["untargeted", "targeted_text"], default="untargeted")
    parser.add_argument("--target-answer", default=None)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--max-eps", type=float, default=32 / 255)
    parser.add_argument("--profile", choices=["simple", "full"], default="simple")
    parser.add_argument(
        "--untargeted-placeholder",
        default="",
        help="Dummy target text passed to the interface in untargeted mode.",
    )
    args = parser.parse_args()

    hyperparameters = {
        "mode": args.mode,
        "steps": args.steps,
        "alpha": args.alpha,
        "max_eps": args.max_eps,
        "profile": args.profile,
    }
    if args.mode == "targeted_text":
        hyperparameters["question"] = None

    attack = AttackVLMTextAdapter(
        device=args.device,
        strength=args.strength,
        hyperparameters=hyperparameters,
    )

    output_dir = args.output_dir
    originals_dir = output_dir / "clean"
    adversarial_dir = output_dir / "adversarial"
    metadata_path = output_dir / "metadata.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = _download_chartx_zip()

    with zipfile.ZipFile(zip_path, "r") as archive, metadata_path.open("w", encoding="utf-8") as metadata_file:
        for index, (split_name, row) in enumerate(_load_chartx_rows(args.limit)):
            question = row["QA"]["input"]
            answer = row["QA"]["output"]
            image = _load_image_from_zip(archive, row["img"])
            target = _resolve_targets(
                mode=args.mode,
                question=question,
                answer=answer,
                untargeted_placeholder=args.untargeted_placeholder,
                target_answer=args.target_answer,
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
                "mode": args.mode,
                "success": success,
                "clean_path": str(clean_path),
                "adversarial_path": str(adv_path),
            }
            metadata_file.write(json.dumps(record) + "\n")
            metadata_file.flush()
            print(f"[{stem}] success={success} question={question!r}")


if __name__ == "__main__":
    main()
