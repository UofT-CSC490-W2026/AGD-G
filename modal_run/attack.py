"""
Modal wrapper for the adversarial-attack pipeline.
Applies the AGD attack to preprocessed charts in the database.

    modal run modal_run/attack.py
    modal run modal_run/attack.py -- --limit 50
"""
import modal
from modal_run.image import build_data_pipeline_image

from agdg.data_pipeline.attack import attack_all

app = modal.App(
    "agd-attack-pipeline",
    image=build_data_pipeline_image(),
    secrets=[
        modal.Secret.from_name("aws"),
    ],
)


@app.function(timeout=7200, memory=8192, gpu="A10G:1")
def attack(
    max_rows: int = 0,
    method: str = "targeted_text",
    surrogate: str = "llava",
    steps: int = 300,
    strategy: str | None = None,
):
    return attack_all(
        method=method,
        surrogate=surrogate,
        max_rows=max_rows,
        steps=steps,
        target_strategy=strategy,
    )


@app.local_entrypoint()
def main(
    limit: int = 0,
    method: str = "targeted_text",
    surrogate: str = "llava",
    steps: int = 300,
    strategy: str | None = None,
):
    r = attack.remote(
        max_rows=limit,
        method=method,
        surrogate=surrogate,
        steps=steps,
        strategy=strategy,
    )
    for k, v in r.items():
        print(f"  {k}: {v}")
