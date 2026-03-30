"""
Modal wrapper for evaluation and clean response generation.

    modal run modal_run/evaluate.py --mode clean --limit 100
    modal run modal_run/evaluate.py --mode evaluate --limit 100
"""
import modal
from modal_run.image import build_data_pipeline_image

from agdg.data_pipeline.eval import evaluate_all
from agdg.data_pipeline.clean_response import generate_clean_responses

app = modal.App(
    "agd-evaluate",
    image=build_data_pipeline_image(),
    secrets=[
        modal.Secret.from_name("aws")
    ],
)


@app.function(timeout=3600, gpu="A10G:1")
def evaluate(max_rows: int = 0):
    return evaluate_all(max_rows=max_rows)


@app.function(timeout=3600, gpu="A10G:1")
def generate_clean(max_rows: int = 0, model_id: str = "llava-hf/llava-1.5-7b-hf"):
    return generate_clean_responses(max_rows=max_rows, model_id=model_id)




@app.local_entrypoint()
def main(mode: str = "evaluate", limit: int = 0, model: str = "llava-hf/llava-1.5-7b-hf"):
    if mode == "evaluate":
        r = evaluate.remote(max_rows=limit)
    elif mode == "clean":
        r = generate_clean.remote(max_rows=limit, model_id=model)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'evaluate', 'clean', or 'target')")
    
    for k, v in r.items():
        print(f"  {k}: {v}")
