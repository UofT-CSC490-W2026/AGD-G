"""
Modal wrapper for evaluation and explanation pipelines.

    modal run modal_run/evaluate.py
    modal run modal_run/evaluate.py -- --mode explain --limit 100
"""
import modal
from modal_run.image import build_data_pipeline_image

from agdg.data_pipeline.eval import evaluate_all
from agdg.data_pipeline.explain import explain_all

app = modal.App(
    "agd-evaluate",
    image=build_data_pipeline_image(),
    secrets=[
        modal.Secret.from_name("aws"),
    ],
)


@app.function(timeout=3600, memory=4096)
def evaluate(max_rows: int = 0):
    return evaluate_all(max_rows=max_rows)


@app.function(timeout=3600, memory=4096)
def explain(max_rows: int = 0):
    return explain_all(max_rows=max_rows)


@app.local_entrypoint()
def main(mode: str = "evaluate", limit: int = 0):
    if mode == "evaluate":
        r = evaluate.remote(max_rows=limit)
    elif mode == "explain":
        r = explain.remote(max_rows=limit)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'evaluate' or 'explain')")
    for k, v in r.items():
        print(f"  {k}: {v}")
