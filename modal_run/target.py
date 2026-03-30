"""
Modal wrapper for the target-generation pipeline.
Generates target captions for clean charts using a targeting strategy.

    # Generate targets for all charts
    modal run modal_run/target.py -- --strategy qwen

    # Generate targets for a specific dataset source
    modal run modal_run/target.py -- --strategy qwen --source ChartBench

    # Preview: sample 10 images per dataset, print thinking trace + answer
    modal run modal_run/target.py -- --preview
    modal run modal_run/target.py -- --preview --per-source 5
"""
import modal
from modal_run.image import build_data_pipeline_image

from agdg.data_pipeline.target_response import generate_target_responses, preview_target_responses

app = modal.App(
    "agd-target-pipeline",
    image=build_data_pipeline_image(),
    secrets=[
        modal.Secret.from_name("aws"),
        # modal.Secret.from_name("aws-rds"),
        modal.Secret.from_name("huggingface"),
    ],
)


@app.function(timeout=7200, gpu="H100:1")
def generate_targets(
    max_rows: int = 0,
    strategy: str = "qwen",
    source: str | None = None,
    batch_size: int = 100,
):
    return generate_target_responses(
        targeting_strategy=strategy,
        max_rows=max_rows,
        source=source,
        batch_size=batch_size,
    )


@app.function(timeout=3600, gpu="H100:1")
def preview_targets(
    per_source: int = 10,
    strategy: str = "qwen",
):
    return preview_target_responses(
        targeting_strategy=strategy,
        per_source=per_source,
    )


@app.function(timeout=60)
def fix_bad_targets(strategy: str = "qwen"):
    from agdg.data_pipeline.aws.rds import delete_bad_targets
    return delete_bad_targets(strategy)


@app.local_entrypoint()
def main(
    limit: int = 0,
    strategy: str = "qwen",
    source: str | None = None,
    batch_size: int = 100,
    preview: bool = False,
    per_source: int = 10,
    fix_bad: bool = False,
):
    if fix_bad:
        deleted = fix_bad_targets.remote(strategy=strategy)
        print(f"Deleted {deleted} bad target_answers for strategy={strategy!r}")
        return

    if preview:
        results = preview_targets.remote(per_source=per_source, strategy=strategy)
        for r in results:
            print(f"[{r['source']}] Clean:   {r['clean_answer']}")
            print(f"  Thinking: {r['thinking']}")
            print(f"  Target:   {r['target']}\n")
    else:
        r = generate_targets.remote(
            max_rows=limit,
            strategy=strategy,
            source=source,
            batch_size=batch_size,
        )
        for k, v in r.items():
            print(f"  {k}: {v}")
