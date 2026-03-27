from __future__ import annotations

import json

from agd.eval_chartx_attackvlm_modal import evaluate_chartx_attackvlm
from agd.run_chartx_attackvlm_modal import generate_chartx_attackvlm


def run_pipeline(
    *,
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
    generation = generate_chartx_attackvlm.remote(
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
    evaluation = evaluate_chartx_attackvlm.remote(
        input_subdir=output_subdir,
        max_rows=limit,
    )
    return {
        "generation": generation,
        "evaluation": evaluation,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
