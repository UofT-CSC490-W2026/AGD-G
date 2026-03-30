"""
CLI entrypoint for running the AGD-G pipeline locally (without Modal).

    python -m agdg.cli ingest [--clean] [--limit N] [--skip-import] [--skip-preprocess]
    python -m agdg.cli target --strategy qwen [--source ChartBench] [--max-rows N]
    python -m agdg.cli target --preview [--per-source N] [--strategy qwen]
    python -m agdg.cli attack [--limit N] [--method targeted_text] [--surrogate llava]
    python -m agdg.cli evaluate [--limit N] [--strategy S]
    python -m agdg.cli clean-responses [--limit N] [--model MODEL_ID]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def _load_dotenv():
    """Load .env.local if present (python-dotenv is already a dependency)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for name in (".env.local", ".env"):
        p = Path(name)
        if p.is_file():
            load_dotenv(p, override=False)
            break


def _cmd_ingest(args):
    from agdg.data_pipeline.clean import clean as do_clean
    from agdg.data_pipeline.import_chartbench import import_chartbench
    from agdg.data_pipeline.import_chartx import import_chartx
    from agdg.data_pipeline.import_chartqax import import_chartqax
    from agdg.data_pipeline.preprocess_charts import preprocess_all

    log = logging.getLogger("ingest")

    if args.clean:
        log.info("=== Clean ===")
        do_clean()

    if not args.skip_import:
        log.info("=== Import ChartBench ===")
        import_chartbench(max_rows=args.limit)

        log.info("=== Import ChartX ===")
        import_chartx(max_rows=args.limit or None)

        log.info("=== Import ChartQA-X ===")
        import_chartqax(max_rows=args.limit or None)

    if not args.skip_preprocess:
        log.info("=== Preprocess ===")
        result = preprocess_all()
        log.info("Preprocess result: %s", result)

    log.info("=== Ingest complete ===")


def _cmd_target(args):
    from agdg.data_pipeline.target_response import (
        generate_target_responses,
        preview_target_responses,
    )

    if args.preview:
        results = preview_target_responses(
            targeting_strategy=args.strategy,
            per_source=args.per_source,
        )
        for r in results:
            print(f"[{r['source']}] Clean:   {r['clean_answer']}")
            print(f"  Thinking: {r['thinking']}")
            print(f"  Target:   {r['target']}\n")
    else:
        r = generate_target_responses(
            targeting_strategy=args.strategy,
            max_rows=args.max_rows,
            source=args.source,
            batch_size=args.batch_size,
        )
        for k, v in r.items():
            print(f"  {k}: {v}")


def _cmd_attack(args):
    from agdg.data_pipeline.attack import attack_all

    r = attack_all(
        method=args.method,
        surrogate=args.surrogate,
        max_rows=args.limit,
        steps=args.steps,
        target_strategy=args.strategy,
    )
    for k, v in r.items():
        print(f"  {k}: {v}")


def _cmd_evaluate(args):
    from agdg.data_pipeline.eval import evaluate_all

    r = evaluate_all(max_rows=args.limit, target_strategy=args.strategy)
    for k, v in r.items():
        print(f"  {k}: {v}")


def _cmd_clean_responses(args):
    from agdg.data_pipeline.clean_response import generate_clean_responses

    r = generate_clean_responses(max_rows=args.limit, model_id=args.model)
    for k, v in r.items():
        print(f"  {k}: {v}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agdg",
        description="AGD-G adversarial chart pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Import datasets and preprocess charts")
    p_ingest.add_argument("--clean", action="store_true", help="Wipe DB and S3 before importing")
    p_ingest.add_argument("--limit", type=int, default=0, help="Max rows per dataset (0 = all)")
    p_ingest.add_argument("--skip-import", action="store_true", help="Skip dataset imports")
    p_ingest.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")

    # --- target ---
    p_target = sub.add_parser("target", help="Generate target captions")
    p_target.add_argument("--strategy", default="qwen", help="Targeting strategy (default: qwen)")
    p_target.add_argument("--source", default=None, help="Restrict to dataset source, e.g. ChartBench")
    p_target.add_argument("--max-rows", type=int, default=0, help="Max rows (0 = all)")
    p_target.add_argument("--batch-size", type=int, default=100, help="DB commit batch size")
    p_target.add_argument("--preview", action="store_true", help="Preview mode (no DB writes)")
    p_target.add_argument("--per-source", type=int, default=10, help="Samples per source in preview mode")

    # --- attack ---
    p_attack = sub.add_parser("attack", help="Run adversarial attacks")
    p_attack.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    p_attack.add_argument("--method", default="targeted_text", help="Attack method")
    p_attack.add_argument("--surrogate", default="llava", help="Surrogate model")
    p_attack.add_argument("--steps", type=int, default=300, help="Attack optimisation steps")
    p_attack.add_argument("--strategy", default=None, help="Filter by target strategy")

    # --- evaluate ---
    p_evaluate = sub.add_parser("evaluate", help="Evaluate adversarial charts")
    p_evaluate.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    p_evaluate.add_argument("--strategy", default=None, help="Filter by target strategy")

    # --- clean-responses ---
    p_clean = sub.add_parser("clean-responses", help="Generate clean VLM captions")
    p_clean.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    p_clean.add_argument("--model", default="llava-hf/llava-1.5-7b-hf", help="VLM model ID")

    return parser


_DISPATCH = {
    "ingest": _cmd_ingest,
    "target": _cmd_target,
    "attack": _cmd_attack,
    "evaluate": _cmd_evaluate,
    "clean-responses": _cmd_clean_responses,
}


def main(argv: list[str] | None = None):
    _load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)
    _DISPATCH[args.command](args)


if __name__ == "__main__":
    main()
