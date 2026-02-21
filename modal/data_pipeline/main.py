#!/usr/bin/env python3
"""
Run all dataset imports and preprocessing.
"""
import argparse
import subprocess
import sys
import time


IMPORTERS = [
    ("ChartBench",  "import_chartbench.py"),
    ("ChartX",      "import_chartx.py"),
    ("ChartQA-X",   "import_chartqax.py"),
]

PREPROCESSOR = "preprocess_charts.py"


def run(cmd: list[str], label: str) -> bool:
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {label} ({elapsed:.1f}s)\n")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all dataset imports and preprocessing")
    parser.add_argument("-l", "--limit", type=int, default=None,
                        help="Max rows per dataset (default: all)")
    parser.add_argument("-c", "--clean", action="store_true",
                        help="Wipe S3 + RDS before first import")
    parser.add_argument("--skip-import", action="store_true",
                        help="Skip imports, run preprocess only")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing, run imports only")
    args = parser.parse_args()

    results = {}

    # ── Clean (must finish before imports) ──────────────────────────
    if not args.skip_import and args.clean:
        ok = run(["modal", "run", "clean.py"], "Clean S3 + RDS")
        results["clean"] = ok
        if not ok:
            print("Clean failed — aborting")
            sys.exit(1)

    # ── Imports (parallel) ────────────────────────────────────────────
    if not args.skip_import:
        procs = {}
        for name, script in IMPORTERS:
            cmd = ["modal", "run", script, "--"]
            if args.limit is not None:
                cmd += ["-l", str(args.limit)]

            print(f"\n  Starting {name}...")
            procs[name] = (subprocess.Popen(cmd), time.time())

        for name, (proc, start) in procs.items():
            proc.wait()
            elapsed = time.time() - start
            ok = proc.returncode == 0
            results[name] = ok
            print(f"  {'Passed' if ok else 'Failed'} {name} ({elapsed:.1f}s)")

    # Preprocess
    if not args.skip_preprocess:
        ok = run(["modal", "run", PREPROCESSOR], "Preprocess all")
        results["preprocess"] = ok

    # Summary
    print("  Summary")
    for name, ok in results.items():
        print(f"  {'Passed' if ok else 'Failed'} {name}")

    failed = sum(1 for ok in results.values() if not ok)
    if failed:
        print(f"\n  {failed} step(s) failed")
        sys.exit(1)
    else:
        print(f"\n  All steps passed")


if __name__ == "__main__":
    main()
