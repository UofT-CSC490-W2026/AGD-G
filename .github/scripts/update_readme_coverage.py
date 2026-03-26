from __future__ import annotations

import json
import sys
from pathlib import Path


START_MARKER = "<!-- coverage:modal:start -->"
END_MARKER = "<!-- coverage:modal:end -->"


def badge_color(percent: float) -> str:
    if percent >= 90:
        return "brightgreen"
    if percent >= 80:
        return "green"
    if percent >= 70:
        return "yellow"
    if percent >= 60:
        return "orange"
    return "red"


def build_block(percent: float) -> str:
    rounded = f"{percent:.2f}"
    color = badge_color(percent)
    return "\n".join(
        [
            START_MARKER,
            f"![Modal coverage](https://img.shields.io/badge/modal%20coverage-{rounded}%25-{color})",
            "",
            "| Area | Automation | Latest status |",
            "| --- | --- | --- |",
            f"| `modal/` | `pytest` + `pytest-cov` | `{rounded}%` line coverage |",
            "| `terraform/` | `terraform fmt -check` + `terraform validate` | Coverage not applicable until Terraform tests are added |",
            END_MARKER,
        ]
    )


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: update_readme_coverage.py <coverage.json> <README.md>")

    coverage_path = Path(sys.argv[1])
    readme_path = Path(sys.argv[2])
    percent = json.loads(coverage_path.read_text())["totals"]["percent_covered"]
    replacement = build_block(percent)

    readme = readme_path.read_text()
    if START_MARKER in readme and END_MARKER in readme:
        start = readme.index(START_MARKER)
        end = readme.index(END_MARKER) + len(END_MARKER)
        updated = readme[:start] + replacement + readme[end:]
    else:
        updated = readme.rstrip() + "\n\n## CI Status\n\n" + replacement + "\n"

    readme_path.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
