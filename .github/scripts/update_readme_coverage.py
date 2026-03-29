from __future__ import annotations

import json
import sys
from pathlib import Path


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


def build_block(area: str, percent: float) -> str:
    title = area.upper()
    start_marker = f"<!-- coverage:{area}:start -->"
    end_marker = f"<!-- coverage:{area}:end -->"
    rounded = f"{percent:.2f}"
    color = badge_color(percent)
    
    if area == "project":
        title = "PROJECT"
        badge_label = "coverage"
        description = f"Overall automated line coverage: `{rounded}%`"
    else:
        badge_label = f"{area} coverage"
        description = f"`{area}/` automated line coverage: `{rounded}%`"
        
    return "\n".join(
        [
            start_marker,
            f"![{title} coverage](https://img.shields.io/badge/{badge_label.replace(' ', '%20')}-{rounded}%25-{color})",
            "",
            description,
            end_marker,
        ]
    )


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: update_readme_coverage.py <area> <coverage.json> <README.md>")

    area = sys.argv[1]
    coverage_path = Path(sys.argv[2])
    readme_path = Path(sys.argv[3])
    percent = json.loads(coverage_path.read_text())["totals"]["percent_covered"]
    replacement = build_block(area, percent)
    start_marker = f"<!-- coverage:{area}:start -->"
    end_marker = f"<!-- coverage:{area}:end -->"

    readme = readme_path.read_text()
    if start_marker in readme and end_marker in readme:
        start = readme.index(start_marker)
        end = readme.index(end_marker) + len(end_marker)
        updated = readme[:start] + replacement + readme[end:]
    else:
        updated = readme.rstrip() + "\n\n" + replacement + "\n"

    readme_path.write_text(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
