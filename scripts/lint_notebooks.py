#!/usr/bin/env python
"""CI lint enforcing decision D13 (PROJECT_REPORT.md §R7.4).

"The notebook is an execution and orchestration layer only ... a CI lint
parses every notebook and fails if any code cell contains a `def` or
`class` statement, or defines a model, an aggregator, or a training loop.
Permitted in cells: imports, shell commands, config-name strings, calls
into `safelie`, and display code."

Notebook-only code is invisible to code review, untested by CI, and
unreachable from the local smoke suite -- a result produced by a function
defined in a notebook cell cannot be reproduced by running the
repository.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def check_notebook(path: Path) -> list[str]:
    violations = []
    data = json.loads(path.read_text(encoding="utf-8"))
    for i, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = _cell_source(cell)
        # Strip IPython magics/shell escapes before parsing as Python.
        lines = [ln for ln in source.splitlines() if not ln.strip().startswith(("!", "%"))]
        clean_source = "\n".join(lines)
        if not clean_source.strip():
            continue
        try:
            tree = ast.parse(clean_source)
        except SyntaxError as exc:
            violations.append(f"{path.name} cell {i}: could not parse ({exc})")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                violations.append(
                    f"{path.name} cell {i}: defines `{node.name}` -- notebooks may only "
                    f"orchestrate calls into safelie, per decision D13"
                )
    return violations


def main() -> int:
    if not NOTEBOOK_DIR.exists():
        print("No notebooks/ directory found; nothing to lint.")
        return 0
    all_violations: list[str] = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        all_violations.extend(check_notebook(path))

    if all_violations:
        print("Notebook lint FAILED (decision D13 -- orchestration only):")
        for v in all_violations:
            print(f"  - {v}")
        return 1
    print("Notebook lint passed: no def/class found in any notebook cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
