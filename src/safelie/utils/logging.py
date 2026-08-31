"""Structured JSONL round logging.

Report reference: §8.2 (schema), smoke test S13.

Every training round produces one JSON record per constraint. The
schema mirrors §8.2 of PROJECT_REPORT.md exactly: raw per-source reports,
the retained set, the aggregate, the applied margin, the multiplier
before/after, and a separate `oracle` block. The `oracle` block must be
written by the evaluator, never the learner — see
`safelie.envs.guards` and `safelie.eval.oracle`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


def _default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class JsonlLogger:
    """Append-only JSONL writer for one run.

    Opened in append mode so a resumed run (checkpoint-restore, S14)
    continues the same log file rather than truncating it.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JsonlLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
