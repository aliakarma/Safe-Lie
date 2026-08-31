"""The ground-truth attack ledger.

Report reference: PROJECT_REPORT.md Phase 5 exit criterion; smoke test
S19 — "The ledger reconstructs sum_k 1^T delta_k to 1e-12, names exactly
the corrupted source IDs, and matches the difference between corrupted
and uncorrupted residual streams round by round."
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LedgerEntry:
    round_k: int
    corrupted_source_ids: list[str]
    delta_applied: dict[str, float]


@dataclass
class AttackLedger:
    entries: list[LedgerEntry] = field(default_factory=list)

    def record(self, round_k: int, original: dict[str, float], corrupted_output: dict[str, float]) -> LedgerEntry:
        """Diff the pre- and post-attack residual streams and log exactly
        what changed. This is derived from the two streams, never from the
        attack operator's internal bookkeeping, so it cannot drift from
        what was actually applied (the property S19 checks)."""
        delta = {
            sid: corrupted_output[sid] - original[sid]
            for sid in original
            if abs(corrupted_output[sid] - original[sid]) > 0.0
        }
        entry = LedgerEntry(round_k=round_k, corrupted_source_ids=list(delta.keys()), delta_applied=delta)
        self.entries.append(entry)
        return entry

    def total_mass(self) -> float:
        """sum_k 1^T delta_k across the whole run."""
        return sum(sum(e.delta_applied.values()) for e in self.entries)

    def total_mass_per_source(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for e in self.entries:
            for sid, d in e.delta_applied.items():
                totals[sid] = totals.get(sid, 0.0) + d
        return totals

    def corrupted_source_ids(self) -> set[str]:
        ids: set[str] = set()
        for e in self.entries:
            ids.update(e.corrupted_source_ids)
        return ids
