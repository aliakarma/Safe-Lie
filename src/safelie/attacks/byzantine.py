"""The Byzantine (equivocating) consistency axis.

Report reference: PROJECT_REPORT.md §14.3 item 7; Appendix B gap G15 —
"declared in the taxonomy, never exercised in the [paper's] protocol."

A consistent source reports the same value to every peer; a Byzantine
source may equivocate, reporting a *different* value to each recipient.
This requires the report value to be a function of the receiving peer,
not a scalar — most naive source-layer implementations get this wrong by
storing one value per source (the report warns most builders will do
exactly this). This module provides the per-recipient interface so the
source layer is *designed* to allow it, per decision D3's scope note in
§R5.2 ("the source layer should still be designed to allow per-recipient
values").

`[DECISION]` This attack is implemented but **not wired into the compact
training loop** (`safelie.training.loop`): the compact matrix (§R2.1) uses
the persistent static under-reporting attack only, and Byzantine
equivocation is explicitly out of scope for Stage 2 (§R5.2, gap G15 —
"Not exercised in the pilot"). It is available as a Stage-3 extension
(§R9 item 14) without further source-layer changes.
"""

from __future__ import annotations


def byzantine_attack(
    residuals: dict[str, float],
    corrupted: set[str],
    recipients: list[str],
    k: int,
    B: float,
    rng,
) -> dict[str, dict[str, float]]:
    """Per-recipient corrupted residuals.

    Returns ``{source_id: {recipient_id: reported_value}}``. Honest
    sources report the same value to every recipient (consistent);
    corrupted sources may report a different value per recipient, each
    perturbed by up to `B` in either direction — a strictly more general
    threat than the consistent static attack, since a Byzantine source's
    reports need not even be mutually consistent enough to average
    sensibly against each other.
    """
    out: dict[str, dict[str, float]] = {}
    for sid, g in residuals.items():
        if sid in corrupted:
            out[sid] = {r: g + float(rng.uniform(-B, B)) for r in recipients}
        else:
            out[sid] = {r: g for r in recipients}
    return out
