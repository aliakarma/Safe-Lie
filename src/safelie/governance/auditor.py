"""SourceAuditor: the M >= 2f+1 checklist, over independence classes.

Report reference: PROJECT_REPORT.md Phase 10 exit criterion — "The
auditor correctly rejects the paper's own N=2, M=5 configuration if the
three ensemble replicas share an independence class"; §2.6 ("the most
transferable artifact ... is the operational criterion: count your
genuinely independent reporting sources M and your tolerable
compromised-source count f; if M < 2f+1, your constraint enforcement has
no robustness guarantee at all").

This tool deliberately does not "pass" a configuration by counting report
IDs. It counts distinct independence classes (failure domains) — the
quantity `safelie.sources.registry.effective_M` computes — because the
report's central engineering finding (W4) is that report count is easy
to satisfy incorrectly and independence class count is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from safelie.utils.config import SourcesConfig


@dataclass(frozen=True)
class AuditResult:
    nominal_m: int
    effective_m: int
    assumed_f: int
    required_m: int
    passes: bool
    independence_classes: list[str]
    warnings: list[str]


def audit_sources(sources: SourcesConfig, assumed_f: int) -> AuditResult:
    nominal_m = sources.M
    effective_m = sources.effective_M
    required_m = 2 * assumed_f + 1
    passes = effective_m >= required_m

    warnings = []
    if nominal_m > effective_m:
        warnings.append(
            f"nominal M={nominal_m} but effective_M={effective_m}: "
            f"{nominal_m - effective_m} report(s) share an independence class "
            f"with another report and do not add independent robustness "
            f"(PROJECT_REPORT.md §13.4, W4)."
        )
    if effective_m == nominal_m and not passes:
        warnings.append(
            "Every report is its own independence class, yet the "
            "configuration still fails M >= 2f+1 -- more reporting sources "
            "or a smaller assumed f is required."
        )
    if passes and required_m == effective_m:
        warnings.append(
            "Configuration passes exactly at the boundary (effective_M == "
            "2f+1); Theorem 2's error bound degrades sharply as f -> M/2 "
            "(PROJECT_REPORT.md §6.7) -- consider provisioning margin."
        )

    return AuditResult(
        nominal_m=nominal_m,
        effective_m=effective_m,
        assumed_f=assumed_f,
        required_m=required_m,
        passes=passes,
        independence_classes=sorted({s.independence_class for s in sources.sources}),
        warnings=warnings,
    )


def format_audit_report(result: AuditResult, label: str = "configuration") -> str:
    lines = [
        f"Source audit for {label}:",
        f"  nominal M              = {result.nominal_m}",
        f"  effective M (classes)  = {result.effective_m}",
        f"  assumed f              = {result.assumed_f}",
        f"  required (2f+1)        = {result.required_m}",
        f"  independence classes   = {', '.join(result.independence_classes)}",
        f"  RESULT                 = {'PASS' if result.passes else 'FAIL'}",
    ]
    for w in result.warnings:
        lines.append(f"  WARNING: {w}")
    if not result.passes:
        lines.append(
            "  Theorem 2's bounded-violation guarantee does NOT apply to this "
            "configuration (PROJECT_REPORT.md §R10.3, §13.4)."
        )
    return "\n".join(lines)
