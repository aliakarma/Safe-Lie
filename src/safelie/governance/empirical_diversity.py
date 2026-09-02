"""Empirical source-diversity diagnostic (P0 #9).

Report reference: this repository's own `safelie.governance.auditor`
checks *declared* independence -- whether each `SourceSpec` in a config
carries a distinct `independence_class` string. That is a necessary
accounting exercise (Theorem 2 needs the paper's own M >= 2f+1 to be
counted over genuinely separate failure domains, not over correlated
replicas -- see W4 / decision D10), but a declared label is a claim about
*how a source is deployed* (a different agent, a different process), not
a measurement of whether its *errors* actually behave independently of
the other sources'. Two sources with different declared classes can
still have highly correlated estimation error if they are ultimately
regressed toward the same underlying training signal.

This module never modifies `effective_M` or `SourceAuditor`; it adds a
second, independent, measured quantity computed from a completed run's
own logs:

  - Per-source **bias**, **MAE**, **RMSE**, and **correlation with true
    cost** against the withheld oracle's `true_cost_return` (never a
    quantity the learner could see) -- one number set per (owner,
    source) pair.
  - **Cross-source error correlation**, per owner: the M x M correlation
    matrix of each source's (predicted - true) error series across
    rounds.
  - **Statistical effective diversity**: the *participation ratio* of
    that correlation matrix's eigenvalues,
    `PR = (sum(eigenvalues))^2 / sum(eigenvalues^2) = M^2 / sum(eigenvalues^2)`
    (a correlation matrix's trace, and hence eigenvalue sum, is always
    M). This is a standard measure of "effective number of independent
    components" (participation ratio / inverse Simpson index on a
    spectrum): it equals M when the error correlation matrix is the
    identity (every source's error genuinely independent of every
    other's) and approaches 1 as the sources' errors become collinear.
    It is *not* the same quantity as `safelie.sources.registry.
    effective_M` (which counts declared independence classes) and this
    module never conflates the two: `EmpiricalDiversityReport` reports
    both, side by side, under different names.

The scientific question this answers: are the M sources actually
estimating the same J_C^i, with errors that behave as advertised by their
independence-class labels? Assumption 1(ii) (source independence) should
not be asserted merely because a config passed `SourceAuditor` -- it
should be asserted, if at all, because this diagnostic's
`statistical_effective_m` is close to the nominal `effective_M`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from safelie.utils.logging import read_jsonl

# Below this many rounds, a pairwise Pearson correlation is not a
# reliable diversity measurement in either direction: at n=2 every pair
# of series has |r|=1 by construction (any two points define a line),
# so the diagnostic cannot even distinguish independent sources from
# collinear ones, let alone measure a graded degree of correlation.
# This is deliberately a simple round-number gate, matching this
# repository's existing convention for small-sample warnings
# (`safelie.defenses.rce`'s `min_retained`, `safelie.analysis.stats`'
# `MIN_SEEDS_FOR_INFERENCE`), not a formal power calculation.
MIN_ROUNDS_FOR_RELIABLE_CORRELATION = 30


@dataclass(frozen=True)
class SourceDiagnostic:
    """One (owner, source) pair's measured estimation quality against the
    withheld oracle's true cost return, over every round both logs cover.
    `predicted` is the source's PRE-attack, PRE-aggregation report value
    (`rounds.jsonl`'s `reports` field) -- the source's own estimate of
    J_C^i, not anything the attack or the aggregator has touched."""

    owner: str
    source_id: str
    source_type: str
    independence_class: str
    n_rounds: int
    mean_predicted: float
    mean_true: float
    bias: float  # mean(predicted - true); >0 over-reports, <0 under-reports
    mae: float
    rmse: float
    correlation_with_true: float  # Pearson r(predicted, true) across rounds


@dataclass(frozen=True)
class EmpiricalDiversityReport:
    run_dir: str
    n_rounds_used: int
    nominal_m: int
    nominal_effective_m: int  # declared independence classes (safelie.sources.registry.effective_M)
    per_source: list[SourceDiagnostic]
    cross_source_error_correlation: dict[str, dict[str, float]]  # owner -> "source_a|source_b" -> r
    statistical_effective_m_by_owner: dict[str, float]  # participation ratio, per owner
    statistical_effective_m_overall: float  # min across owners (worst case, conservative)
    assumption_1ii_supported: bool  # statistical_effective_m_overall close to nominal_effective_m
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_source"] = [asdict(s) for s in self.per_source]
        return d


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _participation_ratio(corr: np.ndarray) -> float:
    """PR = (sum eigenvalues)^2 / sum(eigenvalues^2). For a correlation
    matrix, sum(eigenvalues) == trace == M always, so PR = M^2 /
    sum(eigenvalues^2). Ranges [1, M]: 1 when every off-diagonal entry is
    +-1 (fully collinear errors), M when the matrix is the identity
    (fully independent errors)."""
    m = corr.shape[0]
    if m <= 1:
        return float(m)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)  # numerical noise can give tiny negative eigenvalues
    denom = float(np.sum(eigenvalues**2))
    if denom <= 0:
        return float(m)
    return float((np.sum(eigenvalues) ** 2) / denom)


def compute_empirical_diversity(
    run_dir: str | Path,
    nominal_m: int | None = None,
    nominal_effective_m: int | None = None,
    tolerance: float = 0.5,
    source_specs: dict[str, tuple[str, str]] | None = None,
) -> EmpiricalDiversityReport:
    """`source_specs`: optional `{source_id: (source_type,
    independence_class)}`, e.g. built from the run's own
    `ExperimentConfig.sources.sources` as
    `{s.source_id: (s.source_type, s.independence_class) for s in cfg.sources.sources}`.
    Without it, `SourceDiagnostic.source_type`/`.independence_class` are
    "unknown" -- the diagnostic still runs, since the empirical
    error/correlation measurements do not depend on the declared labels
    at all, only the report's own presentation of them does.
    """
    """Compute the full diagnostic from one completed run's `rounds.jsonl`
    + `oracle.jsonl`. `nominal_m`/`nominal_effective_m` are read from the
    run's own logs when not given (M = number of distinct source_ids in
    round 0's `reports`; effective_M is NOT recoverable from logs alone,
    since `independence_class` isn't logged per-report -- pass it
    explicitly from the run's `SourcesConfig` when available, e.g. via
    `safelie.sources.registry.effective_M(cfg.sources.sources)`).
    """
    run_dir = Path(run_dir)
    rounds = read_jsonl(run_dir / "rounds.jsonl")
    oracle = read_jsonl(run_dir / "oracle.jsonl")
    if not rounds or not oracle:
        raise ValueError(f"{run_dir}: rounds.jsonl and oracle.jsonl must both be non-empty")

    rounds_by_k = {r["round_k"]: r for r in rounds}
    oracle_by_k = {o["round_k"]: o for o in oracle}
    common_ks = sorted(set(rounds_by_k) & set(oracle_by_k))
    if not common_ks:
        raise ValueError(f"{run_dir}: no round_k common to both logs")

    agent_ids = list(rounds_by_k[common_ks[0]]["constraints"].keys())
    source_meta = {
        r["source_id"]: r for r in rounds_by_k[common_ks[0]]["constraints"][agent_ids[0]]["reports"]
    }
    # source_type/independence_class are not logged per-report (only
    # source_id + value); recover them from the config if the caller
    # can supply it, else fall back to "unknown" rather than guessing.
    source_ids = list(source_meta.keys())
    if nominal_m is None:
        nominal_m = len(source_ids)

    per_source: list[SourceDiagnostic] = []
    cross_source_corr: dict[str, dict[str, float]] = {}
    stat_eff_m_by_owner: dict[str, float] = {}

    for owner in agent_ids:
        predicted_by_source: dict[str, list[float]] = {sid: [] for sid in source_ids}
        true_series: list[float] = []
        for k in common_ks:
            reports = {r["source_id"]: r["value"] for r in rounds_by_k[k]["constraints"][owner]["reports"]}
            true_val = oracle_by_k[k]["agents"][owner]["true_cost_return"]
            true_series.append(true_val)
            for sid in source_ids:
                predicted_by_source[sid].append(reports.get(sid, float("nan")))

        true_arr = np.asarray(true_series, dtype=float)
        error_by_source: dict[str, np.ndarray] = {}
        for sid in source_ids:
            pred_arr = np.asarray(predicted_by_source[sid], dtype=float)
            valid = np.isfinite(pred_arr) & np.isfinite(true_arr)
            error = pred_arr[valid] - true_arr[valid]
            error_by_source[sid] = error
            spec_type, spec_class = (source_specs or {}).get(sid, ("unknown", "unknown"))
            per_source.append(
                SourceDiagnostic(
                    owner=owner,
                    source_id=sid,
                    source_type=spec_type,
                    independence_class=spec_class,
                    n_rounds=int(valid.sum()),
                    mean_predicted=float(pred_arr[valid].mean()) if valid.any() else float("nan"),
                    mean_true=float(true_arr[valid].mean()) if valid.any() else float("nan"),
                    bias=float(error.mean()) if len(error) else float("nan"),
                    mae=float(np.mean(np.abs(error))) if len(error) else float("nan"),
                    rmse=float(np.sqrt(np.mean(error**2))) if len(error) else float("nan"),
                    correlation_with_true=_pearson(pred_arr[valid], true_arr[valid]),
                )
            )

        m = len(source_ids)
        corr_matrix = np.eye(m)
        pair_corr: dict[str, float] = {}
        for i in range(m):
            for j in range(i + 1, m):
                ei, ej = error_by_source[source_ids[i]], error_by_source[source_ids[j]]
                n = min(len(ei), len(ej))
                r = _pearson(ei[:n], ej[:n])
                r_filled = 0.0 if np.isnan(r) else r
                corr_matrix[i, j] = corr_matrix[j, i] = r_filled
                pair_corr[f"{source_ids[i]}|{source_ids[j]}"] = r if not np.isnan(r) else None
        cross_source_corr[owner] = pair_corr
        stat_eff_m_by_owner[owner] = _participation_ratio(corr_matrix)

    overall = min(stat_eff_m_by_owner.values()) if stat_eff_m_by_owner else float("nan")
    resolved_nominal_effective_m = nominal_effective_m if nominal_effective_m is not None else nominal_m

    warnings: list[str] = []
    n_rounds_used = len(common_ks)
    assumption_supported = (resolved_nominal_effective_m - overall) <= tolerance
    if n_rounds_used < MIN_ROUNDS_FOR_RELIABLE_CORRELATION:
        warnings.append(
            f"Only {n_rounds_used} rounds available (< "
            f"{MIN_ROUNDS_FOR_RELIABLE_CORRELATION}); pairwise correlations are not "
            f"reliable at this sample size in either direction (at n=2, |r|=1 for "
            f"every pair regardless of true structure). Do not treat "
            f"assumption_1ii_supported as a positive finding from this report; a "
            f"clear non-independence finding (e.g. r~1 sustained over many more "
            f"rounds, or an exact-duplicate source) remains meaningful, but "
            f"'supported=True' at low n is not evidence of independence."
        )
        assumption_supported = False

    return EmpiricalDiversityReport(
        run_dir=str(run_dir),
        n_rounds_used=n_rounds_used,
        nominal_m=nominal_m,
        nominal_effective_m=resolved_nominal_effective_m,
        per_source=per_source,
        cross_source_error_correlation=cross_source_corr,
        statistical_effective_m_by_owner=stat_eff_m_by_owner,
        statistical_effective_m_overall=overall,
        assumption_1ii_supported=assumption_supported,
        warnings=warnings,
    )


def write_report(report: EmpiricalDiversityReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path


def format_report(report: EmpiricalDiversityReport) -> str:
    lines = [
        f"Empirical source diversity for {report.run_dir} ({report.n_rounds_used} rounds):",
        f"  nominal M                        = {report.nominal_m}",
        f"  nominal effective M (classes)    = {report.nominal_effective_m}",
        f"  statistical effective M (overall, worst-case owner) = {report.statistical_effective_m_overall:.2f}",
        "  statistical effective M by owner:",
    ]
    for owner, val in report.statistical_effective_m_by_owner.items():
        lines.append(f"    {owner}: {val:.2f}")
    lines.append(
        f"  Assumption 1(ii) (source independence) SUPPORTED by measurement: "
        f"{report.assumption_1ii_supported}"
    )
    for w in report.warnings:
        lines.append(f"  WARNING: {w}")
    if not report.assumption_1ii_supported:
        lines.append(
            "  WARNING: nominal M is presented in the config/paper as the source count, "
            "but this run's measured error correlations give a statistically effective "
            "count well below it. Do not claim Assumption 1(ii) or Theorem 2's M-source "
            "guarantee for this configuration without addressing this gap."
        )
    lines.append("  per-source bias / MAE / RMSE / corr(predicted, true):")
    for s in report.per_source:
        lines.append(
            f"    {s.owner:<10}{s.source_id:<18}bias={s.bias:+8.3f}  mae={s.mae:7.3f}  "
            f"rmse={s.rmse:7.3f}  corr={s.correlation_with_true:+.3f}"
        )
    return "\n".join(lines)
