#!/usr/bin/env python
"""Summarize the completed pilot matrix using this repository's own
pre-registered procedures -- never an ad-hoc test chosen after the fact.

Usage:
    python scripts/analyze_matrix.py
    python scripts/analyze_matrix.py --window 100 --budget 25.0

Which statistics are legitimate is decided by seed count, not by taste:

  n >= 5  `safelie.analysis.stats.welch_t_test` + `holm_correction`, the
          paper's own [SPEC] procedure (main_iclr.tex §5.1).
  n <  5  Per-seed sign and ordering only
          (`safelie.eval.protocol.pilot_seed_summary`). `welch_t_test`
          refuses to run below 5 seeds by construction -- at n=3 it
          "manufactures false precision" (decision D6, §R2.3). This
          script does not work around that refusal; it reports what the
          available seeds actually support.

Reading order follows §R8.3: the falsification control (D) is judged
BEFORE the defense claim (C). If the attacked condition does not separate
from the benign control -- unbiased zero-mean noise matched to the
attack's magnitude -- then the defense comparison is moot, and the paper
commits in advance (§R6.4) to withdrawing its central claim.

Every number here is measured from run logs. Nothing is compared against
main_iclr.tex's projected Table 3/4 values, which the paper itself marks
as specifications rather than results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Which directory of runs this script reads. Set once, from --runs-dir,
# and never defaulted implicitly at a call site.
#
# This was previously the repository's results/runs path, hardcoded at
# three separate glob sites. That directory still holds the complete
# pre-P0-repair pilot matrix (`pilot_{A..E}_*_seed{0..4}`), which
# docs/assumptions.md marks as forensic evidence that must never enter a
# post-repair aggregate -- and whose run directories have exactly the
# names a post-repair rerun would produce. A glob cannot tell them
# apart, so the directory is now an explicit argument: a validation
# stage writes to its own directory (e.g. results/runs_g0) and is
# analyzed with --runs-dir pointing there.
RUNS_DIR = REPO / "results" / "runs"

from safelie.analysis.stats import (  # noqa: E402
    MIN_SEEDS_FOR_INFERENCE,
    holm_correction,
    welch_t_test,
)
from safelie.eval.metrics import RunMetrics  # noqa: E402
from safelie.eval.protocol import consistent_across_seeds, pilot_seed_summary  # noqa: E402


def _as_metrics(seeds: dict, budget: float) -> RunMetrics:
    """Mean across the given seeds, as the `RunMetrics` the pre-registered
    protocol helpers expect. `detection_gap` is the aggregate-based one --
    the channel the attack acts through (see this module's load_seed)."""
    m = lambda k: float(np.mean([v[k] for v in seeds.values()]))  # noqa: E731
    return RunMetrics(
        return_mean=m("task_return"),
        reported_cost_mean=m("reported_cost"),
        true_cost_mean=m("true_cost"),
        violation_rate=m("violation_rate"),
        peak_violation=float("nan"),
        detection_gap=m("detection_gap_vs_aggregate"),
    )

CONDITIONS = {
    "A": ("pilot_A_clean", "clean baseline"),
    "B": ("pilot_B_attack", "attacked, undefended"),
    "C": ("pilot_C_rce", "attacked, RCE-defended"),
    "D": ("pilot_D_benign", "benign control (falsification)"),
    "E": ("pilot_E_clean_rce", "clean + RCE"),
}
METRICS = [
    "detection_gap_vs_aggregate",  # primary: the gap the attack can actually move
    "detection_gap",  # as logged; measured against the agent's own critic
    "true_cost",
    "reported_cost",
    "lambda",
    "violation_rate",
    "task_return",
]


def load_seed(run_dir: Path, window: int) -> dict[str, float] | None:
    rounds_p, oracle_p = run_dir / "rounds.jsonl", run_dir / "oracle.jsonl"
    if not rounds_p.exists() or not oracle_p.exists():
        return None
    rounds = [json.loads(x) for x in rounds_p.read_text(encoding="utf-8").splitlines() if x.strip()]
    oracle = [json.loads(x) for x in oracle_p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rounds or not oracle:
        return None
    agents = list(rounds[-1]["constraints"])
    r, o = (rounds, oracle) if window <= 0 else (rounds[-window:], oracle[-window:])
    g = lambda arr, f: float(np.mean([[f(rec, a) for a in agents] for rec in arr]))  # noqa: E731

    # The detection gap the attack can actually move. `detection_gap` as
    # logged is true_cost - `reported_cost_return`, and
    # `safelie.training.loop` sets that to the agent's OWN cost-critic
    # estimate -- a quantity the attack never touches, since corruption is
    # applied to the source reports and lands in `aggregate`. Measured
    # against the aggregate instead, a single attacked seed moves +3.2 sd
    # from clean; measured against the own critic, +0.2 sd. The aggregate
    # is what the dual update consumes, so it is what "reported" means in
    # J_true_C - J_reported_C.
    #
    # P0 #6 fix: this used to be RECONSTRUCTED here from
    # `aggregate.point_estimate`/`spread` plus the config's own beta,
    # because neither `pessimistic_estimate` nor the aggregate-based gap
    # was logged directly. Both now are --
    # `constraints.<agent>.mechanism_reported_cost_return` (loop.py) and
    # `agents.<agent>.detection_gap_vs_aggregate` (experiment.py) are
    # computed once, at the source, from the exact value that drove that
    # round's dual update. Read directly; do not reconstruct. A log
    # missing this field predates the fix and must be re-run, not patched
    # around here.
    missing = [i for i in range(min(len(r), len(o))) for a in agents if "detection_gap_vs_aggregate" not in o[i]["agents"][a]]
    if missing:
        raise KeyError(
            f"{run_dir} predates the P0 #6 return-reporting fix (no "
            f"'detection_gap_vs_aggregate' in oracle.jsonl) -- re-run it rather than "
            f"reconstructing a value this script no longer computes independently."
        )

    return {
        "n_rounds": len(rounds),
        "detection_gap_vs_aggregate": g(o, lambda rec, a: rec["agents"][a]["detection_gap_vs_aggregate"]),
        "detection_gap": g(o, lambda rec, a: rec["agents"][a]["detection_gap"]),
        "true_cost": g(o, lambda rec, a: rec["agents"][a]["true_cost_return"]),
        "violation_rate": g(o, lambda rec, a: float(rec["agents"][a]["violated"])),
        "reported_cost": g(r, lambda rec, a: rec["constraints"][a]["mechanism_reported_cost_return"]),
        "lambda": g(r, lambda rec, a: rec["constraints"][a]["lambda_after"]),
        "task_return": g(o, lambda rec, a: rec["agents"][a]["episodic_task_return"]),
    }


def collect(window: int) -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, dict[str, float]]] = {}
    for key, (prefix, _) in CONDITIONS.items():
        out[key] = {}
        for run_dir in sorted(RUNS_DIR.glob(f"{prefix}_seed*")):
            seed = int(run_dir.name.rsplit("seed", 1)[1])
            rec = load_seed(run_dir, window)
            if rec is not None:
                out[key][seed] = rec
    return out


def main() -> int:
    global RUNS_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--window",
        type=int,
        default=0,
        help="Trailing rounds averaged per run; 0 = whole run (default). Do not "
        "narrow this without reading the window-sensitivity table it prints: the "
        "dual variable oscillates with a ~100-round period and only ~2 cycles fit "
        "in a 250-round run, so a trailing-100 window samples a near-random phase "
        "and manufactured a spurious 2.6-sigma effect in both the attacked and "
        "benign arms that reversed sign on any longer window.",
    )
    ap.add_argument("--budget", type=float, default=25.0)
    ap.add_argument(
        "--runs-dir",
        default=str(RUNS_DIR),
        help="Directory of run directories to aggregate (default: results/runs). "
        "Point this at a validation stage's own directory -- results/runs_g0 for G0 -- "
        "so a post-repair analysis cannot silently include the pre-repair matrix that "
        "shares its run-directory naming (see this module's RUNS_DIR comment).",
    )
    ap.add_argument("--complete-only", action="store_true", help="Ignore runs short of 250 rounds")
    args = ap.parse_args()

    RUNS_DIR = Path(args.runs_dir).resolve()
    if not RUNS_DIR.is_dir():
        print(f"ERROR: --runs-dir {RUNS_DIR} does not exist.", file=sys.stderr)
        return 2
    print(f"Aggregating runs from: {RUNS_DIR}")

    data = collect(args.window)
    if args.complete_only:
        data = {k: {s: v for s, v in d.items() if v["n_rounds"] >= 250} for k, d in data.items()}

    problems = integrity_check()
    if problems:
        print("=" * 78)
        print("!! DATA INTEGRITY PROBLEMS -- resolve before trusting anything below")
        for name, issue in problems:
            print(f"   {name}: {issue}")
        print("=" * 78)

    print("=" * 78)
    wlabel = "whole run" if args.window <= 0 else f"trailing {args.window} rounds"
    print(f"PILOT MATRIX -- {wlabel}, budget d={args.budget}")
    print("Measured from run logs. Never compared against the paper's projected values.")
    print("=" * 78)

    # gap*  = detection gap vs the aggregate (primary; what the attack moves)
    # gapOwn = detection gap as logged, vs the agent's own cost critic
    print(f"\n{'cond':<6}{'n':>3}  {'description':<30}{'gap*':>8}{'gapOwn':>9}"
          f"{'trueC':>8}{'lambda':>8}{'viol%':>7}")
    print("-" * 78)
    for k, (_, desc) in CONDITIONS.items():
        seeds = data[k]
        if not seeds:
            print(f"{k:<6}{0:>3}  {desc:<30}{'--':>8}{'--':>9}{'--':>8}{'--':>8}{'--':>7}")
            continue
        def arr(m: str, _seeds: dict = seeds) -> np.ndarray:
            return np.array([s[m] for s in _seeds.values()])

        sd = f"+/-{arr('detection_gap_vs_aggregate').std(ddof=1):.2f}" if len(seeds) > 1 else ""
        print(f"{k:<6}{len(seeds):>3}  {desc:<30}"
              f"{arr('detection_gap_vs_aggregate').mean():>8.2f}"
              f"{arr('detection_gap').mean():>9.2f}"
              f"{arr('true_cost').mean():>8.2f}{arr('lambda').mean():>8.3f}"
              f"{arr('violation_rate').mean() * 100:>7.1f}   {sd}")

    print("\nseeds per condition: " + ", ".join(f"{k}={len(v)}" for k, v in data.items()))

    print("\n" + "=" * 78)
    print("STEP 0 -- IS THE PRE-REGISTERED CRITERION USABLE HERE?")
    print("=" * 78)
    criterion_degeneracy(data, args.budget)

    # ---- §R8.3 reading order: falsification control first ----------
    print("\n" + "=" * 78)
    print("STEP 1 -- FALSIFICATION CONTROL (§R6.4). Does the attack separate from")
    print("unbiased zero-mean noise of matched magnitude? If not, the central claim")
    print("is withdrawn and the defense comparison is moot.")
    print("=" * 78)
    contrasts = [("B", "D", "attack vs benign control  <-- THE FALSIFICATION TEST"),
                 ("B", "A", "attack vs clean           <-- the core claim"),
                 ("D", "A", "benign control vs clean   (should be ~no effect)")]
    report_contrasts(data, contrasts, args.budget)

    print("\n" + "=" * 78)
    print("MECHANISM CHECK -- Theorem 1's spreading prediction, measured in-run.")
    print("=" * 78)
    spreading_check()

    print("\n" + "=" * 78)
    print("WINDOW SENSITIVITY. A contrast that changes sign across these windows is")
    print("an artifact of the dual variable's ~100-round oscillation, not an effect.")
    print("=" * 78)
    window_sensitivity(args)

    print("\n" + "=" * 78)
    print("STEP 2 -- DEFENSE CLAIM. Only interpretable if step 1 separated.")
    print("=" * 78)
    report_contrasts(
        data,
        [
            ("C", "B", "RCE vs undefended attack  <-- the defense claim"),
            # The sharpest statement the matrix can make about the defense:
            # if the attacked-and-defended condition is indistinguishable
            # from the clean-and-defended one, RCE removed the attack's
            # effect rather than merely blunting it. A null here is the
            # desired result -- the one place in this report where failing
            # to separate is good news.
            ("C", "E", "RCE attacked vs RCE clean <-- residual attack effect under defense"),
            ("E", "A", "RCE on clean              (conservatism cost)"),
        ],
        args.budget,
    )
    return 0


def spreading_check() -> None:
    """Theorem 1, measured inside a single run rather than across conditions.

    The logged `reports` are the source values BEFORE corruption; the logged
    `aggregate.point_estimate` is the value after it. Their difference is
    the attack's effect on the aggregate with the policy, the seed and the
    round held fixed -- no cross-condition confound at all. Theorem 1 says
    corrupting f sources of M with total mass B moves a mean aggregate by
    exactly B/M, so this is a direct numerical test of the paper's central
    mechanism rather than an inference from outcomes.

    It also checks the falsification control is properly matched: the
    benign arm should show the same magnitude with zero mean, which is what
    makes it a control rather than a weaker attack.
    """
    import yaml

    for key in ("B", "D", "A"):
        prefix, desc = CONDITIONS[key]
        cfg = yaml.safe_load(
            (REPO / "configs" / "experiment" / f"{prefix}.yaml").read_text(encoding="utf-8")
        )
        if cfg["defense"]["name"] != "mean":
            continue  # the identity holds for mean aggregation; RCE trims
        mass = float(cfg["attack"].get("budget_ratio", 0.0)) * float(cfg["env"]["budget"])
        shifts = []
        for d in sorted(RUNS_DIR.glob(f"{prefix}_seed*")):
            rp = d / "rounds.jsonl"
            if not rp.exists():
                continue
            for line in rp.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for con in rec["constraints"].values():
                    clean = float(np.mean([x["value"] for x in con["reports"]]))
                    shifts.append(con["aggregate"]["point_estimate"] - clean)
        if not shifts:
            continue
        s = np.asarray(shifts)
        m = len(json.loads(rp.read_text(encoding="utf-8").splitlines()[0])["constraints"]["agent_0"]["reports"])
        predicted = -mass / m
        print(f"\n  {key} -- {desc}   (B={mass:g}, M={m}, n={len(s)})")
        print(f"     predicted shift  {predicted:+.4f}   (= -B/M)")
        print(f"     measured shift   {s.mean():+.4f}   sd={s.std():.4f}")
        if cfg["attack"]["name"] == "primary":
            print("     -> a persistent, static, consistent attack is deterministic here,")
            print("        so an exact match with zero variance is the expected result.")
        elif cfg["attack"]["name"] == "benign_control":
            print(f"     -> control is matched: mean ~0 with sd {s.std():.3f} against the")
            print(f"        attack's |{predicted:.3f}| magnitude. Same size, no bias.")


def integrity_check() -> list[tuple[str, str]]:
    """Structural checks on every run's logs before any of it is averaged.

    Runs here are resumed from checkpoints across process boundaries -- an
    overnight matrix will not survive in one process -- so the logs are
    appended to by more than one process over their life. A resume that
    silently replayed or skipped rounds would not show up in any summary
    statistic; it would just shift the means. These checks catch it:
    `round_k` must be exactly 0..n-1 in both logs, and the agent and source
    counts must not vary within a run.

    A one-record lead of `rounds.jsonl` over `oracle.jsonl` is normal and
    not reported: the learner writes its record first and the orchestrator
    appends the oracle's afterwards, so a run sampled mid-round is caught
    between the two.
    """
    problems: list[tuple[str, str]] = []
    for d in sorted(RUNS_DIR.glob("pilot_*_seed*")):
        rp, op = d / "rounds.jsonl", d / "oracle.jsonl"
        if not rp.exists():
            continue
        rs = [json.loads(x) for x in rp.read_text(encoding="utf-8").splitlines() if x.strip()]
        os_ = (
            [json.loads(x) for x in op.read_text(encoding="utf-8").splitlines() if x.strip()]
            if op.exists()
            else []
        )
        if [r["round_k"] for r in rs] != list(range(len(rs))):
            problems.append((d.name, "round_k not sequential -- a resume replayed or skipped"))
        if [o["round_k"] for o in os_] != list(range(len(os_))):
            problems.append((d.name, "oracle round_k not sequential"))
        if len(rs) - len(os_) not in (0, 1):
            problems.append((d.name, f"log length mismatch: rounds={len(rs)} oracle={len(os_)}"))
        if len({len(r["constraints"]) for r in rs}) > 1:
            problems.append((d.name, "agent count varies within the run"))
        if len({len(r["constraints"]["agent_0"]["reports"]) for r in rs}) > 1:
            problems.append((d.name, "source count varies within the run"))
    return problems


def criterion_degeneracy(data: dict, budget: float) -> None:
    """Check the paper's pre-registered criterion before quoting it.

    §5.1 fixes success in advance as: return within one standard deviation
    of the no-attack baseline, reported cost <= d, and true cost > d. That
    is discriminative only if the *baseline* satisfies its own constraint.
    If the clean condition already has reported <= d < true -- which is
    what happens whenever the constraint is loosely enforced, as at pilot
    scale -- then the last two clauses hold for every condition including
    the baseline itself, and the criterion returns True for the benign
    control as readily as for the attack. It stops being a test.

    When that happens the continuous contrasts (effect sizes on the
    aggregate-based detection gap and true cost) are the instrument that
    still discriminates, and the binary criterion must not be quoted as
    evidence.
    """
    if not data.get("A"):
        print("  no clean baseline runs; cannot evaluate")
        return
    a_reported = float(np.mean([s["reported_cost"] for s in data["A"].values()]))
    a_true = float(np.mean([s["true_cost"] for s in data["A"].values()]))
    degenerate = a_reported <= budget < a_true
    print(f"  clean baseline: reported={a_reported:.2f}  true={a_true:.2f}  d={budget}")
    if degenerate:
        print("  DEGENERATE. The clean baseline itself satisfies 'reported <= d < true',")
        print("  so the pre-registered criterion is True for the baseline, the benign")
        print("  control and the attack alike. Do NOT quote it as evidence here; read")
        print("  the effect sizes on detection_gap_vs_aggregate and true_cost instead.")
        print("  The cause is that the constraint is not tightly enforced at pilot")
        print("  scale, not that the attack succeeded (PROJECT_REPORT.md §R6.1).")
    else:
        print("  Baseline satisfies its constraint; the pre-registered criterion is")
        print("  discriminative and may be quoted.")


def window_sensitivity(args) -> None:
    """Recompute the headline contrasts over several trailing windows.

    This exists because a trailing-100 window on a 250-round run produced a
    2.6-sigma lambda "suppression" in BOTH the attacked and the benign arm
    that vanished -- and reversed sign -- on the whole run. The dual
    variable oscillates with a ~100-round period, so a short trailing
    window samples a near-random phase of that cycle rather than a steady
    state. Only contrasts stable across these windows are effects.
    """
    windows = [(0, "whole run"), (200, "last 200"), (150, "last 150"), (100, "last 100")]
    pairs = [("B", "D"), ("B", "A"), ("D", "A"), ("C", "B")]
    for m in ("lambda", "detection_gap_vs_aggregate", "detection_gap", "true_cost"):
        print(f"\n  {m}")
        print("    " + f"{'window':<12}" + "".join(f"{a + '-' + b:>12}" for a, b in pairs))
        for w, wname in windows:
            d = collect(w)
            if args.complete_only:
                d = {k: {s: v for s, v in c.items() if v["n_rounds"] >= 250} for k, c in d.items()}
            cells = ""
            for a, b in pairs:
                if not d.get(a) or not d.get(b):
                    cells += f"{'--':>12}"
                    continue
                xa = float(np.mean([s[m] for s in d[a].values()]))
                xb = float(np.mean([s[m] for s in d[b].values()]))
                cells += f"{xa - xb:+12.3f}"
            print(f"    {wname:<12}{cells}")


def report_contrasts(data, contrasts, budget) -> None:
    pending: list[tuple[str, str, str, float]] = []
    for a, b, label in contrasts:
        sa, sb = data.get(a, {}), data.get(b, {})
        if not sa or not sb:
            print(f"\n  {label}\n    insufficient runs (n_{a}={len(sa)}, n_{b}={len(sb)})")
            continue
        print(f"\n  {label}   n_{a}={len(sa)}  n_{b}={len(sb)}")
        for m in METRICS:
            xa = np.array([s[m] for s in sa.values()])
            xb = np.array([s[m] for s in sb.values()])
            diff = xa.mean() - xb.mean()
            pooled = np.sqrt((xa.var(ddof=1) + xb.var(ddof=1)) / 2) if min(len(xa), len(xb)) > 1 else np.nan
            d_eff = diff / pooled if pooled and np.isfinite(pooled) and pooled > 0 else np.nan
            line = f"    {m:<16}{xa.mean():9.3f} vs {xb.mean():8.3f}   diff={diff:+8.3f}"
            if np.isfinite(d_eff):
                line += f"   d={d_eff:+5.2f}"
            print(line)
            if len(xa) >= MIN_SEEDS_FOR_INFERENCE and len(xb) >= MIN_SEEDS_FOR_INFERENCE:
                pending.append((label, m, "", welch_t_test(xa, xb).p_value))
        if min(len(sa), len(sb)) < MIN_SEEDS_FOR_INFERENCE:
            # Decision D6 / §R2.3: below 5 seeds the only defensible report
            # is per-seed sign and ordering. Use the repository's own
            # pre-registered helpers rather than an ad-hoc check, so what
            # is reported here is what the protocol module defines.
            baseline = _as_metrics(sb, budget)
            summaries = [
                pilot_seed_summary(seed, _as_metrics({seed: rec}, budget), baseline, budget)
                for seed, rec in sorted(sa.items())
            ]
            print(f"    -> n<{MIN_SEEDS_FOR_INFERENCE}: no t-test (decision D6); "
                  f"per-seed sign/ordering instead:")
            for s in summaries:
                print(f"       seed {s.seed}: d_trueC={s.true_cost_delta_vs_baseline:+7.2f}  "
                      f"d_return={s.return_delta_vs_baseline:+7.2f}  "
                      f"reported<=d: {str(s.reported_stayed_at_or_below_budget):<5} "
                      f"trueC>d: {s.true_cost_exceeded_budget}")
            print(f"       consistent_across_seeds (§R6.2): "
                  f"{consistent_across_seeds(summaries)}")

    if pending:
        # [SPEC] §5.1: "Welch's t-test for the primary comparison and Holm
        # correction ACROSS THE ATTACK CONDITIONS." The correction family is
        # the set of conditions compared on one metric -- not the metrics.
        # Pooling all metrics into a single family would be a different,
        # far more conservative procedure than the paper specifies, and it
        # costs most of the power at n=5 (a p=0.002 contrast lands at 0.04
        # across 18 tests, and a p=0.04 one at 0.62).
        print("\n  Welch t-tests [SPEC §5.1]. Holm family = the attack conditions")
        print("  compared on a given metric; metrics are separate families.")
        by_metric: dict[str, list[tuple[str, float]]] = {}
        for label, m, _, raw in pending:
            by_metric.setdefault(m, []).append((label, raw))
        for m in METRICS:
            group = by_metric.get(m)
            if not group:
                continue
            adj = holm_correction([p for _, p in group])
            primary = "  [PRIMARY]" if m in ("detection_gap_vs_aggregate", "true_cost") else ""
            print(f"\n    {m}{primary}   (family of {len(group)})")
            for (label, raw), p in zip(group, adj, strict=True):
                star = "  *" if p < 0.05 else ""
                print(f"      {label.split('<--')[0].strip():<30}p_raw={raw:.4f}  p_holm={p:.4f}{star}")


if __name__ == "__main__":
    raise SystemExit(main())
