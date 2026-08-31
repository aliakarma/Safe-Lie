# Security Policy

## What this repository is

This is a research-engineering repository accompanying an academic paper
about an adversarial threat against a specific class of safe
multi-agent reinforcement learning systems (decentralized primal-dual
constrained MARL). It implements:

- a formal threat model and its theoretical analysis,
- a **minimal reference attack** (perturbing a scalar cost-report residual
  by a bounded amount, in a self-contained simulated research
  environment — `safelie.attacks`), and
- a defense (Robust Constraint Estimation, `safelie.defenses.rce`) with a
  proved bounded-violation guarantee under an honest majority of sources.

## Dual-use position

The source paper's own Ethics Statement (`main_iclr.tex`) gives three
reasons for publishing the attack alongside the defense, which this
repository inherits:

1. The attack requires an adversary who **already has write access** to a
   cost-reporting channel. Simulating that perturbation in code does not
   create that access; an attacker who holds it already possesses the
   serious capability.
2. The vulnerability is a property of a **published, widely used
   algorithmic structure** (doubly stochastic consensus on dual
   variables), not an implementation flaw in a specific product. There is
   no vendor to notify privately and no patch to coordinate.
3. The defense (RCE) is published alongside the attack, and gives
   operators a concrete, checkable criterion (`M >= 2f+1` **independence
   classes**, not raw report count — see `safelie.governance.auditor` and
   `PROJECT_REPORT.md` §13.4).

Consistent with the source paper, this repository does **not** ship
attack tooling hardened for use against real deployed systems: the attack
implementations here are minimal, operate only inside the synthetic /
Safe-MAMuJoCo research environments this repository defines, and have no
network, transport, or deployment-targeting code of any kind.

## Reporting a vulnerability in this codebase

If you find a security issue in the *software* (not the research
threat model it studies) — e.g., unsafe deserialization, path traversal
in a script, or a dependency with a known CVE — please open a GitHub
issue and mark it security-sensitive, or contact the maintainers listed
in the repository's profile before public disclosure.

## Reporting a gap in the oracle isolation boundary

`safelie.eval.oracle` and `safelie.envs.guards` implement the boundary that
keeps the learner from observing true cost (see
`tests/isolation/test_oracle_isolation.py`). If you find a path by which
learner-side code can read true cost, please treat this as a correctness
bug that invalidates any detection-gap metric computed by the affected
code path, and report it with the same urgency as a security issue —
every downstream number depends on this boundary holding.

## Supported versions

This is a research prototype (see `IMPLEMENTATION_STATUS.md`). There is
no long-term support policy; fixes land on `main`.
