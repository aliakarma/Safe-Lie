"""safelie — Corrupted-Signal Multi-Agent RL.

Reference implementation of the threat model and defense described in
"When the Safety Signal Lies: Adversarial Corruption of Safety-Cost
Feedback in Constrained Multi-Agent RL".

This package implements the paper's theory (corruption mass conservation,
spreading, multiplicative amplification), the attack taxonomy, the Robust
Constraint Estimation (RCE) defense, and a CPU-only synthetic training
pipeline used to verify the software before any GPU experiment is run.

No experiment reported in the source paper has been reproduced by this
package. See docs/paper_implementation_mapping.md and
IMPLEMENTATION_STATUS.md for exactly what has and has not been validated.
"""

__version__ = "0.1.0"
