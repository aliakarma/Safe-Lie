# G4 -- identity-conditioned source estimator, pre-declared decision rules

**Status: DECLARED BEFORE ANY G4 NUMBER EXISTS.** Written and committed
while `results/g4_identity_source_diagnostic/` does not yet exist. Nothing
below may be edited after a G4 number has been read. If a rule turns out to
have been badly chosen, the correct response is to record that and say so,
not to move it.

## What G4 is for

`docs/g3_gates.md` recorded outcome **B (H2 supported) / verdict FAIL**:
`peer_critic` (a physical peer's own head, trained on that peer's own
target, queried at a different agent's observation) is scientifically
invalid as a `J_C^i` estimator. Normalization does not materially fix it in
either condition; the in-distribution head (V1/V3) is well calibrated once
normalized, but the cross-agent query (V2) stays uncorrelated with the
owner's truth in both conditions.

G4 does **not** re-attempt a fix to that architecture. It asks a different,
narrower question: can a *single identity-conditioned network*,
`h(o, e_i)`, learn each owner's `J_C^i` mapping well enough to replace six
separate per-agent heads -- and, separately, does querying that one network
with six different identity vectors produce anything resembling six
*independent* estimators, or does it remain one statistical source no
matter how many identities are fed into it. These are two different
questions (calibration vs. diversity) and G4 must not let a good answer to
the first stand in for the second.

No PPO update, no GAE, no dual update, no attack, no RCE, and no change to
the environment, the safety budget, or the policy occurs anywhere in this
diagnostic. G4 trains only small regression networks on the **same**
fixed, disk-persisted dataset G3 already collected
(`results/g3_source_diagnostic/dataset/agent_*.npz`) -- no new rollout is
collected.

## Data reuse (pre-declared)

* Reuses `results/g3_source_diagnostic/dataset/agent_{0..5}.npz` verbatim
  -- same frozen checkpoint (`pilot_A_clean_seed0`, round 249), same 60
  rollout rounds seeded from generator seed 777, same masked MC
  `cost_to_go` fit targets, same `obs[0]`/`mc_cost_return` eval
  query/truth pairs `_collect_source_value` uses in production.
* **Training split**: rounds 0-39 (`fit_obs`/`fit_target` in each agent's
  `.npz`) -- pooled across all six agents for the shared-model variants
  (B2/B3), per-agent only for the separate-head variant (B1).
* **Evaluation split**: rounds 40-59 (`eval_query_obs`/`eval_truth`,
  20 rounds/agent) -- untouched by any fit step of any variant. No
  evaluation sample is used to refit any estimator, and no variant's
  training loop reads `eval_truth` or `eval_query_obs` before scoring.
* Because rounds are collected once per multi-agent rollout, round `r`'s
  six per-agent rows come from the *same* simultaneous timestep across
  agents -- this is what makes the cross-identity residual-correlation
  analysis in section K meaningful (it is not comparing unrelated draws).

## Model variants (pre-declared)

### B1 -- separate normalized owner-specific heads (positive control)
Identical construction to G3's *normalized* condition: one
`Linear(63,16) -> Tanh -> Linear(16,1)` head per physical agent, Xavier-
uniform init, Adam, lr=1e-3, 2500 full-batch-bootstrap steps (batch
2048), `RunningMeanStd` fit on that agent's own fit-rows for both input
and target, denormalized before scoring. Re-trained here (not merely
read off the G3 JSON) only so that raw per-sample predictions are
available for this report's tables; the recipe, seeds, and resulting
numbers are the same G3 already reported for V1/V3.

### B2 -- identity-conditioned shared estimator
One shared network, input = `[normalized_obs (63); one_hot_identity (6)]`
-> `Linear(69,32) -> Tanh -> Linear(32,1)`, Xavier-uniform init, Adam,
lr=1e-3, 5000 full-batch-bootstrap steps, batch 2048 sampled uniformly
from the **pooled** fit rows of all six agents (so the identity one-hot
is drawn together with its own agent's `(obs, target)` pair). Hidden
width (32, vs. B1's 16) and step count (5000, vs. B1's 2500) are raised
once, a priori, because this network must fit six owners' mappings
jointly rather than one; this is the only capacity change made, it is
made identically for B2 and B3, and it is not tuned against the
evaluation split. Observation and target normalization use a single
*pooled* `RunningMeanStd` fit across all six agents' fit rows -- a
per-agent target normalizer would hand the network owner information
through the back door and defeat the purpose of testing whether the
identity input itself is what the network needs.

### B3 -- identity-conditioned shared estimator without identity (control)
Identical to B2 in every respect (architecture width, steps, lr, pooled
normalization, training data) except the input is `normalized_obs`
alone (63-dim, no one-hot). This isolates whether the one-hot identity
input is load-bearing for position-dependent cost structure, holding
capacity and training budget fixed.

### B4 -- current peer-style estimator (negative control)
Not retrained. G3's already-computed *normalized* V2 numbers
(`results/g3_source_diagnostic/g3_report.json`, `conditions.normalized.V2_*`)
are reused verbatim as the negative-control reference throughout this
report, per the task's explicit permission to do so.

## Evaluation protocol (pre-declared)

* **Per-agent diagonal**: for each owner `i in {0..5}`, score
  `h(o^i, e_i)` (B2), `h(o^i)` (B3), and `h_i(o^i)` (B1) against
  `J_C^i` = `eval_truth[i]` over the 20 eval rounds.
* **6x6 identity-query matrix (B2 only)**: for every owner `i` and every
  queried identity `j in {0..5}`, score `h(o^i, e_j)` against
  `J_C^i` = `eval_truth[i]`. The diagonal (`i == j`) is the intended
  identity-conditioned use; off-diagonal cells test whether swapping the
  identity input actually changes the learned mapping in the direction
  of agent `j`'s cost level, or merely perturbs the output.
* B3 has no identity axis, so only the diagonal (owner-only) table
  applies to it.
* **Metrics**, identical definitions to `docs/g3_gates.md`: `bias = mean(pred
  - truth)`, `MAE = mean(|pred-truth|)`, `RMSE = sqrt(mean((pred-truth)^2))`,
  `corr = Pearson(pred, truth)`, `FSR = P(pred <= d | truth > d)` with
  `d = 25` (`budget_d` from the reused dataset's `meta.json`), all over the
  20 eval-round samples for that cell.

## Calibration bar (pre-declared, reused from G3/G2 verbatim)

* **Good**: `|bias| <= 5.0` AND `corr >= 0.5`.
* **Poor**: `corr < 0.2` OR `|bias| > 5.0`.
* Anything between is reported as-is and called neither.
* **FSR acceptable**: `<= 0.20`.
* **Identity materially necessary** iff removing identity (B3 vs. B2)
  degrades per-agent calibration such that at least one agent moves from
  "good" to "not good" under the bar above, OR pooled `|bias|` or `MAE`
  increases by `>= 50%` relative to B2 -- symmetric to G3's own
  "materially fixes" convention. Anything short of this is "not
  necessary": B3 explains the data about as well as B2.

## Source-diversity protocol (pre-declared, section 11/K of the task)

For B2's diagonal predictions, compute residuals
`e^i_t = h(o^i_t, e_i) - J_C^i_t` for each agent `i` over the 20 shared
eval rounds `t`. Build the `6x6` Pearson correlation matrix and covariance
matrix of `{e^i}` across agents (over the 20 rounds), and the participation
ratio `PR = (sum(lambda))^2 / sum(lambda^2)` of that covariance matrix's
eigenvalues `lambda` (`PR = 1` means all variance sits on one axis --
maximally non-diverse; `PR = 6` means six equal, uncorrelated axes --
maximally diverse). **Pre-declared interpretation**: regardless of the
measured `PR` or correlation values, B2 is classified `M_model = 1`
(section G4d) -- a low pairwise residual correlation does not by itself
establish independent failure modes, because all six outputs are read from
one shared parameter vector and a single adversarial input perturbation to
that one network can in principle move every identity's output at once.
`PR` and the correlation matrix are reported as descriptive diagnostics,
not as a re-derivation of `M`.

## Independent-replica protocol (pre-declared, section 12/L of the task)

If run: three B2 networks (`h_1, h_2, h_3`), identical architecture/
training recipe, differing only in the Xavier-init/bootstrap-sampling RNG
seed. For each pair `(h_a, h_b)`, compute the Pearson correlation of their
diagonal residuals `e^i_t` pooled over all `(agent, round)` cells, and the
same participation-ratio measure across the three replicas' residual
vectors. This is a diagnostic on the *architecture's* achievable
diversity under independent initialization alone (no independent data, no
independent training procedure) -- it is explicitly not a claim about
what a production ensemble with independent data/training would achieve,
and section L must say so.

## G4a/b/c/d/e gates (pre-declared)

* **G4a (owner calibration)**: B2's diagonal pooled `bias`/`MAE`/`RMSE`
  must be within a factor of 2 of B1's pooled numbers (the normalized
  positive reference) to be called comparable; B2 pooled must independently
  clear the "good" bar above to pass outright.
* **G4b (owner-wise consistency)**: G4a's classification is computed and
  reported per agent, not only pooled. B2 cannot pass this gate if any of
  the six agents individually falls in "poor" while the pooled number
  looks "good" -- pooling must not be allowed to hide a per-agent failure,
  per the task's explicit instruction (mirrors G2h/G3's own per-agent
  reporting requirement).
* **G4c (identity necessity)**: pass/fail per the "identity materially
  necessary" bar above, evaluated per-agent (not pooled-only).
* **G4d (structural source count)**: B2 is `M_model = 1` unconditionally,
  regardless of G4a-c's outcome, unless independent replicas (section L)
  are run and demonstrate non-trivial diversity by the participation-ratio
  measure above -- and even then, replica diversity from shared-recipe
  independent initialization is reported as a separate, weaker claim than
  "independent estimators" in the RCE/Theorem-2 sense (section 13 of the
  task). Six identity inputs to one trained network never, by themselves,
  license `M = 6` or `M = 7`.
* **G4e (cross-seed robustness)**: this diagnostic uses exactly one frozen
  checkpoint and one 60-round dataset, identical to G3. It is labeled a
  single-dataset diagnostic; no claim of generality across checkpoints,
  seeds, or training rounds is made. The only "independent partitions"
  available are the pre-declared train/eval round split (section above)
  and, for section L only, independent network initializations -- neither
  substitutes for a genuine multi-checkpoint or multi-seed campaign.

## Decision rule (pre-declared, mirrors task section 14)

* **Outcome A**: B2 clears "good" pooled AND per-agent (G4a+G4b pass), and
  substantially outperforms B4 (G3's V2) on bias/MAE/RMSE/corr.
* **Outcome B**: B2 does not clear "good" (pooled or per-agent) -- identity
  conditioning does not fix the calibration problem either.
* **Outcome C**: Outcome A holds (calibration is good) but G4d is
  unconditionally `M_model = 1` -- i.e. calibration is solved but the
  source-count problem is not, by construction, regardless of the
  calibration result.
* **Outcome D**: only reachable if independent replicas (section L) are
  run AND show low pairwise residual correlation / high participation
  ratio -- otherwise D is not available as a conclusion from this
  diagnostic alone.
* **Outcome E**: neither B1-style owner-specific heads nor B2/B3 nor
  independent replicas clear "good" calibration at all -- would call the
  entire `M > N` peer/replica source strategy into question for this
  environment's cost definition, not just this one architecture.

Per G4d, **Outcome C is the expected default outcome** of this diagnostic
even if calibration (Q1) succeeds -- because prediction validity and
source diversity are independent claims (task section 13) and this
diagnostic's replica arm, even if run, tests only initialization-seed
diversity, not the independent-data/independent-training diversity
Assumption 1(ii) actually requires.

## G4 verdict rule (pre-declared)

* **PASS** -- B2 is a valid owner-specific estimator (G4a-c all pass) AND
  a separately demonstrated independence mechanism (section L, with
  materially low replica error-correlation) supports treating multiple
  queries/replicas of this architecture as more than one source.
* **CONDITIONAL PASS** -- B2 is a valid owner-specific estimator (G4a-c
  pass) but no independence mechanism is demonstrated (G4d stands as
  `M_model = 1`, section L not run or inconclusive). Prediction validity
  is established; the `M`-source requirement is not.
* **FAIL** -- B2 does not clear the calibration bar (G4a or G4b fails):
  identity conditioning does not solve G3's failure, independent of the
  source-count question.

On any verdict, `M_effective` for any future run must still be recomputed
by counting only sources whose calibration clears "good" above, exactly as
`docs/g3_gates.md` already requires; this diagnostic can raise that count
by at most **one** additional statistical source (the shared
identity-conditioned network), never by six, per G4d.

## What this diagnostic explicitly does not establish

Regardless of verdict: no claim about Assumption 1(ii) (source
independence / honest-majority) beyond the narrow initialization-seed
diversity measured in section L if run, no claim about Theorem 2, no
claim about attack stealth or RCE, no claim generalizing beyond
`pilot_A_clean_seed0`'s one frozen policy snapshot and one 60-round
dataset, and no claim that a "good" B1/B2 number here would survive being
re-embedded in the full nonstationary 250-round training loop. These are
the same standing limitations `docs/g2_gates.md` and `docs/g3_gates.md`
already declared, restated here rather than assumed carried over.
