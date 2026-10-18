# M2 gate report

Report date: 2026-10-18

**Verdict: NOT CLEARED** (7 items, 6 failed, 0 without a stated criterion)

| Item | Owner | Verdict | Criterion | Evidence | What failed first |
| --- | --- | --- | --- | --- | --- |
| PPO harness at fixed D3 hyperparameters | Ilan Danial | FAIL | optspread.train exposes a train entry point and one pinned PPO_HYPERPARAMETERS mapping (D3); every pinned key is a Stable-Baselines3 PPO argument; and at least one run under the runs root has a valid manifest recording the learner's library versions | no training entry point | optspread.train does not exist yet |
| Deterministic replay from hash | Ilan Danial | FAIL | optspread.replay.replay_matches re-runs a recorded run from its manifest (config hash and seed) and reproduces every artifact digest the manifest records | no replay entry point | optspread.replay does not exist yet |
| Variance-components pilot | Dominick Dupuy | FAIL | an S x N pilot matrix (matrix.npy, S >= 2, N >= 2, finite) sits beside a valid manifest that records its digest; tau^2 and sigma^2 are reported with standard errors and turned into the detection floor at one-sided alpha = 0.05 and 80% power | no runs/pilot/matrix.npy | pilot not run |
| Analysis pipeline validated against a known analytic case | Dominick Dupuy | PASS | at every effect on the grid, including delta = 0 where the answer is alpha, the decision rule's rejection rate over R paired synthetic replications lies within 4 Monte Carlo standard errors, sqrt(p (1 - p) / R), of the closed-form noncentral-t power at the hand-specified tau^2, sigma^2 and delta; and the mean estimated sigma^2 and Var(grand mean) lie within 4 standard errors of their true values; and each deliberately wrong rule (normal critical values, unpaired arms, episodes as independent) fails the same check | tau^2 = 4, sigma^2 = 100, shared path variance 400, S = 10 x N = 25, one-sided alpha = 0.05; R = 10,000 per effect; rejection rate/closed form: 0.047/0.050 at 0, 0.127/0.130 at 0.5, 0.275/0.271 at 1, 0.662/0.662 at 2, 0.927/0.925 at 3; largest deviation -1.47 MC se (rejection rate at delta = 0); 3 of 3 wrong rules refused; the normal approximation reads 0.723 against 0.662 at delta = 2 | -- |
| Pre-registration committed and tagged | Dominick Dupuy | FAIL | docs/PRE_REGISTRATION.md is tracked with no uncommitted changes; the earliest prereg-* tag points at a commit on this history that contains it; and the file at HEAD begins with the tagged text byte for byte (amendments appended, never edited in place) | committed at a9881ca5 | no prereg-* tag: committed, not tagged |
| Dashboard run browser and gate board | Ilan Danial | FAIL | the dashboard's data layer (optspread.dashboard_data) lists every run directory under the runs root that has a valid manifest, of which there is at least one, and every gate report JSON sidecar under docs/ | no dashboard data layer | optspread.dashboard_data does not exist yet |
| Real data adapter conformance | Hemdutt Rao | FAIL | the real optionmetrics source is available (its SPX panel is present) and clears conformance_report, the provider contract and structural invariants every source meets; SKIPPED is not a pass | synthetic PASS (64 chains); optionmetrics_stub PASS (64 chains); optionmetrics SKIPPED | optionmetrics is unavailable (SKIPPED): no SPX panel for the adapter to serve |

## What failed first

- **PPO harness at fixed D3 hyperparameters**: optspread.train does not exist yet
- **Deterministic replay from hash**: optspread.replay does not exist yet
- **Variance-components pilot**: pilot not run
- **Pre-registration committed and tagged**: no prereg-* tag: committed, not tagged
- **Dashboard run browser and gate board**: optspread.dashboard_data does not exist yet
- **Real data adapter conformance**: optionmetrics is unavailable (SKIPPED): no SPX panel for the adapter to serve
