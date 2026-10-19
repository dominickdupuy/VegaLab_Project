# M2 gate report

Report date: 2026-10-19

Runs root: `$VEGALAB_DATA/runs`; pilot root: `$VEGALAB_DATA/runs/pilot`

**Verdict: NOT CLEARED** (7 items, 5 failed, 0 without a stated criterion)

| Item | Owner | Verdict | Criterion | Evidence | What failed first |
| --- | --- | --- | --- | --- | --- |
| PPO harness at fixed D3 hyperparameters | Ilan Danial | FAIL | optspread.train exposes a train_ppo entry point and one pinned frozen_hyperparameters record (D3) whose values equal the installed Stable-Baselines3 defaults (its matches_library is True; not installed is not a pass); and at least one PPO run (policy ppo with a training config) under the runs root has a valid manifest | 21 pinned values, hash 628d9ce46772fd0e | stable-baselines3 is not installed, so the pinned set was not compared |
| Deterministic replay from hash | Ilan Danial | FAIL | optspread.replay.replay_matches re-runs a recorded run from its manifest (config hash and seed) and reproduces every artifact digest the manifest records; one PPO run and one scripted run are replayed when both kinds are under the runs root, at least one run in any case | nothing to replay | no valid run manifest under $VEGALAB_DATA/runs |
| Variance-components pilot | Dominick Dupuy | FAIL | the pilot's S x N matrices (matrix.npy, the PPO arm, and matrix_oracle.npy, the fixed-delta seller on the same paths) sit beside a valid manifest that records both digests and whose config is the PPO arm with its training config; S >= 5, N >= 20, every entry finite, N equal to the manifest's n_episodes and S to the grid's seed count where the grid record is present; tau^2 and sigma^2 of the paired contrast PPO minus oracle (what the pre-registered test tests) are reported with standard errors beside each arm's, and turned into the exact noncentral-t detection floor at one-sided alpha = 0.05 and 80% power | no $VEGALAB_DATA/runs/pilot/matrix.npy | pilot not run |
| Analysis pipeline validated against a known analytic case | Dominick Dupuy | PASS | at every effect on the grid, including delta = 0 where the answer is alpha, the decision rule's rejection rate over R paired synthetic replications lies within 4 Monte Carlo standard errors, sqrt(p (1 - p) / R), of the closed-form noncentral-t power at the hand-specified tau^2, sigma^2 and delta; and the mean estimated sigma^2 and Var(grand mean) lie within 4 standard errors of their true values; and each deliberately wrong rule (normal critical values, unpaired arms, episodes as independent) fails the same check | tau^2 = 4, sigma^2 = 100, shared path variance 400, S = 10 x N = 25, one-sided alpha = 0.05; R = 10,000 per effect; rejection rate/closed form: 0.047/0.050 at 0, 0.127/0.130 at 0.5, 0.275/0.271 at 1, 0.662/0.662 at 2, 0.927/0.925 at 3; largest deviation -1.47 MC se (rejection rate at delta = 0); 3 of 3 wrong rules refused; the normal approximation reads 0.723 against 0.662 at delta = 2 | -- |
| Pre-registration committed and tagged | Dominick Dupuy | FAIL | docs/PRE_REGISTRATION.md is tracked with no uncommitted changes; the earliest prereg-* tag points at a commit on this history that contains it; and the file at HEAD begins with the tagged text byte for byte (amendments appended, never edited in place) | committed at 2b56a6fa | no prereg-* tag: committed, not tagged |
| Dashboard run browser and gate board | Ilan Danial | FAIL | the dashboard's data layer (optspread.dashboard_data.scan_runs and load_gate_boards) lists every run directory under the runs root that has a valid manifest recording its run.json, of which there is at least one, and reads every gate report JSON sidecar under docs/ | run browser lists 0 of 0 valid manifests under $VEGALAB_DATA/runs (0 rows in all); gate board reads 2 of 2 sidecars | no valid run manifest under $VEGALAB_DATA/runs: nothing real to browse |
| Real data adapter conformance | Hemdutt Rao | PASS | the real optionmetrics source is available (its SPX panel is present) and clears conformance_report, the provider contract and structural invariants every source meets; SKIPPED is not a pass | synthetic PASS (64 chains); optionmetrics_stub PASS (64 chains); optionmetrics PASS (64 chains) | -- |

## What failed first

- **PPO harness at fixed D3 hyperparameters**: stable-baselines3 is not installed, so the pinned set was not compared
- **Deterministic replay from hash**: no valid run manifest under $VEGALAB_DATA/runs
- **Variance-components pilot**: pilot not run
- **Pre-registration committed and tagged**: no prereg-* tag: committed, not tagged
- **Dashboard run browser and gate board**: no valid run manifest under $VEGALAB_DATA/runs: nothing real to browse
