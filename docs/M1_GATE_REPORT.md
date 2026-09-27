# M1 gate report

Report date: 2026-09-27

**Verdict: NOT CLEARED** (8 items, 1 failed, 0 without a stated criterion)

| Item | Owner | Verdict | Criterion | Evidence | What failed first |
| --- | --- | --- | --- | --- | --- |
| Wave 0 no-edge gate | Hemdutt Rao | PASS | flat P&L exactly 0 with and without costs; no-cost always-on |t| <= 3; with-cost always-on negative and below no-cost; random bleeds fastest | 100 episodes/arm: always-on -509 free (t=-0.44), -1,049 costed | -- |
| Cost, fill and margin model | Hemdutt Rao | PASS | for every scripted seller: costed P&L equals free P&L minus what was charged, per seed; a cash settlement charges nothing; no structure leaves the book unpaid | 4 sellers, 100 episodes each; measured drag matches prediction to 0.00 dollars | -- |
| Data adapter boundary | Hemdutt Rao | PASS | every available source clears the provider contract and the structural invariants; an unavailable source reports SKIPPED and never PASS | 2 sources conform (synthetic, optionmetrics_stub); 1 skipped (optionmetrics) | -- |
| Leak detection suite | Dominick Dupuy | PASS | no observation field at t predicts the true return into t+1 at a Bonferroni-adjusted alpha; every field in OBS_NAMES is tested under the probe or records why not; every field has a positive control that fires | 630 pairs over 10 seeds; 13 of 14 fields tested, untested: atm_iv; all 14 controls fire and are attributed correctly | -- |
| Leak detection floor | Dominick Dupuy | PASS | the smallest planted leak the detector still catches is recorded | grid (1.0, 0.5, 0.25, 0.1, 0.05), caught (True, True, True, True, False), floor eps = 0.1 | -- |
| Run artifacts and manifests | Ilan Danial | PASS | every run writes a hashed manifest with config, commit, seed and package versions; a manifest missing any required field is refused, not written | run_id adddb0b7e5f604ad-s1, git abbd72c9, 5 versions recorded; an unknown commit is refused | -- |
| Throughput, measured | Ilan Danial | PASS | the schedule uses a measured steps/s figure for the steady-state workload, not the document's unvalidated 500-3,000 estimate | 1,813 steps/s/core over 20,034 steps (always_on) | -- |
| PPO-inclusive throughput | Ilan Danial | FAIL | the training-update arm reports a measured steps/s figure | the ppo arm is wired at the fixed hyperparameters of decision D3 | stable-baselines3 is not installed, so the arm reports 'not measured' |

## What failed first

- **PPO-inclusive throughput**: stable-baselines3 is not installed, so the arm reports 'not measured'
