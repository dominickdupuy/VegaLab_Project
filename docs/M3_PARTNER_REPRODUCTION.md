# M3 partner reproduction: PASS

Team: Vega Lab
Commit: 1ee83b218fd4a701202aba8af8de91ec5dac0aab; tag cutoff-2026-11-01 at 34872b2f51d8
Platform: macOS-26.3-arm64-arm-64bit, Python 3.12.8
Versions: gymnasium 1.4.0, numpy 2.5.4, optspread 0.1.0, pandas 3.1.1, pyarrow 26.0.1, pydantic 2.14.3, python 3.12.8, scipy 1.18.1, stable_baselines3 2.9.2, torch 2.15.0
Generated: 2026-11-15T04:55:14+00:00, 18.5 minutes in all

| Step | Result | Seconds | Detail |
| --- | --- | --- | --- |
| ruff | PASS | 1 |  |
| mypy --strict | PASS | 44 |  |
| pytest | PASS | 503 |  |
| wave0 gate | PASS | 9 |  |
| cost drag | PASS | 6 |  |
| leak suite | PASS | 5 |  |
| poisoned demo | PASS | 0 | eps = 0.5 over 10 seeds: poisoned flagged True (worst ret_1d, p = 3.7e-99), its past moved True; clean flagged False, its past moved False |
| sweep cells | PASS | 36 | 3 cell(s) regenerated: 3 within tolerance, 0 byte-identical |
| ppo cell | PASS | 506 | 1 cell(s) regenerated: 1 within tolerance, 0 byte-identical; PPO checked statistically, since its training is not bit-reproducible across platforms (mean within 3 se, trades, no refused open, residual share) |

Regenerated sweep cells:

- PASS delta_seller at 0 block 0 (65046cadd80098cf-s100000): bytes differ, max relative difference 9.6e-13 over 1005 values (tolerance 1e-09), 12s
- PASS delta_seller at 4 block 0 (7ed5d3842e4d92d8-s100000): bytes differ, max relative difference 2.1e-12 over 1005 values (tolerance 1e-09), 12s
- PASS delta_seller at 1 block 3 (7198f9de012d36ed-s101500): bytes differ, max relative difference 1.3e-12 over 1005 values (tolerance 1e-09), 12s
- PASS ppo at 4 block 4 (e608f959c816fc46-s102000): checked statistically (PPO training is not bit-reproducible across platforms): mean P&L z = -0.44 (within 3 se), 20 episodes: 100 opens, 0 refused, residual share 1.3%; bytes differ, 506s

Send back docs/M3_PARTNER_REPRODUCTION.json and the .md beside it (a pull request adding both under docs/, or by email to the Vega Lab team). If a step failed, send them anyway: the failure is the finding.
