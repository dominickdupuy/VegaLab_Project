# Cost model calibration

Owner: Hemdutt Rao. Built by `scripts/calibrate_costs.py`; its docstring has the method.

Data: real SPX closing quotes, 2023 to 2025, 548,586 contract-days between 0.05 and 0.16 delta at 20 to 70 calendar days (databento, optionmetrics).

| | min_half_spread | half_spread_frac | mean abs error per share |
|---|---|---|---|
| current | 0.05 | 0.0200 | 0.156 |
| fitted | 0.15 | 0.0075 | 0.053 |

Median half-spread per share by tenor and delta, observed against each model:

| tenor | delta_bucket | n | median mid | median half-spread | fitted | current |
|---|---|---|---|---|---|---|
| ~21 trading days | (0.049, 0.08] | 131083 | 8.3 | 0.15 | 0.15 | 0.166 |
| ~21 trading days | (0.08, 0.12] | 128843 | 12.5 | 0.15 | 0.15 | 0.25 |
| ~21 trading days | (0.12, 0.16] | 105500 | 17.65 | 0.2 | 0.15 | 0.353 |
| ~42 trading days | (0.049, 0.08] | 67287 | 12.75 | 0.15 | 0.15 | 0.255 |
| ~42 trading days | (0.08, 0.12] | 64914 | 19.25 | 0.15 | 0.15 | 0.385 |
| ~42 trading days | (0.12, 0.16] | 50959 | 27.1 | 0.2 | 0.203 | 0.542 |

## Effect on the gates

Both tables were first run with the fitted parameters set temporarily and then
reverted. Every gate holds either way.

**Adopted 2026-10-25 for the Nov 1 freeze (#8).** `CostConfig` now defaults to
the fitted `min_half_spread = 0.15` and `half_spread_frac = 0.0075`; the
functional form, `max(min_half_spread, half_spread_frac * mid)` per share plus
the commission, is unchanged. `make gate` and `make drag` on the adopted values
reproduce the fitted column below to the dollar, and the README tables are
those runs. Size slippage is not modelled and does not apply: every leg is one
contract.

Wave 0 (`make gate`), mean net P&L per 63-day episode, 100 episodes per arm:

| arm | costs | current | fitted |
|---|---|---|---|
| flat | on | 0 | 0 |
| always_on | off | -509 | -509 |
| always_on | on | -1,049 | -786 |
| random | off | -333 | -333 |
| random | on | -9,843 | -4,583 |

Cost drag (`make drag`), dollars per episode; measured equals predicted to the dollar in both:

| seller | current | fitted |
|---|---|---|
| hold_condor | 484 | 257 |
| hold_put_spread | 242 | 128 |
| roll_condor | 603 | 350 |
| roll_put_spread | 370 | 197 |

The current parameters overstate the cost of the legs the environment trades by
about two and a half times at 16 delta. Adopting the fit roughly halves the null
world's cost drag, which moves the oracle's zero-premium point and therefore the
whole sweep; it has to be decided before the pilot runs.
