# Drift sensitivity: the confirmatory sweep in the fair world (A11)

**Post hoc and not pre-registered. The bias was found by an outside review of the code after the confirmatory sweep ran and after its M4 analysis was written; this analysis was written and run on 2026-11-14 and is recorded as amendment A11. It changes nothing in A1, A2 or A7 and reruns or retrains nothing: the confirmatory study, in the world it ran in, stays the headline. Read what follows as a sensitivity of that result to the world's rate, not as a second test.**

Run 2026-11-14 by `python -m scripts.drift_sensitivity` on 6 workers in 9.4 minutes; `optspread/drift.py` defines every number. Synthetic, costs on. Analysis code 1ee83b21; sweep cells written at 8b47f99d.

Grid `grid-7d9d53d340006f02.json` under `~/VegaLab_Project/runs/dom/runs`: S = 5 blocks x N = 500 episodes, premiums 0, 0.25, 0.5, 1, 2, 4 vol points, arms delta_seller, ppo.

## The worlds

The study's path is a zero-drift GBM, but its chain was priced at r = 0.04 and the book earns nothing on cash, so its null world was not fair: a put was priced below its expected payoff under the path and a call above it. The fair world is r = 0 on the same path. Every cell is re-simulated in both, on its own seeds, with the scripted policy or the PPO policy it saved (scored, never retrained):

- **saved**: the run.json rows, as the sweep wrote them (the headline, docs/M4_ANALYSIS.md).
- **original**: the recorded config re-simulated here (the study's world, r = 0.04).
- **fair**: the recorded config with market.r = 0, re-simulated here (the fair world).

**What moves besides the carry.** Two things change with the rate other than what a contract is worth against its payoff. A strike picked by delta moves, since delta depends on r: the seller opens a slightly different structure in the fair world (a 16-delta put about one strike lower on the day-0 chain). And a learned policy was trained in the study's world: scored, not retrained, in the fair one it reads prices and greeks it did not train on, so its fair-world figures mix the bias's removal with a policy evaluated outside the world it learned in. Retraining at r = 0, which A11 leaves out, would separate the two.

**The original world reproduces the run.** 2 of 60 cells reproduce bit for bit here, 56 agree within 1e-09 (P&L and cost, same trades and steps), 2 differ.
- ppo at 1, block 0 (`ea5be29e60901c1a-s100000`), seed 100148: saved P&L 2,510.69, re-simulated 4,096.75 (cost 765.17 against 751.03); the cell's other episodes agree. Known: a cross-platform path split (docs/DEFECTS.md, "One edge episode differs").
- ppo at 2, block 3 (`4a48a04bb0099da1-s101500`), seed 101623: saved P&L -18,213.69, re-simulated -18,245.81 (cost 1,292.66 against 1,006.64); the cell's other episodes agree. Known: the defined-risk margin check, fixed after the sweep (4642b1a, GH-39; docs/DEFECTS.md, "An open defect in the frozen margin check").

The comparison below is original against fair, both re-simulated here. The original world gives the headline's verdicts, floor statuses and rules.

## Mean P&L per premium

Dollars per episode, pooled over the blocks; the fair world minus the original is the bias the arm carried at that premium.

| arm | premium | saved (run.json) | original | fair | fair - original | trades (original, fair) | episodes trading differently |
|---|---|---|---|---|---|---|---|
| delta_seller | 0 | -673 | -673 | -591 | +83 | 4.00, 4.00 | 0% |
| delta_seller | 0.25 | -462 | -462 | -381 | +82 | 4.00, 4.00 | 0% |
| delta_seller | 0.5 | -257 | -257 | -150 | +106 | 4.00, 4.00 | 0% |
| delta_seller | 1 | 148 | 148 | 257 | +109 | 4.00, 4.00 | 0% |
| delta_seller | 2 | 822 | 822 | 1,046 | +224 | 4.00, 4.00 | 0% |
| delta_seller | 4 | 2,030 | 2,030 | 2,291 | +261 | 4.00, 4.00 | 0% |
| ppo | 0 | -335 | -335 | -290 | +45 | 1.87, 1.79 | 4% |
| ppo | 0.25 | 148 | 148 | 71 | -77 | 3.49, 3.41 | 11% |
| ppo | 0.5 | 179 | 179 | 120 | -58 | 3.58, 3.39 | 14% |
| ppo | 1 | 1,061 | 1,062 | 691 | -371 | 9.33, 8.93 | 30% |
| ppo | 2 | 2,341 | 2,341 | 1,868 | -473 | 11.58, 11.54 | 32% |
| ppo | 4 | 4,469 | 4,469 | 4,187 | -282 | 7.97, 7.59 | 26% |

## The decisions (A1, A2), original against fair

Each arm at p against itself at zero, one-sided t on the seed means. Dollars per episode.

| arm | premium | edge | edge (fair) | change | t | t (fair) | detected | detected (fair) |
|---|---|---|---|---|---|---|---|---|
| delta_seller | 0.25 | 211 | 210 | -1 | 13.95 | 16.12 | yes | yes |
| delta_seller | 0.5 | 417 | 440 | +23 | 23.12 | 25.27 | yes | yes |
| delta_seller | 1 | 821 | 847 | +26 | 60.90 | 32.51 | yes | yes |
| delta_seller | 2 | 1,495 | 1,636 | +141 | 69.82 | 90.66 | yes | yes |
| delta_seller | 4 | 2,703 | 2,881 | +178 | 51.62 | 78.11 | yes | yes |
| ppo | 0.25 | 483 | 361 | -121 | 2.05 | 1.55 | no | no |
| ppo | 0.5 | 514 | 411 | -103 | 1.60 | 1.53 | no | no |
| ppo | 1 | 1,397 | 981 | -416 | 3.81 | 3.88 | yes | yes |
| ppo | 2 | 2,676 | 2,159 | -518 | 7.52 | 15.74 | yes | yes |
| ppo | 4 | 4,804 | 4,478 | -326 | 5.45 | 6.31 | yes | yes |

## The power curve, the floor and the gap

Power of a fresh experiment of this design at each premium (M4's median-unbiased noncentrality, its band in brackets), original against fair.

| arm | premium | power | power (fair) | noncentrality | noncentrality (fair) |
|---|---|---|---|---|---|
| delta_seller | 0.25 | 1.000 (0.980 to 1.000) | 1.000 (0.996 to 1.000) | 12.79 | 14.78 |
| delta_seller | 0.5 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 21.18 | 23.16 |
| delta_seller | 1 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 55.79 | 29.79 |
| delta_seller | 2 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 63.96 | 83.05 |
| delta_seller | 4 | 1.000 (1.000 to 1.000) | 1.000 (1.000 to 1.000) | 47.29 | 71.55 |
| ppo | 0.25 | 0.475 (0.023 to 0.969) | 0.331 (0.011 to 0.913) | 1.91 | 1.45 |
| ppo | 0.5 | 0.346 (0.012 to 0.921) | 0.324 (0.011 to 0.909) | 1.50 | 1.43 |
| ppo | 1 | 0.889 (0.117 to 1.000) | 0.898 (0.122 to 1.000) | 3.53 | 3.59 |
| ppo | 2 | 1.000 (0.556 to 1.000) | 1.000 (0.995 to 1.000) | 6.91 | 14.43 |
| ppo | 4 | 0.991 (0.287 to 1.000) | 0.999 (0.396 to 1.000) | 5.03 | 5.80 |

| arm | floor (vol points) | floor (fair) | rule | rule (fair) |
|---|---|---|---|---|
| delta_seller | 0.0594 (below_grid; band 0.0324 to 0.165) | 0.0514 (below_grid; band 0.0281 to 0.141) | t | t |
| ppo | 0.88 (reached; band 0.174 to above the grid) | 0.873 (reached; band 0.206 to 1.53) | t | t |

**Gap** (ppo floor minus delta_seller floor): 0.821 vol points in the original world, 0.822 in the fair world.

## Reading

No verdict changes: every premium detected in the study's world is detected in the fair one and every premium not detected is not, and no floor status or rule in force changes. delta_seller: its mean at zero premium moves by +83 dollars an episode in the fair world, its edge by -1 (-1%) at 0.25, +23 (+6%) at 0.5, +26 (+3%) at 1, +141 (+9%) at 2, +178 (+7%) at 4 dollars, and its floor moves from 0.0594 to 0.0514 vol points (-0.00801); 0% of its episodes trade a different number of times. ppo: its mean at zero premium moves by +45 dollars an episode in the fair world, its edge by -121 (-25%) at 0.25, -103 (-20%) at 0.5, -416 (-30%) at 1, -518 (-19%) at 2, -326 (-7%) at 4 dollars, and its floor moves from 0.88 to 0.873 vol points (-0.00689); 19% of its episodes trade a different number of times. The gap moves from 0.821 to 0.822 vol points (+0.00112).
