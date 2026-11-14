# M4 analysis: the power curve, the detection floor and the gap (#12, #13, #14)

Run 2026-11-14 by `python -m scripts.m4_analysis`; `optspread/m4analysis.py` defines every number. Synthetic, costs on.

Grid `grid-7d9d53d340006f02.json` (hash 7d9d53d340006f02) under `$VEGALAB_DATA/runs`: S = 5 blocks x N = 500 episodes, premiums 0, 0.25, 0.5, 1, 2, 4 vol points, arms delta_seller, ppo, 1,000,000 training steps. Analysis code c9d117e0; sweep cells written at 8b47f99d.

**Confirmatory**: the A7 design (grid 7d9d53d340006f02), every cell valid.

**When these rules were written.** The decisions are A1 and A2's, pre-registered. The rest of this analysis (the curve's interpolation and its anchor at zero, the floor, the gap, Brown-Forsythe on seed means, the bootstrap agreement tolerance with its Bonferroni split, and the switch to the bootstrap rule) was written on 2026-10-28, after the sweep ran (2026-10-26) and after its raw arm means were posted in issue #9, and before the analysis code read the sweep's files. These rules are not pre-registered: they enter the pre-registration only when A9 (issue #37) records them, and A9 says they were written after the sweep.

## The decisions (A1, A2)

Each arm at p against itself at zero, one-sided t on the 5 seed means, 4 df, critical 2.132. Dollars per episode.

| arm | premium | estimate | se | t | 95% lower bound | detected | tau^2 | sigma^2 | power (plug-in) |
|---|---|---|---|---|---|---|---|---|---|
| delta_seller | 0.25 | 211 | 15 | 13.95 | 179 | yes | 556.4 | 2.947e+05 | 1.000 |
| delta_seller | 0.5 | 417 | 18 | 23.12 | 378 | yes | 636.8 | 4.947e+05 | 1.000 |
| delta_seller | 1 | 821 | 13 | 60.90 | 793 | yes | 0 (clamped) | 7.284e+05 | 1.000 |
| delta_seller | 2 | 1,495 | 21 | 69.82 | 1,450 | yes | 0 (clamped) | 3.017e+06 | 1.000 |
| delta_seller | 4 | 2,703 | 52 | 51.62 | 2,592 | yes | 0 (clamped) | 8.899e+06 | 1.000 |
| ppo | 0.25 | 483 | 236 | 2.05 | -20 | no | 6.517e+04 | 1.062e+08 | 0.520 |
| ppo | 0.5 | 514 | 320 | 1.60 | -169 | no | 3.812e+05 | 6.608e+07 | 0.378 |
| ppo | 1 | 1,396 | 366 | 3.81 | 615 | yes | 4.088e+05 | 1.311e+08 | 0.926 |
| ppo | 2 | 2,676 | 356 | 7.52 | 1,918 | yes | 3.3e+05 | 1.514e+08 | 1.000 |
| ppo | 4 | 4,804 | 881 | 5.45 | 2,926 | yes | 3.647e+06 | 1.156e+08 | 0.997 |

## The power curve (#12)

Power of a fresh 5 x 500 experiment at each premium: the noncentral t at the median-unbiased noncentrality from the observed t, with the exact 95% band; linear in noncentrality between sweep points, from power 0.05 at zero. Power reaches 80% at noncentrality 3.040.

| arm | premium | noncentrality (band) | power | band | plug-in |
|---|---|---|---|---|---|
| delta_seller | 0.25 | 12.79 (4.60 to 23.45) | 1.000 | 0.980 to 1.000 | 1.000 |
| delta_seller | 0.5 | 21.18 (7.89 to 38.69) | 1.000 | 1.000 to 1.000 | 1.000 |
| delta_seller | 1 | 55.79 (21.13 to 101.68) | 1.000 | 1.000 to 1.000 | 1.000 |
| delta_seller | 2 | 63.96 (24.24 to 116.56) | 1.000 | 1.000 to 1.000 | 1.000 |
| delta_seller | 4 | 47.29 (17.89 to 86.21) | 1.000 | 1.000 to 1.000 | 1.000 |
| ppo | 0.25 | 1.91 (-0.42 to 4.36) | 0.475 | 0.023 to 0.969 | 0.520 |
| ppo | 0.5 | 1.50 (-0.70 to 3.77) | 0.346 | 0.012 to 0.921 | 0.378 |
| ppo | 1 | 3.53 (0.54 to 6.95) | 0.889 | 0.116 to 1.000 | 0.926 |
| ppo | 2 | 6.92 (2.16 to 12.87) | 1.000 | 0.556 to 1.000 | 1.000 |
| ppo | 4 | 5.03 (1.30 to 9.53) | 0.991 | 0.287 to 1.000 | 0.997 |

## The detection floor and the gap (#12)

The smallest premium at which the curve reaches 80%, in vol points:

- **delta_seller**: 0.0594 vol points (band 0.0324 to 0.165), below the first sweep point (0.25), so it rests on the interpolation from the anchor at zero; the measured noncentrality per vol point is 51.2, 42.4, 55.8, 32, 11.8 at 0.25, 0.5, 1, 2, 4, and the floor holds the first of these down to zero.
- **ppo**: 0.88 vol points (band 0.174 to above the grid), between the sweep points 0.5 and 1.

A floor taken where a noisy curve first crosses 80% comes early: on linear truths at S = 5, simulated, its median is 8 to 11% below the true floor and it lies below it in 57 to 61% of draws, so read each floor as about a tenth low; its band [lo, hi] covered the true floor in 97 to 99% of draws.

**Gap** (ppo floor minus delta_seller floor): 0.821 vol points. Band range 0.00898 to --; between sweep points 0.25 to 1 (ranges, not confidence intervals). The delta_seller floor lies below the first sweep point, so the gap inherits its interpolation from the anchor at zero.

## The power the design had (A7's plan against the sweep's variances)

**Planned (A7, on the pilot's PPO variances tau^2 = 51,838, sigma^2 = 105,996,836):** power 0.898 at delta = 824.3 (the seller's pilot effect at 1 vol point), exact-t floor 698.3 dollars per episode, at S = 5 x N = 500. These are pilot estimates, not what the sweep achieved.

**Measured (the sweep's own variances):** the same S = 5 x N = 500 design's power at delta = 824.3 and its exact-t floor, with each contrast's tau^2 and sigma^2 from the sweep.

| arm | premium | tau^2 | sigma^2 | power at delta = 824.3 | floor (dollars) |
|---|---|---|---|---|---|
| delta_seller | 0.25 | 556.4 | 2.947e+05 | 1.000 | 46 |
| delta_seller | 0.5 | 636.8 | 4.947e+05 | 1.000 | 55 |
| delta_seller | 1 (A6's premium) | 0 | 7.284e+05 | 1.000 | 52 |
| delta_seller | 2 | 0 | 3.017e+06 | 1.000 | 106 |
| delta_seller | 4 | 0 | 8.899e+06 | 1.000 | 181 |
| ppo | 0.25 | 6.517e+04 | 1.062e+08 | 0.885 | 716 |
| ppo | 0.5 | 3.812e+05 | 6.608e+07 | 0.681 | 974 |
| ppo | 1 (A6's premium) | 4.088e+05 | 1.311e+08 | 0.584 | 1,114 |
| ppo | 2 | 3.3e+05 | 1.514e+08 | 0.605 | 1,081 |
| ppo | 4 | 3.647e+06 | 1.156e+08 | 0.195 | 2,677 |

At 1 vol point the ppo contrast's measured variances give the design power 0.584 and a floor of 1,114 dollars, against the planned 0.898 and 698.3: quote these, not the plan, as the power the confirmatory design had.

## Is tau^2 the same across the sweep? (#13)

Brown-Forsythe across premiums on seed means, alpha 0.05: on each contrast's seed means (p > 0), and on the arm's own seed means (every premium). It tests the variance of seed means, tau^2 + sigma^2 / N; the split is below. Five means a group: a rejection is a finding, a non-rejection is weak evidence.

| arm | seed means of | W | df | p | equal variance rejected |
|---|---|---|---|---|---|
| delta_seller | the contrast | 0.697 | 4, 20 | 0.603 | no |
| delta_seller | the arm | 0.106 | 5, 24 | 0.990 | no |
| ppo | the contrast | 0.431 | 4, 20 | 0.785 | no |
| ppo | the arm | 0.696 | 5, 24 | 0.632 | no |

- **The test's own size and power at 5 groups of 5** (normal seed means, 1,000 replications): size 0.003 (MC se 0.002) at alpha 0.05; power 0.012 / 0.052 / 0.172 when the largest variance is 4 / 10 / 25 times the smallest, spread geometrically.
- **The test's own size and power at 6 groups of 5** (normal seed means, 1,000 replications): size 0.003 (MC se 0.002) at alpha 0.05; power 0.015 / 0.062 / 0.159 when the largest variance is 4 / 10 / 25 times the smallest, spread geometrically.
- **delta_seller**: its contrasts' seed-mean variance spans 15.1-fold across the premiums.
- **ppo**: its contrasts' seed-mean variance spans 14-fold across the premiums.

So a non-rejection here does not show that tau^2 is the same across premiums.

| arm | premium | contrast tau^2 | contrast sigma^2 / N | arm tau^2 | arm sigma^2 / N |
|---|---|---|---|---|---|
| delta_seller | 0 | -- | -- | 0 | 1.757e+05 |
| delta_seller | 0.25 | 556.4 | 589.4 | 0 | 1.748e+05 |
| delta_seller | 0.5 | 636.8 | 989.3 | 0 | 1.728e+05 |
| delta_seller | 1 | 0 | 1,457 | 0 | 1.695e+05 |
| delta_seller | 2 | 0 | 6,033 | 0 | 1.333e+05 |
| delta_seller | 4 | 0 | 1.78e+04 | 0 | 1.233e+05 |
| ppo | 0 | -- | -- | 8.452e+04 | 1.187e+05 |
| ppo | 0.25 | 6.517e+04 | 2.124e+05 | 2.265e+04 | 1.021e+05 |
| ppo | 0.5 | 3.812e+05 | 1.322e+05 | 3.439e+05 | 1.303e+05 |
| ppo | 1 | 4.088e+05 | 2.622e+05 | 9.143e+04 | 3.049e+05 |
| ppo | 2 | 3.3e+05 | 3.028e+05 | 6.712e+04 | 3.216e+05 |
| ppo | 4 | 3.647e+06 | 2.312e+05 | 2.899e+06 | 2.405e+05 |

## Bootstrap validation (#14)

10,000 resamples of the 5 seed means at each level; the t test's detection rate on them against the analytic rate (`bootstrap_detection`'s: the power with the observed t taken as the noncentrality). The tolerance (written after the sweep; see the top): a level agrees when its gap lies inside what normal seed means give, two-sided reference p at least 0.05 over the arm's levels (1,000 normal replications).

| arm | premium | bootstrap | analytic | gap | normal 95% range | p | agrees | t 95% interval | bootstrap 95% interval | width ratio (normal) |
|---|---|---|---|---|---|---|---|---|---|---|
| delta_seller | 0.25 | 1.000 | 1.000 | +0.000 | +0.000 to +0.000 | 1.000 | yes | 169 to 253 | 187 to 240 | 0.631 (0.623) |
| delta_seller | 0.5 | 1.000 | 1.000 | +0.000 | +0.000 to +0.000 | 1.000 | yes | 367 to 467 | 387 to 447 | 0.600 (0.623) |
| delta_seller | 1 | 1.000 | 1.000 | +0.000 | +0.000 to +0.000 | 1.000 | yes | 784 to 859 | 799 to 846 | 0.634 (0.622) |
| delta_seller | 2 | 1.000 | 1.000 | +0.000 | +0.000 to +0.000 | 1.000 | yes | 1,436 to 1,555 | 1,455 to 1,531 | 0.638 (0.623) |
| delta_seller | 4 | 1.000 | 1.000 | +0.000 | +0.000 to +0.000 | 1.000 | yes | 2,558 to 2,849 | 2,599 to 2,773 | 0.595 (0.622) |
| ppo | 0.25 | 0.512 | 0.520 | -0.007 | -0.097 to +0.216 | 0.617 | yes | -172 to 1,137 | 82 to 883 | 0.612 (0.623) |
| ppo | 0.5 | 0.326 | 0.378 | -0.052 | -0.103 to +0.219 | 0.256 | yes | -376 to 1,403 | 3 to 1,080 | 0.606 (0.622) |
| ppo | 1 | 0.942 | 0.926 | +0.016 | -0.048 to +0.194 | 0.965 | yes | 379 to 2,413 | 667 to 1,918 | 0.615 (0.623) |
| ppo | 2 | 1.000 | 1.000 | +0.000 | +0.000 to +0.032 | 0.823 | yes | 1,689 to 3,664 | 1,982 to 3,189 | 0.611 (0.623) |
| ppo | 4 | 1.000 | 0.997 | +0.003 | +0.000 to +0.121 | 0.807 | yes | 2,359 to 7,249 | 3,032 to 5,917 | 0.590 (0.623) |

- **delta_seller**: every level agrees; the t test stands.
- **ppo**: every level agrees; the t test stands.

At S = 5 the percentile interval is narrower than t by construction (the normal ratio above), and the bootstrap rule itself fires at 0.098 (MC se 0.009) on 5 normal seed means with nothing planted, against 0.05. Agreement says the seed means show nothing normal data would not; at this S it cannot say more.

Its limits at this S. The check barely sees non-normal seed means: by simulation at S = 5 (80 arms of five levels, 1,000 resamples, 300 reference sets, MC se about 0.025) an arm switched rules at 0.025 of normal seed means, 0.050 of t(3) ones and 0.062 of lognormal ones. The rule it switches to is anticonservative, about 0.10 against 0.05 (above). And a switch changes the verdict only: the curve, the floor and the gap above are t-based whichever rule is in force.
