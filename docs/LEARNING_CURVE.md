# PPO learning curve (#4)

Run 2026-10-25 by `python -m scripts.learning_curve`; its docstring states the rule. Synthetic, costs on.

PPO at the frozen D3 hyperparameters, 4 vol points planted, 3 training seeds (2000, 2001, 2002), each evaluated every 250,000 steps on the same 200 paths (seeds 300,000 on). 2.04 core-hours in all.

Mean net P&L per 63-day episode, dollars, with its standard error over episodes:

| steps | seed 2000 mean (se) | seed 2001 mean (se) | seed 2002 mean (se) | mean over seeds |
|---|---|---|---|---|
| 251,904 | 4,948 (933) | 5,318 (902) | 1,908 (485) | 4,058 |
| 501,760 | 4,787 (939) | 4,842 (904) | 4,986 (870) | 4,872 |
| 751,616 | 4,832 (958) | 4,887 (919) | 4,804 (697) | 4,841 |
| 1,001,472 | 5,569 (942) | 5,038 (924) | 4,759 (877) | 5,122 |
| 1,251,328 | 5,758 (891) | 5,202 (965) | 5,659 (881) | 5,540 |
| 1,501,184 | 5,204 (843) | 5,087 (947) | 5,251 (939) | 5,181 |
| 1,751,040 | 5,223 (905) | 4,746 (1,095) | 5,519 (874) | 5,163 |
| 2,000,896 | 5,742 (918) | 5,470 (890) | 5,102 (929) | 5,438 |

The fixed-delta seller on the same paths: 2,083 (se 575).

Gain from each checkpoint to the last (2,000,896 steps), averaged over seeds; flat when the gain is at most 2 times the larger standard error:

| from | gain | se between seeds | se within seeds | |
|---|---|---|---|---|
| 251,904 | 1,379 | 926 | 373 | flat |
| 501,760 | 566 | 244 | 257 | rising |
| 751,616 | 596 | 177 | 286 | rising |
| 1,001,472 | 316 | 76 | 280 | flat |
| 1,251,328 | -102 | 242 | 215 | flat |
| 1,501,184 | 257 | 208 | 269 | flat |
| 1,751,040 | 275 | 351 | 285 | flat |

**Levelled at:** 251,904 steps (every later gain flat).

**Budget:** `total_timesteps = 1,000,000` per seed (levelled by 1,000,000).

## Reading

The rule's first levelled checkpoint, 251,904 steps, is levelled only because one
seed (2002) lagged there and the between-seed standard error of every gain from it is
wide (about 900 to 1,300 dollars); 501,760 and 751,616 still read as rising against
the last checkpoint. The checkpoint that decides the budget is the one a run of
`total_timesteps = 1,000,000` stops at, 1,001,472 (Stable-Baselines3 rounds up to whole
2,048-step rollouts), and it is levelled in its own right: its gains to 1.25M, 1.5M,
1.75M and 2M are 418, 59, 41 and 316 dollars against standard errors of 260 to 360,
all within two. The budget is 1,000,000 by either reading.

At 4 vol points the learner earns about 5,400 dollars an episode against the
fixed-delta seller's 2,083 on the same paths: it finds more than the oracle's trade,
which is the oracle-to-PPO gap of section 4.4 seen from the other side.
