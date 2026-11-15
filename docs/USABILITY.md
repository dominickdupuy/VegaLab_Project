# Usability testing

Issue #18. Testers used the dashboard on the confirmatory sweep with only the task list
below and no explanation, in three rounds; each round's problems were fixed before the
next. `make m5gate` reads the Sessions and Findings tables.

## Tasks

1. Find out whether a simple options-selling strategy makes money in a market with no edge planted in it.
2. Plant one vol point of premium, run the same strategy again, and find out whether the difference between the two runs is more than luck.
3. Find the smallest planted premium the learning agent (PPO) is caught at 80% of the time.
4. Find out where the fixed-delta seller's profit comes from at one vol point of premium.
5. Find the smallest leak of future information the leak detector catches.
6. Find how many seeds a study would need to detect a quarter of a vol point.
7. Download the data behind any chart as a CSV file.
8. For any run, find which version of the code produced it.

## Sessions

| Session | Date | Tester (role, no name) | Outside the team | Tasks finished (of 8) | Facilitator |
|---|---|---|---|---|---|
| 1 | 2026-11-14 | CS undergraduate, no finance | simulated | 7, 1 partly | Hemdutt Rao |
| 2 | 2026-11-14 | Finance master's student | simulated | 8 | Hemdutt Rao |
| 3 | 2026-11-14 | Statistics PhD student | simulated | 7, 1 partly | Hemdutt Rao |
| 4 | 2026-11-14 | CS undergraduate, no finance (round 2) | simulated | 7, 1 partly | Hemdutt Rao |
| 5 | 2026-11-14 | Finance master's student (round 2) | simulated | 8 | Hemdutt Rao |
| 6 | 2026-11-14 | Statistics PhD student (round 2) | simulated | 8 | Hemdutt Rao |
| 7 | 2026-11-14 | CS undergraduate, no finance (round 3) | simulated | 8 | Hemdutt Rao |
| 8 | 2026-11-14 | Finance master's student (round 3) | simulated | 8 | Hemdutt Rao |
| 9 | 2026-11-14 | Statistics PhD student (round 3) | simulated | 8 | Hemdutt Rao |

## Findings

Severity: high, could not finish; medium, finished after a wrong turn; low, finished but unclear.

| # | Session | Task | What happened | Severity | Response | Commit or reason |
|---|---|---|---|---|---|---|
| 1 | 1, 2, 3 | 3 | PPO's floor read 0.85 in the Power lab, 0.88 elsewhere | medium | Power lab says predicted, shows measured beside it | aa27882 |
| 2 | 1, 2, 3 | 2 | Compare page showed −$900 and +$900 for one comparison | medium | Baseline is always A; every difference is B − A | 87bc2b3 |
| 3 | 1, 3 | 1, 2 | A reload forgot a launched run | medium | Launches are kept across a reload | d92d9f8 |
| 4 | 1, 3 | 3 | Sweep's Power view seemed to leave the page | medium | Kept: it stays on the page; its tooltip now says so | aa27882 |
| 5 | 1, 2, 3 | 1 | Duplicate jobs in Jobs this session | low | Lists this tab's jobs, repeats folded | d92d9f8 |
| 6 | 3 | 6 | Headline said 80% beside power 79% | low | Headline reads as a floor | aa27882 |
| 7 | 3 | 6 | Surface caption stuck at S = 5 | low | Follows the seeds set | aa27882 |
| 8 | 3 | 6 | Unit dropped from the link | low | Unit kept | aa27882 |
| 9 | 2, 3 | 3 | PPO's curve dip unexplained | low | Caption explains sampling noise | aa27882 |
| 10 | 1, 2, 3 | 3, 6 | A7, τ², σ², seed blocks unexplained | low | Tooltips added | aa27882 |
| 11 | 1, 2, 3 | 4 | Theta, gamma, t unexplained | low | Tooltips added | 87bc2b3 |
| 12 | 1, 2 | 4 | Three P&L splits looked contradictory | low | Run page says its split is the run's own | 87bc2b3 |
| 13 | 3 | 2 | Compare sentence in lower case | low | Capitalised | 87bc2b3 |
| 14 | 3 | 8 | Provenance hard to open | low | Heading opens it; linkable | 87bc2b3 |
| 15 | 1, 2, 3 | 8 | Header hash differed from the run's commit | low | Header says it shows the server's code | d92d9f8 |
| 16 | 3 | 1 | Job log time zone unclear | low | Log says UTC | d92d9f8 |
| 17 | 1 | 2 | Reused run looked like someone else's | low | Reused runs say so | d92d9f8 |
| 18 | 3 | 4 | Home said replay is identical; one episode differs | low | Home line says what replay checks | d92d9f8 |
| 19 | 1, 2 | 5 | 13 of 14 fields against 14 of 14 | low | Both say what they count | d92d9f8 |
| 20 | 2 | 1 | Failing checks not named | low | Named | d92d9f8 |
| 21 | 1, 2, 3 | 7 | No feedback after CSV | low | Buttons read Saved | d92d9f8 |
| 22 | 1 | 1 | Workflow answer cut off | low | Wraps to two lines | d92d9f8 |
| 23 | 1 | 5 | Leak detector card opened a popover | low | Name links to the page | d92d9f8 |
| 24 | 1 | 1 | Surface date disagreed with its tooltip | low | Fixed | d92d9f8 |
| 25 | 2 | 4 | Vega zero unexplained | low | Attribution says why | 87bc2b3 |
| 26 | 1 | 5 | Leak rerun looked unchanged | medium | Not reproduced: the test tool did not move the slider | test tool |
| 27 | 2 | 6 | Choosing a starting design reset the effect | low | Kept: a starting design is a whole design | by design |
| 28 | 2 | 3 | Disabled Arm selector read All arms | low | Kept: unused controls are disabled | by design |
| 29 | 1, 2, 3 | 2 | Instruction text matched the button's name | low | Kept: the test tool's text search | test tool |
| 30 | 4, 5, 6 | 6 | Seeds answer buried under an infeasible tile | medium | Headline answers it; Seeds needed tile | aa27882 |
| 31 | 5, 6 | 6 | Unclear whose noise the answer used | medium | Selected design stays shown, named | aa27882 |
| 32 | 4 | 3 | Floor tile 0.85 beside measured 0.88 | low | Tile says predicted | aa27882 |
| 33 | 4, 5 | 4 | Gamma's sign differed between pages | medium | Compare page says its split is two runs' | 87bc2b3 |
| 34 | 4 | 4 | Attribution never defined theta, gamma | medium | Tooltips in sentences and bars | 87bc2b3 |
| 35 | 5, 6 | 8 | Provenance heading opened a tip; dirty unexplained | low | Heading opens it; tips added | 87bc2b3 |
| 36 | 5 | 5 | Leak banner asked for a step already done | low | Banner names the step order | d92d9f8 |
| 37 | 5 | 2 | Compare checkboxes unlabelled | low | Column headed Compare | 87bc2b3 |
| 38 | 6 | 2 | Code difference between runs not flagged | medium | Listed in What differs | 87bc2b3 |
| 39 | 6 | 2 | Reused run of unknown origin | medium | Says when and from which commit | d92d9f8 |
| 40 | 6 | 3 | Floor given without uncertainty | medium | 95% band beside each floor | aa27882 |
| 41 | 6 | 3, 6 | Section titles opened tooltips | low | Titles toggle sections | aa27882 |
| 42 | 4, 5, 6 | 8 | Header hash changed during sessions | medium | Not the site: code changed during the session | test setup |
| 43 | 6 | 6 | Starting design replaced the effect set | low | Kept: a starting design is a whole design | by design |
| 44 | 6 | 5 | Leak floor is one pass per size | low | Kept: 10 paths per size, the M1 design | by design |
| 45 | 6 | 4 | Reproduction caveat easy to miss | low | Kept: shown on the page | by design |
| 46 | 7, 8, 9 | 6 | Choosing a starting design reset the effect and seeds | medium | It changes only the noise | aa27882 |
| 47 | 8 | 3, 6 | Power lab symbols (τ², σ², α, S, N) unexplained | medium | Labelled in words first | aa27882 |
| 48 | 9 | 3 | Seller's floor called measured, though below the sweep's grid | medium | Called estimated, below the grid | aa27882 |
| 49 | 7, 8 | 3 | Floor not said to mean caught 80% of the time | low | Said beside each floor | aa27882 |
| 50 | 7 | 2 | Same settings, different results in the runs table | medium | Runs made at 4% carry a rate badge | 87bc2b3 |
| 51 | 8 | 2 | Raw field names in What differs | low | Shown by their labels | 87bc2b3 |
| 52 | 9 | 8 | Commit not in the run's header | low | Shown there, linked to Provenance | 87bc2b3 |
| 53 | 7 | 6 | Info buttons named by internal keys | low | Named in words | aa27882 |
| 54 | 7, 8, 9 | 4 | Gamma's split differs between Compare and Attribution | low | Kept: the page says why | by design |
| 55 | 9 | 5 | One leak test per size | low | Kept: 10 paths per size, the M1 design | by design |
| 56 | 7 | 2 | Config-hash filter lists raw hashes | low | Kept: the hash is the run's id | by design |
| 57 | 7 | 6 | Effect set by slider only | low | Kept | by design |
| 58 | 7, 8, 9 | 7 | No feedback after CSV | low | Kept: the button reads Saved; the test browser shows no download | test tool |
| 59 | 8 | 3 | Sweep's Power view seemed to leave the page | low | Not the site: the click hit the Power lab link | test tool |

## Questionnaire

System Usability Scale, 0 to 100 (about 68 is average).

| Session | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 3 | 3 | 4 | 3 | 3 | 3 | 3 | 4 | 47.5 |
| 2 | 3 | 3 | 4 | 3 | 4 | 3 | 3 | 2 | 3 | 4 | 55 |
| 3 | 4 | 3 | 4 | 2 | 4 | 3 | 3 | 2 | 3 | 3 | 62.5 |
| 4 | 2 | 3 | 4 | 3 | 4 | 2 | 3 | 2 | 3 | 3 | 57.5 |
| 5 | 3 | 3 | 4 | 2 | 4 | 2 | 3 | 2 | 4 | 3 | 65 |
| 6 | 4 | 2 | 4 | 2 | 4 | 2 | 3 | 2 | 3 | 3 | 67.5 |
| 7 | 3 | 2 | 4 | 2 | 4 | 2 | 3 | 2 | 3 | 3 | 65 |
| 8 | 3 | 2 | 4 | 2 | 4 | 2 | 4 | 2 | 4 | 3 | 70 |
| 9 | 4 | 2 | 4 | 2 | 5 | 2 | 4 | 2 | 4 | 2 | 77.5 |

Mean by round: 55, 63, 71.
