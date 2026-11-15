# Collaboration report

Vega Lab, CIS 4914, Fall 2026. Evaluation 3 (M4, 2026-11-19). Issue #17.

## Team and lanes

The project document (section 9) splits the work into three lanes that meet at a small
number of interfaces, so each could proceed against fakes until the others landed.

| Member | Lane | What the lane owns |
|---|---|---|
| Dominick Dupuy (lead) | Statistical spine | Leak detection suite and its floor, the pre-registration, the variance-components pilot, the power-sweep analysis, validation of the analysis, P&L attribution |
| Hemdutt Rao | Market layer | Pricing, greeks and the option chain; costs, fills and margin; scripted policies; the VRP mapping; the data adapter and its conformance suite |
| Ilan Danial | Machinery | Configuration, manifests and reproducibility; CI and quality gates; the environment and observation pipeline; training-harness wiring; sweep orchestration; the dashboard |

Advisor: Ashish Aggarwal.

## How we worked

- **Meetings.** We met with our advisor, Ashish Aggarwal, every two weeks, and as a team
  every week on our own. The weekly meeting set the week's issues, settled cross-lane
  questions and checked the last week's work against its acceptance criteria.
- **Communication.** Between meetings we worked over calls and a WhatsApp group, which is
  where blockers, design questions and handoffs between lanes were raised as they came up.
- **Constant review.** We reviewed each other's work throughout rather than only at the
  end: a change was read by another member before it was merged, and the platform was
  built iteratively, each lane's work tested against the others' as it landed.
- **One issue per piece of work.** 38 GitHub issues so far (31 closed). Each states its
  context, a recommendation, the ask, acceptance criteria and what it fits with. Issues that
  needed the statistics lane's decision were assigned to Dominick with a recommendation for
  him to confirm or change.
- **A branch and a pull request per week.** Week 5 (#2), week 6 (#33), week 8 (#40), then
  fixes, the refactor and the M4 reports (#41 to #45). Commits name their issue
  (`GH-N type(scope): summary`), so every change traces to the question it answers.
- **Gates, not opinions.** Every milestone closes on a generated report (`make m1` to
  `make m5gate`) that reads the runs and the code and says PASS, FAIL or NOT CLEARED per
  item, with its evidence and its owner. CI runs four jobs on every push: the checks
  (ruff, `mypy --strict`, the test suite with a coverage floor), the dashboard, the
  training extra and the web build.
- **Decisions fixed before the data.** The detection rule, the training budget and the
  confirmatory design were recorded in `docs/PRE_REGISTRATION.md` and tagged (`prereg-m2`)
  before the sweep ran. Anything decided after the sweep is recorded as an amendment that
  says so (A9, A10).
- **A freeze.** Everything that changes a run's numbers (costs, margin, actions,
  observation) was frozen at the `cutoff-2026-11-01` tag; changes after it are listed in
  `docs/DEFECTS.md`, section 1.
- **Shared data and compute.** Market data lives in one shared folder and is never
  committed (it is licensed). The pilot and the 60-cell confirmatory sweep ran on the team's
  laptops; no cluster was used.

## Cross-lane dependencies, and how they went

The project document names two:

1. **The RNG contract (market layer) to the perturbation tests (statistics).** The contract
   was agreed in the week-one interface session, so the leak tests were written against it
   before the refactor landed.
2. **The training harness (machinery) to the variance pilot (statistics).** The pilot
   needed seed-level PPO results; it ran once the harness and the job pool were in place
   (week 6, #33).

Others appeared along the way: the real-data date split (#28) bound the market layer and
the statistics lane, and the dashboard's views waited on the sweep's analysis (#17).

## Contributions by member

**Dominick Dupuy.** The leak detection suite and floor; the pre-registration and its
amendments A1 to A9; the variance-components pilot and the confirmatory sweep (run on his
machine, grid 7d9d53d340006f02); the M4 analysis (power curves, floors, the gap,
homogeneity and bootstrap checks); P&L attribution and the edge decomposition; the M2 to
M4 gate reports; the partner-reproduction instructions and their Windows fixes (#43).

**Hemdutt Rao.** The market layer (pricing, chain, costs, margin, actions, the book,
scripted sellers); the real SPX panel and its data adapter, conformance suite and
OptionMetrics stub; the VRP mapping (the real premium, +0.64 vol points); the
arbitrage-free surface; the extension points (#32); the dashboard showcase; issue
tracking and the weekly integration of each lane's work into one pull request.

**Ilan Danial.** Configuration, manifests and reproducibility (every run re-creates itself
from its record); CI; the environment; the PPO training harness and job pool; sweep
orchestration; the dashboard (every view, the guided workflows, the 3D views); the
refactor into subpackages (#42).

## Problems we hit, and what we did

- **PPO training is not bit-reproducible across platforms**, even at the same library
  versions. We changed what "reproducible" means for a retrained policy to a statistical
  check, and kept bit-for-bit for scripted runs and saved policies (`DEFECTS.md`, section 2).
- **The M4 rules were written after the sweep ran.** We recorded them as amendment A9
  rather than backdating them, and every M4 number is labelled a post-sweep analysis of a
  pre-registered test.
- **The design had less power than planned** (0.584 measured against 0.898 planned). The
  M4 report gives both, labelled.
- **The null world carried a small bias.** An outside review of the code found that the
  simulated market priced options at a 4% rate while its path had no drift, so a put
  credit spread lost about $406 and an iron condor about $95 by construction. We fixed
  the default for future runs, kept the confirmatory study as it ran, and measured the
  bias's effect on it: no verdict changes (`docs/DRIFT_SENSITIVITY.md`, amendment A11).
- **Usability testing** ran in three rounds, each round's problems fixed before the next
  (`docs/USABILITY.md`).
- **Integration across three laptops** (Windows and macOS) surfaced path, line-ending and
  floating-point differences; each is a test now.

## What is left

The final report, the consolidated gate reports and the demo (#19, 2026-12-02). After the
course: the model lab (#46), a dataset generator with planted alpha that a researcher's own
model plugs into.
