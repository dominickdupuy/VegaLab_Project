"""Job functions for the dead-worker tests of the job runner (#30): no torch, still a job.

Not collected (no ``test_`` prefix). Module-level, so a process pool can pickle them
by name. Each runs ``execute_job`` unless the spec's seed says its worker is to die
without raising, which is what the out-of-memory killer does: ``os._exit`` skips every
handler, so the pool learns of it only from the dead process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from optspread.dashboard.jobs.spec import JOBS_DIR, JobSpec
from optspread.dashboard.jobs.worker import execute_job

DIES_RUNNING = 666  # writes its progress file first, as a job killed mid-run has
DIES_ON_START = 667  # dies before it has written anything, every time it is launched


def dying_job(job_id: str, spec_json: str, runs_root: str) -> str:
    seed = JobSpec.model_validate_json(spec_json).seed
    if seed == DIES_RUNNING:
        state = Path(runs_root) / JOBS_DIR / f"{job_id}.json"
        state.write_text(json.dumps({"done": 0, "total": 1, "phase": "evaluating"}))
        os._exit(137)
    if seed == DIES_ON_START:
        os._exit(137)
    return execute_job(job_id, spec_json, runs_root)
