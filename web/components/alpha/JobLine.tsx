"use client";

import { useEffect, useRef } from "react";
import { Progress, Verdict } from "@/components/Bits";
import ui from "@/components/ui.module.css";
import { type AlphaJob, type ApiError, api } from "@/lib/api";
import { ARM_LABEL } from "@/lib/chart";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";

/** What a job line says when the server no longer knows the job (a 404 after a restart). */
export const JOB_GONE = "This job is no longer on the server; it may have restarted. Run it again.";

/** Why a job line stopped asking, in a sentence: the job is gone, or the server answered with an error. */
export function jobLost(error: ApiError | undefined): string | undefined {
  if (!error) return undefined;
  return error.status === 404 ? JOB_GONE : `Could not read this job: ${error.message}`;
}

/**
 * One benchmark or sweep job, polled until it ends; `onEnd` hears once when it does, with
 * the job, or with nothing when the server no longer has it (it may have written the result
 * before it restarted, so the page reads its lists again). A failed read stops the polling.
 */
export function JobLine({ id, onEnd }: { id: string; onEnd: (job?: AlphaJob) => void }) {
  const job = useFetch((s) => api.alphaJob(id, s), `alpha-job-${id}`);
  const state = job.data?.state;
  const lost = jobLost(job.error);
  const live = !lost && (!state || state === "queued" || state === "running");
  usePoll([job], live, 1500);
  const told = useRef(false);
  useEffect(() => {
    if (!live && (job.data || lost) && !told.current) {
      told.current = true;
      onEnd(job.data);
    }
  }, [live, job.data, lost, onEnd]);
  if (lost) {
    return (
      <div className={ui.stack}>
        <span>
          Job <span className="mono">{id}</span>
        </span>
        <p className={ui.fieldError}>{lost}</p>
      </div>
    );
  }
  if (!job.data) return null;
  const d = job.data;
  return (
    <div className={ui.stack}>
      <div className={ui.spread}>
        <span>
          {ARM_LABEL[d.policy] ?? d.policy} {d.kind === "sweep" ? "sweep" : "benchmark"}, job <span className="mono">{d.job_id}</span>
        </span>
        <Verdict value={d.state} />
      </div>
      <Progress fraction={d.fraction} state={d.state} />
      {d.error ? <p className={ui.fieldError}>{d.error}</p> : null}
      {live && d.log_tail.length ? <pre className={ui.log}>{d.log_tail.slice(-4).join("\n")}</pre> : null}
    </div>
  );
}
