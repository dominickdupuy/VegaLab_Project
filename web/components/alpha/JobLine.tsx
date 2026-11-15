"use client";

import { useEffect, useRef } from "react";
import { Progress, Verdict } from "@/components/Bits";
import ui from "@/components/ui.module.css";
import { type AlphaJob, api } from "@/lib/api";
import { ARM_LABEL } from "@/lib/chart";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";

/** One benchmark or sweep job, polled until it ends; `onEnd` hears once when it does. */
export function JobLine({ id, onEnd }: { id: string; onEnd: (job: AlphaJob) => void }) {
  const job = useFetch((s) => api.alphaJob(id, s), `alpha-job-${id}`);
  const state = job.data?.state;
  const live = !state || state === "queued" || state === "running";
  usePoll([job], live, 1500);
  const told = useRef(false);
  useEffect(() => {
    if (!live && job.data && !told.current) {
      told.current = true;
      onEnd(job.data);
    }
  }, [live, job.data, onEnd]);
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
