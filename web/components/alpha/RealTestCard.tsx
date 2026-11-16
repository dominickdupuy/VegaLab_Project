"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, Chip, Progress, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Reading } from "@/components/PageBits";
import { Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, type Benchmark, get, request } from "@/lib/api";
import { ARM_LABEL, useTokens } from "@/lib/chart";
import { num, usd } from "@/lib/format";
import { guide } from "@/lib/guide";
import type { components } from "@/lib/schema";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";
import { benchKey } from "./AgentsCard";
import { jobLost } from "./JobLine";
import { type ForestItem, forestOption } from "./charts";

/**
 * The selected benchmark's policies played on every real SPX window of A8's evaluation
 * dates, beside the same benchmark's held-out synthetic result. Exploratory, as every
 * real-data result on the site. Where the server has no panel it says so and offers nothing.
 */

type Schemas = components["schemas"];
type RealState = Schemas["AlphaRealOut"];
type RealJob = Schemas["AlphaRealJobOut"];
type RealTest = Schemas["RealTest"];
type RealRow = Schemas["RealRow"];

function specOf(b: Benchmark) {
  return { dataset: b.dataset, arm: b.arm, total_timesteps: b.total_timesteps ?? null, n_seeds: b.n_seeds ?? 1 };
}

function plusMinus(mean: number, se: number): string {
  return `${usd(mean)} ± ${usd(2 * se)}`;
}

/** The test's job, polled until it ends. `onEnd` hears when it is done, or with why when the
 * server no longer has it (or cannot be read), so the card offers the test again. */
function RealJobLine({ id, onEnd }: { id: string; onEnd: (lost?: string) => void }) {
  const job = useFetch((s) => get<RealJob>(`/api/alpha/real/jobs/${encodeURIComponent(id)}`, undefined, s), `alpha-real-job-${id}`);
  const state = job.data?.state;
  const lost = jobLost(job.error);
  const live = !lost && (!state || state === "queued" || state === "running");
  usePoll([job], live, 1500);
  const told = useRef(false);
  useEffect(() => {
    if ((state === "done" || lost) && !told.current) {
      told.current = true;
      onEnd(lost);
    }
  }, [state, lost, onEnd]);
  if (!job.data) return null;
  const d = job.data;
  return (
    <div className={ui.stack}>
      <div className={ui.spread}>
        <span>
          Real-data test, job <span className="mono">{d.job_id}</span>
        </span>
        <Verdict value={d.state} />
      </div>
      <Progress fraction={d.fraction} state={d.state} />
      {d.error ? <p className={ui.fieldError}>{d.error}</p> : null}
      {live && d.log_tail.length ? <pre className={ui.log}>{d.log_tail.slice(-3).join("\n")}</pre> : null}
    </div>
  );
}

const NAME: Record<string, string> = { arm: "Agent", oracle: "Oracle", flat: "Flat" };

function Result({ r }: { r: RealTest }) {
  const t = useTokens();
  const items: ForestItem[] = useMemo(() => {
    const color: Record<string, string> = { arm: t.s1, oracle: t.s2 };
    return r.rows
      .filter((x) => x.policy !== "flat")
      .flatMap((x: RealRow) => [
        { label: `{p|${NAME[x.policy]}}{w|synthetic, held-out}`, mean: x.synthetic_mean, se: x.synthetic_se, color: color[x.policy] ?? t.s5, tip: `${NAME[x.policy]}, synthetic held-out<br/>${plusMinus(x.synthetic_mean, x.synthetic_se)}, n ${x.synthetic_n}` },
        { label: `{p|${NAME[x.policy]}}{w|real SPX windows}`, mean: x.real_mean, se: x.real_se, color: color[x.policy] ?? t.s5, tip: `${NAME[x.policy]}, real SPX windows<br/>${plusMinus(x.real_mean, x.real_se)}, ${num(r.n_windows)} windows` },
      ]);
  }, [t, r]);
  const option = useMemo(() => forestOption(t, items, "Mean net P&L per episode, $", (v) => usd(v)), [t, items]);
  const edge = r.edge;
  const z = edge.value !== null && edge.value !== undefined && edge.se ? edge.value / edge.se : null;
  return (
    <div className={ui.stack}>
      <Reading note={r.note}>{r.verdict}</Reading>
      <div className={ui.tableWrap}>
        <table className={ui.table}>
          <thead>
            <tr>
              <th>Policy</th>
              <th className={ui.num}>Synthetic, held-out (± 2 se)</th>
              <th className={ui.num}>Real SPX windows (± 2 se)</th>
            </tr>
          </thead>
          <tbody>
            {r.rows.map((x) => (
              <tr key={x.policy}>
                <td>
                  {NAME[x.policy] ?? x.policy}
                  {x.policy === "oracle" ? `, ${r.oracle.replaceAll("_", " ")}` : ""}
                </td>
                <td className={ui.num}>{plusMinus(x.synthetic_mean, x.synthetic_se)}</td>
                <td className={ui.num}>{plusMinus(x.real_mean, x.real_se)}</td>
              </tr>
            ))}
            <tr>
              <td>Agent minus flat, real days</td>
              <td />
              <td className={ui.num}>
                {edge.value === null || edge.value === undefined ? "none" : plusMinus(edge.value, edge.se ?? 0)}
                {z === null ? "" : `, z ${num(z, 1)}`}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <Chart
        option={option}
        height={60 + 34 * items.length}
        label="Mean P&L per episode with two-standard-error bars, synthetic held-out paths against real SPX windows"
        rows={() => r.rows.map((x) => ({ policy: x.policy, synthetic_mean: x.synthetic_mean, synthetic_se: x.synthetic_se, real_mean: x.real_mean, real_se: x.real_se, n_windows: r.n_windows }))}
      />
      <p>
        {num(r.n_windows)} windows of {r.n_days} trading days, one starting on every day from {r.first_date} to the last that ends by {r.last_date} (A8&apos;s evaluation dates; the holdout is not read). Neighbouring windows share most of their days, so the se is a stationary bootstrap with a mean block of {num(r.block, 0)} windows, never shorter than the episode
        {(r.n_seeds ?? 1) > 1 ? `, plus the spread of the ${r.n_seeds} training seeds` : ""}.
        {r.reproduced === true ? " The policies were rebuilt from the benchmark's config and seeds and reproduce its held-out synthetic mean exactly." : ""}
        {r.reproduced === false
          ? ` The rebuilt policies do not reproduce the benchmark's held-out synthetic mean (${usd(r.reproduced_mean ?? 0)} here), so the real-day figures are for policies retrained on this machine.`
          : ""}
      </p>
    </div>
  );
}

export function RealTestCard({ bench }: { bench?: Benchmark }) {
  const [tick, setTick] = useState(0);
  const [job, setJob] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError>();
  const key = bench ? `${bench.dataset}-${benchKey(bench)}` : "";
  const spec = bench ? specOf(bench) : undefined;
  const state = useFetch(
    (s) => get<RealState>("/api/alpha/real", spec ? { dataset: spec.dataset, arm: spec.arm, n_seeds: spec.n_seeds, ...(spec.total_timesteps ? { total_timesteps: spec.total_timesteps } : {}) } : undefined, s),
    `alpha-real-${key}-${tick}`,
    Boolean(bench),
  );
  useEffect(() => {
    setJob(undefined);
    setError(undefined);
  }, [key]);
  const ended = useCallback((lost?: string) => {
    setJob(undefined);
    setTick((n) => n + 1);
    if (lost) setError(new ApiError(404, lost));
  }, []);
  if (!bench) return null;
  const start = async () => {
    setBusy(true);
    setError(undefined);
    try {
      setJob((await request<RealJob>("POST", "/api/alpha/real", spec)).job_id);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    }
    setBusy(false);
  };
  const d = state.data;
  const arm = ARM_LABEL[bench.arm] ?? bench.arm;
  return (
    <Card
      title="On real SPX days"
      right={<Chip term={false}>exploratory</Chip>}
      info="The same policies, rebuilt from the benchmark's config and seeds, play every window of the real SPX panel's evaluation dates (pre-registration A8: 2019-01-02 to 2022-08-24) with the same episode length, environment and costs. Real chains carry the premium the market priced, not the planted one. Real-data results are exploratory: the pre-registered analysis is the synthetic sweep."
      source={d ? <code>{d.source}</code> : undefined}
    >
      {state.loading && !d ? <Loading what="the real-data test" /> : null}
      {state.error ? <p className={ui.fieldError}>{state.error.message}</p> : null}
      {d ? (
        <div className={ui.stack}>
          {d.result ? <Result r={d.result} /> : null}
          {!d.available ? <p className={ui.callout}>{d.message}</p> : null}
          {d.available && d.refused ? <p className={ui.callout}>{d.refused}</p> : null}
          {d.available && !d.refused && !d.result && !job ? (
            <>
              <p>
                Score {arm} ({bench.total_timesteps ? `${num(bench.total_timesteps)} steps, ${bench.n_seeds ?? 1} seed${(bench.n_seeds ?? 1) === 1 ? "" : "s"}` : "scripted"}) on every real window from {d.first_date} to {d.last_date}, beside its held-out synthetic result.
                {bench.total_timesteps ? " Each seed is retrained first, so this takes about as long as half the benchmark." : ""}
              </p>
              <div className={ui.row}>
                <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy} onClick={start} {...guide("alpha-real-run")}>
                  {busy ? "Submitting…" : "Test on real SPX days"}
                </button>
              </div>
            </>
          ) : null}
          {job ? <RealJobLine id={job} onEnd={ended} /> : null}
          {error ? <p className={ui.fieldError}>{error.message}</p> : null}
        </div>
      ) : null}
    </Card>
  );
}
