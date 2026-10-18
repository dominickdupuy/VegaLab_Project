"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { Card, PageHead, Progress, Tile, Verdict } from "@/components/Bits";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type Job, type JobOptions, type Knob, type Spec } from "@/lib/api";
import { ARM_LABEL } from "@/lib/chart";
import { hours, num, when } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";

const GROUP_ORDER = ["Arm", "Market", "Costs", "Structure", "Account", "Run", "Training"];
const SCRIPTED_RATE = 1_800; // steps/s/core, the M1-measured steady state (README, Throughput)

function bounds(k: Knob): string {
  const lo = k.minimum ?? k.exclusive_minimum;
  const hi = k.maximum ?? k.exclusive_maximum;
  if (lo === null && hi === null) return "";
  const l = lo === null ? "" : `${k.exclusive_minimum !== null ? ">" : "≥"} ${lo}`;
  const h = hi === null ? "" : `${k.exclusive_maximum !== null ? "<" : "≤"} ${hi}`;
  return [l, h].filter(Boolean).join(", ");
}

function checkKnob(k: Knob, v: unknown): string {
  if (k.kind === "number" || k.kind === "integer") {
    if (typeof v !== "number" || !Number.isFinite(v)) return "a number is required";
    if (k.kind === "integer" && !Number.isInteger(v)) return "a whole number is required";
    if (k.minimum !== null && v < k.minimum) return `at least ${k.minimum}`;
    if (k.maximum !== null && v > k.maximum) return `at most ${k.maximum}`;
    if (k.exclusive_minimum !== null && v <= k.exclusive_minimum) return `above ${k.exclusive_minimum}`;
    if (k.exclusive_maximum !== null && v >= k.exclusive_maximum) return `below ${k.exclusive_maximum}`;
  }
  return "";
}

/** The cross-field rules JobSpec enforces, checked here first so the form says why. */
function checkSpec(spec: Spec): Record<string, string> {
  const errs: Record<string, string> = {};
  const tenors = (spec.tenor_days as number[]) ?? [];
  if (!tenors.length || tenors.some((x, i) => !Number.isInteger(x) || x < 1 || (i > 0 && x <= (tenors[i - 1] ?? 0))))
    errs.tenor_days = "distinct positive whole days, increasing";
  const idx = spec.expiry_idx as number;
  if (Number.isInteger(idx) && idx >= tenors.length) errs.expiry_idx = `there are only ${tenors.length} tenors`;
  if ((spec.wing_delta as number) >= (spec.short_delta as number)) errs.wing_delta = "must be smaller than the short delta";
  const opening = tenors[idx];
  if (opening !== undefined && (spec.roll_dte as number) >= opening) errs.roll_dte = `must be shorter than the opening tenor (${opening})`;
  if (spec.policy === "ppo" && !spec.total_timesteps) errs.total_timesteps = "PPO needs a training budget";
  return errs;
}

function KnobField({ k, value, error, onChange }: { k: Knob; value: unknown; error?: string; onChange: (v: unknown) => void }) {
  const label = (
    <span className={ui.fieldLabel}>
      {k.name.replaceAll("_", " ")}
      <span>{bounds(k)}</span>
    </span>
  );
  if (k.kind === "boolean")
    return (
      <div className={ui.field}>
        <label className={ui.check}>
          <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
          {k.name.replaceAll("_", " ")}
        </label>
        <span className={ui.fieldHelp}>{k.help}</span>
      </div>
    );
  if (k.kind === "choice")
    return (
      <label className={ui.field}>
        {label}
        <select className={ui.select} value={String(value)} onChange={(e) => onChange(e.target.value)}>
          {k.choices.map((c) => (
            <option key={c} value={c}>
              {ARM_LABEL[c] ?? c}
            </option>
          ))}
        </select>
        <span className={ui.fieldHelp}>{k.help}</span>
      </label>
    );
  if (k.kind === "tenors")
    return (
      <label className={ui.field}>
        {label}
        <input
          className={`${ui.input} ${error ? ui.inputBad : ""}`}
          value={((value as number[]) ?? []).join(", ")}
          onChange={(e) =>
            onChange(
              e.target.value
                .split(/[,\s]+/)
                .filter(Boolean)
                .map(Number),
            )
          }
        />
        <span className={error ? ui.fieldError : ui.fieldHelp}>{error || k.help}</span>
      </label>
    );
  return (
    <label className={ui.field}>
      {label}
      <input
        className={`${ui.input} ${error ? ui.inputBad : ""}`}
        type="number"
        step={k.kind === "integer" ? 1 : "any"}
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
      />
      <span className={error ? ui.fieldError : ui.fieldHelp}>{error || k.help}</span>
    </label>
  );
}

function JobCard({ job }: { job: Job }) {
  return (
    <div className={ui.stack}>
      <div className={ui.spread}>
        <span>
          <b>{ARM_LABEL[job.policy] ?? job.policy}</b> <span className={`${ui.dim} mono`}>job {job.job_id}</span>
        </span>
        <Verdict value={job.state} />
      </div>
      <Progress fraction={job.fraction} state={job.state} />
      <div className={ui.spread}>
        <span className={ui.dim}>
          {job.phase} · {num(job.done)} / {num(job.total)} episodes
        </span>
        {job.run_key ? (
          <Link href={`/runs/?key=${encodeURIComponent(job.run_key)}`}>Open the run →</Link>
        ) : job.run_id ? (
          <span className={`${ui.dim} mono`}>{job.run_id}</span>
        ) : null}
      </div>
      {job.error ? <p className={ui.fieldError}>{job.error}</p> : null}
      {job.log_tail.length ? <pre className={ui.log}>{job.log_tail.join("\n")}</pre> : null}
    </div>
  );
}

function Form({ options }: { options: JobOptions }) {
  const [spec, setSpec] = useState<Spec>({ ...options.defaults });
  const [preset, setPreset] = useState<string>("null_world");
  const [sent, setSent] = useState<{ error?: ApiError; busy: boolean }>({ busy: false });
  const [current, setCurrent] = useState<string | null>(null);
  const jobs = useFetch((s) => api.jobs(s), "jobs");
  const running = (jobs.data ?? []).some((j) => j.state === "queued" || j.state === "running");
  usePoll([jobs], running, 700);

  const knobs = options.knobs.filter((k) => k.name !== "total_timesteps" || spec.policy === "ppo");
  const errs = useMemo(() => {
    const out = checkSpec(spec);
    for (const k of knobs) {
      const e = checkKnob(k, spec[k.name]);
      if (e && !(k.name === "total_timesteps" && spec.policy !== "ppo")) out[k.name] = e;
    }
    return out;
  }, [spec, knobs]);
  const ok = Object.keys(errs).length === 0;
  const steps = (spec.n_episodes as number) * (spec.n_days as number);

  const set = (name: string, v: unknown) => {
    setPreset("");
    setSpec((s) => {
      const next = { ...s, [name]: v };
      if (name === "policy") next.total_timesteps = v === "ppo" ? (s.total_timesteps ?? 100_000) : null;
      return next;
    });
  };

  const launch = async () => {
    setSent({ busy: true });
    try {
      const body = { ...spec };
      if (body.policy !== "ppo") delete body.total_timesteps;
      const job = await api.launch(body);
      setCurrent(job.job_id);
      setSent({ busy: false });
      jobs.reload();
    } catch (err) {
      setSent({ busy: false, error: err instanceof ApiError ? err : new ApiError(0, String(err)) });
    }
  };

  const groups = GROUP_ORDER.map((g) => ({ g, ks: knobs.filter((k) => k.group === g) })).filter((x) => x.ks.length);
  const currentJob = (jobs.data ?? []).find((j) => j.job_id === current);

  return (
    <div className={ui.split}>
      <div className={`${ui.stack} ${ui.sticky}`}>
        <Card title="Presets" sub="Start from a named configuration, then adjust.">
          <div className={ui.stack}>
            {options.presets.map((p) => (
              <button
                key={p.name}
                type="button"
                className={`${ui.btn} ${preset === p.name ? ui.btnPrimary : ""}`}
                style={{ justifyContent: "flex-start", height: "auto", padding: "8px 12px", whiteSpace: "normal", textAlign: "left", flexDirection: "column", alignItems: "flex-start", gap: 2 }}
                onClick={() => {
                  setSpec({ ...options.defaults, ...p.spec });
                  setPreset(p.name);
                }}
              >
                <span>{p.label}</span>
                <span style={{ fontWeight: 400, fontSize: 12.5, opacity: 0.85 }}>{p.description}</span>
              </button>
            ))}
          </div>
        </Card>
        <Card title="This run">
          <div className={ui.grid2}>
            <Tile label="Env steps" value={num(steps)} note={`${num(spec.n_episodes as number)} × ${num(spec.n_days as number)} days`} />
            <Tile label="Estimate" value={hours(steps / SCRIPTED_RATE / 3600)} note="at 1,800 steps/s/core" />
          </div>
          <div className={ui.stack} style={{ marginTop: 12 }}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={!ok || sent.busy} onClick={launch}>
              {sent.busy ? "Submitting…" : "Launch run"}
            </button>
            {!ok ? <span className={ui.fieldError}>Fix {Object.keys(errs).length} field(s) first.</span> : null}
            {sent.error ? <span className={ui.fieldError}>{sent.error.message}</span> : null}
            <span className={ui.fieldHelp}>
              Written through the harness and <code>write_run</code> to <code>runs/launched/&lt;config hash&gt;-s&lt;seed&gt;</code>. The same knobs twice
              reuse the run on disk.
            </span>
          </div>
        </Card>
        {currentJob ? (
          <Card title="Launched">
            <JobCard job={currentJob} />
          </Card>
        ) : null}
      </div>
      <div className={ui.stack}>
        <Card title="Knobs" sub={`Defaults are optspread.config's. ${options.ppo_available ? "PPO is available." : "PPO is hidden: optspread.train or its learner is not installed here."}`}>
          <div className={ui.stack}>
            {groups.map(({ g, ks }) => (
              <div key={g}>
                <p className={ui.groupTitle}>{g}</p>
                <div className={ui.grid3}>
                  {ks.map((k) => (
                    <KnobField key={k.name} k={k} value={spec[k.name]} error={errs[k.name]} onChange={(v) => set(k.name, v)} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
        <Card title="Jobs this session" sub="Newest first. Jobs run in a bounded process pool; the server never blocks on one.">
          {jobs.error ? <ErrorBox what="jobs" error={jobs.error} onRetry={jobs.reload} /> : null}
          {jobs.data && jobs.data.length === 0 ? <Empty title="No jobs yet">Launch a run and its progress appears here.</Empty> : null}
          <div className={ui.stack}>
            {(jobs.data ?? []).map((j) => (
              <div key={j.job_id} className={ui.card} style={{ boxShadow: "none", border: "1px solid var(--line)" }}>
                <JobCard job={j} />
                <p className={ui.dim} style={{ fontSize: 12, marginTop: 6 }}>
                  submitted {when(j.created_at)}
                </p>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

export default function LaunchPage() {
  const options = useFetch((s) => api.jobOptions(s), "options");
  return (
    <>
      <PageHead
        eyebrow="Launch"
        title="Start a run"
        lede={
          <>
            Every knob maps onto a <code>RunConfig</code> field. The result is an ordinary run: hashed, keyed by its config and seed, and listed in{" "}
            <Link href="/runs/">Runs</Link> like any other. Mean P&L is net of costs, in dollars per episode.
          </>
        }
      />
      {options.error ? <ErrorBox what="the launch options" error={options.error} onRetry={options.reload} /> : null}
      {options.loading ? <Loading what="the knobs" /> : null}
      {options.data ? <Form options={options.data} /> : null}
    </>
  );
}
