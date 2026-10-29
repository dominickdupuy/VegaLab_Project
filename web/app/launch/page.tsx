"use client";

import Link from "next/link";
import { Suspense, useEffect, useMemo, useState } from "react";
import { Card, Chip, PageHead, Progress, Segmented, SelectField, Verdict } from "@/components/Bits";
import { InfoTip, Labelled } from "@/components/InfoTip";
import { Fold } from "@/components/PageBits";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type Job, type JobOptions, type Knob, type Preset, type Spec } from "@/lib/api";
import { type Preview, runsApi } from "@/lib/api_runs";
import { ARM_LABEL, armColor, SOURCE_LABEL, useTokens } from "@/lib/chart";
import { define } from "@/lib/glossary";
import { hours, num, pct, prem, usd, when } from "@/lib/format";
import { guide } from "@/lib/guide";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";
import { type QueryValue, useQuery, useQuerySync } from "@/lib/url";
import s from "./launch.module.css";

// the knobs the guided steps set; every other one folds away under Advanced
const STEP_KNOBS = new Set(["policy", "source", "iv_premium_vol_pts", "n_episodes", "seed", "total_timesteps"]);
const ADVANCED_ORDER = ["Market", "Costs", "Structure", "Account"];
const PREMIUMS = [0, 0.5, 1, 2, 4];
const EPISODES = [50, 100, 200, 500];
const STEPS = [100_000, 200_000, 500_000, 1_000_000];
const ARM_ORDER = ["delta_seller", "always_on", "random", "flat", "ppo"];
const ARM_DOES: Record<string, string> = {
  delta_seller: "Sells an iron condor at a fixed delta and rolls it before expiry. The rule a planted premium should pay.",
  always_on: "Holds an iron condor all the time, to expiry, then reopens. Selling with no timing.",
  random: "Picks a structure at random each day. Shows what trading costs alone do.",
  flat: "Never trades: the zero line.",
  ppo: "Learns when to sell and which structure from simulated paths, then is scored on fresh ones.",
};
const MAX_TRAIN_SEED = 32768;

/** A knob's value from the URL, parsed by its kind; undefined when absent or malformed. */
function fromUrl(k: Knob, raw: string | null): unknown {
  if (raw === null) return undefined;
  if (k.kind === "boolean") return raw === "1" || raw === "true";
  if (k.kind === "tenors") {
    const xs = raw.split(/[,\s]+/).filter(Boolean).map(Number);
    return xs.every(Number.isFinite) && xs.length ? xs : undefined;
  }
  if (k.kind === "choice") return k.choices.includes(raw) ? raw : undefined;
  const v = Number(raw);
  return raw.trim() !== "" && Number.isFinite(v) ? v : undefined;
}

function toUrl(v: unknown): QueryValue {
  if (Array.isArray(v)) return v.join(",");
  if (typeof v === "boolean") return v ? "1" : "0";
  return v === null || v === undefined ? null : (v as string | number);
}

function allPresets(o: JobOptions): Preset[] {
  return o.ppo_preset ? [...o.presets, o.ppo_preset] : o.presets;
}

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
  if (spec.policy === "ppo" && Number(spec.seed) >= MAX_TRAIN_SEED) errs.seed = `PPO trains on its first seed, which must be below ${MAX_TRAIN_SEED.toLocaleString("en-US")}`;
  return errs;
}

function KnobField({ k: knob, value, error, onChange, off }: { k: Knob; value: unknown; error?: string; onChange: (v: unknown) => void; off?: string }) {
  const k = { ...knob, help: define(knob.name) ?? knob.help };
  const name = k.label || k.name.replaceAll("_", " ");
  const label = (
    <span className={ui.fieldLabel}>
      {name}
      <span>{off ? "not used" : bounds(k)}</span>
    </span>
  );
  if (off)
    return (
      <label className={ui.field}>
        {label}
        <input className={ui.input} disabled value={value === null || value === undefined ? "" : Array.isArray(value) ? value.join(", ") : String(value)} readOnly aria-describedby={`off-${k.name}`} />
        <span className={ui.fieldHelp} id={`off-${k.name}`}>
          {off}
        </span>
      </label>
    );
  if (k.kind === "boolean")
    return (
      <div className={ui.field}>
        <label className={ui.check}>
          <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
          {name}
        </label>
        <span className={ui.fieldHelp}>{k.help}</span>
      </div>
    );
  if (k.kind === "choice")
    return (
      <div className={ui.field}>
        <SelectField label={name} term={k.name} value={String(value)} options={k.choices} render={(c) => ARM_LABEL[c] ?? c} onChange={(v) => v !== "" && onChange(v)} />
        <span className={ui.fieldHelp}>{k.help}</span>
      </div>
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

function initial(options: JobOptions, q: ReturnType<typeof useQuery>): { spec: Spec; preset: string } {
  const named = allPresets(options).find((p) => p.name === q.str("preset"));
  const usable = named && (named.spec.policy !== "ppo" || options.ppo_available) ? named : undefined;
  const spec: Spec = { ...options.defaults, ...(usable?.spec ?? {}) };
  let edited = false;
  for (const k of options.knobs) {
    const v = fromUrl(k, q.params.get(k.name));
    if (v !== undefined) {
      spec[k.name] = v;
      if (k.name !== "source") edited = true; // the data source rides along with a preset
    }
  }
  if (spec.policy !== "ppo") spec.total_timesteps = null;
  if (!(options.sources ?? ["synthetic"]).includes(String(spec.source ?? "synthetic"))) spec.source = "synthetic";
  return { spec, preset: usable && !edited ? usable.name : usable ? "" : edited ? "" : "null_world" };
}

/** A spec settled for a moment: the preview asks once the knobs stop moving. */
function useSettled<T>(value: T, ms = 250): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setV(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return v;
}

function StepHead({ n, title, tip }: { n: number; title: string; tip?: string }) {
  return (
    <span className={s.step}>
      <span className={s.num} aria-hidden="true">
        {n}
      </span>
      <span>
        {title} {tip ? <InfoTip term={tip} /> : null}
      </span>
    </span>
  );
}

function Choices<T extends number>({ value, options, render, onChange, label }: { value: T; options: T[]; render: (v: T) => string; onChange: (v: T) => void; label: string }) {
  return (
    <div className={s.choices} role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button key={o} type="button" role="radio" aria-checked={o === value} className={`${ui.btn} ${ui.btnSmall} ${o === value ? ui.btnPrimary : ""}`} onClick={() => onChange(o)}>
          {render(o)}
        </button>
      ))}
    </div>
  );
}

function PreviewCard({ p, loading, spec }: { p: Preview | undefined; loading: boolean; spec: Spec }) {
  const t = useTokens();
  if (!p) return <Card title="Preview">{loading ? <Loading what="the preview" /> : null}</Card>;
  if (!p.ok)
    return (
      <Card title="Preview" sub="This run cannot start yet:">
        <ul className={ui.fieldError} style={{ margin: 0, paddingLeft: 18 }}>
          {p.errors.map((e) => (
            <li key={e}>{e}</li>
          ))}
        </ul>
      </Card>
    );
  const d = p.day0;
  const est = p.estimate;
  return (
    <Card title="What this run will do" sub={p.world} source={`Run id ${p.run_id}: the config hash and the first seed. The same choices always give the same id and the same numbers.`}>
      <div className={ui.stack}>
        <div>
          <p className={ui.groupTitle} style={{ margin: "0 0 4px" }}>
            Day 0 of the first episode {d?.reference ? <InfoTip text="PPO decides for itself once trained; shown is what the fixed-delta seller opens on the same day, for scale." /> : null}
          </p>
          {d ? (
            <>
              <p className={s.explain}>{d.plain}</p>
              {d.legs.length ? (
                <table className={s.legs} style={{ marginTop: 6 }}>
                  <thead>
                    <tr>
                      <th>Leg</th>
                      <th>Strike</th>
                      <th>Delta</th>
                      <th>IV</th>
                      <th>Price</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.legs.map((l) => (
                      <tr key={`${l.right}${l.strike}${l.qty}`}>
                        <td>
                          {l.qty < 0 ? "short" : "long"} {l.right === "P" ? "put" : "call"}
                        </td>
                        <td>{num(l.strike)}</td>
                        <td>{num(l.delta, 2)}</td>
                        <td>{pct(l.iv, 1)}</td>
                        <td>{num(l.price, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
              {d.credit !== null ? (
                <p className={ui.dim} style={{ fontSize: 12.5, margin: "6px 0 0" }}>
                  index {num(d.spot)} · credit {usd(d.credit)} · margin {usd(d.margin)} <InfoTip term="margin" />
                  {d.max_loss !== null ? ` · max loss ${usd(d.max_loss)}` : ""}
                  {d.breakevens.length ? ` · breakevens ${d.breakevens.map((b) => num(b)).join(" and ")}` : ""}
                </p>
              ) : null}
            </>
          ) : (
            <p className={ui.dim}>{p.day0_note || "No day 0 to show."}</p>
          )}
        </div>
        <div>
          <p className={ui.groupTitle} style={{ margin: "0 0 4px" }}>
            Time on one core
          </p>
          <span className={s.big}>{hours(est?.total_hours)}</span>{" "}
          <span className={ui.dim} style={{ fontSize: 12.5 }}>
            {est ? (est.train_steps ? `training ${hours(est.train_hours)} + scoring ${hours(est.eval_hours)}` : `${num(est.eval_steps)} simulated days`) : ""}
          </span>
          {p.exists ? (
            <p className={ui.fieldHelp} style={{ margin: "4px 0 0" }}>
              This exact run is already on disk: launching opens it instead of running it again.
            </p>
          ) : null}
        </div>
        <div>
          <p className={ui.groupTitle} style={{ margin: "0 0 4px" }}>
            It will be read against <InfoTip term="paired_seeds" />
          </p>
          {p.peers.length ? (
            p.peers.slice(0, 6).map((x) => (
              <div key={x.key} className={s.peer}>
                <span>
                  <Chip color={armColor(t, x.policy)}>{ARM_LABEL[x.policy] ?? x.policy}</Chip>{" "}
                  <Link href={`/runs/?key=${encodeURIComponent(x.key)}`}>{x.source !== "synthetic" ? "real SPX" : `${prem(x.premium)} vol pts`}</Link>
                  <div className={ui.dim}>
                    {x.note}
                    {x.paired ? ", on the same seeds" : /-s(\d+)$/.test(x.key) ? `, seeds from ${Number(x.key.match(/-s(\d+)$/)?.[1]).toLocaleString("en-US")}` : ""}
                    {x.n_episodes ? ` · ${num(x.n_episodes)} episodes` : ""}
                  </div>
                </span>
                <span className={ui.num} style={{ whiteSpace: "nowrap" }}>{usd(x.mean_pnl)}</span>
              </div>
            ))
          ) : (
            <p className={ui.dim} style={{ margin: 0 }}>
              No run in the same world yet{Number(spec.iv_premium_vol_pts) !== 0 ? ": launch the same arm at zero premium too, so the premium can be read as a paired difference" : ""}.
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}

function Form({ options }: { options: JobOptions }) {
  const t = useTokens();
  const q = useQuery();
  const [start] = useState(() => initial(options, q));
  const [spec, setSpec] = useState<Spec>(start.spec);
  const [preset, setPreset] = useState<string>(start.preset);
  // the URL: the preset, or every knob that differs from the defaults
  const changed: Record<string, QueryValue> = {};
  for (const k of options.knobs) changed[k.name] = (preset && k.name !== "source") || JSON.stringify(spec[k.name]) === JSON.stringify(options.defaults[k.name]) ? null : toUrl(spec[k.name]);
  useQuerySync({ preset: preset || null, ...changed });
  const settled = useSettled(spec);
  const body = useMemo(() => {
    const b = { ...settled };
    if (b.policy !== "ppo") delete b.total_timesteps;
    return b;
  }, [settled]);
  const preview = useFetch((sig) => runsApi.preview(body, sig), `preview|${JSON.stringify(body)}`);
  const [sent, setSent] = useState<{ error?: ApiError; busy: boolean }>({ busy: false });
  const [current, setCurrent] = useState<string | null>(null);
  const jobs = useFetch((sig) => api.jobs(sig), "jobs");
  const running = (jobs.data ?? []).some((j) => j.state === "queued" || j.state === "running");
  usePoll([jobs], running, 700);

  const knobs = options.knobs.filter((k) => k.name !== "source" && (k.name !== "total_timesteps" || spec.policy === "ppo"));
  const sources = options.sources?.length ? options.sources : ["synthetic"];
  const source = String(spec.source ?? "synthetic");
  const real = source !== "synthetic";
  const [evalFirst, evalLast] = options.splits?.eval ?? ["", ""];
  const [trainFirst, trainLast] = options.splits?.train ?? ["", ""];
  const errs = useMemo(() => {
    const out = checkSpec(spec);
    for (const k of knobs) {
      const e = checkKnob(k, spec[k.name]);
      if (e && !(k.name === "total_timesteps" && spec.policy !== "ppo")) out[k.name] = e;
    }
    return out;
  }, [spec, knobs]);
  const serverErrors = preview.data && !preview.data.ok ? preview.data.errors : [];
  const ok = Object.keys(errs).length === 0 && !serverErrors.length;

  const set = (name: string, v: unknown) => {
    if (name !== "source") setPreset("");
    setSpec((x) => {
      const next = { ...x, [name]: v };
      if (name === "policy") next.total_timesteps = v === "ppo" ? (x.total_timesteps ?? 100_000) : null;
      return next;
    });
  };

  const launch = async () => {
    setSent({ busy: true });
    try {
      const b = { ...spec };
      if (b.policy !== "ppo") delete b.total_timesteps;
      const job = await api.launch(b);
      setCurrent(job.job_id);
      setSent({ busy: false });
      jobs.reload();
    } catch (err) {
      setSent({ busy: false, error: err instanceof ApiError ? err : new ApiError(0, String(err)) });
    }
  };

  const advanced = ADVANCED_ORDER.map((g) => ({ g, ks: knobs.filter((k) => k.group === g && !STEP_KNOBS.has(k.name)) })).filter((x) => x.ks.length);
  const currentJob = (jobs.data ?? []).find((j) => j.job_id === current);
  const arms = ARM_ORDER.filter((a) => options.knobs.find((k) => k.name === "policy")?.choices.includes(a) || a === "ppo");
  const premium = Number(spec.iv_premium_vol_pts ?? 0);
  const sigma = Number(spec.sigma ?? 0.2);
  const episodes = Number(spec.n_episodes ?? 0);
  const fieldOf = (name: string) => knobs.find((k) => k.name === name);
  const errorList = [...Object.entries(errs).map(([k, v]) => `${fieldOf(k)?.label || k}: ${v}`), ...serverErrors.filter((e) => !Object.values(errs).some((v) => e.includes(v)))];

  return (
    <div className={s.layout}>
      <div className={ui.stack}>
        <div className={ui.row} aria-label="Quick starts">
          <span className={ui.fieldLabel}>Quick start</span>
          {allPresets(options).map((p) => (
            <button
              key={p.name}
              type="button"
              {...guide(`launch-preset-${p.name}`)}
              disabled={p.spec.policy === "ppo" && !options.ppo_available}
              title={p.description}
              aria-pressed={preset === p.name}
              className={`${ui.btn} ${ui.btnSmall} ${preset === p.name ? ui.btnPrimary : ""}`}
              onClick={() => {
                setSpec({ ...options.defaults, ...p.spec, source: spec.source });
                setPreset(p.name);
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        <Card title={<StepHead n={1} title="Pick a strategy" />} sub="What trades. Each arm is a fixed rule except PPO, which learns.">
          <div className={s.arms}>
            {arms.map((a) => {
              const off = a === "ppo" && !options.ppo_available;
              return (
                <button key={a} type="button" disabled={off} aria-pressed={spec.policy === a} className={`${s.arm} ${spec.policy === a ? s.armOn : ""}`} onClick={() => set("policy", a)} title={off ? "PPO needs the train extra on the server" : undefined}>
                  <span className={s.armName}>
                    <span className={ui.swatch} style={{ background: armColor(t, a) }} aria-hidden="true" />
                    {ARM_LABEL[a] ?? a}
                  </span>
                  <span className={s.armText}>{ARM_DOES[a]}</span>
                </button>
              );
            })}
          </div>
          {spec.policy ? <p className={s.explain} style={{ marginTop: 10 }}>{define(String(spec.policy))}</p> : null}
        </Card>

        <Card title={<StepHead n={2} title="Pick a world" tip="world" />} sub="Where it trades: a simulated market with a premium you plant, or real SPX history.">
          <div className={ui.stack}>
            <Segmented label="World" value={source} options={sources.map((x) => ({ value: x, label: x === "synthetic" ? "Simulated, with a planted premium" : `Real SPX history (${SOURCE_LABEL[x] ?? x})` }))} onChange={(v) => set("source", v)} />
            {real ? (
              <p className={s.explain}>
                Episodes start on real closes and trade each day&apos;s real option surface; nothing is planted, so the premium is whatever the market paid. Every arm is scored only on {evalFirst} to {evalLast}
                {spec.policy === "ppo" ? `, and PPO first trains on ${trainFirst} to ${trainLast}, so it is never scored on a date it learned from` : ""}. <InfoTip term="a8_dates" />
              </p>
            ) : (
              <>
                <span className={ui.fieldLabel}>
                  <Labelled term="premium">Planted premium (vol points)</Labelled>
                </span>
                <div className={s.choices}>
                  <Choices label="Planted premium" value={premium} options={PREMIUMS.includes(premium) ? PREMIUMS : [...PREMIUMS, premium].sort((a, b) => a - b)} render={(v) => (v === 0 ? "0 (null world)" : String(v))} onChange={(v) => set("iv_premium_vol_pts", v)} />
                  <input className={ui.input} style={{ width: 90 }} type="number" step="any" aria-label="Planted premium, any value" value={String(spec.iv_premium_vol_pts ?? "")} onChange={(e) => set("iv_premium_vol_pts", e.target.value === "" ? null : Number(e.target.value))} />
                </div>
                <p className={s.explain}>
                  {premium
                    ? `Options are priced at ${pct(sigma + premium / 100, 0)} vol while the index moves at ${pct(sigma, 0)}: a seller who is paid ${prem(premium)} vol point${premium === 1 ? "" : "s"} too much should earn it, a learner should find it.`
                    : `Options are priced at the same ${pct(sigma, 0)} vol the index moves at. Nothing is there to earn, so a good arm should not make money here before costs; this is the control every premium is compared with.`}
                </p>
              </>
            )}
            {errs.iv_premium_vol_pts ? <span className={ui.fieldError}>{errs.iv_premium_vol_pts}</span> : null}
          </div>
        </Card>

        <Card title={<StepHead n={3} title="Pick a size" />} sub="More episodes narrow the interval; the time on the right follows every choice.">
          <div className={ui.stack}>
            <span className={ui.fieldLabel}>
              <Labelled term="n_episodes">Episodes</Labelled>
            </span>
            <div className={s.choices}>
              <Choices label="Episodes" value={episodes} options={EPISODES.includes(episodes) ? EPISODES : [...EPISODES, episodes].sort((a, b) => a - b)} render={(v) => num(v)} onChange={(v) => set("n_episodes", v)} />
              <input className={ui.input} style={{ width: 90 }} type="number" step={1} aria-label="Episodes, any number" value={String(spec.n_episodes ?? "")} onChange={(e) => set("n_episodes", e.target.value === "" ? null : Number(e.target.value))} />
            </div>
            {errs.n_episodes ? <span className={ui.fieldError}>{errs.n_episodes}</span> : null}
            <label className={ui.field} style={{ maxWidth: 260 }}>
              <span className={ui.fieldLabel}>
                <Labelled term="seed_first">First seed</Labelled>
              </span>
              <input className={`${ui.input} ${errs.seed ? ui.inputBad : ""}`} type="number" step={1} value={String(spec.seed ?? "")} onChange={(e) => set("seed", e.target.value === "" ? null : Number(e.target.value))} />
              <span className={errs.seed ? ui.fieldError : ui.fieldHelp}>{errs.seed || "Runs that share a first seed share their paths, so they compare as pairs."}</span>
            </label>
            {spec.policy === "ppo" ? (
              <>
                <span className={ui.fieldLabel}>
                  <Labelled term="total_timesteps">Training steps</Labelled>
                </span>
                <Choices label="Training steps" value={Number(spec.total_timesteps ?? 0)} options={STEPS} render={(v) => (v >= 1_000_000 ? `${v / 1_000_000}M` : `${v / 1000}k`)} onChange={(v) => set("total_timesteps", v)} />
                {errs.total_timesteps ? <span className={ui.fieldError}>{errs.total_timesteps}</span> : null}
              </>
            ) : null}
          </div>
        </Card>

        <Fold title="Advanced knobs" sub="the market, costs, structure and account; defaults are the study's">
          <div className={ui.stack}>
            {advanced.map(({ g, ks }) => (
              <div key={g}>
                <p className={ui.groupTitle}>{g}</p>
                <div className={ui.grid3}>
                  {ks.map((k) => (
                    <KnobField key={k.name} k={k} value={spec[k.name]} error={errs[k.name]} onChange={(v) => set(k.name, v)} off={real && k.synthetic_only ? `Not used on ${SOURCE_LABEL[source] ?? source}: the real chain brings its own surface, path and carry.` : undefined} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Fold>

        <Card title="Jobs this session">
          {jobs.error ? <ErrorBox what="jobs" error={jobs.error} onRetry={jobs.reload} /> : null}
          {jobs.data && jobs.data.length === 0 ? <Empty title="No jobs yet" /> : null}
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

      <div className={s.side}>
        <PreviewCard p={preview.data} loading={preview.loading} spec={spec} />
        <Card>
          <div className={ui.stack}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={!ok || sent.busy} onClick={launch} {...guide("launch-run")}>
              {sent.busy ? "Submitting…" : preview.data?.exists ? "Launch (reuses the run on disk)" : `Launch ${ARM_LABEL[String(spec.policy)] ?? "run"}`}
            </button>
            {errorList.length ? (
              <ul className={ui.fieldError} style={{ margin: 0, paddingLeft: 18 }}>
                {errorList.map((e) => (
                  <li key={e}>{e}</li>
                ))}
              </ul>
            ) : null}
            {sent.error ? <span className={ui.fieldError}>{sent.error.message}</span> : null}
          </div>
        </Card>
        {currentJob ? (
          <Card title="Launched">
            <JobCard job={currentJob} />
          </Card>
        ) : null}
      </div>
    </div>
  );
}

export default function LaunchPage() {
  return (
    <Suspense fallback={<Loading what="the launch form" />}>
      <Launch />
    </Suspense>
  );
}

function Launch() {
  const options = useFetch((sig) => api.launchOptions(sig), "options");
  return (
    <>
      <PageHead eyebrow="Launch" title="What would a strategy make?" lede="Launch one to find out, in three choices: what trades, where, and for how long. The panel on the right shows what the run will do before you start it." />
      {options.error ? <ErrorBox what="the launch options" error={options.error} onRetry={options.reload} /> : null}
      {options.loading ? <Loading what="the launch form" /> : null}
      {options.data ? <Form options={options.data} /> : null}
    </>
  );
}
