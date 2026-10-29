"use client";

import { Suspense, useEffect, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, PageHead, Segmented } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { FigureFoot } from "@/components/Export";
import lab from "@/components/lab.module.css";
import { InfoTip, Labelled } from "@/components/InfoTip";
import { Facts, Fold, Reading } from "@/components/PageBits";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { PowerSurfaceCard } from "@/components/views/PowerSurface";
import { api, type Power, type PowerParams } from "@/lib/api";
import { axis, base, type Tokens, useTokens } from "@/lib/chart";
import { hours, num, pct, usd } from "@/lib/format";
import { define } from "@/lib/glossary";
import { guide } from "@/lib/guide";
import { useFetch } from "@/lib/useFetch";
import { useQuery, useQuerySync } from "@/lib/url";

/**
 * The pre-registration's numbers (docs/PRE_REGISTRATION.md A6, A7): the PPO contrast's
 * variance components from the pilot, the confirmatory S and N, the 1,000,000-step
 * budget and the rates measured on the pilot machine. The effect is in vol points of
 * planted premium (D2), converted at A6's delta: the fixed-delta seller's contrast at
 * one vol point, $824.3 of mean P&L per episode.
 */
const PER_VOL = 824.3;
const A7: PowerParams = {
  tau2: 51_838,
  sigma2: 105_996_836,
  n_seeds: 5,
  n_episodes: 500,
  delta: PER_VOL,
  alpha: 0.05,
  target_power: 0.8,
  train_steps: 1_000_000,
  episode_days: 63,
  train_rate: 804,
  eval_rate: 1538,
  cores: 1,
};
const PRESETS = [
  { value: "a7", label: "PPO contrast (A7)", params: A7 },
  { value: "seller", label: "Seller contrast (A7)", params: { ...A7, tau2: 1_026, sigma2: 828_609, train_steps: 0 } },
] as const;
type PresetName = (typeof PRESETS)[number]["value"];
const UNITS = [
  { value: "vol", label: "Vol pts" },
  { value: "usd", label: "$" },
] as const;
type Unit = (typeof UNITS)[number]["value"];
const KEYS = Object.keys(A7) as (keyof PowerParams)[];

function presetOf(p: PowerParams): PresetName | "" {
  return PRESETS.find((x) => KEYS.every((k) => x.params[k] === p[k]))?.value ?? "";
}

/** A slider over log10(value), for the variance components that span decades. */
function LogSlider({ id, label, term, value, min, max, onChange, unit }: { id: string; label: string; term: string; value: number; min: number; max: number; onChange: (v: number) => void; unit: string }) {
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  const pos = value > 0 ? Math.log10(value) : lo;
  return (
    <div className={ui.field} {...guide(id)}>
      <span className={ui.fieldLabel}>
        <Labelled term={term}>
          <label htmlFor={id}>{label}</label>
        </Labelled>
        <span>
          {value === 0 ? "0" : value.toExponential(2)} {unit}
        </span>
      </span>
      <input id={id} className={ui.range} type="range" min={lo - 0.25} max={hi} step={0.01} value={value === 0 ? lo - 0.25 : pos} onChange={(e) => {
        const x = Number(e.target.value);
        onChange(x < lo ? 0 : Number((10 ** x).toPrecision(3)));
      }} />
    </div>
  );
}

function Slider({ id, label, term, value, min, max, step, onChange, show }: { id: string; label: string; term: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; show?: string }) {
  return (
    <div className={ui.field} {...guide(id)}>
      <span className={ui.fieldLabel}>
        <Labelled term={term}>
          <label htmlFor={id}>{label}</label>
        </Labelled>
        <span>{show ?? num(value)}</span>
      </span>
      <input id={id} className={ui.range} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

const inUnit = (x: number, unit: Unit) => (unit === "vol" ? x / PER_VOL : x);
const fmt = (x: number | null | undefined, unit: Unit) => (unit === "vol" ? `${num(x === null || x === undefined ? null : x / PER_VOL, 2)} vol pts` : usd(x));

function powerOption(t: Tokens, p: Power, unit: Unit): EChartsOption {
  const pts = (ys: number[]) => p.curve.delta.map((d, i) => [inUnit(d, unit), ys[i] ?? null]);
  return {
    ...base(t),
    legend: { ...base(t).legend, data: [`Your design (S = ${p.params.n_seeds})`, ...(p.params.n_seeds === 5 ? [] : ["S = 5"]), ...(p.params.n_seeds === 50 ? [] : ["S = 50"])] },
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => pct(v as number, 1) },
    xAxis: { type: "value", name: unit === "vol" ? "Planted premium (vol pts)" : "Effect ($ of mean P&L per episode)", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", min: 0, max: 1, name: "Detection probability", nameLocation: "middle", ...axis(t), nameGap: 40, axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => pct(v) } },
    grid: { ...base(t).grid, left: 60, right: 44, top: 40 },
    series: [
      {
        id: "design",
        name: `Your design (S = ${p.params.n_seeds})`,
        type: "line",
        data: pts(p.curve.power),
        showSymbol: false,
        lineStyle: { width: 2.5, color: t.s1 },
        itemStyle: { color: t.s1 },
        markLine: {
          symbol: "none",
          silent: true,
          label: { color: t.muted, fontSize: 11 },
          lineStyle: { color: t.s5, type: "solid", width: 1 },
          data: [
            { yAxis: p.params.target_power, label: { formatter: `${pct(p.params.target_power)} target` } },
            { xAxis: inUnit(p.params.delta, unit), label: { formatter: "your effect" } },
          ],
        },
      },
      {
        id: "design-point",
        name: `Your design (S = ${p.params.n_seeds})`,
        type: "scatter",
        data: [[inUnit(p.params.delta, unit), p.power]],
        symbolSize: 11,
        itemStyle: { color: t.s1, borderColor: t.surface, borderWidth: 2 },
        tooltip: { show: false },
        z: 5,
      },
      // the comparison lines, minus the one this design already draws
      ...(p.params.n_seeds === 5 ? [] : [{ id: "s5", name: "S = 5", type: "line" as const, data: pts(p.curve.power_s5), showSymbol: false, lineStyle: { width: 1.5, color: t.s2 }, itemStyle: { color: t.s2 } }]),
      ...(p.params.n_seeds === 50 ? [] : [{ id: "s50", name: "S = 50", type: "line" as const, data: pts(p.curve.power_s50), showSymbol: false, lineStyle: { width: 1.5, color: t.s3 }, itemStyle: { color: t.s3 } }]),
    ],
  };
}

function floorOption(t: Tokens, p: Power, unit: Unit): EChartsOption {
  const f = p.floor_vs_seeds;
  const y = (v: number) => inUnit(v, unit);
  const show = (v: number) => (unit === "vol" ? `${num(v, 2)} vol pts` : usd(v));
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, formatter: (x: unknown) => { const q = (x as { value: [number, number] }[])[0]; return q ? `S = ${q.value[0]}<br/>floor <b>${show(q.value[1])}</b>` : ""; } },
    xAxis: { type: "log", logBase: 10, name: "Seeds S (log scale)", nameLocation: "middle", min: 1, max: 100, ...axis(t) },
    yAxis: { type: "value", name: unit === "vol" ? "Detection floor (vol pts)" : "Detection floor ($)", nameLocation: "middle", ...axis(t), nameGap: 50 },
    grid: { ...base(t).grid, left: 70 },
    series: [
      {
        id: "floor",
        name: "floor",
        type: "line",
        data: f.seeds.map((s, i) => [s, f.floor[i] === undefined ? null : y(f.floor[i] ?? 0)]),
        symbolSize: 5,
        lineStyle: { width: 2.5, color: t.s1 },
        itemStyle: { color: t.s1 },
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: t.s5, width: 1, type: "solid" },
          label: { color: t.text2, fontSize: 11, formatter: (q: { name?: string }) => q.name ?? "" },
          data: [
            { xAxis: 5, name: `S = 5: ${show(y(f.floor_s5))}` },
            { xAxis: 50, name: `S = 50: ${show(y(f.floor_s50))}` },
          ],
        },
      },
      {
        id: "marks",
        name: "marks",
        type: "scatter",
        data: [
          [5, y(f.floor_s5)],
          [50, y(f.floor_s50)],
        ],
        symbolSize: 11,
        itemStyle: { color: t.s2, borderColor: t.surface, borderWidth: 2 },
      },
    ],
  };
}

export default function PowerPage() {
  return (
    <Suspense fallback={<Loading what="the power lab" />}>
      <PowerLab />
    </Suspense>
  );
}

function initial(q: ReturnType<typeof useQuery>): PowerParams {
  const named = PRESETS.find((x) => x.value === q.str("preset"))?.params ?? A7;
  const out = { ...named };
  for (const k of KEYS) out[k] = q.num(k, named[k]);
  out.n_seeds = Math.round(Math.min(100, Math.max(2, out.n_seeds)));
  out.n_episodes = Math.round(Math.max(1, out.n_episodes));
  return out;
}

/** The current design in one plain sentence, and what it would take to reach the target. */
function reading(d: Power, unit: Unit) {
  const p = d.params;
  const floor = unit === "vol" ? `${num(d.floor / PER_VOL, 2)} vol points` : `${usd(d.floor)} per episode`;
  const effect = unit === "vol" ? `${num(p.delta / PER_VOL, 2)} vol pt${p.delta === PER_VOL ? "" : "s"}` : usd(p.delta);
  const main = (
    <>
      With <b>{p.n_seeds} seeds</b> of <b>{num(p.n_episodes)} episodes</b> you detect a premium of <b>{floor}</b> {pct(p.target_power)} of the time.
    </>
  );
  const enough = d.power >= p.target_power;
  const need =
    d.episodes_needed === null
      ? `no number of episodes reaches it with ${p.n_seeds} seeds: add seeds (${num(d.seeds_needed)} at this N)`
      : `you need ${num(d.episodes_needed)} episodes per seed at S = ${p.n_seeds}, or ${num(d.seeds_needed)} seeds at N = ${num(p.n_episodes)}`;
  const note = enough
    ? `At the ${effect} you set, power is ${pct(d.power, 0)}. The design costs ${hours(d.design.total_hours)} of compute.`
    : `At the ${effect} you set, power is only ${pct(d.power, 0)}; to reach ${pct(p.target_power)} ${need}.`;
  return { main, note };
}

function PowerLab() {
  const t = useTokens();
  const query = useQuery();
  const [p, setP] = useState<PowerParams>(() => initial(query));
  const [unit, setUnit] = useState<Unit>(() => query.oneOf("unit", ["vol", "usd"] as const, "vol"));
  const [q, setQ] = useState<PowerParams>(p);
  useEffect(() => {
    const id = window.setTimeout(() => setQ(p), 120);
    return () => window.clearTimeout(id);
  }, [p]);
  const preset = presetOf(p);
  const base7 = preset ? PRESETS.find((x) => x.value === preset)?.params ?? A7 : A7;
  useQuerySync(
    { preset: preset && preset !== "a7" ? preset : null, unit, ...Object.fromEntries(KEYS.map((k) => [k, p[k] === base7[k] ? null : p[k]])) },
    { unit: "vol" },
  );
  const power = useFetch((s) => api.power(q, s), JSON.stringify(q));
  const [last, setLast] = useState<Power | undefined>();
  useEffect(() => {
    if (power.data) setLast(power.data);
  }, [power.data]);
  const d = power.data ?? last;
  const set = (k: keyof PowerParams) => (v: number) => setP((x) => ({ ...x, [k]: v }));
  const design = `τ² ${num(q.tau2)} · σ² ${num(q.sigma2)} · S ${q.n_seeds} · N ${num(q.n_episodes)} · α ${q.alpha} · 1 vol pt = ${usd(PER_VOL)}`;
  const read = d ? reading(d, unit) : null;

  return (
    <>
      <PageHead
        eyebrow="Power lab"
        title="How small an edge can you detect?"
        right={
          <>
            <span {...guide("power-unit")}>
              <Segmented label="Effect unit" value={unit} options={UNITS} onChange={setUnit} />
            </span>
            <InfoTip term="power.unit" label="About the effect unit" />
            <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => setP(A7)} title={define("power.reset")}>
              Reset
            </button>
          </>
        }
      />
      <div className={ui.split}>
        <div className={`${ui.stack} ${ui.sticky}`}>
          <Card title="Your design">
            <div className={ui.stack}>
              <Segmented label="Preset" title="Start from" term="power.preset" guide="power-preset" value={preset} options={PRESETS.map((x) => ({ value: x.value, label: x.label }))} onChange={(v) => setP({ ...(PRESETS.find((x) => x.value === v)?.params ?? A7) })} />
              {unit === "vol" ? (
                <Slider id="power-effect" label="Effect to detect" term="power.effect" value={Number((p.delta / PER_VOL).toFixed(2))} min={0.05} max={8} step={0.05} onChange={(v) => setP((x) => ({ ...x, delta: Number((v * PER_VOL).toFixed(2)) }))} show={`${(p.delta / PER_VOL).toFixed(2)} vol pts`} />
              ) : (
                <Slider id="power-effect" label="Effect to detect" term="power.effect" value={p.delta} min={50} max={10000} step={50} onChange={set("delta")} show={usd(p.delta)} />
              )}
              <Slider id="power-seeds" label="Seeds, S" term="seeds" value={p.n_seeds} min={2} max={100} step={1} onChange={set("n_seeds")} />
              <Slider id="power-episodes" label="Episodes per seed, N" term="episodes" value={p.n_episodes} min={1} max={2000} step={1} onChange={set("n_episodes")} />
              <Slider id="power-alpha" label="False-alarm rate, α" term="alpha" value={p.alpha} min={0.005} max={0.2} step={0.005} onChange={set("alpha")} show={`${p.alpha.toFixed(3)}, one-sided`} />
              <Slider id="power-target" label="Target power" term="power.target" value={p.target_power} min={0.55} max={0.99} step={0.01} onChange={set("target_power")} show={pct(p.target_power)} />
              <details className={lab.subFold}>
                <summary>Noise: τ² and σ²</summary>
                <div className={ui.stack}>
                  <LogSlider id="power-tau2" label="Between-seed, τ²" term="tau2" value={p.tau2} min={1e2} max={1e9} unit="$²" onChange={set("tau2")} />
                  <LogSlider id="power-sigma2" label="Within-seed, σ²" term="sigma2" value={p.sigma2} min={1e4} max={1e10} unit="$²" onChange={set("sigma2")} />
                </div>
              </details>
              <details className={lab.subFold}>
                <summary>Compute</summary>
                <div className={ui.stack}>
                  <Slider id="power-steps" label="PPO steps per seed" term="power.steps" value={p.train_steps} min={0} max={5_000_000} step={50_000} onChange={set("train_steps")} />
                  <Slider id="power-train-rate" label="Training rate" term="power.trainRate" value={p.train_rate} min={100} max={5000} step={1} onChange={set("train_rate")} show={`${num(p.train_rate)} steps/s/core`} />
                  <Slider id="power-eval-rate" label="Evaluation rate" term="power.evalRate" value={p.eval_rate} min={100} max={5000} step={1} onChange={set("eval_rate")} show={`${num(p.eval_rate)} steps/s/core`} />
                  <Slider id="power-cores" label="Cores" term="power.cores" value={p.cores} min={1} max={128} step={1} onChange={set("cores")} />
                </div>
              </details>
            </div>
          </Card>
        </div>
        <div className={ui.stack}>
          {power.error ? <ErrorBox what="the power lab" error={power.error} onRetry={power.reload} /> : null}
          {!d ? <Loading what="the power curves" /> : null}
          {d && read ? (
            <Card>
              <div {...guide("power-reading")}>
                <Reading note={read.note}>{read.main}</Reading>
              </div>
            </Card>
          ) : null}
          <div {...guide("power-surface")}>
            <PowerSurfaceCard params={{ ...q, delta: PER_VOL }} seeds={q.n_seeds} mark={{ premium: q.delta / PER_VOL, power: d && d.params.delta === q.delta && d.params.n_seeds === q.n_seeds ? d.power : null }} />
          </div>
          {d ? (
            <>
              <Facts
                items={[
                  { label: "Power at your effect", value: pct(d.power, 1), note: `${fmt(d.params.delta, unit)} · S = ${d.params.n_seeds}, N = ${num(d.params.n_episodes)}`, term: "power" },
                  { label: "Detection floor", value: fmt(d.floor, unit), note: `${unit === "vol" ? usd(d.floor) : `${num(d.floor / PER_VOL, 2)} vol pts`} at ${pct(d.params.target_power)} power`, term: "floor" },
                  { label: "Episodes needed", value: d.episodes_needed === null ? "infeasible" : num(d.episodes_needed), note: d.episodes_needed === null ? "τ²/S alone exceeds the budget" : `per seed at S = ${d.params.n_seeds}; or ${num(d.seeds_needed)} seeds at this N`, term: "power.episodesNeeded" },
                  { label: "Compute", value: hours(d.design.total_hours), note: `train ${hours(d.design.train_hours)} · evaluate ${hours(d.design.eval_hours)}`, term: "power.compute" },
                ]}
              />
              <div className={ui.grid2}>
                <Card title="Power against effect size" info="power.powerCurve" sub="Your design against S = 5 and S = 50.">
                  <Chart option={powerOption(t, d, unit)} height={320} label="Detection probability against effect size for three seed counts" stamp={design} />
                </Card>
                <Card title="Detection floor against seeds" info="power.floorCurve" sub={`At N = ${num(d.params.n_episodes)}; S = 5 needs ${num(d.floor_vs_seeds.ratio, 2)}× the premium S = 50 does.`}>
                  <Chart option={floorOption(t, d, unit)} height={320} label="Detection floor against the number of seeds" stamp={design} />
                </Card>
              </div>
              <Fold title="Sample-size frontier" term="power.frontier" sub={`Fewest episodes per seed count for ${pct(d.params.target_power)} power at ${fmt(d.params.delta, unit)}, and the compute.`}>
                <div className={ui.tableWrap}>
                  <table className={ui.table}>
                    <thead>
                      <tr>
                        <th className={ui.num}>Seeds</th>
                        <th className={ui.num}>Episodes per seed</th>
                        <th className={ui.num}>Power</th>
                        <th className={ui.num}>Train</th>
                        <th className={ui.num}>Evaluate</th>
                        <th className={ui.num}>Total</th>
                      </tr>
                    </thead>
                    <tbody>
                      {d.frontier.map((row) => (
                        <tr key={row.n_seeds} className={row.n_seeds === d.params.n_seeds ? ui.rowSelected : undefined}>
                          <td className={ui.num}>{row.n_seeds}</td>
                          {row.n_episodes === null ? (
                            <td colSpan={5} className={ui.dim}>
                              infeasible: seed-to-seed noise alone exceeds the budget at any N
                            </td>
                          ) : (
                            <>
                              <td className={ui.num}>{num(row.n_episodes)}</td>
                              <td className={ui.num}>{pct(row.power, 1)}</td>
                              <td className={ui.num}>{hours(row.train_hours)}</td>
                              <td className={ui.num}>{hours(row.eval_hours)}</td>
                              <td className={ui.num}>
                                <b>{hours(row.total_hours)}</b>
                              </td>
                            </>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <FigureFoot name="sample-size frontier" stamp={`${design} · ${d.source}`} rows={() => d.frontier.map((r) => ({ ...r }))} />
                <p className={ui.dim} style={{ fontSize: 12.5 }}>
                  Var(grand mean) = τ²/S + σ²/(S·N); standard error {fmt(d.se, unit)}, variance budget {num(d.variance_budget)} at exact t with S = {d.params.n_seeds}.
                </p>
              </Fold>
            </>
          ) : null}
        </div>
      </div>
    </>
  );
}
