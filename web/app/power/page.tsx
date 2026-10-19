"use client";

import { useEffect, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, PageHead, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type Power, type PowerParams } from "@/lib/api";
import { axis, base, type Tokens, useTokens } from "@/lib/chart";
import { hours, num, pct, usd } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

const DEFAULTS: PowerParams = {
  tau2: 250_000,
  sigma2: 1.3e8,
  n_seeds: 10,
  n_episodes: 100,
  delta: 1500,
  alpha: 0.05,
  target_power: 0.8,
  train_steps: 1_000_000,
  episode_days: 63,
  train_rate: 1_000,
  eval_rate: 1_800,
  cores: 1,
};

/** A slider over log10(value), for the variance components that span decades. */
function LogSlider({ label, value, min, max, onChange, unit }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void; unit: string }) {
  const lo = Math.log10(min);
  const hi = Math.log10(max);
  const pos = value > 0 ? Math.log10(value) : lo;
  return (
    <label className={ui.field}>
      <span className={ui.fieldLabel}>
        {label}
        <span>
          {value === 0 ? "0" : value.toExponential(2)} {unit}
        </span>
      </span>
      <input className={ui.range} type="range" min={lo - 0.25} max={hi} step={0.01} value={value === 0 ? lo - 0.25 : pos} onChange={(e) => {
        const x = Number(e.target.value);
        onChange(x < lo ? 0 : Number((10 ** x).toPrecision(3)));
      }} />
    </label>
  );
}

function Slider({ label, value, min, max, step, onChange, show }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; show?: string }) {
  return (
    <label className={ui.field}>
      <span className={ui.fieldLabel}>
        {label}
        <span>{show ?? num(value)}</span>
      </span>
      <input className={ui.range} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  );
}

function powerOption(t: Tokens, p: Power): EChartsOption {
  const pts = (ys: number[]) => p.curve.delta.map((d, i) => [d, ys[i] ?? null]);
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => pct(v as number, 1) },
    xAxis: { type: "value", name: "Effect size δ ($ of mean P&L per episode)", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", min: 0, max: 1, name: "Detection probability", nameLocation: "middle", ...axis(t), nameGap: 40, axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => pct(v) } },
    grid: { ...base(t).grid, left: 60 },
    series: [
      {
        name: `This design (S = ${p.params.n_seeds})`,
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
            { xAxis: p.params.delta, label: { formatter: "δ" } },
          ],
        },
      },
      { name: "S = 5 (typical published)", type: "line", data: pts(p.curve.power_s5), showSymbol: false, lineStyle: { width: 1.5, color: t.s2 }, itemStyle: { color: t.s2 } },
      { name: "S = 50", type: "line", data: pts(p.curve.power_s50), showSymbol: false, lineStyle: { width: 1.5, color: t.s3 }, itemStyle: { color: t.s3 } },
    ],
  };
}

function floorOption(t: Tokens, p: Power): EChartsOption {
  const f = p.floor_vs_seeds;
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, formatter: (x: unknown) => { const q = (x as { value: [number, number] }[])[0]; return q ? `S = ${q.value[0]}<br/>floor <b>${usd(q.value[1])}</b>` : ""; } },
    xAxis: { type: "log", logBase: 10, name: "Seeds S (log scale)", nameLocation: "middle", min: 1, max: 100, ...axis(t) },
    yAxis: { type: "value", name: "Detection floor ($)", nameLocation: "middle", ...axis(t), nameGap: 50 },
    grid: { ...base(t).grid, left: 70 },
    series: [
      {
        name: "floor",
        type: "line",
        data: f.seeds.map((s, i) => [s, f.floor[i] ?? null]),
        symbolSize: 5,
        lineStyle: { width: 2.5, color: t.s1 },
        itemStyle: { color: t.s1 },
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: t.s5, width: 1, type: "solid" },
          label: { color: t.text2, fontSize: 11, formatter: (q: { name?: string }) => q.name ?? "" },
          data: [
            { xAxis: 5, name: `S = 5: ${usd(f.floor_s5)}` },
            { xAxis: 50, name: `S = 50: ${usd(f.floor_s50)}` },
          ],
        },
      },
      {
        name: "marks",
        type: "scatter",
        data: [
          [5, f.floor_s5],
          [50, f.floor_s50],
        ],
        symbolSize: 11,
        itemStyle: { color: t.s2, borderColor: t.surface, borderWidth: 2 },
      },
    ],
  };
}

export default function PowerPage() {
  const t = useTokens();
  const [p, setP] = useState<PowerParams>(DEFAULTS);
  const [q, setQ] = useState<PowerParams>(DEFAULTS);
  useEffect(() => {
    const id = window.setTimeout(() => setQ(p), 120);
    return () => window.clearTimeout(id);
  }, [p]);
  const power = useFetch((s) => api.power(q, s), JSON.stringify(q));
  const [last, setLast] = useState<Power | undefined>();
  useEffect(() => {
    if (power.data) setLast(power.data);
  }, [power.data]);
  const d = power.data ?? last;
  const set = (k: keyof PowerParams) => (v: number) => setP((x) => ({ ...x, [k]: v }));

  return (
    <>
      <PageHead
        eyebrow="Power lab"
        title="Detection floor"
        lede={<code>Var(grand mean) = τ²/S + σ²/(S·N)</code>}
        right={
          <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => setP(DEFAULTS)}>
            Reset
          </button>
        }
      />
      <div className={ui.split}>
        <div className={`${ui.stack} ${ui.sticky}`}>
          <Card title="Design" sub="$ of mean P&L per episode.">
            <div className={ui.stack}>
              <LogSlider label="τ², between-seed variance" value={p.tau2} min={1e2} max={1e9} unit="$²" onChange={set("tau2")} />
              <LogSlider label="σ², within-seed variance" value={p.sigma2} min={1e4} max={1e10} unit="$²" onChange={set("sigma2")} />
              <Slider label="S, seeds" value={p.n_seeds} min={1} max={100} step={1} onChange={set("n_seeds")} />
              <Slider label="N, episodes per seed" value={p.n_episodes} min={1} max={2000} step={1} onChange={set("n_episodes")} />
              <Slider label="δ, effect to detect" value={p.delta} min={50} max={10000} step={50} onChange={set("delta")} show={usd(p.delta)} />
              <Slider label="α, one-sided" value={p.alpha} min={0.005} max={0.2} step={0.005} onChange={set("alpha")} show={p.alpha.toFixed(3)} />
              <Slider label="Target power" value={p.target_power} min={0.55} max={0.99} step={0.01} onChange={set("target_power")} show={pct(p.target_power)} />
              <p className={ui.groupTitle}>Compute</p>
              <Slider label="PPO steps per seed" value={p.train_steps} min={0} max={5_000_000} step={50_000} onChange={set("train_steps")} />
              <Slider label="Training rate, steps/s/core" value={p.train_rate} min={100} max={5000} step={50} onChange={set("train_rate")} />
              <Slider label="Evaluation rate, steps/s/core" value={p.eval_rate} min={100} max={5000} step={50} onChange={set("eval_rate")} />
              <Slider label="Cores" value={p.cores} min={1} max={128} step={1} onChange={set("cores")} />
            </div>
          </Card>
        </div>
        <div className={ui.stack}>
          {power.error ? <ErrorBox what="the power lab" error={power.error} onRetry={power.reload} /> : null}
          {!d ? <Loading what="the power curves" /> : null}
          {d ? (
            <>
              <div className={ui.grid4}>
                <Tile accent label="Power at δ" value={pct(d.power, 1)} note={`S = ${d.params.n_seeds}, N = ${num(d.params.n_episodes)}`} />
                <Tile accent label="Detection floor" value={usd(d.floor)} note={`smallest δ at ${pct(d.params.target_power)} power`} />
                <Tile label="SE of the grand mean" value={usd(d.se)} note={`budget ${usd(Math.sqrt(d.variance_budget))} at δ`} />
                <Tile label="Floor, S = 5 vs S = 50" value={`${num(d.floor_vs_seeds.ratio, 2)}×`} note={`${usd(d.floor_vs_seeds.floor_s5)} vs ${usd(d.floor_vs_seeds.floor_s50)}`} />
                <Tile label="Episodes needed at this S" value={d.episodes_needed === null ? "infeasible" : num(d.episodes_needed)} note={d.episodes_needed === null ? "τ²/S exceeds the budget" : undefined} />
                <Tile label="Seeds needed at this N" value={num(d.seeds_needed)} />
                <Tile label="Compute, this design" value={hours(d.design.total_hours)} note={`train ${hours(d.design.train_hours)} · eval ${hours(d.design.eval_hours)}`} />
                <Tile label="Variance budget" value={num(d.variance_budget)} />
              </div>
              <Card title="Power against effect size" sub="This design, S = 5 and S = 50." source={d.source}>
                <Chart option={powerOption(t, d)} height={340} label="Detection probability against effect size for three seed counts" />
              </Card>
              <div className={ui.grid2}>
                <Card title="Detection floor against seed count" source={d.source}>
                  <Chart option={floorOption(t, d)} height={300} label="Detection floor against the number of seeds" />
                </Card>
                <Card title="Sample-size frontier" sub="Fewest episodes per seed count at δ." source="optspread.power.frontier and budget">
                  <div className={ui.tableWrap}>
                    <table className={ui.table}>
                      <thead>
                        <tr>
                          <th className={ui.num}>S</th>
                          <th className={ui.num}>N</th>
                          <th className={ui.num}>Power</th>
                          <th className={ui.num}>Train</th>
                          <th className={ui.num}>Eval</th>
                          <th className={ui.num}>Total</th>
                        </tr>
                      </thead>
                      <tbody>
                        {d.frontier.map((row) => (
                          <tr key={row.n_seeds}>
                            <td className={ui.num}>{row.n_seeds}</td>
                            {row.n_episodes === null ? (
                              <td colSpan={5} className={ui.dim}>
                                infeasible: τ²/S exceeds the variance budget at any N
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
                </Card>
              </div>
            </>
          ) : null}
        </div>
      </div>
    </>
  );
}
