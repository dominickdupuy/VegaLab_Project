"use client";

import type { EChartsOption } from "echarts";
import { Suspense, useEffect, useMemo, useState } from "react";
import { Card, PageHead, Progress, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Reading } from "@/components/PageBits";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import {
  type AlphaDataset,
  type AlphaJob,
  type AlphaKnob,
  type AlphaOptions,
  ApiError,
  api,
  type Benchmark,
  type BenchRow,
  type BenchStat,
  type Spec,
} from "@/lib/api";
import { ARM_LABEL, axis, base, type Tokens, useTokens } from "@/lib/chart";
import { num, pct, short, usd, when } from "@/lib/format";
import { type Loadable, useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";
import { setQuery, useQuery } from "@/lib/url";

/**
 * Plant an edge of known size in a synthetic market, freeze the paths, train an agent on
 * exactly those paths, and compare what it earned with what the planted edge paid (the
 * oracle) and with what it earns where nothing is planted (the noise twin, and a copy of the
 * agent trained on it). The open dataset lives in the URL (?dataset=<hash>), so a reload or
 * a shared link shows the same data and its saved results.
 */

const POLICY: Record<string, string> = { arm: "Agent", noise_control: "Agent trained on noise", oracle: "Oracle", flat: "Flat" };
const WORLD: Record<string, string> = {
  "signal/train": "signal, train",
  "signal/heldout": "signal, held-out",
  "twin/train": "noise twin, train",
  "twin/heldout": "noise twin, held-out",
};

function toError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(0, String(err));
}

function NumberField({ k, value, onChange }: { k: AlphaKnob; value: unknown; onChange: (v: number | null) => void }) {
  return (
    <label className={ui.field}>
      <span className={ui.fieldLabel}>{k.label}</span>
      <input
        className={ui.input}
        type="number"
        step={k.kind === "integer" ? 1 : "any"}
        min={k.minimum ?? undefined}
        max={k.maximum ?? undefined}
        title={k.help}
        value={value === null || value === undefined ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
      />
    </label>
  );
}

// ---------- generate ----------

function GenerateCard({ options, dataset, onOpen }: { options: AlphaOptions; dataset?: AlphaDataset; onOpen: (d: AlphaDataset) => void }) {
  const [kind, setKind] = useState(options.generators[1]?.kind ?? "noise");
  const gen = options.generators.find((g) => g.kind === kind) ?? options.generators[0];
  const [params, setParams] = useState<Record<string, Spec>>(() =>
    Object.fromEntries(options.generators.map((g) => [g.kind, Object.fromEntries(g.knobs.map((k) => [k.name, k.default]))])),
  );
  const [spec, setSpec] = useState<Spec>(() => Object.fromEntries(options.spec_knobs.map((k) => [k.name, k.default])));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError>();
  const recent = useFetch((s) => api.alphaDatasets(s), `alpha-recent-${dataset?.hash ?? ""}`);

  // an opened dataset shows the settings it was built with
  useEffect(() => {
    if (!dataset) return;
    const { generator, ...rest } = dataset.spec as Spec & { generator: Spec & { kind: string } };
    setKind(generator.kind);
    setParams((p) => ({ ...p, [generator.kind]: { ...generator } }));
    setSpec((s) => ({ ...s, ...rest }));
  }, [dataset]);

  const build = async () => {
    setBusy(true);
    setError(undefined);
    try {
      onOpen(await api.alphaBuild({ ...spec, generator: { kind, ...params[kind] } }));
    } catch (err) {
      setError(toError(err));
    }
    setBusy(false);
  };
  const open = async (hash: string) => {
    setError(undefined);
    try {
      onOpen(await api.alphaDataset(hash));
    } catch (err) {
      setError(toError(err));
    }
  };

  return (
    <Card title="Generate a world with a known edge" info="The paths are generated once and frozen to disk with their noise twin: the same random shocks with nothing planted.">
      <div className={ui.stack}>
        <label className={ui.field}>
          <span className={ui.fieldLabel}>Generator</span>
          <select className={ui.select} value={kind} onChange={(e) => setKind(e.target.value)}>
            {options.generators.map((g) => (
              <option key={g.kind} value={g.kind}>
                {g.title}
              </option>
            ))}
          </select>
        </label>
        {gen ? (
          <dl className={ui.kv}>
            <dt>Planted</dt>
            <dd>{gen.edge}</dd>
            <dt>Who can take it</dt>
            <dd>{gen.taker}</dd>
          </dl>
        ) : null}
        {gen && gen.knobs.length ? (
          <div className={ui.grid2}>
            {gen.knobs.map((k) => (
              <NumberField key={k.name} k={k} value={params[kind]?.[k.name]} onChange={(v) => setParams((p) => ({ ...p, [kind]: { ...p[kind], [k.name]: v } }))} />
            ))}
          </div>
        ) : null}
        <div className={ui.grid2}>
          {options.spec_knobs.map((k) => (
            <NumberField key={k.name} k={k} value={spec[k.name]} onChange={(v) => setSpec((s) => ({ ...s, [k.name]: v }))} />
          ))}
        </div>
        <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy} onClick={build}>
          {busy ? "Generating…" : "Generate and freeze"}
        </button>
        {error ? <p className={ui.fieldError}>{error.message}</p> : null}
        {recent.data && recent.data.length ? (
          <div className={ui.tableWrap}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>Earlier datasets</th>
                  <th>Hash</th>
                  <th>Built</th>
                </tr>
              </thead>
              <tbody>
                {recent.data.slice(0, 6).map((m) => (
                  <tr key={m.hash} className={`${ui.clickRow} ${m.hash === dataset?.hash ? ui.rowSelected : ""}`} onClick={() => open(m.hash)}>
                    <td>{m.title}</td>
                    <td className="mono">{short(m.hash)}</td>
                    <td style={{ whiteSpace: "nowrap" }}>{when(m.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </Card>
  );
}

// ---------- train ----------

function TrainCard({ options, ds, job, onJob }: { options: AlphaOptions; ds?: AlphaDataset; job: Loadable<AlphaJob>; onJob: (id: string) => void }) {
  const [arm, setArm] = useState(options.learners_available ? "ppo" : "delta_seller");
  const [steps, setSteps] = useState<number | null>(options.default_timesteps);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError>();
  const learned = options.learned_arms.includes(arm);

  const launch = async () => {
    if (!ds) return;
    setBusy(true);
    setError(undefined);
    try {
      onJob((await api.alphaBench({ dataset: ds.hash, arm, total_timesteps: learned ? steps : null })).job_id);
    } catch (err) {
      setError(toError(err));
    }
    setBusy(false);
  };

  return (
    <Card title="Train on exactly that data" info="A learned arm trains twice with the same seed and budget: once on the dataset's train paths, once on their noise twin. Whatever the second copy earns is memorised noise.">
      {!ds ? (
        <p>Generate or open a dataset first.</p>
      ) : (
        <div className={ui.stack}>
          <div className={ui.grid2}>
            <label className={ui.field}>
              <span className={ui.fieldLabel}>Arm</span>
              <select className={ui.select} value={arm} onChange={(e) => setArm(e.target.value)}>
                {options.arms.map((a) => (
                  <option key={a} value={a}>
                    {ARM_LABEL[a] ?? a}
                  </option>
                ))}
              </select>
            </label>
            {learned ? (
              <label className={ui.field}>
                <span className={ui.fieldLabel}>Timesteps per learner</span>
                <input className={ui.input} type="number" step={1} min={2048} value={steps ?? ""} onChange={(e) => setSteps(e.target.value === "" ? null : Number(e.target.value))} />
              </label>
            ) : null}
          </div>
          <p>
            {learned
              ? `${ARM_LABEL[arm] ?? arm} trains on the ${num(ds.n_train)} frozen paths of dataset ${short(ds.hash)}, then a second copy on their noise twin.`
              : "A scripted arm needs no training; it is scored on the same paths in seconds."}
          </p>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy} onClick={launch}>
            {busy ? "Submitting…" : "Train and score"}
          </button>
          {error ? <p className={ui.fieldError}>{error.message}</p> : null}
          {job.data ? (
            <div className={ui.stack}>
              <div className={ui.spread}>
                <span>
                  {ARM_LABEL[job.data.policy] ?? job.data.policy}, job <span className="mono">{job.data.job_id}</span>
                </span>
                <Verdict value={job.data.state} />
              </div>
              <Progress fraction={job.data.fraction} state={job.data.state} />
              {job.data.error ? <p className={ui.fieldError}>{job.data.error}</p> : null}
              {job.data.log_tail.length ? <pre className={ui.log}>{job.data.log_tail.join("\n")}</pre> : null}
            </div>
          ) : null}
          {job.error ? <ErrorBox what="the benchmark job" error={job.error} onRetry={job.reload} /> : null}
        </div>
      )}
    </Card>
  );
}

// ---------- the data ----------

function pathOption(t: Tokens, ds: AlphaDataset, field: "spot" | "iv" | "signal", withTwin: boolean, fmt: (v: number) => string): EChartsOption {
  const p = ds.preview;
  const lines = (name: string, rows: number[][], color: string, opacity: number) =>
    rows.map((row, i) => ({
      name,
      type: "line" as const,
      data: row.map((v, d) => [p.days[d], v]),
      showSymbol: false,
      silent: i > 0,
      lineStyle: { width: 1.2, color, opacity },
      itemStyle: { color },
      emphasis: { disabled: true },
    }));
  return {
    ...base(t),
    tooltip: { show: false },
    legend: { ...base(t).legend, show: withTwin, data: ["Signal world", "Noise twin"] },
    grid: { ...base(t).grid, left: 60 },
    xAxis: { type: "value", min: 0, max: p.days[p.days.length - 1], name: "Trading day", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", scale: true, ...axis(t), axisLabel: { ...axis(t).axisLabel, formatter: fmt } },
    series: [...(withTwin ? lines("Noise twin", p.twin[field], t.muted, 0.35) : []), ...lines("Signal world", p.signal[field], t.s1, 0.6)],
  };
}

function DataCards({ ds }: { ds: AlphaDataset }) {
  const t = useTokens();
  const rows = (field: "spot" | "iv" | "signal") => () =>
    (["signal", "twin"] as const).flatMap((w) => ds.preview[w][field].flatMap((path, i) => path.map((v, day) => ({ world: w, path: i, day, [field]: v }))));
  return (
    <>
      <Card title={`${ds.title}, dataset ${short(ds.hash)}`} info={false} source={<code>{ds.source}</code>}>
        <Reading>{`${num(ds.n_train)} train and ${num(ds.n_heldout)} held-out paths of ${num(Number(ds.spec.n_days))} days. Planted: ${ds.edge.charAt(0).toLowerCase()}${ds.edge.slice(1)}`}</Reading>
      </Card>
      <div className={ui.grid3}>
        <Card title="Spot" info="The first held-out paths, signal world over its noise twin.">
          <Chart option={pathOption(t, ds, "spot", true, (v) => num(v))} height={240} label="Spot paths, signal world against its noise twin" rows={rows("spot")} />
        </Card>
        <Card title="At-the-money implied vol" info="The atm_iv a policy observes each day.">
          <Chart option={pathOption(t, ds, "iv", true, (v) => pct(v, 0))} height={240} label="At-the-money implied vol, signal world against its noise twin" rows={rows("iv")} />
        </Card>
        <Card title={`Planted signal, ${ds.signal_label}`} info={false}>
          <Chart option={pathOption(t, ds, "signal", false, (v) => num(v, 1))} height={240} label={`The planted signal: ${ds.signal_label}`} rows={rows("signal")} />
        </Card>
      </div>
    </>
  );
}

// ---------- compare ----------

const ROW_ORDER: [string, string][] = [
  ["oracle", "signal/heldout"],
  ["oracle", "twin/heldout"],
  ["arm", "signal/train"],
  ["arm", "signal/heldout"],
  ["arm", "twin/heldout"],
  ["noise_control", "twin/train"],
  ["noise_control", "twin/heldout"],
  ["flat", "signal/heldout"],
];

function ordered(rows: BenchRow[]): BenchRow[] {
  return ROW_ORDER.map(([p, w]) => rows.find((r) => r.policy === p && `${r.world}/${r.split}` === w)).filter((r): r is BenchRow => Boolean(r));
}

function forestOption(t: Tokens, rows: BenchRow[]): EChartsOption {
  const color: Record<string, string> = { arm: t.s1, noise_control: t.s4, oracle: t.s2, flat: t.muted };
  return {
    ...base(t),
    animation: false, // a results chart has nothing to reveal, and a re-render must not restart it
    grid: { left: 300, right: 24, top: 8, bottom: 44 },
    tooltip: {
      ...base(t).tooltip,
      trigger: "item",
      formatter: (p: unknown) => {
        const r = rows[(p as { dataIndex: number }).dataIndex];
        return r ? `${POLICY[r.policy]}, ${WORLD[`${r.world}/${r.split}`]}<br/>${usd(r.mean)} ± ${usd(2 * r.se)}, n ${r.n}` : "";
      },
    },
    legend: { show: false },
    xAxis: { type: "value", name: "Mean net P&L per episode, $", nameLocation: "middle", ...axis(t) },
    yAxis: {
      type: "category",
      inverse: true,
      data: rows.map((r) => `{p|${POLICY[r.policy]}}{w|${WORLD[`${r.world}/${r.split}`]}}`),
      ...axis(t, { splitLine: { show: false } }),
      axisLabel: {
        margin: 10,
        rich: {
          p: { width: 160, align: "left", color: t.text, fontFamily: t.font, fontSize: 12.5 },
          w: { width: 120, align: "left", color: t.text2, fontFamily: t.font, fontSize: 12.5 },
        },
      },
    },
    series: [
      {
        type: "custom",
        name: "± 2 se",
        renderItem: (_params: unknown, api: unknown) => {
          const a = api as { value: (i: number) => number; coord: (v: [number, number]) => [number, number] };
          const i = a.value(0);
          const r = rows[i];
          if (!r) return null;
          const lo = a.coord([r.mean - 2 * r.se, i]);
          const hi = a.coord([r.mean + 2 * r.se, i]);
          const c = color[r.policy] ?? t.s5;
          const line = (x1: number, y1: number, x2: number, y2: number) => ({ type: "line", shape: { x1, y1, x2, y2 }, style: { stroke: c, lineWidth: 1.5 } });
          return { type: "group", children: [line(lo[0], lo[1], hi[0], hi[1]), line(lo[0], lo[1] - 4, lo[0], lo[1] + 4), line(hi[0], hi[1] - 4, hi[0], hi[1] + 4)] };
        },
        data: rows.map((r, i) => [i, r.mean - 2 * r.se, r.mean + 2 * r.se]),
        encode: { x: [1, 2], y: 0 },
        z: 1,
      },
      {
        type: "scatter",
        name: "Mean",
        symbol: "rect",
        symbolSize: 8,
        data: rows.map((r, i) => ({ value: [r.mean, i], itemStyle: { color: color[r.policy] ?? t.s5 } })),
        markLine: { silent: true, symbol: "none", lineStyle: { color: t.axis }, label: { show: false }, data: [{ xAxis: 0 }] },
        z: 2,
      },
    ],
  } as EChartsOption;
}

function headline(b: Benchmark): string {
  const arm = ARM_LABEL[b.arm] ?? b.arm;
  const edge = b.attainable.value ?? null;
  const r = b.capture_ratio;
  const c = b.captured;
  const z = c.value !== null && c.value !== undefined && c.se ? c.value / c.se : 0;
  const ratio = r.value !== null && r.value !== undefined && r.se !== null && r.se !== undefined ? `capture ratio ${pct(r.value)} ± ${pct(2 * r.se)}` : "";
  if (edge === null || !ratio) return `There was no measurable edge to take on this dataset; ${arm}: ${c.verdict}.`;
  if (z > 2) return `${arm} took ${pct(r.value ?? 0)} (± ${pct(2 * (r.se ?? 0))}) of a ${usd(edge)} edge per episode.`;
  if (z < -2) return `${arm} did worse where the edge was planted than on its noise twin, so it took none of the ${usd(edge)} edge per episode (${ratio}).`;
  return `${arm}'s gain from the planted signal is not distinguishable from zero; the oracle took ${usd(edge)} per episode (${ratio}).`;
}

function Measure({ name, s, money = true }: { name: string; s?: BenchStat | null; money?: boolean }) {
  if (!s) return null;
  const fmt = (x: number) => (money ? usd(x) : pct(x));
  const value = s.value ?? null;
  const se = s.se ?? null;
  const z = value !== null && se ? value / se : null;
  return (
    <tr>
      <td>{name}</td>
      <td className={ui.num}>{value === null ? "none" : fmt(value)}</td>
      <td className={ui.num}>{se === null ? "" : fmt(2 * se)}</td>
      <td className={ui.num}>{z === null ? "" : num(z, 1)}</td>
      <td>{s.verdict.charAt(0).toUpperCase() + s.verdict.slice(1)}</td>
    </tr>
  );
}

function CompareCard({ bench }: { bench?: Benchmark }) {
  const t = useTokens();
  const rows = useMemo(() => (bench ? ordered(bench.rows) : []), [bench]);
  const option = useMemo(() => forestOption(t, rows), [t, rows]);
  if (!bench) {
    return (
      <Card title="Compare: from the signal, or from noise?" info={false}>
        <Empty title="No result yet">Train an arm on the dataset; the scoreboard appears here when it finishes.</Empty>
      </Card>
    );
  }
  return (
    <Card
      title="Compare: from the signal, or from noise?"
      info="Every comparison is on paths the agent never trained on unless the row says train. The oracle is the best scripted policy on 200 fresh paths the agent never saw, and it must beat flat by 2 se there. The twin shares every shock and cost with the signal world, so signal minus twin is what the planted edge paid. Short premium is negatively skewed: on a few dozen paths a seller often looks profitable because the rare large loss was not drawn."
      source={<code>{`<runs root>/alpha/bench/${bench.dataset}-${bench.arm}-${bench.total_timesteps ?? 0}/benchmark.json`}</code>}
    >
      <div className={ui.stack}>
        <Reading>{headline(bench)}</Reading>
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>Measure</th>
                <th className={ui.num}>Value</th>
                <th className={ui.num}>± 2 se</th>
                <th className={ui.num}>z</th>
                <th>Reading</th>
              </tr>
            </thead>
            <tbody>
              <Measure name="Attainable edge (oracle, signal minus twin)" s={bench.attainable} />
              <Measure name="Agent's edge from the signal (signal minus twin)" s={bench.captured} />
              <Measure name="Capture ratio" s={bench.capture_ratio} money={false} />
              <Measure name="Overfit gap (train minus held-out)" s={bench.overfit_gap} />
              <Measure name="Noise fit (noise-trained, in minus out of sample)" s={bench.noise_fit} />
            </tbody>
          </table>
        </div>
        <Chart
          option={option}
          height={330}
          label="Mean P&L per episode with two-standard-error bars, by policy and data split"
          rows={() => rows.map((r) => ({ policy: POLICY[r.policy] ?? r.policy, world: r.world, split: r.split, mean: r.mean, se: r.se, n: r.n }))}
        />
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>Oracle candidate</th>
                <th className={ui.num}>Fresh-sample mean</th>
                <th className={ui.num}>se</th>
              </tr>
            </thead>
            <tbody>
              {bench.menu.map((m) => (
                <tr key={m.policy} className={m.policy === bench.oracle ? ui.rowSelected : undefined}>
                  <td>
                    {m.policy.replaceAll("_", " ")}
                    {m.policy === bench.oracle ? ", chosen" : ""}
                  </td>
                  <td className={ui.num}>{usd(m.mean)}</td>
                  <td className={ui.num}>{usd(m.se)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Card>
  );
}

// ---------- the page ----------

export default function AlphaPage() {
  return (
    <Suspense fallback={<Loading what="the generators" />}>
      <Alpha />
    </Suspense>
  );
}

function Alpha() {
  const query = useQuery();
  const options = useFetch((s) => api.alphaOptions(s), "alpha-options");
  const [dataset, setDataset] = useState<AlphaDataset>();
  const [jobId, setJobId] = useState<string>();
  const job = useFetch((s) => api.alphaJob(jobId ?? "", s), `alpha-job-${jobId ?? ""}`, Boolean(jobId));
  const running = job.data ? job.data.state === "queued" || job.data.state === "running" : Boolean(jobId);
  usePoll([job], running, 1500);
  const saved = useFetch((s) => api.alphaSaved(dataset?.hash ?? "", s), `alpha-saved-${dataset?.hash ?? ""}-${job.data?.state ?? ""}`, Boolean(dataset));
  const fresh = job.data?.benchmark?.dataset === dataset?.hash ? job.data?.benchmark : undefined;
  const bench = fresh ?? saved.data?.[0];

  const wanted = query.str("dataset");
  useEffect(() => {
    if (wanted && wanted !== dataset?.hash) api.alphaDataset(wanted).then(setDataset, () => undefined);
    // only the URL drives this; opening a dataset writes the URL below
  }, [wanted]);
  const open = (d: AlphaDataset) => {
    setDataset(d);
    setJobId(undefined);
    setQuery({ dataset: d.hash });
  };

  return (
    <>
      <PageHead
        eyebrow="Synthetic alpha"
        title="Does the agent find an edge we planted?"
        lede="Plant an edge of known size, train the agent on exactly that frozen data, and compare what it earned with what was there to earn and with what it earns on pure noise."
      />
      {options.error ? <ErrorBox what="the generators" error={options.error} onRetry={options.reload} /> : null}
      {options.loading ? <Loading what="the generators" /> : null}
      {options.data ? (
        <div className={ui.split}>
          <div className={ui.stack}>
            <GenerateCard options={options.data} dataset={dataset} onOpen={open} />
            <TrainCard options={options.data} ds={dataset} job={job} onJob={setJobId} />
          </div>
          <div className={ui.stack}>
            {dataset ? <DataCards ds={dataset} /> : <Empty title="No dataset open">Generate one, or open an earlier one from the list.</Empty>}
            <CompareCard bench={bench} />
          </div>
        </div>
      ) : null}
    </>
  );
}
