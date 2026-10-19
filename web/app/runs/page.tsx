"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, Chip, PageHead, Tile, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { RunTable } from "@/components/RunTable";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type Compare, type RunDetail, type RunFilters, type RunRow } from "@/lib/api";
import { ARM_LABEL, armColor, axis, base, histBars, type PremiumPoint, premiumChart, type Tokens, useTokens } from "@/lib/chart";
import { num, pct, short, usd, when } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

export default function RunsPage() {
  return (
    <Suspense fallback={<Loading what="runs" />}>
      <Runs />
    </Suspense>
  );
}

function Runs() {
  const params = useSearchParams();
  const key = params.get("key");
  const a = params.get("a");
  const b = params.get("b");
  if (a && b) return <CompareView a={a} b={b} />;
  if (key) return <DetailView runKey={key} />;
  return <ListView initialQuery={params.get("q") ?? ""} />;
}

// ---------------------------------------------------------------- the list

function FilterSelect<T extends string | number>({
  label,
  value,
  options,
  onChange,
  render,
}: {
  label: string;
  value: T | "";
  options: T[];
  onChange: (v: T | "") => void;
  render?: (v: T) => string;
}) {
  return (
    <label className={ui.field}>
      <span className={ui.fieldLabel}>{label}</span>
      <select
        className={ui.select}
        value={String(value)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") return onChange("");
          const hit = options.find((o) => String(o) === raw);
          onChange(hit === undefined ? "" : hit);
        }}
      >
        <option value="">All</option>
        {options.map((o) => (
          <option key={String(o)} value={String(o)}>
            {render ? render(o) : String(o)}
          </option>
        ))}
      </select>
    </label>
  );
}

function runCurves(rows: RunRow[]): Record<string, PremiumPoint[]> {
  const out: Record<string, PremiumPoint[]> = {};
  for (const r of rows) {
    if (r.premium === null || r.mean_pnl === null) continue;
    (out[r.policy] ??= []).push({ premium: r.premium, mean: r.mean_pnl, se: r.se_pnl, label: `${short(r.config_hash, 10)}-s${r.seed ?? ""}` });
  }
  return out;
}

function ListView({ initialQuery }: { initialQuery: string }) {
  const t = useTokens();
  const router = useRouter();
  const [f, setF] = useState<{ policy: string; premium: number | ""; seed: number | ""; config_hash: string; commit: string; sweep: string; source: string; q: string }>({
    policy: "",
    premium: "",
    seed: "",
    config_hash: "",
    commit: "",
    sweep: "",
    source: "",
    q: initialQuery,
  });
  const [picked, setPicked] = useState<string[]>([]);
  const filters: RunFilters = {
    policy: f.policy ? [f.policy] : undefined,
    premium: f.premium !== "" ? [f.premium] : undefined,
    seed: f.seed !== "" ? [f.seed] : undefined,
    config_hash: f.config_hash ? [f.config_hash] : undefined,
    commit: f.commit ? [f.commit] : undefined,
    sweep: f.sweep ? [f.sweep] : undefined,
    source: f.source ? [f.source] : undefined,
    q: f.q || undefined,
  };
  const runs = useFetch((s) => api.runs(filters, s), JSON.stringify(f));
  const data = runs.data;
  const facets = data?.facets;
  const charted = useMemo(() => (data?.rows ?? []).filter((r) => r.status === "OK" && r.premium !== null), [data]);
  const toggle = (k: string) => setPicked((p) => (p.includes(k) ? p.filter((x) => x !== k) : [...p.slice(-1), k]));

  return (
    <>
      <PageHead
        eyebrow="Run browser"
        title="Runs"
        right={
          <button
            type="button"
            className={`${ui.btn} ${ui.btnPrimary}`}
            disabled={picked.length !== 2}
            onClick={() => router.push(`/runs/?a=${encodeURIComponent(picked[0] ?? "")}&b=${encodeURIComponent(picked[1] ?? "")}`)}
          >
            Compare selected ({picked.length}/2)
          </button>
        }
      />

      <Card>
        <div className={ui.filterBar}>
          <FilterSelect label="Arm" value={f.policy} options={facets?.policies ?? []} onChange={(v) => setF({ ...f, policy: v })} render={(v) => ARM_LABEL[v] ?? v} />
          <FilterSelect label="Premium (vol pts)" value={f.premium} options={facets?.premiums ?? []} onChange={(v) => setF({ ...f, premium: v })} />
          <FilterSelect label="Seed" value={f.seed} options={facets?.seeds ?? []} onChange={(v) => setF({ ...f, seed: v })} />
          <FilterSelect label="Config hash" value={f.config_hash} options={facets?.config_hashes ?? []} onChange={(v) => setF({ ...f, config_hash: v })} />
          <FilterSelect label="Commit" value={f.commit} options={facets?.commits ?? []} onChange={(v) => setF({ ...f, commit: v })} />
          <FilterSelect label="Sweep" value={f.sweep} options={facets?.sweeps ?? []} onChange={(v) => setF({ ...f, sweep: v })} render={(v) => short(v, 12)} />
          <FilterSelect label="Data source" value={f.source} options={facets?.sources ?? []} onChange={(v) => setF({ ...f, source: v })} />
          <label className={ui.field} style={{ flex: 1, minWidth: 200 }}>
            <span className={ui.fieldLabel}>Search run id, hash or commit</span>
            <input className={ui.input} value={f.q} placeholder="e.g. adddb0b7" onChange={(e) => setF({ ...f, q: e.target.value })} />
          </label>
        </div>
      </Card>

      {runs.error ? <ErrorBox what="runs" error={runs.error} onRetry={runs.reload} /> : null}
      {runs.loading && !data ? <Loading what="runs" /> : null}

      {data ? (
        <>
          <div className={ui.grid4}>
            <Tile label="Shown" value={num(data.n_shown)} note={`of ${num(data.trials.n_runs)} runs indexed`} accent />
            <Tile label="OK" value={num(data.trials.n_ok)} note={`${num(data.trials.n_runs - data.trials.n_ok)} flagged`} />
            <Tile label="Distinct configs" value={num(data.trials.n_configs)} />
            <Tile label="Distinct seeds" value={num(data.trials.n_seeds)} />
          </div>

          {data.rows.length ? (
            <Card
              title="Runs"
              sub="Tick two to compare."
              source={
                <>
                  <code>{data.root}</code>
                </>
              }
            >
              <RunTable rows={data.rows} selected={picked} onToggle={toggle} />
            </Card>
          ) : (
            <Empty title="No runs match">
              <p>
                {data.trials.n_runs ? "Clear a filter." : "No runs yet: "}
                {data.trials.n_runs ? null : <code>python scripts/oracle_sweep.py --episodes 50</code>}
              </p>
            </Empty>
          )}

          {charted.length > 1 ? (
            <Card title="Mean P&L against planted premium" sub="OK runs shown above, per arm, ±1 se." source="run.json of each run">
              <Chart option={premiumChart(t, runCurves(charted))} height={320} label="Mean P&L per episode against the planted premium, by arm" />
            </Card>
          ) : null}
        </>
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------- one run

function histOption(t: Tokens, edges: number[], counts: number[], color: string, name: string, xName = "P&L per episode ($)"): EChartsOption {
  const width = edges.length > 1 ? (edges[1] ?? 0) - (edges[0] ?? 0) : 1;
  return {
    ...base(t),
    legend: { show: false },
    tooltip: {
      ...base(t).tooltip,
      trigger: "item",
      formatter: (p: unknown) => {
        const v = (p as { value: [number, number] }).value;
        return `${usd(v[0] - width / 2)} to ${usd(v[0] + width / 2)}<br/><b>${v[1]}</b> episode${v[1] === 1 ? "" : "s"}`;
      },
    },
    xAxis: { type: "value", name: xName, nameLocation: "middle", scale: true, ...axis(t) },
    yAxis: { type: "value", name: "Episodes", nameLocation: "middle", minInterval: 1, ...axis(t), nameGap: 36 },
    series: [
      {
        name,
        type: "bar",
        barWidth: "92%",
        barCategoryGap: 0,
        itemStyle: { color, borderRadius: [3, 3, 0, 0] },
        data: histBars(edges, counts),
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: t.text2, type: "solid", width: 1 },
          label: { color: t.muted, formatter: "0" },
          data: [{ xAxis: 0 }],
        },
      },
    ],
  };
}

function equityOption(t: Tokens, d: RunDetail): EChartsOption {
  const eq = d.equity;
  if (!eq) return {};
  const len = Math.max(...eq.paths.map((p) => p.length));
  const mean = Array.from({ length: len }, (_, i) => {
    const vals = eq.paths.map((p) => p[i]).filter((v): v is number => v !== undefined);
    return vals.reduce((s, v) => s + v, 0) / Math.max(vals.length, 1);
  });
  const color = armColor(t, d.row.policy);
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, trigger: "axis", formatter: (p: unknown) => `Day ${(p as { dataIndex: number }[])[0]?.dataIndex ?? 0}<br/>mean ${usd(mean[(p as { dataIndex: number }[])[0]?.dataIndex ?? 0])}` },
    xAxis: { type: "category", data: Array.from({ length: len }, (_, i) => i), name: "Trading day", nameLocation: "middle", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "Cumulative P&L ($)", nameLocation: "middle", ...axis(t), nameGap: 50 },
    grid: { ...base(t).grid, left: 70 },
    series: [
      ...eq.paths.map((p, i) => ({
        name: `seed ${eq.seeds[i]}`,
        type: "line" as const,
        data: p,
        showSymbol: false,
        silent: true,
        lineStyle: { width: 1, color, opacity: 0.18 },
      })),
      { name: "mean", type: "line" as const, data: mean, showSymbol: false, lineStyle: { width: 2.5, color: t.text } },
    ],
  };
}

function bySeedOption(t: Tokens, d: RunDetail): EChartsOption {
  const color = armColor(t, d.row.policy);
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, trigger: "axis", formatter: (p: unknown) => { const x = (p as { name: string; value: number }[])[0]; return x ? `seed ${x.name}<br/><b>${usd(x.value)}</b>` : ""; } },
    xAxis: { type: "category", data: d.episodes.map((e) => e.seed), name: "Episode seed", nameLocation: "middle", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "P&L ($)", nameLocation: "middle", ...axis(t), nameGap: 50 },
    grid: { ...base(t).grid, left: 70 },
    series: [{ type: "bar", data: d.episodes.map((e) => ({ value: e.pnl, itemStyle: { color: e.pnl >= 0 ? color : t.s5 } })), barCategoryGap: "20%" }],
  };
}

function RiskTiles({ d }: { d: RunDetail }) {
  const r = d.risk;
  return (
    <div className={ui.grid4}>
      <Tile accent label="Mean P&L / episode" value={usd(r.mean)} note={`se ${usd(r.se)} · t ${num(d.row.t_stat, 2)}`} />
      <Tile label="Sortino" value={num(r.sortino, 2)} />
      <Tile label="CVaR 5%" value={usd(r.cvar5)} />
      <Tile label="Max drawdown" value={usd(r.max_drawdown)} note="episodes in seed order" />
      <Tile label="Hit rate" value={pct(r.hit_rate)} />
      <Tile label="Turnover" value={num(r.turnover, 1)} note="trades / episode" />
      <Tile label="Cost / episode" value={usd(d.row.mean_cost)} />
      <Tile label="Episodes" value={num(r.n)} note={d.episodes.length ? `seeds ${d.episodes[0]?.seed}…${d.episodes[d.episodes.length - 1]?.seed}` : "none recorded"} />
    </div>
  );
}

function flatten(prefix: string, v: unknown, out: [string, unknown][]) {
  if (v && typeof v === "object" && !Array.isArray(v)) {
    for (const [k, x] of Object.entries(v as Record<string, unknown>)) flatten(prefix ? `${prefix}.${k}` : k, x, out);
  } else out.push([prefix, v]);
}

function ManifestCard({ d }: { d: RunDetail }) {
  const m = d.manifest;
  const versions = (m.versions ?? {}) as Record<string, string>;
  const config: [string, unknown][] = [];
  flatten("", m.config, config);
  return (
    <div className={ui.grid2}>
      <Card title="Manifest" source={<code>{d.provenance.manifest}</code>}>
        <dl className={ui.kv}>
          <dt>Run id</dt>
          <dd className="mono">{d.row.run_id}</dd>
          <dt>Created</dt>
          <dd>{d.row.created_at || "–"}</dd>
          <dt>Git SHA</dt>
          <dd className="mono">
            {d.row.git_sha || "–"}
            {d.row.git_dirty ? <span className={ui.dim}> (working tree dirty)</span> : null}
          </dd>
          <dt>Config hash</dt>
          <dd className="mono">{d.row.config_hash}</dd>
          <dt>Schema</dt>
          <dd>v{String(m.schema_version ?? "?")}</dd>
          <dt>Replicas</dt>
          <dd>{d.provenance.replicas} run(s) in the index share this config hash</dd>
          {d.row.sweep ? (
            <>
              <dt>Sweep</dt>
              <dd>
                <span className="mono">{d.row.sweep}</span> · {d.provenance.sweep_cells} cells
              </dd>
            </>
          ) : null}
        </dl>
        <p className={ui.groupTitle} style={{ marginTop: 14 }}>
          Library versions
        </p>
        <div className={ui.row}>
          {Object.entries(versions).map(([k, v]) => (
            <Chip key={k}>
              {k} <b>{v}</b>
            </Chip>
          ))}
        </div>
        <p className={ui.groupTitle} style={{ marginTop: 14 }}>
          Artifact digests (recomputed now)
        </p>
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>File</th>
                <th>sha256 recorded</th>
                <th>Check</th>
              </tr>
            </thead>
            <tbody>
              {d.artifacts.map((a) => (
                <tr key={a.path}>
                  <td className="mono">{a.path}</td>
                  <td className="mono" title={a.recorded}>
                    {short(a.recorded, 16)}…
                  </td>
                  <td>
                    <Verdict value={a.ok ? "PASS" : "FAIL"} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      <Card title="Config" source={<code>manifest.json → config</code>}>
        <div className={ui.tableWrap} style={{ maxHeight: 460, overflow: "auto" }}>
          <table className={ui.table}>
            <tbody>
              {config.map(([k, v]) => (
                <tr key={k}>
                  <td className="mono">{k}</td>
                  <td className={ui.num}>{Array.isArray(v) ? v.join(", ") : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

function DetailView({ runKey }: { runKey: string }) {
  const t = useTokens();
  const run = useFetch((s) => api.run(runKey, s), runKey);
  if (run.error) return <ErrorBox what={`run ${runKey}`} error={run.error} onRetry={run.reload} />;
  if (!run.data) return <Loading what="the run" />;
  const d = run.data;
  return (
    <>
      <PageHead
        eyebrow="Run record"
        title={`${short(d.row.config_hash, 16)}-s${d.row.seed ?? "?"}`}
        lede={
          <>
            <Chip color={armColor(t, d.row.policy)}>{ARM_LABEL[d.row.policy] ?? d.row.policy}</Chip> planted premium {num(d.row.premium, 1)} vol pts · source{" "}
            {d.row.source} · created {when(d.row.created_at)}
          </>
        }
        right={
          <>
            <Verdict value={d.row.status} />
            <Link href="/runs/" className={`${ui.btn} ${ui.btnSmall}`}>
              ← All runs
            </Link>
          </>
        }
      />
      {d.row.problems.length ? (
        <Card title="Problems">
          <ul>
            {d.row.problems.map((p) => (
              <li key={p}>{p}</li>
            ))}
          </ul>
        </Card>
      ) : null}
      <RiskTiles d={d} />
      <div className={ui.grid2}>
        <Card title="P&L distribution" sub={`${d.risk.n} episodes, net of costs`} source={<code>{d.provenance.results}</code>}>
          {d.histogram.counts.length ? (
            <Chart option={histOption(t, d.histogram.edges, d.histogram.counts, armColor(t, d.row.policy), "episodes")} height={300} label="Histogram of per-episode P&L" />
          ) : (
            <Empty title="No episodes recorded" />
          )}
        </Card>
        <Card
          title="Equity paths"
          sub={d.equity ? `${d.equity.paths.length} of ${d.equity.n_recorded} paths, and the mean` : undefined}
          source={d.equity ? <code>{d.equity.file}</code> : "run schema v1 records per-episode results only"}
        >
          {d.equity ? (
            <Chart option={equityOption(t, d)} height={300} label="Cumulative P&L paths per episode" />
          ) : (
            <Empty title="Not recorded for this run" />
          )}
        </Card>
      </div>
      {d.episodes.length ? (
        <Card title="P&L by episode seed" source={<code>{d.provenance.results}</code>}>
          <Chart option={bySeedOption(t, d)} height={240} label="Per-episode P&L by seed" />
        </Card>
      ) : null}
      <ManifestCard d={d} />
    </>
  );
}

// ---------------------------------------------------------------- compare

function CompareView({ a, b }: { a: string; b: string }) {
  const t = useTokens();
  const cmp = useFetch((s) => api.compare(a, b, s), `${a}|${b}`);
  if (cmp.error) return <ErrorBox what="the comparison" error={cmp.error} onRetry={cmp.reload} />;
  if (!cmp.data) return <Loading what="the comparison" />;
  const c: Compare = cmp.data;
  const seeds = c.a.episodes.map((e) => e.seed);
  const bySeedB = new Map(c.b.episodes.map((e) => [e.seed, e.pnl]));
  const option: EChartsOption = {
    ...base(t),
    xAxis: { type: "category", data: seeds, name: "Episode seed", nameLocation: "middle", ...axis(t), splitLine: { show: false } },
    yAxis: { type: "value", name: "P&L ($)", nameLocation: "middle", ...axis(t), nameGap: 50 },
    grid: { ...base(t).grid, left: 70 },
    series: [
      { name: `A · ${ARM_LABEL[c.a.row.policy] ?? c.a.row.policy}`, type: "line", data: c.a.episodes.map((e) => e.pnl), symbolSize: 6, lineStyle: { width: 1.5, color: t.s1 }, itemStyle: { color: t.s1 } },
      { name: `B · ${ARM_LABEL[c.b.row.policy] ?? c.b.row.policy}`, type: "line", data: seeds.map((s) => bySeedB.get(s) ?? null), symbolSize: 6, lineStyle: { width: 1.5, color: t.s2 }, itemStyle: { color: t.s2 } },
    ],
  };
  const side = (label: string, d: RunDetail) => (
    <Card
      title={
        <>
          {label} · <Link href={`/runs/?key=${encodeURIComponent(d.row.key)}`} className="mono">{short(d.row.config_hash, 12)}-s{d.row.seed}</Link>
        </>
      }
      right={<Verdict value={d.row.status} />}
      source={<code>{d.provenance.run_dir}</code>}
    >
      <dl className={ui.kv}>
        <dt>Arm</dt>
        <dd>{ARM_LABEL[d.row.policy] ?? d.row.policy}</dd>
        <dt>Premium</dt>
        <dd>{num(d.row.premium, 1)} vol pts</dd>
        <dt>Mean P&L</dt>
        <dd>
          {usd(d.risk.mean)} (se {usd(d.risk.se)})
        </dd>
        <dt>Sortino</dt>
        <dd>{num(d.risk.sortino, 2)}</dd>
        <dt>CVaR 5%</dt>
        <dd>{usd(d.risk.cvar5)}</dd>
        <dt>Max drawdown</dt>
        <dd>{usd(d.risk.max_drawdown)}</dd>
        <dt>Turnover</dt>
        <dd>{num(d.risk.turnover, 1)} trades/ep</dd>
        <dt>Commit</dt>
        <dd className="mono">{d.row.commit}</dd>
      </dl>
    </Card>
  );
  return (
    <>
      <PageHead
        eyebrow="Compare"
        title="Paired on shared seeds"
        right={
          <Link href="/runs/" className={`${ui.btn} ${ui.btnSmall}`}>
            ← All runs
          </Link>
        }
      />
      <div className={ui.grid4}>
        <Tile accent label="Mean of B − A" value={usd(c.diff_mean)} note={`se ${usd(c.diff_se)}`} />
        <Tile label="Paired t" value={num(c.diff_t, 2)} />
        <Tile label="Shared seeds" value={num(c.shared_seeds)} note={`A has ${c.a.episodes.length}, B has ${c.b.episodes.length}`} />
        <Tile label="Config fields that differ" value={num(c.config_diff.length)} />
      </div>
      <div className={ui.grid2}>
        {side("A", c.a)}
        {side("B", c.b)}
      </div>
      <div className={ui.grid2}>
        <Card title="P&L by seed">
          <Chart option={option} height={300} label="Per-seed P&L of both runs" />
        </Card>
        <Card title="Paired differences" sub="B − A per shared seed.">
          {c.diff_histogram.counts.length ? (
            <Chart option={histOption(t, c.diff_histogram.edges, c.diff_histogram.counts, t.s4, "differences", "B − A per seed ($)")} height={300} label="Histogram of paired differences" />
          ) : (
            <Empty title="No shared seeds" />
          )}
        </Card>
      </div>
      <Card title="What differs">
        {c.config_diff.length ? (
          <div className={ui.tableWrap}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>Field</th>
                  <th>A</th>
                  <th>B</th>
                </tr>
              </thead>
              <tbody>
                {c.config_diff.map((r) => (
                  <tr key={r.key}>
                    <td className="mono">{r.key}</td>
                    <td>{JSON.stringify(r.a)}</td>
                    <td>{JSON.stringify(r.b)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty title="Identical configs" />
        )}
      </Card>
    </>
  );
}
