"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, Chip, PageHead, SelectField, Tile, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { RunMixCard } from "@/components/ActionMix";
import { CompareRuns, histOption } from "@/components/CompareRuns";
import { Fold, Reading } from "@/components/PageBits";
import { PremiumView } from "@/components/PremiumView";
import { RunCatalog } from "@/components/RunCatalog";
import { AttributionCard, CapitalCard, capitalPending, EquityCard, ReplayControl, ReplayResult, useCapital } from "@/components/RunExtras";
import { RunTable } from "@/components/RunTable";
import { InfoTip } from "@/components/InfoTip";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type RunDetail, type RunFilters } from "@/lib/api";
import { runsApi } from "@/lib/api_runs";
import { ARM_LABEL, armColor, axis, base, runStamp, SOURCE_LABEL, type Tokens, useTokens } from "@/lib/chart";
import { num, pct, prem, short, usd, when } from "@/lib/format";
import { guide } from "@/lib/guide";
import { useFetch } from "@/lib/useFetch";
import { setQuery, useHashScroll, useQuery, useQuerySync } from "@/lib/url";

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
  if (a && b) return <CompareRuns a={a} b={b} group={params.get("group") ?? undefined} />;
  if (key === "latest") return <Latest />;
  if (key) return <DetailView runKey={key} />;
  return <ListView />;
}

const FILTER_KEYS = ["policy", "premium", "seed", "config_hash", "commit", "sweep", "source", "q"] as const;

function filtersFrom(q: ReturnType<typeof useQuery>): Filters {
  const n = (k: string) => {
    const v = q.str(k);
    return v === "" || !Number.isFinite(Number(v)) ? ("" as const) : Number(v);
  };
  return { policy: q.str("policy"), premium: n("premium"), seed: n("seed"), config_hash: q.str("config_hash"), commit: q.str("commit"), sweep: q.str("sweep"), source: q.str("source"), q: q.str("q") };
}

function toApi(f: Filters): RunFilters {
  return {
    policy: f.policy ? [f.policy] : undefined,
    premium: f.premium !== "" ? [f.premium] : undefined,
    seed: f.seed !== "" ? [f.seed] : undefined,
    config_hash: f.config_hash ? [f.config_hash] : undefined,
    commit: f.commit ? [f.commit] : undefined,
    sweep: f.sweep ? [f.sweep] : undefined,
    source: f.source ? [f.source] : undefined,
    q: f.q || undefined,
  };
}

/** `key=latest` (with any filters): the newest OK run that matches, opened in place. */
function Latest() {
  const q = useQuery();
  const f = filtersFrom(q);
  const runs = useFetch((s) => api.runs(toApi(f), s), `latest|${JSON.stringify(f)}`);
  const hit = runs.data?.rows.find((r) => r.status === "OK");
  useEffect(() => {
    if (hit) setQuery({ key: hit.key, ...Object.fromEntries(FILTER_KEYS.map((k) => [k, null])) });
  }, [hit]);
  if (runs.error) return <ErrorBox what="runs" error={runs.error} onRetry={runs.reload} />;
  if (runs.data && !hit)
    return (
      <Empty title="No run matches">
        <p>
          <Link href="/launch/">Launch one</Link> or <Link href="/runs/">browse every run</Link>.
        </p>
      </Empty>
    );
  return hit ? <DetailView runKey={hit.key} /> : <Loading what="the latest run" />;
}

// ---------------------------------------------------------------- the list

interface Filters {
  policy: string;
  premium: number | "";
  seed: number | "";
  config_hash: string;
  commit: string;
  sweep: string;
  source: string;
  q: string;
}

function ListView() {
  const t = useTokens();
  const router = useRouter();
  const query = useQuery();
  const [f, setF] = useState<Filters>(() => filtersFrom(query));
  useQuerySync({ ...f });
  const [picked, setPicked] = useState<string[]>([]);
  const filters = toApi(f);
  const runs = useFetch((s) => api.runs(filters, s), JSON.stringify(f));
  const premiums = useFetch((s) => runsApi.premiums(filters, s), `premiums|${JSON.stringify(f)}`);
  const catalog = useFetch((s) => runsApi.catalog(s), "catalog");
  const data = runs.data;
  const facets = data?.facets;
  const labels = catalog.data?.available ? catalog.data.labels : undefined;
  const toggle = (k: string) => setPicked((p) => (p.includes(k) ? p.filter((x) => x !== k) : [...p.slice(-1), k]));
  const filtered = FILTER_KEYS.some((k) => f[k] !== "");
  const arms = facets?.policies ?? [];

  // in the head and on the table's own card, so it is beside the ticks wherever they are
  const compareButton = (
    <button
      type="button"
      className={`${ui.btn} ${ui.btnPrimary}`}
      disabled={picked.length !== 2}
      {...guide("runs-compare")}
      onClick={() => router.push(`/runs/?a=${encodeURIComponent(picked[0] ?? "")}&b=${encodeURIComponent(picked[1] ?? "")}`)}
    >
      Compare selected ({picked.length}/2)
    </button>
  );

  return (
    <>
      <PageHead
        eyebrow="Runs"
        title="What has each strategy made?"
        lede={
          data ? (
            <>
              {num(data.trials.n_runs)} runs of {arms.length} arm{arms.length === 1 ? "" : "s"}
              {catalog.data?.available ? ", grouped below by the question each answers" : ""}. Open one to see its equity, capital, attribution and trades, or tick two to compare them on the same seeds.
            </>
          ) : undefined
        }
        right={compareButton}
      />

      {arms.length ? (
        <div className={ui.row} aria-label="The arms">
          {arms.map((a) => (
            <span key={a} style={{ whiteSpace: "nowrap" }}>
              <Chip color={armColor(t, a)}>{ARM_LABEL[a] ?? a}</Chip>
            </span>
          ))}
        </div>
      ) : null}

      {catalog.data?.available && !filtered ? <RunCatalog catalog={catalog.data} /> : null}

      <Card>
        <div className={ui.filterBar}>
          <SelectField label="Arm" all="All" value={f.policy} options={facets?.policies ?? []} onChange={(v) => setF({ ...f, policy: v })} render={(v) => ARM_LABEL[v] ?? v} />
          <SelectField label="Premium (vol pts)" term="premium" all="All" value={f.premium} options={facets?.premiums ?? []} onChange={(v) => setF({ ...f, premium: v })} />
          <SelectField label="First seed" term="seed_first" all="All" value={f.seed} options={facets?.seeds ?? []} onChange={(v) => setF({ ...f, seed: v })} />
          <SelectField label="Config hash" term="config_hash" all="All" value={f.config_hash} options={facets?.config_hashes ?? []} onChange={(v) => setF({ ...f, config_hash: v })} />
          <SelectField label="Commit" all="All" value={f.commit} options={facets?.commits ?? []} onChange={(v) => setF({ ...f, commit: v })} />
          <SelectField label="Sweep" all="All" value={f.sweep} options={facets?.sweeps ?? []} onChange={(v) => setF({ ...f, sweep: v })} render={(v) => short(v, 12)} />
          <SelectField label="Data" all="All" value={f.source} options={facets?.sources ?? []} onChange={(v) => setF({ ...f, source: v })} render={(v) => SOURCE_LABEL[v] ?? v} />
          <label className={ui.field} style={{ flex: 1, minWidth: 200 }}>
            <span className={ui.fieldLabel}>Search run id, hash or commit</span>
            <input className={ui.input} value={f.q} placeholder="e.g. adddb0b7" onChange={(e) => setF({ ...f, q: e.target.value })} />
          </label>
        </div>
      </Card>

      {runs.error ? <ErrorBox what="runs" error={runs.error} onRetry={runs.reload} /> : null}
      {runs.loading && !data ? <Loading what="runs" /> : null}

      {premiums.data && premiums.data.groups.length ? <PremiumView data={premiums.data} /> : premiums.error ? <ErrorBox what="the premium view" error={premiums.error} onRetry={premiums.reload} /> : null}

      {data ? (
        data.rows.length ? (
          <Card
            title={`${num(data.n_shown)} of ${num(data.trials.n_runs)} runs`}
            sub={`${num(data.trials.n_ok)} OK${data.trials.n_runs - data.trials.n_ok ? `, ${num(data.trials.n_runs - data.trials.n_ok)} flagged` : ""} · ${num(data.trials.n_configs)} configs · ${num(data.trials.n_seeds)} first seeds. Tick two to compare.`}
            source={`Every run directory under the runs root (${data.root}) with a manifest; numbers from each run's run.json, checked against the manifest's digests.`}
            right={compareButton}
          >
            <RunTable rows={data.rows} selected={picked} onToggle={toggle} labels={labels} />
          </Card>
        ) : (
          <Empty title="No runs match">
            <p>
              {data.trials.n_runs ? "Clear a filter." : "No runs yet: "}
              {data.trials.n_runs ? null : <Link href="/launch/">launch one</Link>}
            </p>
          </Empty>
        )
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------- one run

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
      <Tile label="Mean P&L / episode" value={usd(r.mean)} note={`se ${usd(r.se)} · t ${num(d.row.t_stat, 2)}`} />
      <Tile label="Sortino" value={num(r.sortino, 2)} note="mean over downside spread" />
      <Tile label="CVaR 5%" value={usd(r.cvar5)} note="the worst 5% of episodes" />
      <Tile label="Hit rate" value={pct(r.hit_rate)} note="episodes that made money" />
      <Tile label="Turnover" value={num(r.turnover, 1)} note="trades / episode" />
      <Tile label="Cost / episode" value={usd(d.row.mean_cost)} note="spread and commission" />
      <Tile label="Drawdown, seed order" value={usd(r.max_drawdown)} info="drawdown_seed" note="over episodes strung together" />
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
      <Card title="Manifest" source={d.provenance.manifest}>
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
          <dt>Data source</dt>
          <dd>
            {d.row.source}
            {d.row.panel ? <span className="mono"> · panel {d.row.panel}</span> : null}
            {d.row.source_recorded ? null : <span className={ui.dim}> (inferred: the manifest predates the source record)</span>}
          </dd>
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
      <Card title="Config" info={"manifest.json → config: every knob the run was made with; its hash is the run's identity."}>
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

/** One sentence on the run: what it made per episode, how sure, and its tail. */
function runSentence(d: RunDetail): string {
  const r = d.risk;
  if (r.mean === null) return "No episodes recorded.";
  const half = r.se !== null ? 1.96 * r.se : null;
  const arm = ARM_LABEL[d.row.policy] ?? d.row.policy;
  const where = d.row.source !== "synthetic" ? "on real SPX history" : `with ${prem(d.row.premium)} vol pt${d.row.premium === 1 ? "" : "s"} planted`;
  const sure = half === null ? "" : r.mean - half > 0 ? ": clearly positive" : r.mean + half < 0 ? ": clearly negative" : ": not clearly different from zero";
  return `${arm} ${where} made ${usd(r.mean)} per episode over ${num(r.n)} episodes (95% ${half === null ? "–" : `${usd(r.mean - half)} to ${usd(r.mean + half)}`})${sure}; its worst 5% of episodes average ${usd(r.cvar5)}.`;
}

function DetailView({ runKey }: { runKey: string }) {
  const t = useTokens();
  const run = useFetch((s) => api.run(runKey, s), runKey);
  const catalog = useFetch((s) => runsApi.catalog(s), "catalog");
  useHashScroll(Boolean(run.data));
  if (run.error) return <ErrorBox what={`run ${runKey}`} error={run.error} onRetry={run.reload} />;
  if (!run.data) return <Loading what="the run" />;
  return <Detail d={run.data} label={catalog.data?.labels[run.data.row.key] ?? ""} question={catalog.data?.questions[run.data.row.key] ?? ""} t={t} />;
}

function Detail({ d, label, question, t }: { d: RunDetail; label: string; question: string; t: Tokens }) {
  const cap = useCapital(d);
  const stamp = runStamp(d.row);
  const id = `${short(d.row.config_hash, 16)}-s${d.row.seed ?? "?"}`;
  return (
    <>
      <PageHead
        eyebrow={question ? `Run record · ${question}` : "Run record"}
        title={label || id}
        lede={
          <>
            <Chip color={armColor(t, d.row.policy)}>{ARM_LABEL[d.row.policy] ?? d.row.policy}</Chip>{" "}
            {d.row.source !== "synthetic" ? (
              <InfoTip term="real">real SPX history</InfoTip>
            ) : (
              <>
                <InfoTip term="premium">planted premium</InfoTip> {prem(d.row.premium)} vol pts
              </>
            )}{" "}
            · {label ? <span className="mono">{id} · </span> : null}created {when(d.row.created_at)}
          </>
        }
        right={
          <>
            <Verdict value={d.row.status} />
            <ReplayControl d={d} />
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
      <Reading>{runSentence(d)}</Reading>
      <RiskTiles d={d} />
      <div className={ui.grid2}>
        <EquityCard d={d} cap={cap} />
        <Card
          title="P&L per episode"
          sub={`${d.risk.n} episodes, net of costs; the line is zero.`}
          source={`From ${d.provenance.results}.`}
        >
          {d.histogram.counts.length ? (
            <Chart
              option={histOption(t, d.histogram.edges, d.histogram.counts, armColor(t, d.row.policy), "episodes")}
              height={300}
              label="Histogram of per-episode P&L"
              stamp={stamp}
              rows={() => d.histogram.counts.map((c, i) => ({ from: d.histogram.edges[i], to: d.histogram.edges[i + 1], episodes: c }))}
            />
          ) : (
            <Empty title="No episodes recorded" />
          )}
        </Card>
      </div>
      <ReplayResult d={d} />
      <CapitalCard d={d} cap={cap} />
      {/* computed one at a time on the server: the capital re-simulation above the fold first */}
      <AttributionCard d={d} wait={capitalPending(cap)} />
      <RunMixCard d={d} wait={capitalPending(cap)} />
      {d.episodes.length ? (
        <Fold title="P&L by episode seed" sub="every episode's P&L, in seed order">
          <Chart option={bySeedOption(t, d)} height={240} label="Per-episode P&L by seed" stamp={stamp} rows={() => d.episodes.map((e) => ({ ...e }))} />
        </Fold>
      ) : null}
      <Fold title="Provenance" sub="the manifest, its digests and the config">
        <ManifestCard d={d} />
      </Fold>
    </>
  );
}
