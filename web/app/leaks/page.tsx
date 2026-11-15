"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, PageHead, Segmented, SelectField, Source, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { FigureFoot } from "@/components/Export";
import lab from "@/components/lab.module.css";
import { InfoTip, Labelled } from "@/components/InfoTip";
import { Fold, Reading } from "@/components/PageBits";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type LeakCoverage, type LeakDemo, type LeakField, type LeakFloor, type LeakSide } from "@/lib/api";
import { axis, base, type Tokens, useTokens } from "@/lib/chart";
import { esc, num, pval } from "@/lib/format";
import { fieldOf, fieldWord, luck, shareOf, shareWord, shareWords as words } from "@/lib/leakwords";
import { define } from "@/lib/glossary";
import { guide } from "@/lib/guide";
import { useQuery, useQuerySync } from "@/lib/url";

/** A field's signal strength, |t|; null when it was not tested (constant under the probe). */
const absT = (f: LeakSide["features"][number]) => (f.tested && f.t_stat !== null ? Math.abs(f.t_stat) : null);

/** erfc, Numerical Recipes' erfcc: relative error below 1.2e-7. */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}

/** The threshold in |t|: from the payload, else the normal approximation (n is thousands of pairs). */
function tCrit(side: { t_critical?: number | null; threshold: number }): number {
  if (side.t_critical) return side.t_critical;
  let lo = 0;
  let hi = 40;
  for (let i = 0; i < 60; i++) {
    const mid = (lo + hi) / 2;
    if (erfc(mid / Math.SQRT2) > side.threshold) lo = mid;
    else hi = mid;
  }
  return lo;
}

/** A few field names in full; more as a count, which the Every field view lists. */
const few = (names: string[]) => (names.length > 3 ? `${names.length} fields` : names.map(fieldWord).join(", "));

/** Each field in words, as a chart's axis names it; its code name is in the tooltip. */
function fieldLabels(features: LeakSide["features"]): string[] {
  return features.map((f) => (f.tested ? fieldWord(f.name) : `${fieldWord(f.name)} (never moves)`));
}

function threshold(t: Tokens, crit: number) {
  return {
    symbol: "none",
    silent: true,
    lineStyle: { color: t.fail, width: 1.5, type: "dashed" as const },
    label: { color: t.fail, fontSize: 11, formatter: `alarm line ${crit.toFixed(1)}`, position: "insideEndTop" as const },
    data: [{ xAxis: crit }],
  };
}

/** Each field's |t| against tomorrow's return, clean and poisoned, with the threshold. */
function strengthOption(t: Tokens, d: LeakDemo): EChartsOption {
  const crit = tCrit(d.clean);
  const names = fieldLabels(d.clean.features);
  const top = Math.max(crit * 1.4, ...d.poisoned.features.map((f) => absT(f) ?? 0), ...d.clean.features.map((f) => absT(f) ?? 0));
  return {
    ...base(t),
    legend: { ...base(t).legend, data: ["Clean copy", "Poisoned copy"] },
    tooltip: { ...base(t).tooltip, trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: unknown) => (typeof v === "number" ? `signal ${v.toFixed(2)} (|t|)` : "not tested: it never moves here") },
    grid: { left: 200, right: 28, top: 34, bottom: 42 },
    xAxis: { type: "value", name: "Signal strength, |t|", nameLocation: "middle", ...axis(t), min: 0, max: Math.ceil(top) },
    yAxis: { type: "category", data: names, inverse: true, ...axis(t), axisLine: { show: false }, splitLine: { show: false }, axisLabel: { ...axis(t).axisLabel, fontSize: 11.5, color: t.text2 } },
    series: [
      { id: "clean", name: "Clean copy", type: "bar", data: d.clean.features.map(absT), barWidth: 7, barGap: "40%", itemStyle: { color: t.s1, borderRadius: [0, 3, 3, 0] }, markLine: threshold(t, crit) },
      { id: "poisoned", name: "Poisoned copy", type: "bar", data: d.poisoned.features.map(absT), barWidth: 7, itemStyle: { color: t.s2, borderRadius: [0, 3, 3, 0] } },
    ],
  };
}

/** The leak planted in one field: every field's |t|, the planted one in its own colour. */
function fieldOption(t: Tokens, f: LeakField): EChartsOption {
  const crit = tCrit(f);
  const top = Math.max(crit * 1.4, ...f.features.map((x) => absT(x) ?? 0));
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: unknown) => (typeof v === "number" ? `signal ${v.toFixed(2)} (|t|)` : "not tested: it never moves here") },
    grid: { left: 200, right: 28, top: 16, bottom: 42 },
    xAxis: { type: "value", name: "Signal strength, |t|", nameLocation: "middle", ...axis(t), min: 0, max: Math.ceil(top) },
    yAxis: { type: "category", data: fieldLabels(f.features), inverse: true, ...axis(t), axisLine: { show: false }, splitLine: { show: false }, axisLabel: { ...axis(t).axisLabel, fontSize: 11.5, color: t.text2 } },
    series: [
      {
        id: "field",
        name: `planted in ${fieldWord(f.field)}`,
        type: "bar",
        barWidth: 9,
        data: f.features.map((x) => ({ value: absT(x), itemStyle: { color: x.name === f.field ? t.s2 : t.s1, borderRadius: [0, 3, 3, 0] } })),
        markLine: threshold(t, crit),
      },
    ],
  };
}

/** Caught or missed at each planted size: a filled dot is caught, a ring is missed. */
function floorOption(t: Tokens, f: LeakFloor): EChartsOption {
  // a share as a fraction over its number, "1/20" over "0.05": six fit side by side at phone width
  const cats = f.eps.map((e) => `${e >= 1 ? "all" : Number.isInteger(1 / e) ? `1/${1 / e}` : shareWord(e)}\n${e}`);
  const at = f.floor === null ? -1 : f.eps.indexOf(f.floor);
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, trigger: "item", formatter: (p: unknown) => { const q = p as { dataIndex: number }; const e = f.eps[q.dataIndex] ?? 0; return `${esc(shareWord(e))} of tomorrow's move (${e})<br/><b>${f.flagged[q.dataIndex] ? "caught" : "missed"}</b>`; } },
    grid: { left: 24, right: 24, top: 30, bottom: 58 },
    xAxis: { type: "category", data: cats, name: "How much of tomorrow leaks, largest first", nameLocation: "middle", ...axis(t, { nameGap: 42 }), axisLabel: { color: t.muted, fontFamily: t.font, fontSize: 11, lineHeight: 14, interval: 0 }, splitLine: { show: false }, axisLine: { lineStyle: { color: t.axis } } },
    yAxis: { type: "value", min: -1, max: 1, show: false },
    series: [
      {
        id: "floor",
        name: "detector",
        type: "scatter",
        symbolSize: 18,
        data: f.flagged.map((c, i) => ({ value: [i, 0], itemStyle: c ? { color: t.pass, borderColor: t.surface, borderWidth: 2 } : { color: t.surface, borderColor: t.fail, borderWidth: 2.5 } })),
        label: { show: true, position: "top", distance: 10, color: t.muted, fontSize: 11, formatter: (p: { dataIndex: number }) => (f.flagged[p.dataIndex] ? "caught" : "missed") },
        markLine: at >= 0 ? { symbol: "none", silent: true, lineStyle: { color: t.text2, width: 1, type: "dashed" }, label: { color: t.text2, fontSize: 11, formatter: "smallest caught" }, data: [{ xAxis: at }] } : undefined,
      },
    ],
  };
}

const VIEWS = [
  { value: "demo", label: "Clean vs poisoned" },
  { value: "field", label: "Every field" },
  { value: "floor", label: "Smallest leak caught" },
] as const;
type View = (typeof VIEWS)[number]["value"];

type Run<T> = { busy: boolean; data?: T; error?: ApiError };

function useRun<T, A extends unknown[] = []>(go: (...args: A) => Promise<T>): [Run<T>, (...args: A) => Promise<void>] {
  const [state, setState] = useState<Run<T>>({ busy: false });
  const run = async (...args: A) => {
    setState((s) => ({ ...s, busy: true, error: undefined }));
    try {
      setState({ busy: false, data: await go(...args) });
    } catch (err) {
      setState({ busy: false, error: err instanceof ApiError ? err : new ApiError(0, String(err)) });
    }
  };
  return [state, run];
}

export default function LeaksPage() {
  return (
    <Suspense fallback={<Loading what="the leak test" />}>
      <Leaks />
    </Suspense>
  );
}

type Line = { text: string; detail: string; term: string; verdict: string; result: string };

/** The criterion as three lines, each with the current result beside it. */
function Criterion({ lines }: { lines: Line[] }) {
  return (
    <ul className={lab.checks}>
      {lines.map((l) => (
        <li key={l.text} className={lab.check}>
          <span>
            <Verdict value={l.verdict} />
          </span>
          <span className={lab.checkText}>
            <Labelled term={l.term}>{l.text}</Labelled>
            <span>{l.detail}</span>
          </span>
          <span className={lab.checkResult}>{l.result}</span>
        </li>
      ))}
    </ul>
  );
}

function Leaks() {
  const t = useTokens();
  const q = useQuery();
  const [view, setView] = useState<View>(() => q.oneOf("view", ["demo", "floor", "field"] as const, "demo"));
  const [eps, setEps] = useState(() => q.num("eps", 0.5, 0.05, 1));
  const [seeds, setSeeds] = useState(() => q.int("seeds", 10, 3, 40));
  const [field, setField] = useState(() => q.str("field", "ret_1d"));
  useQuerySync({ view, eps, seeds, field: view === "field" ? field : null }, { view: "demo", eps: 0.5, seeds: 10 });

  const [demo, runDemo] = useRun(() => api.leakDemo(eps, seeds));
  const [cover, runCover] = useRun<LeakCoverage>(() => api.leakCoverage(eps, seeds));
  const [floor, runFloor] = useRun(() => api.leakFloor(seeds));
  const [one, plant] = useRun((f: string) => api.leakField(f, eps, seeds));
  const runOne = () => plant(field);
  const checking = demo.busy || cover.busy;
  const busy = checking || floor.busy || one.busy;
  // the check: the clean and poisoned simulators first (under a second), then every field in turn
  const check = async () => {
    await runDemo();
    await runCover();
  };

  // run=1 in the URL runs the chosen view once on arrival, as a workflow step does
  const auto = useRef(q.bool("run"));
  useEffect(() => {
    if (!auto.current) return;
    auto.current = false;
    if (view === "floor") void runFloor();
    else {
      void check();
      if (view === "field") void runOne();
    }
  });

  const d = demo.data;
  const c = cover.data;
  const stamp = `optspread.leaks · default_env · seeds 0–${seeds - 1}`;
  const crit = d ? tCrit(d.clean) : null;
  const worstClean = d ? Math.max(0, ...d.clean.features.map((f) => absT(f) ?? 0)) : null;
  const pending = (busy: boolean) => (busy ? "running" : "not run");
  const nFields = c?.controls.length ?? d?.clean.features.length ?? 14;
  const caughtFields = c ? c.controls.filter((x) => x.flagged && x.named_correctly).length : null;
  const untested = d ? d.clean.features.filter((f) => !f.tested).map((f) => f.name) : [];
  const lines: Line[] = [
    {
      text: "The poisoned copy is caught",
      detail: `With ${shareOf(eps)} of tomorrow's move slipped into today, the detector fires.`,
      term: "leaks.caught",
      verdict: d ? (d.caught ? "PASS" : "FAIL") : pending(demo.busy),
      result: d ? (d.poisoned.flagged ? `caught on ${fieldWord(d.poisoned.worst_feature)}, signal ${num(Math.abs(d.poisoned.worst_t ?? 0), 1)}` : `missed: all under the alarm line, ${num(crit, 1)}`) : "",
    },
    {
      text: "It is caught in every field",
      detail: `Planted in each of the ${nFields} numbers the trader sees, one at a time, it is caught and named.`,
      term: "leaks.everyField",
      verdict: c ? (c.passed ? "PASS" : "FAIL") : pending(cover.busy || (demo.busy && !d)),
      result: c ? `caught and named in ${caughtFields} of ${c.controls.length}${c.uncovered.length ? ` · missed ${few(c.uncovered)}` : ""}${c.misattributed.length ? ` · misnamed ${few(c.misattributed)}` : ""}` : cover.busy ? `planting in ${nFields} fields…` : "",
    },
    {
      text: "The clean copy stays quiet",
      detail: `Nothing planted: no field reaches the alarm line.${untested.length ? ` ${untested.map(fieldWord).join(", ").replace(/^./, (x) => x.toUpperCase())} never move${untested.length === 1 ? "s" : ""} here, so ${untested.length === 1 ? "it is" : "they are"} checked only when planted, above.` : ""}`,
      term: "leaks.quiet",
      verdict: d ? (!d.clean.flagged && d.clean.past_unchanged && !(c?.clean_flagged ?? false) ? "PASS" : "FAIL") : pending(demo.busy),
      // a field constant in the clean world has nothing to test there; its planted control still counts above
      result: d ? `strongest signal ${num(worstClean, 1)}, alarm line ${num(crit, 1)} · ${d.clean.features.length - untested.length} of ${d.clean.features.length} fields checked` : "",
    },
  ];
  const verdicts = lines.map((l) => l.verdict);
  const allPass = verdicts.every((v) => v === "PASS");
  const anyFail = verdicts.includes("FAIL");

  return (
    <>
      <PageHead
        eyebrow="Leak detector"
        title="Can the simulator see the future?"
        lede="A leak lets a trader deciding today see a bit of tomorrow and profit on money that never existed. Like a smoke alarm, the detector must go off on a copy poisoned with tomorrow and stay quiet on the clean one."
      />
      <Card title="The test" right={d && c ? <Verdict value={allPass ? "PASS" : "FAIL"} /> : undefined}>
        <div className={ui.stack}>
          <Reading>
            Say the index is 5,000 today and 5,050 tomorrow; a bug showing 5,025 today gives the rise away. We need both checks: an alarm that never fires might be broken, and a clean market that fires would mean ours leaks.
          </Reading>
          <div {...guide("leaks-criterion")}>
            <Criterion lines={lines} />
          </div>
          <div className={lab.bar}>
            <div className={ui.field} style={{ minWidth: 270 }} {...guide("leaks-eps")}>
              <span className={ui.fieldLabel}>
                <Labelled term="leaks.eps">
                  <label htmlFor="leaks-eps">How much of tomorrow leaks</label>
                </Labelled>
                <span>{shareOf(eps)}</span>
              </span>
              <input id="leaks-eps" className={ui.range} type="range" min={0.05} max={1} step={0.05} value={eps} onChange={(e) => setEps(Number(e.target.value))} />
            </div>
            <div className={ui.field} style={{ minWidth: 200 }} {...guide("leaks-seeds")}>
              <span className={ui.fieldLabel}>
                <Labelled term="leaks.seeds">
                  <label htmlFor="leaks-seeds">Paths</label>
                </Labelled>
                <span>{seeds} × 63 days</span>
              </span>
              <input id="leaks-seeds" className={ui.range} type="range" min={3} max={40} step={1} value={seeds} onChange={(e) => setSeeds(Number(e.target.value))} />
            </div>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={() => void check()} disabled={busy} {...guide("leaks-run")} title={define("leaks.run")}>
              {checking ? "Running the detector…" : d ? "Run again" : "Run the detector"}
            </button>
            {d ? (
              <span className={lab.barNote}>
                ran in {(d.seconds + (c?.seconds ?? 0)).toFixed(1)} s · {num(d.clean.n_pairs)} days, each read against the next
                <Source>{`${stamp} · ${d.source}${c ? ` · ${c.source}` : ""}`}</Source>
              </span>
            ) : null}
          </div>
          {demo.error ? <ErrorBox what="the leak test" error={demo.error} onRetry={() => void check()} /> : null}
          {cover.error ? <ErrorBox what="the field coverage" error={cover.error} onRetry={runCover} /> : null}
          {anyFail && d && eps < 0.15 ? <p className={ui.dim} style={{ fontSize: 13 }}>A leak this small can slip under the alarm with {seeds} paths; Smallest leak caught shows where that floor is.</p> : null}
        </div>
      </Card>

      <div {...guide("leaks-view")}>
        <Segmented label="View" value={view} options={VIEWS} onChange={setView} />
      </div>

      {view === "demo" ? (
        d ? (
          <Card title="How strongly each field predicts tomorrow" info="leaks.strength">
            <div className={ui.stack}>
              <Reading>
                {d.poisoned.flagged && !d.clean.flagged ? (
                  <>
                    The poisoned copy set the alarm off on <b>{fieldWord(d.poisoned.worst_feature)}</b> (signal {num(Math.abs(d.poisoned.worst_t ?? 0), 1)}); the clean one stayed quiet.
                  </>
                ) : d.poisoned.flagged ? (
                  <>
                    Both copies set the alarm off: the clean one too, on <b>{fieldWord(d.clean.worst_feature)}</b>, a false alarm that counts as a failure.
                  </>
                ) : d.clean.flagged ? (
                  <>The poisoned copy stayed quiet but the clean one set the alarm off, on <b>{fieldWord(d.clean.worst_feature)}</b>: both count as failures.</>
                ) : (
                  <>Neither copy set the alarm off: with {shareOf(d.eps)} of tomorrow leaking, the leak stayed under the alarm line of {num(crit, 1)}.</>
                )}
              </Reading>
              <Chart option={strengthOption(t, d)} height={60 + 28 * d.clean.features.length} label="Per-field detection strength, clean against poisoned" stamp={`${stamp} · eps ${d.eps}`} />
            </div>
          </Card>
        ) : (
          <Card>
            <p className={ui.dim}>Run the detector to see how well each field predicts tomorrow.</p>
          </Card>
        )
      ) : null}

      {view === "field" ? (
        <Card title="A leak in each field alone" info="leaks.everyField" sub={c ? `Each field in turn gets a leak of ${shareWord(c.eps)} of its usual swing (${c.eps} sd); ✓ is caught and named. Click one to see every field.` : "Run the detector to plant a leak in every field in turn."}>
          <div className={ui.stack}>
            {c ? (
              <div className={lab.fieldGrid} {...guide("leaks-fields")}>
                {c.controls.map((x) => (
                  <button
                    key={x.field}
                    type="button"
                    className={`${lab.fieldCell} ${x.field === field && one.data?.field === x.field ? lab.fieldOn : ""}`}
                    title={`${fieldOf(x.field)}: ${x.flagged ? (x.named_correctly ? "caught" : "caught, but blamed on another field") : "missed"} · chance this is luck: ${luck(x.p_value)} (p = ${pval(x.p_value)})`}
                    onClick={() => {
                      setField(x.field);
                      void plant(x.field);
                    }}
                  >
                    <span>{fieldWord(x.field)}</span>
                    <span className={x.flagged && x.named_correctly ? lab.ok : lab.bad}>{x.flagged ? (x.named_correctly ? "✓" : "misnamed") : "✕"}</span>
                  </button>
                ))}
              </div>
            ) : null}
            <div className={lab.bar}>
              <div className={lab.barItem}>
                <SelectField id="leaks-field" label="One field" term="leaks.field" guide="leaks-field" value={field} options={c?.controls.map((x) => x.field) ?? d?.clean.features.map((f) => f.name) ?? [field]} render={fieldOf} onChange={(v) => v && setField(v)} />
              </div>
              <button type="button" className={ui.btn} onClick={() => void runOne()} disabled={busy} {...guide("leaks-plant")}>
                {one.busy ? "Planting…" : `Plant in ${fieldWord(field)}`}
              </button>
            </div>
            {one.error ? <ErrorBox what="the field control" error={one.error} onRetry={() => void runOne()} /> : null}
            {one.data ? (
              <>
                <Reading>
                  Planted only in <b>{fieldWord(one.data.field)}</b>, {shareWord(one.data.eps)} of its usual swing:{" "}
                  {one.data.flagged ? (one.data.named_correctly ? "the alarm went off and named that field." : `the alarm went off, but named ${fieldWord(one.data.worst_feature)}.`) : "the alarm stayed quiet, a miss."}
                </Reading>
                <Chart option={fieldOption(t, one.data)} height={60 + 26 * one.data.features.length} label={`Per-field detection strength with the leak planted in ${fieldWord(one.data.field)}`} stamp={`${stamp} · ${one.data.field} · eps ${one.data.eps}`} />
              </>
            ) : null}
            {c ? <FigureFoot name="leak field coverage" stamp={`${stamp} · eps ${c.eps} · clean ${c.clean_flagged ? "flagged" : "quiet"}`} rows={() => c.controls.map((x) => ({ ...x }))} /> : null}
          </div>
        </Card>
      ) : null}

      {view === "floor" ? (
        <Card title="The smallest leak still caught" info="leaks.floor" right={
          <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => void runFloor()} disabled={busy} {...guide("leaks-floor")}>
            {floor.busy ? "Planting smaller leaks…" : floor.data ? "Run again" : "Find it"}
          </button>
        }>
          <div className={ui.stack}>
            {floor.error ? <ErrorBox what="the detector floor" error={floor.error} onRetry={runFloor} /> : null}
            {floor.data ? (
              <>
                <Reading>
                  {floor.data.floor === null ? (
                    <>With {floor.data.seeds.length} paths the detector caught none of the planted leaks.</>
                  ) : (
                    <>
                      With {floor.data.seeds.length} paths the alarm caught {words(floor.data.eps.filter((_, i) => floor.data?.flagged[i]))} of tomorrow&apos;s move
                      {floor.data.flagged.some((x) => !x) ? <> and missed {words(floor.data.eps.filter((_, i) => !floor.data?.flagged[i]))}</> : null}: a quiet alarm rules out a leak of <b>{shareWord(floor.data.floor)}</b> or more.
                    </>
                  )}
                </Reading>
                <Chart option={floorOption(t, floor.data)} height={184} label="Whether the detector catches a planted leak at each size" stamp={`${stamp} · ${floor.data.source} · ${floor.data.seconds.toFixed(1)} s`} rows={() => floor.data?.eps.map((e, i) => ({ eps: e, caught: floor.data?.flagged[i] })) ?? []} />
              </>
            ) : floor.busy ? (
              <Loading what="the detector floor" />
            ) : (
              <p className={ui.dim}>Plants smaller and smaller leaks, from all of tomorrow&apos;s move down to a fortieth, with {seeds} paths.</p>
            )}
          </div>
        </Card>
      ) : null}

      {d ? (
        <Fold title="The formal criterion" sub="As the gate report states it.">
          <pre className={ui.pre}>{d.criterion}</pre>
        </Fold>
      ) : null}
    </>
  );
}
