"use client";

import { useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, PageHead, Tile, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { ErrorBox } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type LeakDemo, type LeakSide } from "@/lib/api";
import { axis, base, type Tokens, useTokens } from "@/lib/chart";
import { num, sci } from "@/lib/format";

function strengthOption(t: Tokens, d: LeakDemo): EChartsOption {
  const names = d.clean.features.map((f) => f.name);
  const score = (s: LeakSide) => s.features.map((f) => (f.tested && f.p_value !== null ? -Math.log10(Math.max(f.p_value, 1e-300)) : null));
  const line = -Math.log10(d.clean.threshold);
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (v: unknown) => (typeof v === "number" ? `−log10 p = ${v.toFixed(2)}` : "untested") },
    grid: { left: 130, right: 24, top: 34, bottom: 40 },
    xAxis: { type: "value", name: "−log10 p  (right of the line: flagged)", nameLocation: "middle", ...axis(t), max: (v: { max: number }) => Math.max(v.max, line * 1.6) },
    yAxis: { type: "category", data: names, inverse: true, ...axis(t), splitLine: { show: false }, axisLabel: { ...axis(t).axisLabel, fontFamily: "ui-monospace, Menlo, monospace" } },
    series: [
      {
        name: "Clean env",
        type: "bar",
        data: score(d.clean),
        itemStyle: { color: t.s1, borderRadius: [0, 3, 3, 0] },
        barGap: "15%",
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: t.fail, width: 1.5, type: "solid" },
          label: { color: t.fail, formatter: "Bonferroni threshold", position: "insideEndTop" },
          data: [{ xAxis: line }],
        },
      },
      { name: `Poisoned env (eps = ${d.eps})`, type: "bar", data: score(d.poisoned), itemStyle: { color: t.s2, borderRadius: [0, 3, 3, 0] } },
    ],
  };
}

function Side({ title, side, expectFlag }: { title: string; side: LeakSide; expectFlag: boolean }) {
  const right = side.flagged === expectFlag && side.past_unchanged !== expectFlag;
  return (
    <Card title={title} right={<Verdict value={right ? "PASS" : "FAIL"} />}>
      <div className={ui.grid2}>
        <Tile label="Statistical detector" value={side.flagged ? "LEAK DETECTED" : "quiet"} note={side.flagged ? `worst field ${side.worst_feature}, p = ${sci(side.p_value)}` : `worst p = ${sci(side.p_value)}, threshold ${sci(side.threshold)}`} />
        <Tile label="Perturbation property" value={side.past_unchanged ? "past unchanged" : "past moved"}  />
      </div>
      <p className={ui.dim} style={{ marginTop: 10, fontSize: 13 }}>
        {num(side.n_pairs)} (observation, next return) pairs; {side.n_tested} of {side.features.length} fields tested
        {side.features.some((f) => !f.tested) ? `; untested: ${side.features.filter((f) => !f.tested).map((f) => `${f.name} (${f.reason})`).join(", ")}` : ""}.
      </p>
    </Card>
  );
}

export default function LeaksPage() {
  const t = useTokens();
  const [eps, setEps] = useState(0.5);
  const [seeds, setSeeds] = useState(10);
  const [state, setState] = useState<{ busy: boolean; data?: LeakDemo; error?: ApiError }>({ busy: false });

  const run = async () => {
    setState((s) => ({ ...s, busy: true, error: undefined }));
    try {
      setState({ busy: false, data: await api.leakDemo(eps, seeds) });
    } catch (err) {
      setState({ busy: false, error: err instanceof ApiError ? err : new ApiError(0, String(err)) });
    }
  };
  const d = state.data;

  return (
    <>
      <PageHead
        eyebrow="Leak demo"
        title="Clean vs poisoned"
      />
      <Card>
        <div className={ui.filterBar}>
          <label className={ui.field} style={{ minWidth: 260 }}>
            <span className={ui.fieldLabel}>
              Leak size eps <span>{eps.toFixed(2)} of tomorrow&apos;s return</span>
            </span>
            <input className={ui.range} type="range" min={0.05} max={1} step={0.05} value={eps} onChange={(e) => setEps(Number(e.target.value))} />
          </label>
          <label className={ui.field} style={{ minWidth: 220 }}>
            <span className={ui.fieldLabel}>
              Seeds <span>{seeds} × 63 days</span>
            </span>
            <input className={ui.range} type="range" min={3} max={40} step={1} value={seeds} onChange={(e) => setSeeds(Number(e.target.value))} />
          </label>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={run} disabled={state.busy}>
            {state.busy ? "Running the detector…" : "Run the detector live"}
          </button>
          <span className={ui.fieldHelp} style={{ maxWidth: 360 }}>
            The detector&apos;s stated floor is eps = 0.10 at 10 seeds: below it a leak may go uncaught, and the page will say so.
          </span>
        </div>
      </Card>

      {state.error ? <ErrorBox what="the leak demo" error={state.error} onRetry={run} /> : null}

      {d ? (
        <>
          <div className={ui.grid4}>
            <Tile accent label="Demo verdict" value={<Verdict value={d.caught ? "PASS" : "FAIL"} />} note={d.caught ? "poisoned caught" : "poisoned missed"} />
            <Tile label="Clean env" value={d.clean.flagged ? "flagged" : "quiet"} note={`worst p ${sci(d.clean.p_value)}`} />
            <Tile label="Poisoned env" value={d.poisoned.flagged ? "caught" : "missed"} note={d.poisoned.flagged ? `on ${d.poisoned.worst_feature}, t = ${num(d.poisoned.worst_t, 1)}` : `worst p ${sci(d.poisoned.p_value)}`} />
            <Tile label="Ran in" value={`${d.seconds.toFixed(2)} s`} note={`seeds ${d.seeds[0]}…${d.seeds[d.seeds.length - 1]}`} />
          </div>
          <div className={ui.grid2}>
            <Side title="Clean environment" side={d.clean} expectFlag={false} />
            <Side title={`Poisoned environment, eps = ${d.eps}`} side={d.poisoned} expectFlag={true} />
          </div>
          <Card title="Every observation field" sub="Past the line: flagged." source={d.source}>
            <Chart option={strengthOption(t, d)} height={480} label="Per-field detection strength, clean against poisoned" />
          </Card>
          <Card title="The stated criterion" source="optspread.leaks.PASS_CRITERION">
            <pre className={ui.pre}>{d.criterion}</pre>
          </Card>
        </>
      ) : state.busy ? null : (
        <Card>
          <p className={ui.dim}>Run the demo.</p>
        </Card>
      )}
    </>
  );
}
