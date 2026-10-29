"use client";

import type { CustomSeriesRenderItemAPI, CustomSeriesRenderItemParams, CustomSeriesRenderItemReturn, EChartsOption, SeriesOption } from "echarts";
import { Card, Chip, Verdict } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { FigureFoot } from "@/components/Export";
import lab from "@/components/lab.module.css";
import { Labelled } from "@/components/InfoTip";
import { ArmPhrase, armPhrase, Facts, Fold, Reading } from "@/components/PageBits";
import { ErrorBox, Loading, Pending } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, docUrl, type M4, type M4Analysis, type M4Arm } from "@/lib/api";
import { ARM_LABEL, armColor, axis, base, type Tokens, useTokens } from "@/lib/chart";
import { MINUS, num, pct, pval, short, usd } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

const vp = (x: number | null | undefined, d = 2) => (x === null || x === undefined ? "–" : `${x < 0 ? MINUS : ""}${Math.abs(x).toFixed(d)}`);

function floorText(a: M4Arm): string {
  const f = a.floor;
  if (f.status === "not_reached") return `above ${vp(f.bracket[0])} (max power ${pct(f.max_power, 0)})`;
  return `${vp(f.premium)} vol pts${f.status === "below_grid" ? " (below the grid)" : ""}`;
}

type BandPoint = { premium: number; power_lo: number | null; power_hi: number | null };

/** The pointwise band as one quadrilateral per step of the interpolated curve. */
function band(pts: BandPoint[], color: string) {
  return (params: CustomSeriesRenderItemParams, api: CustomSeriesRenderItemAPI): CustomSeriesRenderItemReturn => {
    const i = params.dataIndex;
    const a = pts[i];
    const b = pts[i + 1];
    if (!a || !b || a.power_lo === null || a.power_hi === null || b.power_lo === null || b.power_hi === null) return null;
    const corners = [api.coord([a.premium, a.power_lo]), api.coord([b.premium, b.power_lo]), api.coord([b.premium, b.power_hi]), api.coord([a.premium, a.power_hi])];
    return { type: "polygon", shape: { points: corners.map((c) => [c[0] ?? 0, c[1] ?? 0]) }, style: { fill: color, opacity: 0.14 } };
  };
}

function powerOption(t: Tokens, m: M4): EChartsOption {
  const a = m.analysis as M4Analysis;
  const target = a.settings.target_power;
  const series: SeriesOption[] = [];
  const floors: { xAxis: number; name: string; lineStyle: { color: string } }[] = [];
  for (const arm of a.arm_order) {
    const x = a.arms[arm];
    if (!x) continue;
    const color = armColor(t, arm);
    const name = ARM_LABEL[arm] ?? arm;
    const pts = x.curve.interpolated;
    series.push(
      {
        name,
        type: "custom",
        data: pts.map((p) => [p.premium, p.power_lo, p.power_hi]),
        renderItem: band(pts, color),
        itemStyle: { color },
        silent: true,
        tooltip: { show: false },
        z: 1,
      },
      { name, type: "line", data: pts.map((p) => [p.premium, p.power]), showSymbol: false, lineStyle: { width: 2.5, color }, itemStyle: { color }, z: 3 },
      {
        name,
        type: "scatter",
        data: x.curve.points.filter((p) => p.measured).map((p) => [p.premium, p.power]),
        symbolSize: 9,
        itemStyle: { color, borderColor: t.surface, borderWidth: 2 },
        tooltip: { show: true },
        z: 4,
      },
    );
    if (x.floor.premium !== null) floors.push({ xAxis: x.floor.premium, name: `${name} floor ${vp(x.floor.premium)}`, lineStyle: { color } });
  }
  const real = m.vrp;
  series.push({
    name: "markers",
    type: "line",
    data: [],
    markLine: {
      symbol: "none",
      silent: true,
      label: { color: t.text2, fontSize: 11, formatter: (q: { name?: string }) => q.name ?? "" },
      lineStyle: { type: "dashed", width: 1 },
      data: [{ yAxis: target, name: `${pct(target)} power`, label: { position: "insideEndTop" }, lineStyle: { color: t.s5 } }, ...floors, ...(real ? [{ xAxis: real.estimate, name: `real ${vp(real.estimate)}`, lineStyle: { color: t.text } }] : [])] as never,
    },
    markArea: real
      ? { silent: true, itemStyle: { color: t.dark ? "rgba(232,238,247,0.07)" : "rgba(19,35,61,0.06)" }, label: { color: t.muted, fontSize: 11, position: "insideTop", formatter: "real premium, 95%" }, data: [[{ xAxis: Math.max(0, real.interval[0]) }, { xAxis: real.interval[1] }]] as never }
      : undefined,
  });
  return {
    ...base(t),
    animation: false, // a figure of record: drawn once, whole
    legend: { ...base(t).legend, data: a.arm_order.map((x) => ARM_LABEL[x] ?? x) },
    tooltip: { ...base(t).tooltip, trigger: "item", formatter: (p: unknown) => { const q = p as { seriesName: string; value: [number, number] }; return `${q.seriesName}<br/>${vp(q.value[0])} vol pts · power <b>${pct(q.value[1], 1)}</b>`; } },
    xAxis: { type: "value", name: "Planted premium (vol pts)", nameLocation: "middle", ...axis(t), min: 0 },
    yAxis: { type: "value", min: 0, max: 1, name: "Detection probability", nameLocation: "middle", ...axis(t), nameGap: 42, axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => pct(v) } },
    grid: { ...base(t).grid, left: 62 },
    series,
  };
}

/** "0.06 vol pts", or "above 4 vol pts" when the grid never reaches the target. */
function floorShort(a: M4Arm): string {
  const f = a.floor;
  if (f.status === "not_reached") return `above ${vp(f.bracket[0])} vol pts`;
  return `${vp(f.premium)} vol pts`;
}

/** The power view in a sentence: each arm's floor, then each arm's power at the real premium. */
function m4Reading(m: M4) {
  const a = m.analysis as M4Analysis;
  const name = armPhrase;
  const floors = a.arm_order.flatMap((arm) => {
    const x = a.arms[arm];
    return x ? [`${name(arm)} from ${floorShort(x)}`] : [];
  });
  const main = (
    <>
      With {a.design.n_seeds} seeds of {num(a.design.n_episodes)} episodes, the edge is detected {pct(a.settings.target_power)} of the time: {floors.join(", ")}.
    </>
  );
  const real = m.vrp && m.mapped.length ? `At the real SPX premium, ${vp(m.vrp.estimate)} vol pts, power is ${m.mapped.map((r) => `${pct(r.power, 0)} for ${name(r.arm)}`).join(" and ")}.` : "";
  const gap = a.gap && a.gap.value !== null ? ` ${ArmPhrase(a.gap.learned)} needs ${a.gap.kind === "lower_bound" ? "at least " : ""}${vp(a.gap.value)} vol pts more premium than ${name(a.gap.oracle)} to be detected as often.` : "";
  return { main, note: `${real}${gap}`.trim() };
}

export function M4View() {
  const t = useTokens();
  const res = useFetch((s) => api.m4(s), "m4");
  if (res.error) return <ErrorBox what="the analysis" error={res.error} onRetry={res.reload} />;
  const m = res.data;
  if (!m) return <Loading what="the analysis" />;
  if (!m.available || !m.analysis)
    return (
      <Card title="Power curves, floors and the gap">
        <Pending title="Appears once the analysis is generated">
          <p>
            {m.error ? `${m.error}. ` : ""}
            {m.waiting_for}.
          </p>
        </Pending>
      </Card>
    );
  const a = m.analysis;
  const prov = a.provenance;
  const stamp = `grid ${prov.grid_hash} · analysis git ${short(prov.analysis_git_sha, 10)}${prov.analysis_git_dirty ? "+dirty" : ""} · sweep git ${prov.sweep_git_shas.map((x) => short(x, 10)).join(", ")} · ${a.report_date} · synthetic`;
  const gap = a.gap;
  const read = m4Reading(m);
  return (
    <>
      {!a.confirmatory ? (
        <Card title="Exploratory analysis" right={<Chip>not the confirmatory design</Chip>}>
          <ul>
            {a.deviations.map((d) => (
              <li key={d}>{d}</li>
            ))}
          </ul>
        </Card>
      ) : null}
      <Card title="Power against planted premium" info="sweep.powerCurve">
        <div className={ui.stack}>
          <Reading note={read.note || undefined}>{read.main}</Reading>
          <Chart
            option={powerOption(t, m)}
            height={400}
            label="Detection probability against planted premium per arm with bands, floors and the real premium's interval"
            stamp={`${stamp} · docs/${m.file}`}
            rows={() => a.arm_order.flatMap((arm) => (a.arms[arm]?.curve.interpolated ?? []).map((p) => ({ arm, ...p })))}
            json={() => a}
          />
          <p className={ui.dim} style={{ fontSize: 12.5 }}>
            Line: median-unbiased estimate; band: its exact {pct(a.settings.curve_level)} interval; dots: the premiums the grid ran. Shaded: the real premium&apos;s 95% interval.
          </p>
        </div>
      </Card>
      <Fold title="Floors and the gap" term="floor" sub="Each arm's floor with its band, and how far apart they are.">
        <Facts
          items={[
            ...a.arm_order.flatMap((arm) => {
              const x = a.arms[arm];
              return x ? [{ label: `Floor · ${ARM_LABEL[arm] ?? arm}`, value: floorText(x), note: `band ${vp(x.floor.lo)} to ${x.floor.hi === null ? "above the grid" : vp(x.floor.hi)} · between ${vp(x.floor.bracket[0])} and ${vp(x.floor.bracket[1])}`, term: "floor" }] : [];
            }),
            { label: "Gap, PPO − seller", value: gap && gap.value !== null ? `${gap.kind === "lower_bound" ? "≥ " : ""}${vp(gap.value)} vol pts` : "–", note: gap ? `${gap.kind.replace("_", " ")}${gap.band_range[0] !== null || gap.band_range[1] !== null ? ` · range ${vp(gap.band_range[0])} to ${gap.band_range[1] === null ? "above the grid" : vp(gap.band_range[1])}` : ""}` : "needs both arms", term: "gap" },
            { label: "Design", value: `S = ${a.design.n_seeds} × N = ${num(a.design.n_episodes)}`, note: `${prov.cells_used} of ${prov.cells_total} cells · ${a.confirmatory ? "confirmatory" : "exploratory"}` },
          ]}
        />
      </Fold>
      {m.mapped.length ? (
        <Fold title="At the real premium" term="sweep.realPremium" sub={m.vrp ? `${vp(m.vrp.estimate)} vol pts, 95% [${vp(m.vrp.interval[0])}, ${vp(m.vrp.interval[1])}]` : undefined}>
          <div className={ui.tableWrap}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>Arm</th>
                  <th className={ui.num}>Power at the estimate</th>
                  <th className={ui.num}>At the low end</th>
                  <th className={ui.num}>At the high end</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {m.mapped.map((r) => (
                  <tr key={r.arm}>
                    <td>{ARM_LABEL[r.arm] ?? r.arm}</td>
                    <td className={ui.num}>{pct(r.power, 1)}</td>
                    <td className={ui.num}>{pct(r.power_low, 1)}</td>
                    <td className={ui.num}>{pct(r.power_high, 1)}</td>
                    <td className={ui.dim}>{r.reason || (r.clipped ? "an end lies outside the grid; held at the end value" : "")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <FigureFoot name="power at the real premium" stamp={`${stamp}${m.vrp ? ` · ${m.vrp.label}` : ""}`} rows={() => m.mapped.map((r) => ({ ...r }))} />
        </Fold>
      ) : null}
      <Fold title="The test at each premium" term="sweep.decisions" sub={`One-sided t on ${a.design.n_seeds} seed means, ${a.settings.df} df, critical ${vp(a.settings.critical, 3)}.`}>
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>Arm</th>
                <th className={ui.num}>Premium</th>
                <th className={ui.num}>Estimate</th>
                <th className={ui.num}>SE</th>
                <th className={ui.num}>t</th>
                <th className={ui.num}>Lower bound</th>
                <th className={ui.num}>95% interval</th>
                <th className={ui.num}>τ²</th>
                <th className={ui.num}>Plug-in power</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {a.arm_order.flatMap((arm) =>
                (a.arms[arm]?.decisions ?? []).map((d) => (
                  <tr key={`${arm}-${d.premium}`}>
                    <td>{ARM_LABEL[arm] ?? arm}</td>
                    <td className={ui.num}>{vp(d.premium)}</td>
                    <td className={ui.num}>{usd(d.estimate)}</td>
                    <td className={ui.num}>{usd(d.se)}</td>
                    <td className={ui.num}>{num(d.t, 2)}</td>
                    <td className={ui.num}>{usd(d.lower_bound)}</td>
                    <td className={ui.num}>
                      [{usd(d.t_interval[0])}, {usd(d.t_interval[1])}]
                    </td>
                    <td className={ui.num}>{num(d.components.tau2 as number | null)}</td>
                    <td className={ui.num}>{pct(d.power_plugin, 1)}</td>
                    <td>
                      <Chip color={d.detected ? t.pass : t.s5}>{d.detected ? "detected" : "not detected"}</Chip>
                    </td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
        <p className={ui.dim} style={{ fontSize: 12.5 }}>
          The full analysis:{" "}
          <a href={docUrl(m.file)} target="_blank" rel="noreferrer">
            {m.file}
          </a>
        </p>
        <FigureFoot name="m4 decisions" stamp={stamp} rows={() => a.arm_order.flatMap((arm) => (a.arms[arm]?.decisions ?? []).map((d) => ({ arm, premium: d.premium, estimate: d.estimate, se: d.se, t: d.t, df: d.df, lower_bound: d.lower_bound, detected: d.detected, t_lo: d.t_interval[0], t_hi: d.t_interval[1], power_plugin: d.power_plugin, ...d.components })))} />
      </Fold>
      <Fold title="Checks on the test" term="sweep.homogeneity" sub="Equal seed-to-seed noise across premiums; the bootstrap against the t test.">
        <p className={lab.section}>Seed-to-seed noise across premiums</p>
        <div>
          <div className={ui.tableWrap}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>Arm</th>
                  <th>On</th>
                  <th className={ui.num}>W</th>
                  <th className={ui.num}>df</th>
                  <th className={ui.num}>p</th>
                  <th>Equal τ²</th>
                </tr>
              </thead>
              <tbody>
                {a.arm_order.flatMap((arm) =>
                  (["contrast", "arm"] as const).map((k) => {
                    const h = a.arms[arm]?.homogeneity[k];
                    return h ? (
                      <tr key={`${arm}-${k}`}>
                        <td>{ARM_LABEL[arm] ?? arm}</td>
                        <td>{k === "contrast" ? "contrasts, p > 0" : "the arm, every p"}</td>
                        <td className={ui.num}>{num(h.statistic, 2)}</td>
                        <td className={ui.num}>
                          {h.df[0]}, {h.df[1]}
                        </td>
                        <td className={ui.num}>{pval(h.p_value)}</td>
                        <td>{h.rejected ? <Chip color={t.fail}>rejected</Chip> : <Chip color={t.s5}>not rejected</Chip>}</td>
                      </tr>
                    ) : null;
                  }),
                )}
              </tbody>
            </table>
          </div>
          <p className={ui.dim} style={{ fontSize: 12.5, marginTop: 8 }}>
            Brown-Forsythe W on seed means. Not rejected is weak evidence of equal τ², not proof.
          </p>
        </div>
        <p className={lab.section}>
          <Labelled term="sweep.bootstrap">Bootstrap against the t test</Labelled>
        </p>
        <div>
          <p className={ui.dim} style={{ fontSize: 12.5, marginBottom: 8 }}>
            {num(a.settings.n_boot)} resamples; agreement against {num(a.settings.reference_replications)} normal references.
          </p>
          <div className={ui.tableWrap}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>Arm</th>
                  <th className={ui.num}>Premium</th>
                  <th className={ui.num}>Rate</th>
                  <th className={ui.num}>Analytic</th>
                  <th className={ui.num}>Gap</th>
                  <th className={ui.num}>p</th>
                  <th>Agrees</th>
                </tr>
              </thead>
              <tbody>
                {a.arm_order.flatMap((arm) =>
                  (a.arms[arm]?.bootstrap.levels ?? []).map((l) => (
                    <tr key={`${arm}-${l.premium}`}>
                      <td>{ARM_LABEL[arm] ?? arm}</td>
                      <td className={ui.num}>{vp(l.premium)}</td>
                      <td className={ui.num}>{pct(l.rate, 1)}</td>
                      <td className={ui.num}>{pct(l.analytic, 1)}</td>
                      <td className={ui.num}>{num(l.gap, 3)}</td>
                      <td className={ui.num}>{pval(l.p_value)}</td>
                      <td>
                        <Verdict value={l.agrees ? "PASS" : "FAIL"} />
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
          <p className={ui.dim} style={{ fontSize: 12.5, marginTop: 8 }}>
            Rule in force: {a.arm_order.map((arm) => `${ARM_LABEL[arm] ?? arm} ${a.arms[arm]?.bootstrap.rule_in_force === "t" ? "t test" : `bootstrap (diverged at ${a.arms[arm]?.bootstrap.diverged_at.join(", ")})`}`).join(" · ")} · bootstrap rule size {pct(a.bootstrap_rule_size.size, 1)} at S = {a.bootstrap_rule_size.n_seeds}
          </p>
        </div>
      </Fold>
    </>
  );
}
