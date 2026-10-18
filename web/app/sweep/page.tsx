"use client";

import Link from "next/link";
import { useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, PageHead, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Empty, ErrorBox, Loading, Pending } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type Sweep } from "@/lib/api";
import { ARM_LABEL, armColor, axis, base, type Tokens, useTokens } from "@/lib/chart";
import { num, short, usd } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

function oracleOption(t: Tokens, s: Sweep): EChartsOption {
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => usd(v as number) },
    xAxis: { type: "value", name: "Planted premium (vol pts)", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", name: "Mean P&L per episode ($)", nameLocation: "middle", ...axis(t), nameGap: 50 },
    grid: { ...base(t).grid, left: 70 },
    series: Object.entries(s.oracle).flatMap(([arm, pts]) => [
      {
        name: ARM_LABEL[arm] ?? arm,
        type: "line" as const,
        data: pts.map((p) => [p.premium, p.mean_pnl]),
        symbolSize: 8,
        lineStyle: { width: 2, color: armColor(t, arm) },
        itemStyle: { color: armColor(t, arm), borderColor: t.surface, borderWidth: 2 },
      },
      {
        name: `${ARM_LABEL[arm] ?? arm} ± 2 se`,
        type: "line" as const,
        data: pts.map((p) => [p.premium, p.mean_pnl + 2 * p.se_pnl]),
        showSymbol: false,
        silent: true,
        lineStyle: { width: 1, color: armColor(t, arm), opacity: 0.35 },
        tooltip: { show: false },
      },
      {
        name: `${ARM_LABEL[arm] ?? arm} ± 2 se`,
        type: "line" as const,
        data: pts.map((p) => [p.premium, p.mean_pnl - 2 * p.se_pnl]),
        showSymbol: false,
        silent: true,
        lineStyle: { width: 1, color: armColor(t, arm), opacity: 0.35 },
        tooltip: { show: false },
      },
    ]),
    legend: { ...base(t).legend, data: Object.keys(s.oracle).map((a) => ARM_LABEL[a] ?? a) },
  };
}

function SweepView({ sweepKey }: { sweepKey: string }) {
  const t = useTokens();
  const sweep = useFetch((s) => api.sweep(sweepKey, s), sweepKey);
  if (sweep.error) return <ErrorBox what="the sweep" error={sweep.error} onRetry={sweep.reload} />;
  if (!sweep.data) return <Loading what="the sweep" />;
  const s = sweep.data;
  return (
    <>
      <div className={ui.grid4}>
        <Tile accent label="Cells" value={num(s.summary.n_cells)} note={`${s.summary.premiums.length} premiums × ${s.summary.arms.length} arms`} />
        <Tile label="Paired seeds" value={`${s.summary.seed}…${s.summary.seed + s.summary.n_episodes - 1}`} note="every arm at every premium on the same paths" />
        <Tile label="Premiums (vol pts)" value={s.summary.premiums.join(", ")} note="the D2 effect-size axis" />
        <Tile label="Sweep hash" value={<span className="mono" style={{ fontSize: 18 }}>{short(s.summary.sweep_hash, 12)}</span>} note="config hash of the sweep" />
      </div>
      <Card title="The oracle curve" sub="Mean P&L along the planted premium per arm, with ±2 se. Scripted arms have no training variance, so this is what the PPO curve is read against." source={<code>{s.source}</code>}>
        <Chart option={oracleOption(t, s)} height={340} label="Mean P&L against planted premium, by arm" />
      </Card>
      <Card title="Cells" sub="Lift is against the first premium for the same arm: a paired difference, since the arms share seeds.">
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>Arm</th>
                <th className={ui.num}>Premium</th>
                <th className={ui.num}>Mean P&L</th>
                <th className={ui.num}>SE</th>
                <th className={ui.num}>Lift</th>
                <th>Run</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(s.oracle).flatMap(([arm, pts]) =>
                pts.map((p) => (
                  <tr key={`${arm}-${p.premium}`}>
                    <td>{ARM_LABEL[arm] ?? arm}</td>
                    <td className={ui.num}>{num(p.premium, 1)}</td>
                    <td className={ui.num}>{usd(p.mean_pnl)}</td>
                    <td className={ui.num}>{usd(p.se_pnl)}</td>
                    <td className={ui.num}>{usd(p.lift)}</td>
                    <td>
                      <Link href={`/runs/?q=${p.run_id}`} className="mono">
                        {short(p.run_id, 14)}…
                      </Link>
                    </td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
        </div>
      </Card>
      <div className={ui.grid2}>
        <Card title="Power curve" sub="Detection probability against planted effect size, scripted and PPO overlaid.">
          <Pending title={s.power_curve.status}>{s.power_curve.waiting_for}.</Pending>
        </Card>
        <Card title="Attribution" sub="P&L split by greek, with the residual shown explicitly.">
          <Pending title={s.attribution.status}>{s.attribution.waiting_for}.</Pending>
        </Card>
      </div>
    </>
  );
}

export default function SweepPage() {
  const list = useFetch((s) => api.sweeps(s), "sweeps");
  const [chosen, setChosen] = useState("");
  const sweeps = list.data?.sweeps ?? [];
  const key = chosen || sweeps[sweeps.length - 1]?.key || "";

  return (
    <>
      <PageHead
        eyebrow="Sweep"
        title="The sweep, read from its cells"
        lede="A sweep is a grid of ordinary run directories, one per premium and arm, on paired seeds. The power curve and attribution views read the confirmatory sweep when it lands (M3, M4); until then they say what they are waiting for."
        right={
          sweeps.length > 1 ? (
            <label className={ui.field}>
              <span className={ui.fieldLabel}>Sweep</span>
              <select className={ui.select} value={key} onChange={(e) => setChosen(e.target.value)}>
                {sweeps.map((s) => (
                  <option key={s.key} value={s.key}>
                    {s.key} · {s.n_cells} cells
                  </option>
                ))}
              </select>
            </label>
          ) : null
        }
      />
      {list.error ? <ErrorBox what="sweeps" error={list.error} onRetry={list.reload} /> : null}
      {list.loading ? <Loading what="sweeps" /> : null}
      {list.data && !sweeps.length ? (
        <>
          <Empty title="No sweep under the runs root yet">
            <p>
              <code>python scripts/oracle_sweep.py --episodes 50 --root runs/sweeps</code> writes one; it appears here with its oracle curve.
            </p>
          </Empty>
          <div className={ui.grid2}>
            <Card title="Power curve">
              <Pending title={list.data.power_curve.status}>{list.data.power_curve.waiting_for}.</Pending>
            </Card>
            <Card title="Attribution">
              <Pending title={list.data.attribution.status}>{list.data.attribution.waiting_for}.</Pending>
            </Card>
          </div>
        </>
      ) : null}
      {key ? <SweepView sweepKey={key} /> : null}
    </>
  );
}
