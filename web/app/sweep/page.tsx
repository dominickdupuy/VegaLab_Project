"use client";

import Link from "next/link";
import { useState } from "react";
import { Card, PageHead, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Empty, ErrorBox, Loading, Pending } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type Sweep } from "@/lib/api";
import { ARM_LABEL, type PremiumPoint, premiumChart, useTokens } from "@/lib/chart";
import { num, short, usd } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

function oracleCurves(s: Sweep): Record<string, PremiumPoint[]> {
  return Object.fromEntries(
    Object.entries(s.oracle).map(([arm, pts]) => [arm, pts.map((p) => ({ premium: p.premium, mean: p.mean_pnl, se: p.se_pnl, label: p.run_id }))]),
  );
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
        <Tile label="Paired seeds" value={`${s.summary.seed}…${s.summary.seed + s.summary.n_episodes - 1}`} />
        <Tile label="Premiums (vol pts)" value={s.summary.premiums.join(", ")} />
        <Tile label="Sweep hash" value={<span className="mono" style={{ fontSize: 18 }}>{short(s.summary.sweep_hash, 12)}</span>} />
      </div>
      <Card title="Mean P&L against planted premium" sub="Per arm, ±1 se." source={<code>{s.source}</code>}>
        <Chart option={premiumChart(t, oracleCurves(s))} height={340} label="Mean P&L against planted premium, by arm" />
      </Card>
      <Card title="Cells" sub="Lift: against the first premium, same arm.">
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
        <Card title="Power curve">
          <Pending title={s.power_curve.status}>{s.power_curve.waiting_for}.</Pending>
        </Card>
        <Card title="Attribution">
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
        title="Sweep"
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
          <Empty title="No sweep yet">
            <p>
              <code>python scripts/oracle_sweep.py --episodes 50 --root runs/sweeps</code>
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
