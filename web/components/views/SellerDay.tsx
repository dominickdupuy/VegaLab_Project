"use client";

import type { EChartsOption } from "echarts";
import { Reading } from "@/components/PageBits";
import { Card, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { FigureFoot } from "@/components/Export";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { dataApi, type SellerRead } from "@/lib/api_data";
import { axis, base, type Tokens, useTokens } from "@/lib/chart";
import { num, usd } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/** The condor's dollar P&L at expiry across SPX: flat in the middle, capped by the wings. */
function payoffOption(t: Tokens, d: SellerRead): EChartsOption {
  const strikes = d.legs.map((l) => l.strike);
  const lo = Math.min(...strikes);
  const hi = Math.max(...strikes);
  const pad = (hi - lo) * 0.35;
  const xs = [lo - pad, ...strikes, ...d.breakevens, d.spot, hi + pad].sort((a, b) => a - b);
  const pnl = (x: number) => d.legs.reduce((acc, l) => acc + l.qty * (Math.max(l.right === "C" ? x - l.strike : l.strike - x, 0) - l.price), 0) * 100;
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, trigger: "axis", formatter: (ps: unknown) => { const q = (ps as { value: [number, number] }[])[0]; return q ? `SPX ${num(q.value[0], 0)} at expiry<br/>P&amp;L <b>${usd(q.value[1])}</b>` : ""; } },
    grid: { left: 70, right: 20, top: 24, bottom: 44 },
    xAxis: { type: "value", name: "SPX at expiry", nameLocation: "middle", min: Math.round(lo - pad), max: Math.round(hi + pad), ...axis(t), nameGap: 28 },
    yAxis: { type: "value", name: "P&L ($)", nameLocation: "middle", ...axis(t), nameGap: 54 },
    series: [
      {
        name: "P&L at expiry",
        type: "line",
        data: xs.map((x) => [x, pnl(x)]),
        showSymbol: false,
        lineStyle: { width: 2, color: t.s1 },
        areaStyle: { color: t.s1, opacity: 0.08 },
        markLine: {
          symbol: "none",
          silent: true,
          data: [
            { yAxis: 0, lineStyle: { color: t.axis, type: "solid", width: 1 }, label: { show: false } },
            { xAxis: d.spot, lineStyle: { color: t.s5, type: "dashed", width: 1 }, label: { color: t.muted, formatter: "spot" } },
          ],
        },
      },
    ],
  };
}

/** What the fixed-delta seller would open at this close: the reading, its legs, credit, risk and margin. */
export function SellerDay({ date, gridded }: { date: string; gridded: boolean }) {
  const seller = useFetch((s) => dataApi.seller(date, s), `seller|${date}`, Boolean(date) && gridded);
  const d = seller.data;
  const t = useTokens();
  return (
    <div className={ui.stack}>
      <Reading loading={gridded && seller.loading}>{d?.reading}</Reading>
      {!gridded ? <Empty title={`No gridded surface on ${date}`} /> : null}
      {seller.error ? <ErrorBox what="the seller's structure" error={seller.error} onRetry={seller.reload} /> : null}
      {gridded && seller.loading && !d ? <Loading what="the structure" /> : null}
      {d ? (
        <>
          {d.margined_as_naked ? (
            <p className={ui.callout}>
              Why the margin is above the max loss: the frozen harness decides whether risk is defined with an absolute tolerance that this condor&apos;s payoffs, at SPX {num(d.spot, 0)}, round past, so it margins the condor as naked shorts (open defect #39, docs/DEFECTS.md). The sweep ran with that check, so the margin is shown as the harness computes it; the max loss is computed from the legs.
            </p>
          ) : null}
          <div className={ui.grid4}>
            <Tile info="data.credit" label="Credit" value={usd(d.credit)} note={`one iron condor · ${d.tenor_trading_days} trading days (${num(d.tenor_calendar_days, 0)} calendar)`} />
            <Tile
              info="data.max_loss"
              label="Max loss"
              value={d.capped_loss !== null ? usd(d.capped_loss) : d.max_loss !== null ? usd(d.max_loss) : "not capped"}
              note={d.margined_as_naked ? "capped by the wings, from the legs" : d.capped_loss !== null || d.max_loss !== null ? "the wider wing less the credit" : "no wing on one side"}
            />
            <Tile info="data.margin" label="Margin" value={usd(d.margin)} note={d.margined_as_naked ? "margined as naked shorts (#39)" : "posted at the open: the max loss"} />
            <Tile info="data.breakevens" label="Breakevens" value={d.breakevens.map((b) => num(b, 0)).join(" – ")} note={`SPX ${num(d.spot, 0)} at the close`} />
          </div>
          <div className={ui.grid2}>
          <Card title="What it makes or loses at expiry" info="data.iron_condor" sub={d.note}>
            <Chart option={payoffOption(t, d)} height={300} label={`The iron condor's profit and loss at expiry against SPX, opened on ${d.date}`} stamp={`${d.date} · one condor`} />
          </Card>
          <Card title="The four legs" info="data.legs">
            <div className={ui.stack}>
              <div className={ui.tableWrap}>
                <table className={ui.table}>
                  <thead>
                    <tr>
                      <th>Leg</th>
                      <th className={ui.num}>Strike</th>
                      <th className={ui.num}>K / spot</th>
                      <th className={ui.num}>Delta</th>
                      <th className={ui.num}>IV</th>
                      <th className={ui.num}>Mid</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.legs.map((l) => (
                      <tr key={`${l.right}${l.strike}`}>
                        <td>
                          {l.qty < 0 ? "Sell" : "Buy"} {l.right === "P" ? "put" : "call"}
                          {l.qty < 0 ? "" : " (wing)"}
                        </td>
                        <td className={ui.num}>{num(l.strike)}</td>
                        <td className={ui.num}>{l.moneyness.toFixed(3)}</td>
                        <td className={ui.num}>{l.delta.toFixed(3)}</td>
                        <td className={ui.num}>{(l.iv * 100).toFixed(1)}</td>
                        <td className={ui.num}>{l.price.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <FigureFoot name={`seller ${d.date}`} stamp={`${d.date} · short ${d.short_delta}Δ, wings ${d.wing_delta}Δ`} rows={() => d.legs.map((l) => ({ ...l }))} json={() => d} />
            </div>
          </Card>
          </div>
        </>
      ) : null}
    </div>
  );
}
