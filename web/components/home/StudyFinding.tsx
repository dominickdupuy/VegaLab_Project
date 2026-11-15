"use client";

import Link from "next/link";
import { Card, Tile } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import ui from "@/components/ui.module.css";
import { belowGrid, floorShort, powerOption } from "@/components/views/M4View";
import type { M4, Vrp } from "@/lib/api";
import { useTokens } from "@/lib/chart";
import { MINUS, num, pct } from "@/lib/format";

const vp = (x: number | null | undefined) => (x === null || x === undefined ? "–" : `${x < 0 ? MINUS : "+"}${Math.abs(x).toFixed(2)}`);

/** "only about half the time (53%)", in plain words from the served power. */
function howOften(p: number): string {
  const share = pct(p, 0);
  if (p < 0.4) return `less than half the time (${share})`;
  if (p <= 0.6) return `only about half the time (${share})`;
  if (p < 0.8) return `${share} of the time, short of the 80% a design should have`;
  return `${share} of the time`;
}

/**
 * The study's result on Home: the measured power curves (the Sweep page's power view, the
 * same chart), the two floors and the real premium, and what they mean in one sentence.
 * Nothing at all until the analysis is on the server.
 */
export function StudyFinding({ m4, vrp }: { m4?: M4; vrp?: Vrp }) {
  const t = useTokens();
  const a = m4?.available ? m4.analysis : undefined;
  if (!m4 || !a) return null;
  const seller = a.arms.delta_seller;
  const ppo = a.arms.ppo;
  const h = vrp?.mapping?.headline;
  const est = m4.vrp?.estimate ?? h?.mean ?? null;
  const lo = m4.vrp?.interval[0] ?? h?.mean_low ?? null;
  const hi = m4.vrp?.interval[1] ?? h?.mean_high ?? null;
  const atReal = m4.mapped.find((r) => r.arm === "ppo")?.power;
  const design = `${a.design.n_seeds} seeds × ${num(a.design.n_episodes)} episodes`;
  const caught = `the smallest premium caught ${pct(a.settings.target_power)} of the time`;
  return (
    <Card
      title="What the study found"
      info="sweep.powerCurve"
      sub={`Measured from the confirmatory sweep, ${design}.`}
      right={
        <Link href="/sweep/?view=power" className={ui.dim} style={{ fontSize: 12.5 }}>
          See the full results →
        </Link>
      }
    >
      <div className={ui.stack}>
        <Chart option={powerOption(t, m4, true)} height={250} label="Detection probability against planted premium for the fixed-delta seller and PPO, with the 80% line and the real premium's interval" stamp={`grid ${a.provenance.grid_hash} · ${a.report_date} · docs/${m4.file}`} rows={() => a.arm_order.flatMap((arm) => (a.arms[arm]?.curve.interpolated ?? []).map((p) => ({ arm, ...p })))} name="what the study found" />
        <div className={ui.grid3}>
          {seller ? <Tile label="Seller's floor" info="floor" value={floorShort(seller)} note={belowGrid(seller.floor) || caught} /> : null}
          {ppo ? <Tile label="PPO's floor" info="floor" value={floorShort(ppo)} note={belowGrid(ppo.floor) || caught} /> : null}
          {est !== null ? <Tile label="Real SPX premium" info="vrp" value={`${vp(est)} vol pts`} note={`95% ${vp(lo)} to ${vp(hi)}`} /> : null}
        </div>
        {atReal !== null && atReal !== undefined ? <p className={ui.heroSentence}>At the real premium, a study this size would catch PPO&apos;s edge {howOften(atReal)}.</p> : null}
      </div>
    </Card>
  );
}
