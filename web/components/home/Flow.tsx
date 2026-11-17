"use client";

import Link from "next/link";
import { type ReactNode, useState } from "react";
import { InfoTip } from "@/components/InfoTip";
import ui from "@/components/ui.module.css";
import { floorShort } from "@/components/views/M4View";
import { api, type M4, type M4Analysis, type Vrp } from "@/lib/api";
import { type Catalog, type CatalogRun, runsApi } from "@/lib/api_runs";
import { ARM_LABEL } from "@/lib/chart";
import { MINUS, num, pct, prem, usd } from "@/lib/format";
import { tourHref } from "@/lib/tours";
import { useFetch } from "@/lib/useFetch";
import s from "./flow.module.css";

/**
 * Our solution on Home, as a flow of four cards: plant an edge, run both strategies on
 * the same markets, find the detection floor, hold the real market up to it. Each card
 * shows its step's result as the server has it now (the sweep, a paired comparison of
 * the example runs, the real premium by year) in a small panel, and one action.
 * The drawings follow the charts' colours: the seller and implied vol --s1, PPO and
 * realised vol --s2.
 */

const vp = (x: number | null | undefined) => (x === null || x === undefined || !Number.isFinite(x) ? "–" : `${x < 0 ? MINUS : "+"}${Math.abs(x).toFixed(2)}`);
const money = (x: number | null | undefined) => (x === null || x === undefined || !Number.isFinite(x) ? "–" : `${x < 0 ? MINUS : "+"}${usd(Math.abs(x))}`);

function Step({ title, text, mini, go }: { title: string; text: ReactNode; mini: ReactNode; go: { href: string; label: string } }) {
  return (
    <li className={`${ui.card} ${s.card}`}>
      <h3 className={s.title}>{title}</h3>
      <p className={s.text}>{text}</p>
      <div className={s.mini}>{mini}</div>
      <Link href={go.href} className={`${ui.btn} ${ui.btnPrimary} ${s.cta}`}>
        {go.label}
      </Link>
    </li>
  );
}

const FW = 240;

// hand-shaped wiggles, the same on every load: implied vol calm, realised vol jumpier
const IMPLIED_WAVE = [0, -1.2, 0.8, -0.6, 1.2, -0.8, 0.4, -1.2, 0.8, 0];
const REALISED_WAVE = [0, 2.4, -1.8, 2.8, -1.2, 2, -2.6, 1.4, -1.8, 0.6];

/** 1. The planted premium: the vol options are priced at, above the vol the paths realise, the gap scaled to its size. */
function PlantFig({ size, max }: { size: number; max: number }) {
  const H = 64;
  const base = 46;
  // the gap on a square-root scale, so a quarter of a vol point is still a visible band
  const gap = 6 + 24 * Math.sqrt(size / max);
  const top = base - gap;
  const inside = gap >= 15;
  const x = (i: number) => 4 + (i * (FW - 8)) / (IMPLIED_WAVE.length - 1);
  const upper = IMPLIED_WAVE.map((w, i) => `${x(i)},${top + w}`);
  const lower = REALISED_WAVE.map((w, i) => `${x(i)},${base + w}`);
  return (
    <svg className={s.svg} viewBox={`0 0 ${FW} ${H}`} role="img" aria-label={`Options priced ${prem(size)} vol points above the volatility the market realises`}>
      <polygon points={[...upper, ...[...lower].reverse()].join(" ")} className={s.gap} />
      <polyline points={upper.join(" ")} className={s.implied} />
      <polyline points={lower.join(" ")} className={s.real} />
      <text x={6} y={top - 5} className={s.lab}>
        what options charge
      </text>
      <text x={6} y={base + 14} className={s.lab}>
        what happens
      </text>
      <text x={FW - 6} y={inside ? top + gap / 2 + 4 : top - 5} className={s.labStrong} textAnchor="end">
        the gap = the edge
      </text>
    </svg>
  );
}

/** 2. A paired difference: its interval as a bar, its mean as a dot, zero dashed. */
function PairFig({ mean, lo, hi, label }: { mean: number; lo: number; hi: number; label: string }) {
  const H = 34;
  const min = Math.min(0, lo);
  const max = Math.max(0, hi);
  const pad = (max - min) * 0.12;
  const x = (v: number) => 8 + ((v - (min - pad)) / (max - min + 2 * pad || 1)) * (FW - 16);
  return (
    <svg className={s.svg} viewBox={`0 0 ${FW} ${H}`} role="img" aria-label={label}>
      <line x1={4} x2={FW - 4} y1={14} y2={14} className={s.axis} />
      <line x1={x(0)} x2={x(0)} y1={3} y2={25} className={s.zeroLine} />
      <text x={x(0)} y={H - 1} className={s.lab} textAnchor="middle">
        $0
      </text>
      <line x1={x(lo)} x2={x(hi)} y1={14} y2={14} className={s.ci} />
      <circle cx={x(mean)} cy={14} r={5} className={s.meanDot} />
    </svg>
  );
}

/** 3. Both measured power curves (PPO dashed), the target line and each floor, labelled under the axis. */
function CurvesFig({ a }: { a: M4Analysis }) {
  const H = 88;
  const floors = a.arm_order.map((arm) => a.arms[arm]?.floor.premium ?? null);
  const top = floors.every((f) => f === null || f <= 2) ? 2 : Math.max(...a.design.premiums);
  const x = (p: number) => 8 + (p / top) * (FW - 16);
  const y = (v: number) => 66 - v * 58;
  const target = a.settings.target_power;
  const marks = a.arm_order.map((arm) => ({ arm, f: a.arms[arm]?.floor.premium ?? null })).filter((m): m is { arm: string; f: number } => m.f !== null && m.f <= top);
  return (
    <svg className={s.svg} viewBox={`0 0 ${FW} ${H}`} role="img" aria-label={`Detection probability against planted premium for ${a.arm_order.map((arm) => ARM_LABEL[arm] ?? arm).join(" and ")}, with the ${pct(target)} line and each floor`}>
      <line x1={8} x2={FW - 8} y1={y(0)} y2={y(0)} className={s.axis} />
      <line x1={8} x2={FW - 8} y1={y(target)} y2={y(target)} className={s.target} />
      <text x={FW - 8} y={y(target) + 11} className={s.lab} textAnchor="end">
        {pct(target)}
      </text>
      {a.arm_order.map((arm) => {
        const c = a.arms[arm];
        if (!c) return null;
        const pts = c.curve.interpolated.filter((p) => p.premium <= top && p.power !== null).map((p) => `${x(p.premium)},${y(p.power as number)}`);
        return <polyline key={arm} points={pts.join(" ")} className={`${s.curve} ${arm === "ppo" ? `${s.ppo} ${s.dashed}` : s.seller}`} />;
      })}
      {marks.map(({ arm, f }, i) => {
        const c = a.arms[arm];
        // two floors close together: the second label goes on a lower line
        const low = i > 0 && Math.abs(x(f) - x(marks[i - 1]?.f ?? 0)) < 44;
        return (
          <g key={arm} className={arm === "ppo" ? s.ppo : s.seller}>
            <line x1={x(f)} x2={x(f)} y1={y(target)} y2={y(0) + 4} className={s.floorLine} />
            <circle cx={x(f)} cy={y(target)} r={3.5} className={s.floorDot}>
              <title>{`${ARM_LABEL[arm] ?? arm}'s floor: ${c ? floorShort(c) : ""}`}</title>
            </circle>
            <text x={Math.max(8, Math.min(FW - 8, x(f)))} y={y(0) + (low ? 22 : 13)} className={s.floorLab} textAnchor={x(f) < 20 ? "start" : "middle"}>
              {num(f, 2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** 4. The real premium each year (above zero --s1, below --s2), the whole sample's average dashed. */
function YearsFig({ years, mean }: { years: { label: string; mean: number | null }[]; mean: number }) {
  const H = 78;
  const vals = years.map((r) => r.mean ?? 0);
  const hi = Math.max(0.5, ...vals);
  const lo = Math.min(-0.5, ...vals);
  const y = (v: number) => 14 + ((hi - v) / (hi - lo)) * 50;
  const bw = (FW - 8) / years.length;
  return (
    <svg className={s.svg} viewBox={`0 0 ${FW} ${H}`} role="img" aria-label={`The real SPX premium by year, ${years[0]?.label} to ${years[years.length - 1]?.label}, with the average of ${vp(mean)} vol points`}>
      {years.map((r, i) => {
        const v = r.mean ?? 0;
        return (
          <rect key={r.label} x={4 + i * bw + 1} y={Math.min(y(v), y(0))} width={Math.max(1, bw - 2)} height={Math.max(0.5, Math.abs(y(v) - y(0)))} rx={1.5} className={v >= 0 ? s.up : s.down}>
            <title>{`${r.label}: ${vp(r.mean)} vol pts`}</title>
          </rect>
        );
      })}
      <line x1={4} x2={FW - 4} y1={y(0)} y2={y(0)} className={s.axis} />
      <line x1={4} x2={FW - 4} y1={y(mean)} y2={y(mean)} className={s.avg} />
      <text x={4} y={9} className={s.labStrong}>
        avg {vp(mean)}
      </text>
      <text x={4} y={H - 1} className={s.lab}>
        {years[0]?.label}
      </text>
      <text x={FW - 4} y={H - 1} className={s.lab} textAnchor="end">
        {years[years.length - 1]?.label}
      </text>
    </svg>
  );
}

/** The example runs' fair world and 1 vol pt world for the default seller: same seed, same paths. */
function fairPair(cat: Catalog | undefined): { a: CatalogRun; b: CatalogRun } | null {
  if (!cat?.available) return null;
  const plain = (r: CatalogRun) => (r.policy || String(r.settings.policy ?? "")) === "delta_seller" && Object.keys(r.settings).every((k) => ["n_episodes", "seed", "policy", "iv_premium_vol_pts"].includes(k));
  const runs = cat.groups.flatMap((g) => g.runs).filter((r) => r.present && plain(r));
  const at = (p: number) => runs.find((r) => (r.premium ?? Number(r.settings.iv_premium_vol_pts ?? 0)) === p);
  const a = at(0);
  const b = at(1);
  return a && b && a.settings.seed === b.settings.seed ? { a, b } : null;
}

export function Flow({ m4, vrp }: { m4?: M4; vrp?: Vrp }) {
  const a = m4?.available ? m4.analysis : null;
  const sizes = (a?.design.premiums ?? []).filter((p) => p > 0);
  const [size, setSize] = useState<number | null>(null);
  const planted = size ?? (sizes.includes(1) ? 1 : (sizes[0] ?? 1));
  // the market's own vol, from the launch defaults (MarketConfig.sigma); 20% if the server does not say
  const opts = useFetch((x) => api.jobOptions(x), "jobOptions");
  const sigma = Number(opts.data?.defaults.sigma);
  const vol = Number.isFinite(sigma) && sigma > 0 ? sigma * 100 : 20;

  const cat = useFetch((x) => runsApi.catalog(x), "catalog");
  const pair = fairPair(cat.data);
  const cmp = useFetch((x) => runsApi.compare(pair?.a.key ?? "", pair?.b.key ?? "", undefined, x), `${pair?.a.key}|${pair?.b.key}`, Boolean(pair));
  const c = cmp.data;
  const paired = c && c.diff_mean !== null && c.diff_ci_low !== null && c.diff_ci_high !== null ? { mean: c.diff_mean, lo: c.diff_ci_low, hi: c.diff_ci_high } : null;
  // no example pair on this server: the sweep's own seller contrast at 1 vol pt
  const sweep1 = a?.arms.delta_seller?.decisions.find((d) => d.premium === 1);
  const fallback = !paired && sweep1 && sweep1.estimate !== null && sweep1.t_interval[0] !== null && sweep1.t_interval[1] !== null ? { mean: sweep1.estimate, lo: sweep1.t_interval[0], hi: sweep1.t_interval[1] } : null;
  const shown = paired ?? fallback;
  const pending = !shown && (cat.loading || cmp.loading || !m4);

  const h = vrp?.mapping?.headline;
  const years = (vrp?.mapping?.years ?? []).filter((r) => r.mean !== null);
  const none = (what: string) => <p className={s.empty}>{m4 || vrp ? `${what} is not on this server yet.` : "Loading…"}</p>;

  return (
    <ol className={s.flow}>
      <Step
        title="Plant an edge of known size"
        text="A leak-tested simulated SPX market where we set the premium, so the right answer is known before any test."
        mini={
          <>
            {sizes.length ? (
              <label className={s.chip}>
                Plant
                <select value={planted} onChange={(e) => setSize(Number(e.target.value))} aria-label="Planted premium, vol points">
                  {sizes.map((p) => (
                    <option key={p} value={p}>
                      {prem(p)}
                    </option>
                  ))}
                </select>
                vol pt{planted === 1 ? "" : "s"}
              </label>
            ) : null}
            <PlantFig size={planted} max={Math.max(...sizes, 1)} />
            <p className={s.note}>
              Options priced at <b>{prem(vol + planted)}% vol</b> when the market moves at {prem(vol)}%
            </p>
          </>
        }
        go={{ href: "/synth/", label: "Try it in Synth data →" }}
      />
      <li className={s.arrow} aria-hidden="true">
        →
      </li>
      <Step
        title="Run both on the same markets"
        text="Same strategy, fair world against 1 vol pt planted, on identical days, so luck cancels."
        mini={
          shown ? (
            <>
              <p className={s.big}>
                {money(shown.mean)} <span className={s.unit}>per episode</span>{" "}
                <InfoTip
                  icon="source"
                  label="Where this comes from"
                  text={
                    paired
                      ? `The example runs' fixed-delta seller at 0 and at 1 vol pt planted, on the same ${num(c?.shared_seeds)} market paths, from the paired compare (/api/runs/compare). ${c?.verdict ?? ""}`
                      : `No paired example runs on this server, so this is the sweep's seller contrast at 1 vol pt instead: ${a?.design.n_seeds} seeds × ${num(a?.design.n_episodes)} episodes, against itself at zero, with its t interval (docs/${m4?.file}).`
                  }
                />
              </p>
              <PairFig {...shown} label={`1 vol pt planted against the fair world: ${money(shown.mean)} per episode, 95% interval ${money(shown.lo)} to ${money(shown.hi)}`} />
              <p className={s.note}>
                {shown.lo > 0 ? "Whole range above zero: a real edge" : "Range includes zero: not clear yet"}
              </p>
              {paired ? <p className={s.fine}>From the example runs; your own run will differ a little.</p> : null}
            </>
          ) : pending ? (
            <p className={s.empty}>Loading…</p>
          ) : (
            none("A paired comparison")
          )
        }
        go={{ href: tourHref("edge"), label: "Run it yourself, ~1 min →" }}
      />
      <li className={s.arrow} aria-hidden="true">
        →
      </li>
      <Step
        title="Find each strategy's detection floor"
        text={`A test fixed before the data, over a sweep of premiums, gives the smallest edge caught ${a ? pct(a.settings.target_power) : "80%"} of the time.`}
        mini={
          a ? (
            <>
              <CurvesFig a={a} />
              <p className={s.note}>
                {a.arm_order.map((arm, i) => {
                  const x = a.arms[arm];
                  return x ? (
                    <span key={arm}>
                      {i ? " · " : ""}
                      <span className={`${s.swatch} ${arm === "ppo" ? `${s.ppo} ${s.swatchDashed}` : s.seller}`} aria-hidden="true" />
                      {arm === "ppo" ? "PPO" : "Seller"} <b>{floorShort(x)}</b>
                    </span>
                  ) : null;
                })}{" "}
                <InfoTip term="floor" />
              </p>
            </>
          ) : (
            none("The sweep's analysis")
          )
        }
        go={{ href: "/sweep/?view=power", label: "See the curves →" }}
      />
      <li className={s.arrow} aria-hidden="true">
        →
      </li>
      <Step
        title="Hold the real market up to it"
        text="The premium real SPX options paid, on the same scale: could a study this size have seen it?"
        mini={
          h && h.mean !== null && years.length ? (
            <>
              <YearsFig years={years} mean={h.mean} />
              <p className={s.note}>
                Mean premium by year, {h.first.slice(0, 4)}–{h.last.slice(0, 4)}; the average&apos;s 95% interval {vp(h.mean_low)} to {vp(h.mean_high)} <InfoTip term="vrp" />
              </p>
            </>
          ) : (
            none("The real premium")
          )
        }
        go={{ href: "/data/?q=premium", label: "Today's premium →" }}
      />
    </ol>
  );
}
