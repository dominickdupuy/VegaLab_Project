"use client";

import { useEffect, useMemo, useState } from "react";
import { Card, Segmented } from "@/components/Bits";
import { Chart3D, type Grid3D } from "@/components/gl/Chart3D";
import type { GLHover } from "@/components/gl/GLView";
import { InfoTip, Labelled } from "@/components/InfoTip";
import { ErrorBox, Loading } from "@/components/States";
import { api, type PowerParams, type PowerSurface as Surface } from "@/lib/api";
import { type Tokens, useTokens } from "@/lib/chart";
import { box, type GLOption, line, type Point3, points, surface, volRamp, zColour } from "@/lib/chart3d";
import { num, pct, usd } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * Exact detection probability over planted premium x seeds at the page's N and
 * variance components: optspread.power.power_t in every cell, the target-power contour
 * from detectable_effect_t. The effect scales linearly with the premium from delta,
 * the effect at one vol point; tau^2 and sigma^2 are held where the sliders put them.
 */

const MAX_PREMIUM = [
  { value: "2", label: "0–2" },
  { value: "4", label: "0–4" },
  { value: "8", label: "0–8" },
] as const;
type MaxPremium = (typeof MAX_PREMIUM)[number]["value"];

const MAX_SEEDS = [
  { value: "20", label: "S ≤ 20" },
  { value: "50", label: "S ≤ 50" },
  { value: "100", label: "S ≤ 100" },
] as const;
type MaxSeeds = (typeof MAX_SEEDS)[number]["value"];

const ls = (s: number) => Math.log10(s);

function surfaceOption(t: Tokens, d: Surface, mark: { premium: number; seeds: number; power: number | null }): GLOption {
  const pts: Point3[] = [];
  d.seeds.forEach((s, i) => d.premium.forEach((x, j) => pts.push([x, ls(s), d.power[i]?.[j] ?? NaN])));
  const target = d.params.target_power;
  const top = d.params.max_premium;
  const contour: Point3[] = d.seeds.flatMap((s, i) => {
    const x = d.contour_premium[i];
    return x !== null && x !== undefined && x <= top ? [[x, ls(s), target] as Point3] : [];
  });
  const floor = contour.map(([x, y]) => [x, y, 0] as Point3);
  const ticks = [2, 5, 10, 20, 50, 100, 200];
  return {
    ...box(
      t,
      { name: "Premium (vol pts)", min: 0, max: top },
      {
        name: "Seeds S",
        min: ls(d.seeds[0] ?? 2),
        max: ls(d.seeds[d.seeds.length - 1] ?? 100),
        formatter: (v: number) => {
          const s = Math.round(10 ** v);
          // one label per tick value: the grid step nearest log10(s), not every step within reach
          return ticks.includes(s) && Math.abs(ls(s) - v) < 0.025 ? String(s) : "";
        },
        interval: 0.05,
      },
      { name: "Power", min: 0, max: 1, interval: 0.2, formatter: (v: number) => pct(v) },
      { width: 120, depth: 96, height: 62 },
    ),
    visualMap: [zColour(t, 0, 1, volRamp(t), [0], "power", (v) => pct(v))],
    series: [
      surface(t, "main", pts, d.seeds.length, d.premium.length),
      line("contour", contour, t.s2, 4),
      line("contour-floor", floor, t.s2, 2, { lineStyle: { color: t.s2, width: 2, opacity: 0.55 } }),
      ...(mark.power !== null ? [points("design", [[mark.premium, ls(mark.seeds), mark.power]], t.s2, 14, { silent: false, itemStyle: { color: t.s2, borderColor: t.surface, borderWidth: 2, opacity: 1 } })] : []),
    ],
  };
}

/**
 * The Power lab's main view. `mark` is the page's own design: the effect it set (in
 * vol points) at its S, with the exact power the page computed for it.
 */
export function PowerSurfaceCard({ params, seeds, mark }: { params: PowerParams; seeds: number; mark: { premium: number; power: number | null } }) {
  const t = useTokens();
  const [maxPremium, setMaxPremium] = useState<MaxPremium>("4");
  const [maxSeeds, setMaxSeeds] = useState<MaxSeeds>("50");
  const q = {
    tau2: params.tau2,
    sigma2: params.sigma2,
    n_episodes: params.n_episodes,
    delta: params.delta,
    alpha: params.alpha,
    target_power: params.target_power,
    max_premium: Number(maxPremium),
    max_seeds: Number(maxSeeds),
  };
  const res = useFetch((s) => api.powerSurface(q, s), JSON.stringify(q));
  const [last, setLast] = useState<Surface | undefined>();
  useEffect(() => {
    if (res.data) setLast(res.data);
  }, [res.data]);
  const d = res.data ?? last;

  const built = useMemo(() => {
    if (!d) return null;
    // the design shows when it lies inside the surface's box
    const inside = mark.power !== null && mark.premium <= d.params.max_premium && seeds >= (d.seeds[0] ?? 2) && seeds <= (d.seeds[d.seeds.length - 1] ?? 100);
    const m = { premium: mark.premium, seeds, power: inside ? mark.power : null };
    const pts: Point3[] = [];
    d.seeds.forEach((s, r) => d.premium.forEach((x, c) => pts.push([x, ls(s), d.power[r]?.[c] ?? NaN])));
    const nav: Grid3D = {
      rows: d.seeds.length,
      cols: d.premium.length,
      describe: (r, c) => {
        const p = d.power[r]?.[c];
        const x = d.premium[c] ?? 0;
        const s = d.seeds[r] ?? 0;
        if (p === null || p === undefined) return null;
        return `${s} seeds of ${d.params.n_episodes} episodes detect a ${x.toFixed(2)} vol pt premium ${pct(p, 0)} of the time (${usd(d.params.delta * x)} per episode)`;
      },
      at: (r, c) => {
        const p = pts[r * d.premium.length + c];
        return p && Number.isFinite(p[2]) ? p : null;
      },
    };
    return { option: surfaceOption(t, d, m), nav, mark: m };
  }, [t, d, seeds, mark.premium, mark.power]);

  const hover = (h: GLHover) => {
    if (!d || !built) return null;
    if (h.seriesId === "design" && built.mark.power !== null) return `your design: ${seeds} seeds of ${d.params.n_episodes} episodes, ${built.mark.premium.toFixed(2)} vol pts · power ${pct(built.mark.power, 1)}`;
    return null;
  };
  const s5 = d ? d.seeds.indexOf(5) : -1;
  const floor5 = d && s5 >= 0 ? d.contour_premium[s5] : null;
  const outside = d && built && mark.power !== null && built.mark.power === null;

  return (
    <Card
      title="Power surface" info="power.surface"
      sub={`Premium × seeds at N = ${params.n_episodes}. Orange line: ${pct(params.target_power)} power${floor5 !== null && floor5 !== undefined ? ` (${floor5.toFixed(2)} vol pts at S = 5)` : ""}. Dot: your design${outside ? ", outside these axes" : ""}.`}
      right={
        <>
          <Segmented label="Premium axis, vol pts" value={maxPremium} options={MAX_PREMIUM} onChange={setMaxPremium} />
          <InfoTip term="power.premiumAxis" label="About the premium axis" />
          <Segmented label="Seed axis" value={maxSeeds} options={MAX_SEEDS} onChange={setMaxSeeds} />
          <InfoTip term="power.seedAxis" label="About the seed axis" />
        </>
      }
    >
      {res.error ? <ErrorBox what="the power surface" error={res.error} onRetry={res.reload} /> : null}
      {!d && !res.error ? <Loading what="the power surface" /> : null}
      {built && d ? (
        <div style={{ opacity: res.loading ? 0.7 : 1, transition: "opacity var(--t-fast)" }}>
          <Chart3D
            option={built.option}
            grid={built.nav}
            describeHover={hover}
            height={480}
            label="Exact detection probability by planted premium and seed count, with the target power contour"
            camera={{ alpha: 22, beta: 40, distance: 215 }}
            stamp={`${d.source} · N = ${d.params.n_episodes} · τ² ${num(d.params.tau2)} · σ² ${num(d.params.sigma2)} · α ${d.params.alpha}`}
          />
        </div>
      ) : null}
    </Card>
  );
}
