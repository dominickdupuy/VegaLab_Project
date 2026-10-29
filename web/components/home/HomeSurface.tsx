"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { EChartsOption } from "echarts";
import { Chart } from "@/components/Chart";
import { Chart3D } from "@/components/gl/Chart3D";
import { gridNav, gridPoints, ivSurfaceOption } from "@/components/views/IvSurface";
import { api, type M4, type Vrp } from "@/lib/api";
import { axis, base, type Tokens, useTokens } from "@/lib/chart";
import { finiteRange, type GLOption } from "@/lib/chart3d";
import { MINUS, pct } from "@/lib/format";
import { useReducedMotion } from "@/lib/motion";
import { useFetch } from "@/lib/useFetch";
import ui from "../ui.module.css";

const DAYS = 40;
const PER_VOL = 824.3;
const A7 = { tau2: 51_838, sigma2: 105_996_836, n_seeds: 5, n_episodes: 500, delta: PER_VOL, alpha: 0.05, target_power: 0.8, train_steps: 1_000_000, episode_days: 63, train_rate: 804, eval_rate: 1538, cores: 1 };
const vp = (x: number | null | undefined) => (x === null || x === undefined ? "–" : `${x < 0 ? MINUS : "+"}${Math.abs(x).toFixed(2)}`);

/** One computed sentence: the market's premium against what the study can detect. */
function sentence(m4?: M4, vrp?: Vrp): string | null {
  const est = m4?.vrp?.estimate ?? vrp?.mapping?.headline.mean ?? null;
  if (est === null || est === undefined) return null;
  const lo = m4?.vrp?.interval[0] ?? vrp?.mapping?.headline.mean_low ?? null;
  const hi = m4?.vrp?.interval[1] ?? vrp?.mapping?.headline.mean_high ?? null;
  const head = `The real SPX premium averages ${vp(est)} vol pts (95% ${vp(lo)} to ${vp(hi)})`;
  const a = m4?.analysis;
  const f = a?.arms.ppo?.floor;
  if (!a || !f) return `${head}.`;
  if (f.status === "not_reached") return `${head}; PPO is not detected anywhere on the sweep's grid, so a study of this size cannot see a premium that small.`;
  const floor = f.premium ?? 0;
  return `${head}; PPO's detection floor is ${vp(floor).replace("+", "")} vol pts at S = ${a.design.n_seeds}, so ${est < floor ? "a study of this size would most likely miss a premium that small" : "a study of this size would see it"}.`;
}

/** The 2D stand-in when there is no panel or no WebGL: power against premium, the real premium marked. */
function powerOption(t: Tokens, curve: { delta: number[]; power: number[] }, vrp?: Vrp): EChartsOption {
  const h = vrp?.mapping?.headline;
  return {
    ...base(t),
    legend: { show: false },
    tooltip: { ...base(t).tooltip, valueFormatter: (v: unknown) => pct(v as number, 0) },
    grid: { ...base(t).grid, left: 52, top: 24 },
    xAxis: { type: "value", name: "Planted premium (vol pts)", nameLocation: "middle", ...axis(t), max: 4 },
    yAxis: { type: "value", min: 0, max: 1, name: "Detection probability", nameLocation: "middle", ...axis(t), nameGap: 38, axisLabel: { ...axis(t).axisLabel, formatter: (v: number) => pct(v) } },
    series: [
      {
        id: "power",
        name: "PPO, S = 5 × N = 500",
        type: "line",
        showSymbol: false,
        data: curve.delta.map((d, i) => [d / PER_VOL, curve.power[i] ?? null]).filter((p) => (p[0] ?? 0) <= 4),
        lineStyle: { width: 2.5, color: t.s2 },
        itemStyle: { color: t.s2 },
        markArea: h && h.mean_low !== null && h.mean_high !== null ? { silent: true, itemStyle: { color: t.dark ? "rgba(232,238,247,0.07)" : "rgba(19,35,61,0.06)" }, label: { color: t.muted, fontSize: 11, formatter: "real premium, 95%" }, data: [[{ xAxis: Math.max(0, h.mean_low) }, { xAxis: h.mean_high }]] as never } : undefined,
        markLine: h ? { symbol: "none", silent: true, lineStyle: { color: t.text, type: "dashed", width: 1 }, label: { color: t.text2, formatter: `real ${vp(h.mean)}` }, data: [{ xAxis: h.mean }] as never } : undefined,
      },
    ],
  };
}

function Fallback({ vrp }: { vrp?: Vrp }) {
  const t = useTokens();
  const power = useFetch((s) => api.power(A7, s), "home-power");
  if (!power.data) return <div className={ui.heroStage} aria-hidden="true" />;
  return <Chart option={powerOption(t, power.data.curve, vrp)} height={360} label="Detection probability against the planted premium for a 5-seed PPO study, with the real premium marked" name="power against premium" />;
}

export default function HomeSurface({ m4, vrp, built }: { m4?: M4; vrp?: Vrp; built?: boolean }) {
  const t = useTokens();
  const reduced = useReducedMotion();
  const dates = useFetch((s) => api.gridDates(s), "home-grid-dates", built === true);
  const recent = useMemo(() => (dates.data ?? []).slice(-DAYS), [dates.data]);
  const frames = useFetch((s) => api.gridFrames(recent[0] ?? "", recent.length, s), `home-frames|${recent[0] ?? ""}`, recent.length > 0);
  const f = frames.data;
  const n = f?.frames.length ?? 0;
  const [idx, setIdx] = useState<number | null>(null); // null: the newest day
  const [playing, setPlaying] = useState(false);
  const [seen, setSeen] = useState(true);
  const stage = useRef<HTMLDivElement>(null);
  const played = useRef(false);

  // play the last trading days once on arrival, then rest on the newest; never under reduced motion
  useEffect(() => {
    if (!n || played.current || reduced) return;
    played.current = true;
    setIdx(0);
    setPlaying(true);
  }, [n, reduced]);
  useEffect(() => {
    if (!playing || !seen) return;
    const id = window.setInterval(() => {
      setIdx((i) => {
        const next = (i ?? 0) + 1;
        if (next >= n) {
          setPlaying(false);
          return null;
        }
        return next;
      });
    }, 140);
    return () => window.clearInterval(id);
  }, [playing, seen, n]);

  // rotate only while on screen and the tab is visible
  useEffect(() => {
    const el = stage.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setSeen(Boolean(e?.isIntersecting) && !document.hidden), { threshold: 0.2 });
    io.observe(el);
    const vis = () => setSeen((s) => (document.hidden ? false : s || true));
    document.addEventListener("visibilitychange", vis);
    return () => {
      io.disconnect();
      document.removeEventListener("visibilitychange", vis);
    };
  }, [n]);

  const z = useMemo<[number, number]>(() => (f ? finiteRange(f.frames.flatMap((fr) => fr.iv.flat().map((v) => (v === null ? NaN : v * 100))), 0.02) : [0, 1]), [f]);
  const shown = f ? f.frames[idx ?? n - 1] : undefined;
  const day = useMemo(() => (f && shown ? { k: f.k, tenor_days: f.tenor_days, iv: shown.iv } : null), [f, shown]);
  const pts = useMemo(() => (day ? gridPoints(day, "k") : []), [day]);
  const rotate = !reduced && !playing && seen;
  const option = useMemo<GLOption | null>(() => {
    if (!day) return null;
    const o = ivSurfaceOption(t, day, "k", z, pts);
    const g = (o.grid3D ?? {}) as GLOption;
    return { ...o, grid3D: { ...g, viewControl: { ...((g.viewControl ?? {}) as GLOption), autoRotate: rotate, autoRotateSpeed: 4, autoRotateAfterStill: 5 } } };
  }, [t, day, z, pts, rotate]);
  const nav = useMemo(() => (day ? gridNav(day, pts) : undefined), [day, pts]);
  const text = sentence(m4, vrp);
  const noPanel = built === false || dates.error || frames.error || (dates.data && !dates.data.length);

  return (
    <div className={ui.heroSide} ref={stage}>
      {noPanel ? (
        <Fallback vrp={vrp} />
      ) : option && shown ? (
        <>
          <Chart3D
            option={option}
            grid={nav}
            height={380}
            label={`The SPX implied-volatility surface on ${shown.date}: log-moneyness by tenor by implied vol`}
            fallback={<Fallback vrp={vrp} />}
            hint="The real SPX vol surface, from our panel · drag to rotate"
            stamp={`panel/surface_grid.parquet · ${f?.frames[0]?.date} to ${f?.frames[n - 1]?.date}`}
          />
          <div className={ui.scrub}>
            <input
              className={ui.range}
              type="range"
              min={0}
              max={Math.max(0, n - 1)}
              step={1}
              value={idx ?? n - 1}
              aria-label="Trading day"
              aria-valuetext={shown.date}
              onChange={(e) => {
                setPlaying(false);
                setIdx(Number(e.target.value));
              }}
            />
            <span className={ui.frameDate}>{shown.date}</span>
            <span className={ui.dim}>
              {String((idx ?? n - 1) + 1).padStart(2, "0")}/{n}
            </span>
          </div>
        </>
      ) : (
        <div className={ui.heroStage} aria-hidden="true" />
      )}
      {text ? <p className={ui.heroSentence}>{text}</p> : null}
    </div>
  );
}
