"use client";

import Link from "next/link";
import { useState } from "react";
import type { EChartsOption } from "echarts";
import { Card, PageHead } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type SynthPaths } from "@/lib/api";
import { axis, base, useTokens, type Tokens } from "@/lib/chart";

// ---- chart helpers --------------------------------------------------------

const PATH_COLORS: Array<keyof Tokens> = ["s1", "s2", "s3", "s4", "s5"];

function pathColor(t: Tokens, i: number): string {
  const key = PATH_COLORS[i % PATH_COLORS.length] ?? "s1";
  return String(t[key]);
}

function spotsOption(t: Tokens, d: SynthPaths): EChartsOption {
  const days = Array.from({ length: d.n_days + 1 }, (_, i) => i);
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, trigger: "axis" },
    xAxis: { type: "category", data: days, name: "Day", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", name: "Spot price", nameLocation: "middle", ...axis(t, { nameGap: 50 }) },
    grid: { ...base(t).grid, left: 64 },
    series: d.spots.map((path, i) => ({
      name: `Path ${i + 1}`,
      type: "line" as const,
      data: path,
      showSymbol: false,
      lineStyle: { width: 1.8, color: pathColor(t, i) },
    })),
  };
}

function atmIvOption(t: Tokens, d: SynthPaths): EChartsOption {
  const days = Array.from({ length: d.n_days + 1 }, (_, i) => i);
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, trigger: "axis" },
    xAxis: { type: "category", data: days, name: "Day", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", name: "ATM IV (%)", nameLocation: "middle", ...axis(t, { nameGap: 50 }) },
    grid: { ...base(t).grid, left: 64 },
    series: d.atm_ivs.map((path, i) => ({
      name: `Path ${i + 1}`,
      type: "line" as const,
      data: path,
      showSymbol: false,
      lineStyle: { width: 1.8, color: pathColor(t, i) },
    })),
  };
}

function smileOption(t: Tokens, d: SynthPaths): EChartsOption {
  const lm = d.smile_log_moneyness.map((v) => v.toFixed(3));
  const avgIv = d.smile_log_moneyness.map((_, j) => {
    const vals = d.smile_ivs.map((p) => p[j]).filter((v): v is number => v !== undefined);
    return vals.reduce((s, v) => s + v, 0) / Math.max(vals.length, 1);
  });
  return {
    ...base(t),
    tooltip: { ...base(t).tooltip, trigger: "axis" },
    xAxis: { type: "category", data: lm, name: "Log-moneyness log(S/K)", nameLocation: "middle", ...axis(t) },
    yAxis: { type: "value", name: "IV (%)", nameLocation: "middle", ...axis(t, { nameGap: 50 }) },
    grid: { ...base(t).grid, left: 64 },
    series: [
      ...d.smile_ivs.map((path, i) => ({
        name: `Path ${i + 1}`,
        type: "line" as const,
        data: path,
        showSymbol: false,
        lineStyle: { width: 1, color: pathColor(t, i), opacity: 0.45 },
      })),
      {
        name: "Average",
        type: "line" as const,
        data: avgIv,
        showSymbol: false,
        lineStyle: { width: 2.5, color: t.text },
        z: 10,
      },
    ],
  };
}

// ---- form -----------------------------------------------------------------

interface FormState {
  sigma: number;
  vol_of_vol: number;
  skew_slope: number;
  alpha_bps: number;
  n_days: number;
  n_paths: number;
  seed: number;
}

const DEFAULTS: FormState = {
  sigma: 0.20,
  vol_of_vol: 0.0,
  skew_slope: 0.0,
  alpha_bps: 0.0,
  n_days: 63,
  n_paths: 5,
  seed: 0,
};

function NumberRow({
  label,
  name,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  name: keyof FormState;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (name: keyof FormState, v: number) => void;
}) {
  return (
    <label className={ui.field}>
      <span className={ui.fieldLabel}>{label}</span>
      <input
        className={ui.input}
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(name, v);
        }}
      />
    </label>
  );
}

// ---- page -----------------------------------------------------------------

export default function SynthPage() {
  const t = useTokens();
  const [form, setForm] = useState<FormState>(DEFAULTS);
  const [data, setData] = useState<SynthPaths | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  function set(name: keyof FormState, v: number) {
    setForm((f) => ({ ...f, [name]: v }));
  }

  async function generate() {
    setLoading(true);
    setErrorMsg(null);
    try {
      const result = await api.synthPaths({
        sigma: form.sigma,
        vol_of_vol: form.vol_of_vol,
        skew_slope: form.skew_slope,
        alpha_bps: form.alpha_bps,
        n_days: form.n_days,
        n_paths: form.n_paths,
        seed: form.seed,
      });
      setData(result);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const midDay = data ? data.n_days >> 1 : 0;

  return (
    <>
      <PageHead
        eyebrow="Synthetic market"
        title="Synth Data"
        lede="Visualise sample price and vol paths before running RL on them. Adjust the knobs and click Generate."
      />
      <div className={ui.split}>
        {/* ---- left: controls ---- */}
        <Card title="Market knobs">
          <div className={ui.stack}>
            <NumberRow label="sigma (annualised vol)" name="sigma" value={form.sigma} min={0.05} max={1.0} step={0.01} onChange={set} />
            <NumberRow label="vol_of_vol" name="vol_of_vol" value={form.vol_of_vol} min={0} max={2.0} step={0.05} onChange={set} />
            <NumberRow label="skew_slope" name="skew_slope" value={form.skew_slope} min={-1} max={1} step={0.05} onChange={set} />
            <NumberRow label="alpha_bps (drift per day)" name="alpha_bps" value={form.alpha_bps} min={0} max={200} step={5} onChange={set} />
            <NumberRow label="n_days" name="n_days" value={form.n_days} min={5} max={252} step={1} onChange={set} />
            <NumberRow label="n_paths" name="n_paths" value={form.n_paths} min={1} max={20} step={1} onChange={set} />
            <NumberRow label="seed" name="seed" value={form.seed} min={0} max={9999} step={1} onChange={set} />
            <button
              type="button"
              className={`${ui.btn} ${ui.btnPrimary}`}
              onClick={() => { void generate(); }}
              disabled={loading}
            >
              {loading ? "Generating…" : "Generate"}
            </button>
            <Link
              href={`/launch/?sigma=${form.sigma}&vol_of_vol=${form.vol_of_vol}&skew_slope=${form.skew_slope}&alpha_bps=${form.alpha_bps}&n_days=${form.n_days}`}
              className={ui.btn}
              style={{ textAlign: "center", textDecoration: "none" }}
            >
              Train on this market →
            </Link>
            {errorMsg && (
              <p style={{ color: "var(--fail)", fontSize: "0.85rem", margin: "0.25rem 0 0" }}>
                {errorMsg}
              </p>
            )}
          </div>
        </Card>

        {/* ---- right: charts ---- */}
        <div className={ui.stack}>
          {loading && <Loading what="paths" />}
          {!loading && !data && (
            <Card title="Price paths">
              <p style={{ color: "var(--muted)", textAlign: "center", padding: "2rem 0" }}>
                Set the knobs and click Generate to preview the synthetic market.
              </p>
            </Card>
          )}
          {data && (
            <>
              <Card title="Price paths">
                <Chart option={spotsOption(t, data)} height={240} label={`Spot price paths (${data.n_paths} paths, ${data.n_days} days)`} />
              </Card>
              <Card title="ATM IV (%)">
                <Chart option={atmIvOption(t, data)} height={240} label={`ATM implied volatility paths (${data.n_paths} paths)`} />
              </Card>
              <Card title={`Smile snapshot — day ${midDay}`}>
                <Chart option={smileOption(t, data)} height={240} label="Volatility smile at episode mid-point" />
              </Card>
            </>
          )}
        </div>
      </div>
    </>
  );
}
