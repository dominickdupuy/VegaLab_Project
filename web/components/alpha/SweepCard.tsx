"use client";

import { useMemo, useState } from "react";
import { Card } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Labelled } from "@/components/InfoTip";
import { Reading } from "@/components/PageBits";
import ui from "@/components/ui.module.css";
import { type AlphaDataset, type AlphaOptions, ApiError, api, type AlphaSweep } from "@/lib/api";
import { ARM_LABEL, useTokens } from "@/lib/chart";
import { num, pct, usd } from "@/lib/format";
import { curveOption } from "./charts";

type Axis = "n_train" | "total_timesteps";
const AXIS_LABEL: Record<Axis, string> = { n_train: "Train paths", total_timesteps: "Training steps" };
const DEFAULT_VALUES: Record<Axis, string> = { n_train: "8, 32, 128", total_timesteps: "10000, 50000, 200000" };

function parse(text: string): number[] {
  return text
    .split(/[,\s]+/)
    .filter(Boolean)
    .map(Number);
}

/** What the sweep says, in one sentence: where overfitting stops and the ratio it reaches. */
function reading(s: AlphaSweep): string {
  const axis = AXIS_LABEL[s.spec.axis as Axis].toLowerCase();
  const pts = s.points;
  const last = pts[pts.length - 1];
  if (!last) return "";
  const z = (v?: number | null, se?: number | null) => (v !== null && v !== undefined && se ? v / se : 0);
  const overfit = pts.filter((p) => z(p.overfit_gap.value, p.overfit_gap.se) > 2).map((p) => num(p.value));
  const ratio = last.capture_ratio.value;
  const tail = ratio === null || ratio === undefined ? "there is no attainable edge to capture" : `at ${num(last.value)} ${axis} it takes ${pct(ratio)} ± ${pct(2 * (last.capture_ratio.se ?? 0))} of the edge`;
  const head = overfit.length ? `Overfit at ${overfit.join(", ")} ${axis}` : "No point overfits detectably";
  return `${head}; ${tail}.`;
}

export function SweepCard({ options, ds, sweeps, onJob }: { options: AlphaOptions; ds: AlphaDataset; sweeps: AlphaSweep[]; onJob: (id: string) => void }) {
  const t = useTokens();
  const [axis, setAxis] = useState<Axis>("n_train");
  const [values, setValues] = useState(DEFAULT_VALUES.n_train);
  const [arm, setArm] = useState(options.learners_available ? "ppo" : "delta_seller");
  const [steps, setSteps] = useState<number>(options.default_timesteps);
  const [seeds, setSeeds] = useState<number>(Math.min(3, options.max_seeds));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError>();
  const learned = options.learned_arms.includes(arm);
  const [shown, setShown] = useState(0);
  const sweep = sweeps[Math.min(shown, sweeps.length - 1)];

  const launch = async () => {
    setBusy(true);
    setError(undefined);
    try {
      const spec = {
        dataset: ds.hash,
        arm,
        axis,
        values: parse(values),
        total_timesteps: axis === "n_train" && learned ? steps : null,
        n_seeds: learned ? seeds : 1,
      };
      onJob((await api.alphaSweep(spec)).job_id);
      setShown(0);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    }
    setBusy(false);
  };

  const curves = useMemo(() => {
    if (!sweep) return null;
    const xName = AXIS_LABEL[sweep.spec.axis as Axis];
    const pts = (pick: (p: AlphaSweep["points"][number]) => { value?: number | null; se?: number | null } | null | undefined) =>
      sweep.points.map((p) => ({ x: p.value, y: pick(p)?.value ?? null, se: pick(p)?.se ?? null }));
    return {
      ratio: curveOption(t, [{ name: "Capture ratio", color: t.s1, points: pts((p) => p.capture_ratio) }], xName, "Share of the edge", (v) => pct(v)),
      gaps: curveOption(
        t,
        [
          { name: "Overfit gap", color: t.s2, points: pts((p) => p.overfit_gap) },
          { name: "Noise fit", color: t.s4, points: pts((p) => p.noise_fit) },
        ],
        xName,
        "$ per episode",
        (v) => usd(v),
      ),
    };
  }, [t, sweep]);

  return (
    <Card
      title="How much data does it take?"
      info="Runs one benchmark per value: the number of train paths (each its own dataset, scored on the same held-out paths, each train set the start of the next) or the training steps. Overfitting shows as an overfit gap and a noise fit that shrink as the train set grows."
    >
      <div className={ui.stack}>
        <div className={ui.grid3}>
          <label className={ui.field}>
            <span className={ui.fieldLabel}>Sweep over</span>
            <select
              className={ui.select}
              value={axis}
              onChange={(e) => {
                const a = e.target.value as Axis;
                setAxis(a);
                setValues(DEFAULT_VALUES[a]);
              }}
            >
              <option value="n_train">Train paths</option>
              <option value="total_timesteps" disabled={!options.learners_available}>
                Training steps
              </option>
            </select>
          </label>
          <label className={ui.field}>
            <span className={ui.fieldLabel}>Values</span>
            <input className={ui.input} value={values} onChange={(e) => setValues(e.target.value)} />
          </label>
          <label className={ui.field}>
            <span className={ui.fieldLabel}>
              <Labelled term="alpha.arm">Arm</Labelled>
            </span>
            <select className={ui.select} value={arm} onChange={(e) => setArm(e.target.value)}>
              {options.arms
                .filter((a) => axis === "n_train" || options.learned_arms.includes(a))
                .map((a) => (
                  <option key={a} value={a}>
                    {ARM_LABEL[a] ?? a}
                  </option>
                ))}
            </select>
          </label>
          {learned && axis === "n_train" ? (
            <label className={ui.field}>
              <span className={ui.fieldLabel}>
                <Labelled term="alpha.budget">Timesteps per learner</Labelled>
              </span>
              <input className={ui.input} type="number" min={2048} step={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
            </label>
          ) : null}
          {learned ? (
            <label className={ui.field}>
              <span className={ui.fieldLabel}>
                <Labelled term="alpha.seeds">Seeds per point</Labelled>
              </span>
              <input className={ui.input} type="number" min={1} max={options.max_seeds} step={1} value={seeds} onChange={(e) => setSeeds(Number(e.target.value))} />
            </label>
          ) : null}
        </div>
        <div className={ui.row}>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy} onClick={launch}>
            {busy ? "Submitting…" : "Run the sweep"}
          </button>
          {learned ? (
            <span>
              {num(parse(values).length * seeds * 2)} trainings in all
            </span>
          ) : null}
        </div>
        {error ? <p className={ui.fieldError}>{error.message}</p> : null}
        {sweeps.length > 1 ? (
          <label className={ui.field}>
            <span className={ui.fieldLabel}>Sweep shown</span>
            <select className={ui.select} value={shown} onChange={(e) => setShown(Number(e.target.value))}>
              {sweeps.map((s, i) => (
                <option key={s.key} value={i}>
                  {ARM_LABEL[s.spec.arm] ?? s.spec.arm} over {AXIS_LABEL[s.spec.axis as Axis].toLowerCase()} ({s.spec.values.map((v) => num(v)).join(", ")}), {s.spec.n_seeds} seed{s.spec.n_seeds === 1 ? "" : "s"}
                </option>
              ))}
            </select>
          </label>
        ) : null}
        {sweep && curves ? (
          <>
            <Reading>{reading(sweep)}</Reading>
            <div className={ui.grid2}>
              <Chart option={curves.ratio} height={260} label="Capture ratio against the swept setting, with two-standard-error bars" rows={() => sweep.points.map((p) => ({ value: p.value, capture_ratio: p.capture_ratio.value, se: p.capture_ratio.se }))} />
              <Chart option={curves.gaps} height={260} label="Overfit gap and noise fit against the swept setting, with two-standard-error bars" rows={() => sweep.points.map((p) => ({ value: p.value, overfit_gap: p.overfit_gap.value, overfit_se: p.overfit_gap.se, noise_fit: p.noise_fit?.value ?? null, noise_se: p.noise_fit?.se ?? null }))} />
            </div>
          </>
        ) : null}
      </div>
    </Card>
  );
}
