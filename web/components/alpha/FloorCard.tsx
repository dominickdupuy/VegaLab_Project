"use client";

import { useMemo, useState } from "react";
import { Card } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { Labelled } from "@/components/InfoTip";
import { Reading } from "@/components/PageBits";
import ui from "@/components/ui.module.css";
import { type AlphaDataset, type AlphaOptions, type AlphaSweep, ApiError, api } from "@/lib/api";
import { ARM_LABEL, useTokens } from "@/lib/chart";
import { num, pct, usd } from "@/lib/format";
import { guide } from "@/lib/guide";
import { alphaArmColor } from "./arms";
import { capturedOption, powerOption } from "./floorCharts";

/**
 * How big must a planted edge be before an arm finds it reliably? An edge-size sweep per
 * arm on the open dataset's generator (the knob it moves is the server's, edge_knob), each
 * size tested on the per-seed captured edge with the study's one-sided t, and the study's
 * power curve and 80% floor per arm (optspread.alpha.floor).
 */

type Floor = NonNullable<AlphaSweep["floor"]>;

const DEFAULT_VALUES: Record<string, string> = { premium_pts: "0, 1, 2, 4", vol_pts: "0, 2, 4, 8", bps_per_day: "0, 2, 5, 10" };

function parse(text: string): number[] {
  return text
    .split(/[,\s]+/)
    .filter(Boolean)
    .map(Number);
}

const label = (arm: string) => ARM_LABEL[arm] ?? arm;
const g3 = (x: number) => (Math.abs(x) >= 10 ? num(x, 1) : String(Number(x.toPrecision(3))));

/** "1 vol point", "2 vol points". */
const sized = (x: number, unit: string) => `${g3(x)} ${x === 1 ? unit.replace(/points$/, "point") : unit}`;

/** One arm's floor in words, for the takeaway and the table. */
function floorWords(f: Floor, unit: string): string {
  if (f.status === "untestable") return "no test (needs two or more seeds)";
  if (f.status === "not_reached") return `not even at ${sized(f.bracket?.[0] ?? 0, unit)} (best power ${pct(f.max_power ?? 0)})`;
  if (f.status === "below_grid") return `at every size tried: under ${sized(f.bracket?.[1] ?? f.edge ?? 0, unit)}`;
  return `from about ${sized(f.edge ?? 0, unit)}`;
}

function takeaway(rows: { arm: string; floor: Floor }[], unit: string): string {
  const parts = rows.filter((r) => r.floor.status !== "untestable").map((r) => `${label(r.arm)} ${floorWords(r.floor, unit)}`);
  if (!parts.length) return "No arm has a floor yet: each size needs at least two training seeds (or path blocks) to test.";
  return `Found 80% of the time: ${parts.join("; ")}.`;
}

function roughNote(rows: { floor: Floor }[]): string {
  const ns = rows.map((r) => r.floor.n);
  const lo = Math.min(...ns);
  const hi = Math.max(...ns);
  const n = lo === hi ? `${lo}` : `${lo} to ${hi}`;
  return `Rough: each size is tested on only ${n} figures (training seeds, or path blocks for a rule), so each floor is a bracket between swept sizes, not a precise number, and tends to read low.`;
}

export function FloorCard({ options, ds, sweeps, onJob }: { options: AlphaOptions; ds: AlphaDataset; sweeps: AlphaSweep[]; onJob: (id: string) => void }) {
  const t = useTokens();
  const gen = options.generators.find((g) => g.kind === (ds.spec as { generator?: { kind?: string } }).generator?.kind);
  const knob = gen?.edge_knob ?? null;
  const unit = gen?.edge_unit ?? "";
  const [values, setValues] = useState(DEFAULT_VALUES[knob ?? ""] ?? "0, 1, 2, 4");
  const [arms, setArms] = useState<string[]>(() => (options.learners_available ? ["ppo", "delta_seller"] : ["delta_seller"]));
  const [steps, setSteps] = useState<number>(options.default_timesteps);
  const [seeds, setSeeds] = useState<number>(Math.min(3, options.max_seeds));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError>();
  const [grid, setGrid] = useState(0);

  // the edge sweeps on this dataset, by grid of sizes; per grid the newest sweep of each arm
  const grids = useMemo(() => {
    const by = new Map<string, Map<string, AlphaSweep>>();
    for (const s of sweeps) {
      if (s.spec.axis !== "edge" || !s.floor) continue;
      const key = s.spec.values.join(", ");
      const arms = by.get(key) ?? new Map<string, AlphaSweep>();
      if (!arms.has(s.spec.arm)) arms.set(s.spec.arm, s);
      by.set(key, arms);
    }
    return [...by.entries()].map(([key, m]) => ({ key, sweeps: [...m.values()] }));
  }, [sweeps]);
  const shown = grids[Math.min(grid, grids.length - 1)];
  const rows = useMemo(() => (shown ? shown.sweeps.map((s) => ({ arm: s.spec.arm, sweep: s, floor: s.floor as Floor })) : []), [shown]);
  const xName = `Planted edge, ${unit}`;

  const charts = useMemo(() => {
    if (!rows.length) return null;
    const color = (arm: string) => alphaArmColor(t, arm, options);
    return {
      power: powerOption(
        t,
        rows.map((r) => ({
          name: label(r.arm),
          color: color(r.arm),
          curve: r.floor.curve as [number, number, number, number][],
          knots: r.sweep.points.filter((p) => p.test && Number(p.value) > 0).map((p) => ({ x: Number(p.value), power: p.test!.power, detected: p.test!.detected })),
          floor: r.floor.edge ?? null,
        })),
        xName,
        g3,
      ),
      captured: capturedOption(
        t,
        rows.map((r) => ({
          name: label(r.arm),
          color: color(r.arm),
          points: r.sweep.points.filter((p) => p.test).map((p) => ({ x: Number(p.value), y: p.test!.estimate, se: p.test!.se })),
        })),
        xName,
        g3,
        (v) => usd(v),
      ),
    };
  }, [t, rows, options, xName]);

  const toggle = (arm: string) => setArms((as) => (as.includes(arm) ? as.filter((a) => a !== arm) : [...as, arm]));
  const launch = async () => {
    setBusy(true);
    setError(undefined);
    try {
      for (const arm of arms) {
        const learned = options.learned_arms.includes(arm);
        const spec = { dataset: ds.hash, arm, axis: "edge", values: parse(values), total_timesteps: learned ? steps : null, n_seeds: learned ? seeds : 1 };
        onJob((await api.alphaSweep(spec)).job_id);
      }
      setGrid(0);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    }
    setBusy(false);
  };
  const learnedPicked = arms.filter((a) => options.learned_arms.includes(a));
  const trainings = parse(values).length * seeds * 2 * learnedPicked.length;

  return (
    <Card
      title="How big must the edge be?"
      info="One benchmark per planted size, the same train and held-out paths at every size (each its own dataset, only the edge knob changed; zero is the null). At each size the captured edge is tested on the training seeds with the study's one-sided t at 5% (a rule has no seeds: it is read on five blocks of held-out paths). The sizes give a power curve, and the floor is where it reaches 80%, the study's own curve and floor."
      source={<code>{"<runs root>/alpha/sweeps/<key>/sweep.json"}</code>}
    >
      {!knob ? (
        <p>This generator has no edge size to sweep. Open a constant premium, mean-reverting premium or drift dataset.</p>
      ) : (
        <div className={ui.stack}>
          <div className={ui.row} role="group" aria-label="Arms to sweep">
            {options.arms.map((a) => (
              <label key={a} className={ui.check} {...guide(`alpha-floor-arm-${a}`)}>
                <input type="checkbox" checked={arms.includes(a)} onChange={() => toggle(a)} />
                {label(a)}
              </label>
            ))}
          </div>
          <div className={ui.grid3}>
            <label className={ui.field}>
              <span className={ui.fieldLabel}>Sizes ({gen?.knobs.find((k) => k.name === knob)?.label ?? knob})</span>
              <input className={ui.input} value={values} onChange={(e) => setValues(e.target.value)} />
            </label>
            {learnedPicked.length ? (
              <label className={ui.field}>
                <span className={ui.fieldLabel}>
                  <Labelled term="alpha.budget">Timesteps per learner</Labelled>
                </span>
                <input className={ui.input} type="number" min={2048} step={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
              </label>
            ) : null}
            {learnedPicked.length ? (
              <label className={ui.field}>
                <span className={ui.fieldLabel}>
                  <Labelled term="alpha.seeds">Seeds per size</Labelled>
                </span>
                <input className={ui.input} type="number" min={2} max={options.max_seeds} step={1} value={seeds} onChange={(e) => setSeeds(Number(e.target.value))} />
              </label>
            ) : null}
          </div>
          <div className={ui.row}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !arms.length} onClick={launch} {...guide("alpha-floor-run")}>
              {busy ? "Submitting…" : "Find each arm's floor"}
            </button>
            <span>{learnedPicked.length ? `${num(trainings)} trainings in all` : "rules only: no training"}</span>
          </div>
          {error ? <p className={ui.fieldError}>{error.message}</p> : null}
          {grids.length > 1 ? (
            <label className={ui.field}>
              <span className={ui.fieldLabel}>Sizes shown</span>
              <select className={ui.select} value={grid} onChange={(e) => setGrid(Number(e.target.value))}>
                {grids.map((g, i) => (
                  <option key={g.key} value={i}>
                    {g.key} {unit}: {g.sweeps.map((s) => label(s.spec.arm)).join(", ")}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          {rows.length && charts ? (
            <>
              <Reading note={roughNote(rows)}>{takeaway(rows, unit)}</Reading>
              <Chart
                option={charts.power}
                height={300}
                label="Power to detect the planted edge against its size, per arm, with each arm's 80% floor marked"
                rows={() => rows.flatMap((r) => r.floor.curve.map((c) => ({ arm: r.arm, edge: c[0], power: c[1], lo: c[2], hi: c[3] })))}
              />
              <div className={ui.tableWrap}>
                <table className={ui.table}>
                  <thead>
                    <tr>
                      <th>Arm</th>
                      <th>Floor (80% power)</th>
                      <th>Status</th>
                      <th>Tested on</th>
                      <th>Detected at</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <tr key={r.arm}>
                        <td>{label(r.arm)}</td>
                        <td>{floorWords(r.floor, unit)}</td>
                        <td>{r.floor.status.replace("_", " ")}</td>
                        <td>
                          {r.floor.n} {r.floor.unit}
                        </td>
                        <td>
                          {r.sweep.points
                            .filter((p) => p.test?.detected)
                            .map((p) => g3(Number(p.value)))
                            .join(", ") || "none"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Chart
                option={charts.captured}
                height={260}
                label="Captured edge (signal minus twin, held-out) against the planted size, per arm, with two-standard-error bars"
                rows={() => rows.flatMap((r) => r.sweep.points.filter((p) => p.test).map((p) => ({ arm: r.arm, edge: p.value, captured: p.test!.estimate, se: p.test!.se, t: p.test!.t, detected: p.test!.detected })))}
              />
            </>
          ) : null}
        </div>
      )}
    </Card>
  );
}
