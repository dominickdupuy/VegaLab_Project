"use client";

import { useMemo } from "react";
import { Card } from "@/components/Bits";
import { Chart } from "@/components/Chart";
import { InfoTip } from "@/components/InfoTip";
import { Reading } from "@/components/PageBits";
import ui from "@/components/ui.module.css";
import type { Benchmark, BenchStat } from "@/lib/api";
import { ARM_LABEL, armColor, useTokens } from "@/lib/chart";
import { num, pct, usd } from "@/lib/format";
import { guide } from "@/lib/guide";
import { forestOption } from "./charts";

/** What identifies a benchmark on one dataset: the arm, its budget and its seeds. */
export function benchKey(b: Benchmark): string {
  return `${b.arm}-${b.total_timesteps ?? 0}-s${b.n_seeds ?? 1}`;
}

function setting(b: Benchmark): string {
  const seeds = b.n_seeds ?? 1;
  return b.total_timesteps ? `${num(b.total_timesteps)} steps, ${seeds} seed${seeds === 1 ? "" : "s"}` : "scripted";
}

function cell(s: BenchStat | null | undefined, fmt: (v: number) => string): string {
  if (!s || s.value === null || s.value === undefined) return "none";
  return s.se === null || s.se === undefined ? fmt(s.value) : `${fmt(s.value)} ± ${fmt(2 * s.se)}`;
}

/** The best capture ratio, read as one sentence, and whether the arms can be told apart. */
function reading(bs: Benchmark[]): string {
  const rated = bs.filter((b) => b.capture_ratio.value !== null && b.capture_ratio.value !== undefined);
  if (!rated.length) return "None of these results has an attainable edge to capture on this dataset.";
  const best = [...rated].sort((a, b) => (b.capture_ratio.value ?? 0) - (a.capture_ratio.value ?? 0))[0]!;
  const r = best.capture_ratio;
  const others = rated.filter((b) => b !== best);
  const apart = others.every((o) => (r.value ?? 0) - 2 * (r.se ?? 0) > (o.capture_ratio.value ?? 0) + 2 * (o.capture_ratio.se ?? 0));
  const head = `${ARM_LABEL[best.arm] ?? best.arm} (${setting(best)}) took the most: ${pct(r.value ?? 0)} ± ${pct(2 * (r.se ?? 0))} of the edge.`;
  if (!others.length) return head;
  return apart ? `${head} Its bar clears every other arm's.` : `${head} Its bar overlaps another arm's, so this dataset does not tell them apart.`;
}

export function AgentsCard({ benches, selected, onSelect }: { benches: Benchmark[]; selected?: string; onSelect: (key: string) => void }) {
  const t = useTokens();
  const rated = benches.filter((b) => b.capture_ratio.value !== null && b.capture_ratio.value !== undefined);
  const option = useMemo(
    () =>
      forestOption(
        t,
        rated.map((b) => ({
          label: `{p|${ARM_LABEL[b.arm] ?? b.arm}}{w|${setting(b)}}`,
          mean: b.capture_ratio.value ?? 0,
          se: b.capture_ratio.se ?? 0,
          color: armColor(t, b.arm),
          tip: `${ARM_LABEL[b.arm] ?? b.arm}, ${setting(b)}<br/>capture ratio ${cell(b.capture_ratio, (v) => pct(v))}`,
        })),
        "Capture ratio (share of the attainable edge)",
        (v) => pct(v),
      ),
    [t, rated],
  );
  if (!benches.length) return null;
  return (
    <Card
      title="Agents on this dataset"
      info="Every result saved for this dataset, one row per arm, budget and seed count. A row with one seed has a se from path noise alone: it says nothing about how another training seed would have done. Click a row to see it in full."
    >
      <div className={ui.stack}>
        <Reading>{reading(benches)}</Reading>
        <div className={ui.tableWrap}>
          <table className={ui.table}>
            <thead>
              <tr>
                <th>
                  Arm <InfoTip term="alpha.arm" />
                </th>
                <th>Setting</th>
                <th className={ui.num}>
                  Capture ratio <InfoTip term="alpha.capture" />
                </th>
                <th className={ui.num}>
                  Edge from the signal <InfoTip term="alpha.captured" label="About the edge from the signal" />
                </th>
                <th className={ui.num}>
                  Overfit gap <InfoTip term="alpha.overfit" />
                </th>
                <th className={ui.num}>
                  Noise fit <InfoTip term="alpha.noiseFit" />
                </th>
              </tr>
            </thead>
            <tbody>
              {benches.map((b) => (
                <tr
                  key={benchKey(b)}
                  className={`${ui.clickRow} ${benchKey(b) === selected ? ui.rowSelected : ""}`}
                  onClick={() => onSelect(benchKey(b))}
                  data-selected={benchKey(b) === selected ? "" : undefined}
                  {...guide(`alpha-bench-${benchKey(b)}`)}
                >
                  <td>{ARM_LABEL[b.arm] ?? b.arm}</td>
                  <td>{setting(b)}</td>
                  <td className={ui.num}>{cell(b.capture_ratio, (v) => pct(v))}</td>
                  <td className={ui.num}>{cell(b.captured, usd)}</td>
                  <td className={ui.num}>{cell(b.overfit_gap, usd)}</td>
                  <td className={ui.num}>{cell(b.noise_fit, usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rated.length ? (
          <Chart
            option={option}
            height={60 + 34 * rated.length}
            label="Capture ratio by arm and setting, with two-standard-error bars"
            rows={() => rated.map((b) => ({ arm: b.arm, total_timesteps: b.total_timesteps, n_seeds: b.n_seeds, capture_ratio: b.capture_ratio.value, se: b.capture_ratio.se }))}
          />
        ) : null}
      </div>
    </Card>
  );
}
