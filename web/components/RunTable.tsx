"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { RunRow } from "@/lib/api";
import { ARM_LABEL, armColor, useTokens } from "@/lib/chart";
import { num, short, usd, when } from "@/lib/format";
import { Chip, Verdict } from "./Bits";
import ui from "./ui.module.css";

type Key = "created_at" | "policy" | "premium" | "seed" | "n_episodes" | "mean_pnl" | "se_pnl" | "t_stat" | "mean_cost" | "mean_trades" | "status";

const COLUMNS: { key: Key; label: string; num?: boolean }[] = [
  { key: "created_at", label: "Created" },
  { key: "policy", label: "Arm" },
  { key: "premium", label: "Premium (vol pts)", num: true },
  { key: "seed", label: "Seed", num: true },
  { key: "n_episodes", label: "Episodes", num: true },
  { key: "mean_pnl", label: "Mean P&L", num: true },
  { key: "se_pnl", label: "SE", num: true },
  { key: "t_stat", label: "t", num: true },
  { key: "mean_cost", label: "Cost/ep", num: true },
  { key: "mean_trades", label: "Trades/ep", num: true },
  { key: "status", label: "Status" },
];

function cmp(a: unknown, b: unknown): number {
  if (a === b) return 0;
  if (a === null || a === undefined) return 1;
  if (b === null || b === undefined) return -1;
  return typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b));
}

export function RunTable({
  rows,
  compact,
  selected,
  onToggle,
}: {
  rows: RunRow[];
  compact?: boolean;
  /** Keys picked for a compare; a checkbox column appears when onToggle is given. */
  selected?: string[];
  onToggle?: (key: string) => void;
}) {
  const t = useTokens();
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({ key: "created_at", dir: -1 });
  const sorted = useMemo(() => [...rows].sort((a, b) => sort.dir * cmp(a[sort.key], b[sort.key])), [rows, sort]);
  const cols = compact ? COLUMNS.filter((c) => !["se_pnl", "mean_cost", "mean_trades"].includes(c.key)) : COLUMNS;

  const head = (c: (typeof COLUMNS)[number]) => (
    <th key={c.key} className={c.num ? ui.num : undefined} aria-sort={sort.key === c.key ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
      <button type="button" className={ui.sortBtn} onClick={() => setSort((s) => ({ key: c.key, dir: s.key === c.key ? ((-s.dir) as 1 | -1) : -1 }))}>
        {c.label}
        <span aria-hidden="true">{sort.key === c.key ? (sort.dir === 1 ? "▲" : "▼") : ""}</span>
      </button>
    </th>
  );

  return (
    <div className={ui.tableWrap}>
      <table className={ui.table}>
        <thead>
          <tr>
            {onToggle ? <th aria-label="Compare" /> : null}
            <th>Run</th>
            {cols.map(head)}
            {compact ? null : <th>Commit</th>}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const picked = selected?.includes(r.key) ?? false;
            return (
              <tr key={r.key} className={picked ? ui.rowSelected : undefined}>
                {onToggle ? (
                  <td>
                    <input type="checkbox" aria-label={`Compare ${r.run_id}`} checked={picked} onChange={() => onToggle(r.key)} />
                  </td>
                ) : null}
                <td>
                  <Link href={`/runs/?key=${encodeURIComponent(r.key)}`} className="mono" title={r.key}>
                    {short(r.config_hash, 10)}-s{r.seed ?? "?"}
                  </Link>
                  {r.sweep ? <div className={ui.dim} style={{ fontSize: 12 }}>sweep {short(r.sweep, 10)}</div> : null}
                </td>
                {cols.map((c) => {
                  const v = r[c.key];
                  let cell: React.ReactNode;
                  if (c.key === "created_at") cell = <span className={ui.dim}>{when(r.created_at)}</span>;
                  else if (c.key === "policy") cell = <Chip color={armColor(t, r.policy)}>{ARM_LABEL[r.policy] ?? (r.policy || "–")}</Chip>;
                  else if (c.key === "status") cell = <Verdict value={r.status} />;
                  else if (c.key === "mean_pnl" || c.key === "se_pnl" || c.key === "mean_cost") cell = usd(v as number | null);
                  else if (c.key === "t_stat") cell = num(v as number | null, 2);
                  else if (c.key === "premium" || c.key === "mean_trades") cell = num(v as number | null, 1);
                  else cell = num(v as number | null);
                  return (
                    <td key={c.key} className={c.num ? ui.num : undefined}>
                      {cell}
                    </td>
                  );
                })}
                {compact ? null : (
                  <td className="mono" title={r.git_sha}>
                    {r.commit || "–"}
                    {r.git_dirty ? <span className={ui.dim}> +dirty</span> : null}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
