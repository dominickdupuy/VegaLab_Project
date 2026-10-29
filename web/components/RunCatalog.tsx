"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { Card, Chip } from "@/components/Bits";
import { InfoTip } from "@/components/InfoTip";
import ui from "@/components/ui.module.css";
import lab from "@/components/lab.module.css";
import type { Catalog, CatalogGroup, CatalogPair } from "@/lib/api_runs";
import { ARM_LABEL, armColor, useTokens } from "@/lib/chart";
import { groupName, pairGuide } from "@/lib/catalog";
import { usd } from "@/lib/format";
import { guide } from "@/lib/guide";
import { setQuery, useQuery } from "@/lib/url";

function Forest({ lo, hi, mean, scale }: { lo: number | null; hi: number | null; mean: number | null; scale: number }) {
  const x = (v: number) => 50 + (v / scale) * 48;
  if (mean === null) return null;
  const l = x(lo ?? mean);
  const h = x(hi ?? mean);
  const clear = lo !== null && hi !== null && (lo > 0 || hi < 0);
  return (
    <div style={{ position: "relative", height: 14 }} aria-hidden="true">
      <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "var(--line-strong)" }} />
      <div style={{ position: "absolute", top: 6, height: 2, left: `${Math.min(l, h)}%`, width: `${Math.max(0.5, Math.abs(h - l))}%`, background: clear ? "var(--accent)" : "var(--muted)" }} />
      <div style={{ position: "absolute", top: 3, width: 8, height: 8, marginLeft: -4, borderRadius: "50%", left: `${x(mean)}%`, background: clear ? "var(--accent)" : "var(--surface-solid)", border: `2px solid ${clear ? "var(--accent)" : "var(--muted)"}` }} />
    </div>
  );
}

/** Each setting against the group's baseline on the same paths: the paired difference first. */
function GroupCard({ g }: { g: CatalogGroup }) {
  const t = useTokens();
  const router = useRouter();
  const label = new Map(g.runs.map((r) => [r.key, r.label]));
  const scale = Math.max(1, ...g.pairs.flatMap((p) => [Math.abs(p.diff_ci_low ?? 0), Math.abs(p.diff_ci_high ?? 0), Math.abs(p.diff_mean ?? 0)]));
  const nClear = g.pairs.filter((p) => p.clear).length;
  const open = (p: CatalogPair) => router.push(`/runs/?a=${encodeURIComponent(p.a)}&b=${encodeURIComponent(p.b)}&group=${encodeURIComponent(g.group)}`);
  return (
    <Card
      title={g.question || groupName(g.group)}
      sub={
        g.pairs.length
          ? `${nClear} of ${g.pairs.length} settings differ clearly from the baseline on the same paths; the rest are within noise at this sample size.`
          : `${g.runs.length} runs; not enough on disk yet to pair.`
      }
      info={"Every run in a group starts on the same seed, so two settings see the same markets. Their per-episode difference cancels the market's luck, which makes it far more precise than either run's own mean."}
    >
      <div className={ui.stack}>
        {g.pairs.length ? (
          <div className={ui.tableWrap}>
            <table className={ui.table} style={{ minWidth: 640 }}>
              <thead>
                <tr>
                  <th>Setting</th>
                  <th>Against</th>
                  <th className={ui.num}>
                    Paired difference <InfoTip term="paired_contrast" />
                  </th>
                  <th style={{ width: "26%" }}>
                    95% interval, 0 in the middle <InfoTip term="interval" />
                  </th>
                  <th aria-label="Open the comparison" />
                </tr>
              </thead>
              <tbody>
                {g.pairs.map((p) => (
                  <tr key={`${p.a}|${p.b}`}>
                    <td>{label.get(p.b) ?? p.b}</td>
                    <td className={ui.dim}>{label.get(p.a) ?? p.a}</td>
                    <td className={ui.num} style={{ whiteSpace: "nowrap" }}>
                      {p.diff_mean !== null ? (p.diff_mean >= 0 ? "+" : "") : ""}
                      {usd(p.diff_mean)}
                      <div className={ui.dim} style={{ fontSize: 12 }}>
                        {p.diff_ci_low !== null && p.diff_ci_high !== null ? `± ${usd((p.diff_ci_high - p.diff_ci_low) / 2)}` : ""} · {p.clear ? "clear" : "noise"}
                      </div>
                    </td>
                    <td>
                      <Forest lo={p.diff_ci_low} hi={p.diff_ci_high} mean={p.diff_mean} scale={scale} />
                    </td>
                    <td>
                      <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={() => open(p)} {...guide(pairGuide(p.a, p.b))}>
                        Compare →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        <details className={lab.subFold}>
          <summary>
            Each run on its own ({g.runs.length}): a single run&apos;s mean is noisy at this sample size
          </summary>
          <div className={ui.tableWrap} style={{ marginTop: 8 }}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Arm</th>
                  <th className={ui.num}>Mean P&L ± se</th>
                </tr>
              </thead>
              <tbody>
                {g.runs.map((r) => (
                  <tr key={r.key}>
                    <td>{r.present ? <Link href={`/runs/?key=${encodeURIComponent(r.key)}`}>{r.label}</Link> : <span className={ui.dim}>{r.label} (not on disk yet)</span>}</td>
                    <td>
                      <Chip color={armColor(t, r.policy)}>{ARM_LABEL[r.policy] ?? r.policy}</Chip>
                    </td>
                    <td className={ui.num}>
                      {usd(r.mean_pnl)}
                      {r.se_pnl !== null ? ` ± ${usd(r.se_pnl)}` : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      </div>
    </Card>
  );
}

/** A runs root's designed matrix: pick a question, read its runs, compare within it. */
export function RunCatalog({ catalog }: { catalog: Catalog }) {
  // the question open is in the URL (?question=costs), so a link or a workflow step opens it
  const q = useQuery();
  const pick = q.str("question");
  const setPick = (group: string) => setQuery({ question: group === catalog.groups[0]?.group ? null : group });
  const g = catalog.groups.find((x) => x.group === pick) ?? catalog.groups[0];
  if (!g) return null;
  return (
    <section className={ui.stack} aria-label="Questions this runs root answers">
      <div className={ui.spread}>
        <p className={ui.groupTitle} style={{ margin: 0 }}>
          Questions these runs answer <InfoTip text={`Runs grouped by the question each answers, from the runs root's catalog.json${catalog.seed !== null ? `; every run starts on seed ${catalog.seed.toLocaleString("en-US")}, so any two pair` : ""}.`} />
        </p>
      </div>
      <div className={ui.row} role="tablist" aria-label="Questions">
        {catalog.groups.map((x) => (
          <button
            key={x.group}
            type="button"
            role="tab"
            aria-selected={x.group === g.group}
            className={`${ui.btn} ${ui.btnSmall} ${x.group === g.group ? ui.btnPrimary : ""}`}
            title={x.question}
            onClick={() => setPick(x.group)}
            {...guide(`runs-question-${x.group}`)}
          >
            {groupName(x.group)}
          </button>
        ))}
      </div>
      <GroupCard g={g} />
    </section>
  );
}
