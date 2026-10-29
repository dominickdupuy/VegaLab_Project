"use client";

import { Card, Verdict } from "@/components/Bits";
import { Empty } from "@/components/States";
import ui from "@/components/ui.module.css";
import { type Check, docUrl, type Study } from "@/lib/api";
import { ARM_LABEL } from "@/lib/chart";
import { hours, num, pct, prem, usd } from "@/lib/format";

/** The confirmatory design as the pre-registration fixes it: the rule, the size, the floor. */
export function StudyDesign({ s }: { s: Study }) {
  return (
    <Card
      title="Study design"
      right={
        <a href={docUrl(s.doc)} target="_blank" rel="noreferrer">
          {s.doc}
        </a>
      }
      source={s.source}
    >
      <dl className={ui.kvTight}>
        <dt>Test</dt>
        <dd>{s.test}</dd>
        <dt>α</dt>
        <dd>{s.alpha}, one-sided</dd>
        <dt>Event</dt>
        <dd>{s.event}</dd>
        <dt>Design</dt>
        <dd>
          S = {s.n_seeds} seed blocks × N = {num(s.n_episodes)} episodes · premiums {s.premiums.join(", ")} vol pts
        </dd>
        <dt>Budget</dt>
        <dd>
          {num(s.train_steps)} PPO steps per seed · {hours(s.core_hours)} core time at {num(s.train_rate)} / {num(s.eval_rate)} steps/s
        </dd>
        {s.measured ? (
          <>
            <dt>Power, measured</dt>
            <dd>
              {pct(s.measured.power, 1)} at {prem(s.measured.premium)} vol pt · {ARM_LABEL[s.measured.arm] ?? s.measured.arm}&apos;s variances from the sweep
            </dd>
            <dt>Floor, measured</dt>
            <dd>
              {num(s.measured.floor_vol_pts, 2)} vol pts ({usd(s.measured.floor)}) at {pct(s.target_power)} power
            </dd>
          </>
        ) : null}
        <dt>{s.measured ? "Planned (A7)" : "Floor, planned (A7)"}</dt>
        <dd>
          {s.measured ? `power ${pct(s.power, 1)} · floor ${num(s.floor_vol_pts, 2)} vol pts (${usd(s.floor)})` : `${num(s.floor_vol_pts, 2)} vol pts (${usd(s.floor)}) at ${pct(s.target_power)} power`} · the pilot&apos;s variances
        </dd>
        {s.measured ? null : (
          <>
            <dt>At 1 vol pt, planned</dt>
            <dd>
              power {pct(s.power, 1)} · δ = {usd(s.delta_per_vol_pt)} per episode
            </dd>
          </>
        )}
        {s.measured ? (
          <>
            <dt>Effect</dt>
            <dd>δ = {usd(s.delta_per_vol_pt)} per episode at 1 vol pt</dd>
          </>
        ) : null}
      </dl>
    </Card>
  );
}

/** Every check, one line each: verdict, what is checked, the evidence (or what failed first), its report. */
export function Checks({ rows }: { rows: Check[] }) {
  const open = rows.filter((r) => r.verdict !== "PASS").length;
  return (
    <Card title="Checks" sub={rows.length ? `${rows.length - open} of ${rows.length} pass · evidence beside each; what failed first, where one fails` : undefined} source="the check reports in docs/, verdicts derived from each item's criterion and evidence">
      {rows.length ? (
        <ul className={ui.lineList}>
          {rows.map((r) => {
            const note = r.verdict === "PASS" ? r.evidence : r.failed_first ? `failed first: ${r.failed_first}` : r.evidence;
            return (
            <li key={`${r.report}-${r.name}`}>
              <Verdict value={r.verdict} />
              <span className={ui.lineName} title={r.label !== r.name ? `reported as "${r.name}"` : undefined}>
                {r.label}
              </span>
              <span className={ui.lineNote} title={`${note}${r.criterion ? `\n\ncriterion: ${r.criterion}` : ""}`}>
                {note}
              </span>
              <span className={ui.lineEnd}>
                <a href={docUrl(r.report)} target="_blank" rel="noreferrer" title={r.report}>
                  report
                </a>
              </span>
            </li>
            );
          })}
        </ul>
      ) : (
        <Empty title="No check reports under docs/" />
      )}
    </Card>
  );
}
