"use client";

import Link from "next/link";
import { useState } from "react";
import { Card, PageHead, Segmented, Tile, Verdict } from "@/components/Bits";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type GateBoard } from "@/lib/api";
import { useFetch } from "@/lib/useFetch";

const STATUS = [
  { value: "all", label: "All" },
  { value: "PASS", label: "Pass" },
  { value: "open", label: "Not passing" },
] as const;

function Board({ board, status }: { board: GateBoard; status: (typeof STATUS)[number]["value"] }) {
  const items = board.items.filter((i) => status === "all" || (status === "PASS" ? i.verdict === "PASS" : i.verdict !== "PASS"));
  const open = board.items.filter((i) => i.verdict !== "PASS");
  return (
    <>
      <div className={ui.grid5}>
        <Tile accent label={`${board.milestone} verdict`} value={<Verdict value={board.verdict} />} note={`report date ${board.report_date || "–"}`} />
        <Tile label="Pass" value={board.counts.PASS ?? 0} note={`of ${board.items.length} items`} />
        <Tile label="Fail" value={board.counts.FAIL ?? 0} note="criterion stated and missed" />
        <Tile label="No criterion" value={board.counts["NO CRITERION"] ?? 0} note="evidence without a rule: not a pass" />
        <Tile label="Not cleared" value={board.counts["NOT CLEARED"] ?? 0} note="no evidence yet" />
      </div>

      {open.length ? (
        <Card title="What failed first" sub="Each item that does not pass, and the first thing that stopped it.">
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {open.map((i) => (
              <li key={i.name} style={{ marginBottom: 4 }}>
                <b>{i.name}</b> <Verdict value={i.verdict} /> <span className={ui.dim}>({i.owner})</span>: {i.failed_first || "no first failure recorded"}
              </li>
            ))}
          </ul>
        </Card>
      ) : null}

      <Card
        title={`${board.milestone} gate board`}
        sub="A milestone closes on this report, not on code that runs. A check whose criterion lives in its author's head reads NO CRITERION."
        source={
          <>
            <code>{board.source}</code> {board.from_json ? "(JSON sidecar)" : "(markdown table; the JSON sidecar is not written yet)"}
          </>
        }
      >
        {items.length ? (
          <div className={ui.tableWrap}>
            <table className={ui.table}>
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Owner</th>
                  <th>Verdict</th>
                  <th>Criterion</th>
                  <th>Evidence</th>
                  <th>What failed first</th>
                </tr>
              </thead>
              <tbody>
                {items.map((i) => (
                  <tr key={i.name}>
                    <td style={{ fontWeight: 700, minWidth: 160 }}>{i.name}</td>
                    <td className={ui.dim} style={{ whiteSpace: "nowrap" }}>
                      {i.owner}
                    </td>
                    <td>
                      <Verdict value={i.verdict} />
                    </td>
                    <td className={ui.wrapText}>{i.criterion || <span className={ui.dim}>none stated</span>}</td>
                    <td className={ui.wrapText}>
                      {i.evidence || <span className={ui.dim}>none</span>}
                      {i.cited_runs.length ? (
                        <div style={{ marginTop: 4 }}>
                          {i.cited_runs.map((r) => (
                            <Link key={r} href={`/runs/?q=${r}`} className="mono" style={{ marginRight: 8 }}>
                              {r} →
                            </Link>
                          ))}
                        </div>
                      ) : null}
                    </td>
                    <td className={ui.wrapText}>{i.failed_first || <span className={ui.dim}>–</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty title="No items match this filter" />
        )}
      </Card>
    </>
  );
}

export default function GatesPage() {
  const gates = useFetch((s) => api.gates(s), "gates");
  const [milestone, setMilestone] = useState<string>("");
  const [status, setStatus] = useState<(typeof STATUS)[number]["value"]>("all");
  const boards = gates.data?.boards ?? [];
  const chosen = boards.find((b) => b.milestone === milestone) ?? boards[boards.length - 1];

  return (
    <>
      <PageHead
        eyebrow="Gate board"
        title="Gates"
        lede="Every gate item per milestone: the criterion it had to meet, the evidence, the verdict, and on a failure what failed first. Four verdicts: PASS, FAIL, NO CRITERION and NOT CLEARED."
        right={
          boards.length ? (
            <>
              <Segmented label="Milestone" value={chosen?.milestone ?? ""} options={boards.map((b) => ({ value: b.milestone, label: b.milestone }))} onChange={setMilestone} />
              <Segmented label="Status" value={status} options={STATUS} onChange={setStatus} />
            </>
          ) : null
        }
      />
      {gates.error ? <ErrorBox what="the gate reports" error={gates.error} onRetry={gates.reload} /> : null}
      {gates.loading ? <Loading what="the gate reports" /> : null}
      {gates.data && !boards.length ? (
        <Empty title="No gate reports yet">
          <p>
            <code>make m1</code> writes <code>docs/M1_GATE_REPORT.md</code>; <code>make m2</code> writes the M2 report. Reports under <code>{gates.data.docs}</code>{" "}
            appear here.
          </p>
        </Empty>
      ) : null}
      {chosen ? <Board board={chosen} status={status} /> : null}
    </>
  );
}
