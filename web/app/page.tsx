"use client";

import Link from "next/link";
import { Card, Chip, Tile, Verdict } from "@/components/Bits";
import { RunTable } from "@/components/RunTable";
import { Empty, ErrorBox, Loading, Pending } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { ARM_LABEL, SOURCE_LABEL, sourceColor, useTokens } from "@/lib/chart";
import { num } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

const VIEWS: { href: string; title: string; text: string }[] = [
  { href: "/runs/", title: "Runs", text: "Every run directory, its manifest and digests, its P&L distribution, and a paired compare." },
  { href: "/launch/", title: "Launch", text: "Set the knobs and start a run. It lands as an ordinary hashed run directory." },
  { href: "/gates/", title: "Gates", text: "Each milestone item: criterion, evidence, verdict, and what failed first." },
  { href: "/power/", title: "Power lab", text: "Power against effect size, the seed-count floor, and what a design costs." },
  { href: "/data/", title: "Real data", text: "The stitched SPX panel: coverage, smiles, the surface, and IV30 against realised vol." },
  { href: "/leaks/", title: "Leak demo", text: "Run the detector live on the clean world and on its poisoned twin." },
  { href: "/sweep/", title: "Sweep", text: "The oracle curve from sweep cells; the power curve and attribution when the sweep lands." },
];

export default function OverviewPage() {
  const t = useTokens();
  const ov = useFetch((s) => api.overview(s), "overview");

  if (ov.error) return <ErrorBox what="the overview" error={ov.error} onRetry={ov.reload} />;
  const d = ov.data;
  const lead = d?.summary[0] ?? "A leak-tested SPX options simulator used to measure how small a trading effect our research setup can actually detect.";
  const team = d?.summary[1];
  const rest = d?.summary.slice(2) ?? [];

  return (
    <>
      <section className={ui.hero}>
        <div>
          <p className={ui.eyebrow} style={{ color: "var(--hero-muted)" }}>
            Vega Lab · optspread
          </p>
          <h1>A calibrated instrument, and a stated detection floor</h1>
          <p className={ui.heroLead}>{lead}</p>
          {rest.map((p) => (
            <p key={p} style={{ marginBottom: 8 }}>
              {p}
            </p>
          ))}
          {team ? <p style={{ fontSize: 13, marginTop: 10 }}>{team}</p> : null}
        </div>
        <p className={ui.heroQuote}>
          The dashboard only shows what the harness recorded, so every number traces back to a run.
          <span>Read-only over manifests, gate reports and the SPX panel. A launched run is written by the same harness as any other.</span>
        </p>
      </section>

      {ov.loading && !d ? <Loading what="the overview" /> : null}

      {d ? (
        <>
          <div className={ui.grid4}>
            {d.gates.map((g) => (
              <Tile
                key={g.milestone}
                accent
                label={`${g.milestone} gate`}
                value={<Verdict value={g.verdict} />}
                note={
                  <>
                    {g.n_pass} of {g.n_items} items pass · {g.report_date}
                  </>
                }
              />
            ))}
            {d.gates.length === 0 ? <Tile label="Gates" value="–" note="No gate report under docs/ yet." /> : null}
            <Tile
              accent
              label="Runs indexed"
              value={num(d.n_runs)}
              note={
                <>
                  {num(d.n_ok)} read cleanly · {d.sweeps} sweep{d.sweeps === 1 ? "" : "s"}
                </>
              }
            />
            <Tile
              accent
              label="SPX panel"
              value={d.data.built ? "Built" : "Not built yet"}
              note={d.data.built ? `${d.data.years.length} years of chains` : `${d.data.missing.length} file(s) missing`}
            />
          </div>

          <div className={ui.grid2}>
            <Card
              title="Data coverage"
              sub="Trading days with quotes, per source. The two overlap where the seam is checked."
              source={<code>{d.data.root}/panel/coverage.parquet</code>}
            >
              {d.coverage.length ? (
                <div className={ui.stack}>
                  {d.coverage.map((s) => (
                    <div key={s.source} className={ui.spread}>
                      <Chip color={sourceColor(t, s.source)}>{SOURCE_LABEL[s.source] ?? s.source}</Chip>
                      <span>
                        {s.first} → {s.last}
                      </span>
                      <span className={ui.dim}>
                        {num(s.n_days)} days · {s.n_missing} missing
                      </span>
                    </div>
                  ))}
                  {d.overlap ? (
                    <p className={ui.callout}>
                      Overlap {d.overlap[0]} → {d.overlap[1]}: {d.overlap[2]} days quoted by both sources.
                    </p>
                  ) : null}
                  <Link href="/data/">Open the real-data views →</Link>
                </div>
              ) : (
                <Pending title="Panel not built yet">
                  <p>The W6-3 build has not written the panel to the shared folder; the real-data views say what is missing.</p>
                </Pending>
              )}
            </Card>

            <Card title="Views" sub="Each is built when the data it reads first exists (spec 10.10).">
              <div className={ui.stack}>
                {VIEWS.map((v) => (
                  <div key={v.href} className={ui.spread}>
                    <Link href={v.href} style={{ fontWeight: 700, minWidth: 90 }}>
                      {v.title}
                    </Link>
                    <span className={ui.dim} style={{ flex: 1, fontSize: 13.5 }}>
                      {v.text}
                    </span>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          <Card
            title="Latest runs"
            sub={
              <>
                Newest first. PPO arm {d.ppo_available ? "available" : "not installed in this checkout"}; arms shown:{" "}
                {Object.values(ARM_LABEL).join(", ")}.
              </>
            }
            right={<Link href="/runs/">All runs →</Link>}
          >
            {d.latest.length ? (
              <RunTable rows={d.latest} compact />
            ) : (
              <Empty title="No runs yet">
                <p>
                  Start one from <Link href="/launch/">Launch</Link>, or run <code>python scripts/oracle_sweep.py</code>; run directories under
                  the runs root appear here.
                </p>
              </Empty>
            )}
          </Card>
        </>
      ) : null}
    </>
  );
}
