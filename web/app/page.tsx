"use client";

import Link from "next/link";
import { Card, Chip, PageHead, Tile } from "@/components/Bits";
import { RunTable } from "@/components/RunTable";
import { Empty, ErrorBox, Loading, Pending } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { SOURCE_LABEL, sourceColor, useTokens } from "@/lib/chart";
import { num } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

export default function OverviewPage() {
  const t = useTokens();
  const ov = useFetch((s) => api.overview(s), "overview");

  if (ov.error) return <ErrorBox what="the overview" error={ov.error} onRetry={ov.reload} />;
  const d = ov.data;

  return (
    <>
      <PageHead eyebrow="Vega Lab" title="optspread platform" />

      {ov.loading && !d ? <Loading what="the overview" /> : null}

      {d ? (
        <>
          <div className={ui.grid4}>
            <Tile accent label="Runs indexed" value={num(d.n_runs)} note={`${num(d.n_ok)} OK · ${d.sweeps} sweep${d.sweeps === 1 ? "" : "s"}`} />
            <Tile
              accent
              label="SPX panel"
              value={d.data.built ? "Built" : "Not built"}
              note={d.data.built ? `${d.data.years.length} years of chains` : `${d.data.missing.length} file(s) missing`}
            />
            <Tile label="Sources" value={num(d.coverage.length)} note={d.coverage.map((s) => SOURCE_LABEL[s.source] ?? s.source).join(" · ") || "none yet"} />
            <Tile label="Overlap days" value={d.overlap ? num(d.overlap[2]) : "–"} note={d.overlap ? `${d.overlap[0]} → ${d.overlap[1]}` : "no overlap"} />
          </div>

          <Card title="Coverage" source={<code>{d.data.root}/panel/coverage.parquet</code>} right={<Link href="/data/">Real data →</Link>}>
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
              </div>
            ) : (
              <Pending title="Panel not built">
                <p>{d.data.missing.length} file(s) missing.</p>
              </Pending>
            )}
          </Card>

          <Card title="Latest runs" right={<Link href="/runs/">All runs →</Link>}>
            {d.latest.length ? (
              <RunTable rows={d.latest} compact />
            ) : (
              <Empty title="No runs yet">
                <p>
                  <Link href="/launch/">Launch a run</Link>
                </p>
              </Empty>
            )}
          </Card>
        </>
      ) : null}
    </>
  );
}
