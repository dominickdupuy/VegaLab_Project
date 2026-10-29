"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { TrustBand } from "@/components/home/TrustBand";
import { WorkflowCard } from "@/components/home/WorkflowCard";
import { InfoTip } from "@/components/InfoTip";
import { ErrorBox, Loading } from "@/components/States";
import { Checks, StudyDesign } from "@/components/Study";
import ui from "@/components/ui.module.css";
import { api, type M4, type Overview, type Study, type SweepList, type Vrp } from "@/lib/api";
import { SOURCE_LABEL } from "@/lib/chart";
import { num, signed } from "@/lib/format";
import { WORKFLOWS } from "@/lib/tours";
import { useFetch } from "@/lib/useFetch";

// the 3D surface is the heaviest chunk on the site: it loads after the text, on idle
const HomeSurface = dynamic(() => import("@/components/home/HomeSurface"), { ssr: false, loading: () => <div className={ui.heroStage} aria-hidden="true" /> });

function useIdle(): boolean {
  const [idle, setIdle] = useState(false);
  useEffect(() => {
    const w = window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number; cancelIdleCallback?: (id: number) => void };
    if (w.requestIdleCallback) {
      const id = w.requestIdleCallback(() => setIdle(true), { timeout: 1200 });
      return () => w.cancelIdleCallback?.(id);
    }
    const id = window.setTimeout(() => setIdle(true), 300);
    return () => window.clearTimeout(id);
  }, []);
  return idle;
}

/** The slim strip under the questions: what the site holds now. An item with nothing to say is left out.
 * The checks and the study design are in the trust band below it. */
function StatusStrip({ ov, vrp, study, sweeps, m4 }: { ov?: Overview; vrp?: Vrp; study?: Study; sweeps?: SweepList; m4?: M4 }) {
  const spans = ov?.coverage ?? [];
  const first = spans.map((s) => s.first).sort()[0];
  const last = spans.map((s) => s.last).sort().slice(-1)[0];
  const h = vrp?.mapping?.headline;
  const grid = sweeps?.sweeps.find((s) => s.kind === "grid" && s.sweep_hash === study?.grid_hash) ?? sweeps?.sweeps.find((s) => s.kind === "grid");
  return (
    <div className={ui.status} aria-label="Status">
      {ov ? (
        <Link href="/data/" className={ui.statusItem}>
          <span className={ui.statusLabel}>Data</span>
          <span className={ui.statusValue}>{first && last ? `SPX ${first.slice(0, 4)}–${last.slice(0, 4)}` : "Synthetic world"}</span>
          <span className={ui.statusNote}>{spans.length ? spans.map((s) => SOURCE_LABEL[s.source] ?? s.source).join(" + ") : "the real panel is not built on this server"}</span>
        </Link>
      ) : null}
      {h ? (
        <Link href="/data/?section=vrp" className={ui.statusItem}>
          <span className={ui.statusLabel}>
            Real premium <InfoTip term="vrp" />
          </span>
          <span className={ui.statusValue}>{signed(h.mean)} vol pts</span>
          <span className={ui.statusNote}>
            95% {signed(h.mean_low)} to {signed(h.mean_high)} · {h.first.slice(0, 4)}–{h.last.slice(0, 4)}
          </span>
        </Link>
      ) : null}
      {grid ? (
        <Link href="/sweep/?view=power" className={ui.statusItem}>
          <span className={ui.statusLabel}>
            Sweep <InfoTip term="sweep" />
          </span>
          <span className={ui.statusValue}>{m4?.available ? "Analysed" : `${num(grid.n_present)} of ${num(grid.n_cells)} cells`}</span>
          <span className={ui.statusNote}>
            {grid.arms.length} arms × {grid.premiums.length} premiums × {grid.n_seeds} seeds{m4?.analysis ? ` · ${m4.analysis.confirmatory ? "confirmatory" : "exploratory"}` : ""}
          </span>
        </Link>
      ) : null}
      {ov && ov.n_runs ? (
        <Link href="/runs/" className={ui.statusItem}>
          <span className={ui.statusLabel}>Runs</span>
          <span className={ui.statusValue}>{num(ov.n_runs)} indexed</span>
          <span className={ui.statusNote}>{num(ov.n_ok)} read cleanly · every one listed</span>
        </Link>
      ) : null}
    </div>
  );
}

function Home() {
  const q = useSearchParams();
  const idle = useIdle();
  const ov = useFetch((s) => api.overview(s), "overview");
  const study = useFetch((s) => api.study(s), "study");
  const checks = useFetch((s) => api.checks(s), "checks");
  const vrp = useFetch((s) => api.vrp(s), "vrp");
  const sweeps = useFetch((s) => api.sweeps(s), "sweeps");
  const m4 = useFetch((s) => api.m4(s), "m4");
  // the questions only: the reviewer's workflow starts from the trust band below them
  const featured = WORKFLOWS.filter((w) => w.featured && !w.reviewer);
  const more = WORKFLOWS.filter((w) => !w.featured && !w.reviewer);

  return (
    <>
      <section className={ui.homeHero}>
        <div className={ui.homeLead}>
          <div className={ui.homeHead}>
            <p className={ui.eyebrow}>optspread platform · Vega Lab</p>
            <h1>What do you want to find out?</h1>
            <p>Pick a question. Each one walks you through the platform: you run, compare and read the result yourself, one step at a time.</p>
          </div>
          <div className={ui.wfGrid2}>
            {featured.map((w) => (
              <WorkflowCard key={w.id} w={w} big />
            ))}
          </div>
        </div>
        {idle ? <HomeSurface m4={m4.data} vrp={vrp.data} built={ov.data ? ov.data.data.built : undefined} /> : <div className={ui.heroStage} aria-hidden="true" />}
      </section>

      {ov.error ? <ErrorBox what="the overview" error={ov.error} onRetry={ov.reload} /> : null}
      <StatusStrip ov={ov.data} vrp={vrp.data} study={study.data} sweeps={sweeps.data} m4={m4.data} />

      <section className={ui.stack}>
        <h2 className={ui.sectionTitle}>More questions</h2>
        <div className={ui.wfGrid}>
          {more.map((w) => (
            <WorkflowCard key={w.id} w={w} />
          ))}
        </div>
      </section>

      <TrustBand checks={checks.data} study={study.data} />
      {checks.error ? <ErrorBox what="the checks" error={checks.error} onRetry={checks.reload} /> : null}
      {q.get("checks") === "1" ? <div id="checks">{checks.data ? <Checks rows={checks.data} /> : checks.error ? null : <Loading what="the checks" />}</div> : null}
      {q.get("study") === "1" ? <div id="study">{study.data ? <StudyDesign s={study.data} /> : study.error ? <ErrorBox what="the study design" error={study.error} /> : <Loading what="the study design" />}</div> : null}
    </>
  );
}

export default function HomePage() {
  return (
    <Suspense fallback={<Loading what="the platform" />}>
      <Home />
    </Suspense>
  );
}
