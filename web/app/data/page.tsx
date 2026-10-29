"use client";

import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { Card, PageHead } from "@/components/Bits";
import { ErrorBox, Loading, PanelNotBuilt } from "@/components/States";
import { useGridPlayback, type XMode } from "@/components/views/IvSurface";
import { COMPARE_VIEWS, type CompareView, RealVsSim } from "@/components/views/RealVsSim";
import { SellerDay } from "@/components/views/SellerDay";
import { signedPts } from "@/components/views/VrpMapping";
import { api, type SigmaFrom } from "@/lib/api";
import { useTokens } from "@/lib/chart";
import { useFetch } from "@/lib/useFetch";
import { setQuery, useQuery, useQuerySync, useSectionScroll } from "@/lib/url";
import s from "./_parts/data.module.css";
import { DateBar, snapTo } from "./_parts/DateBar";
import { PremiumView } from "./_parts/Premium";
import { QualityBody } from "./_parts/Quality";
import { SurfaceView } from "./_parts/Surface";
import { Fold } from "@/components/PageBits";
import { InfoTip, Labelled } from "@/components/InfoTip";

/**
 * The Real data page, led by the four questions a reader brings to it. Each opens a
 * focused view that answers in one plain sentence computed from the panel; the data's
 * own checks (coverage, the vendor seam, FRED, the build's version) fold away at the
 * bottom. Every view and knob is in the URL, so a workflow step can open it:
 *
 *   ?q=premium[&show=levels][&by=year]
 *   ?q=surface&date=2022-06-13[&tenor=90][&x=m]
 *   ?q=sim&date=…[&cmp=diff][&premium=2][&sigma=atm]
 *   ?q=seller&date=…
 *   ?quality=1[&vendor=…][&roots=SPXW][&expiry=…]
 *
 * The old `section=` links still land: vrp on the premium, day on the surface,
 * coverage and seam on the data-quality section, opened.
 */

const QUESTIONS = [
  { value: "premium", text: "What premium does the market pay for volatility?", hint: "Today and since 2010, and where it sits on the sweep's axis.", term: "q_premium" },
  { value: "surface", text: "What does the volatility surface look like on a day?", hint: "The 3D surface with playback, its smile and its term structure.", term: "q_surface" },
  { value: "sim", text: "How does the simulator differ from the market?", hint: "Real against simulated on the same day, and where they part most.", term: "q_sim" },
  { value: "seller", text: "What would the seller have opened on that day?", hint: "The iron condor on a real close: strikes, credit, risk and margin.", term: "q_seller" },
] as const;
type Question = (typeof QUESTIONS)[number]["value"];
const Q_VALUES = QUESTIONS.map((x) => x.value);
const SECTION_TO_Q: Record<string, Question | "quality"> = { vrp: "premium", day: "surface", coverage: "quality", seam: "quality" };
const SIGMAS = ["config", "rv21", "atm"] as const;

export default function DataPage() {
  return (
    <Suspense fallback={<Loading what="the panel" />}>
      <Data />
    </Suspense>
  );
}

function Questions({ value, onPick, teaser }: { value: Question | null; onPick: (q: Question | null) => void; teaser: Partial<Record<Question, string>> }) {
  return (
    <div className={`${s.questions} ${value ? s.compact : ""}`} role="group" aria-label="What do you want to find out?">
      {QUESTIONS.map((x) => (
        <button key={x.value} type="button" aria-pressed={value === x.value} className={`${s.question} ${value === x.value ? s.questionOn : ""}`} onClick={() => onPick(value === x.value ? null : x.value)}>
          {teaser[x.value] ? <span className={s.questionTeaser}>{teaser[x.value]}</span> : null}
          <span className={s.questionText}>{x.text}</span>
          <span className={s.questionHint}>{x.hint}</span>
        </button>
      ))}
    </div>
  );
}

function Data() {
  const t = useTokens();
  const q = useQuery();
  const legacy = SECTION_TO_Q[q.str("section")];
  const [question, setQuestion] = useState<Question | null>(() => {
    const raw = q.str("q");
    if ((Q_VALUES as readonly string[]).includes(raw)) return raw as Question;
    return legacy && legacy !== "quality" ? legacy : null;
  });
  // a link or a workflow step that moves to another question on this page (router.push with
  // a new ?q=) reaches the open view: the page stays mounted, so it follows the URL's q
  const urlQ = q.str("q");
  useEffect(() => {
    if ((Q_VALUES as readonly string[]).includes(urlQ) && urlQ !== question) setQuestion(urlQ as Question);
  }, [urlQ]);
  const [qualityOpen, setQualityOpen] = useState(() => q.bool("quality") || legacy === "quality");
  const [rawDate, setRawDate] = useState(() => q.str("date"));
  const [xMode, setXMode] = useState<XMode>(() => q.oneOf("x", ["k", "m"] as const, "k"));
  const [cmp, setCmp] = useState<CompareView>(() => q.oneOf("cmp", COMPARE_VIEWS, "side"));
  const [premium, setPremium] = useState(() => q.num("premium", 1, 0, 6));
  const [sigma, setSigma] = useState<SigmaFrom>(() => q.oneOf("sigma", SIGMAS, "config"));

  const status = useFetch((sig) => api.dataStatus(sig), "status");
  const missing = status.data?.missing ?? [];
  const usable = Boolean(status.data) && !missing.some((m) => m.endsWith("spx_daily.parquet") || m.endsWith("coverage.parquet"));
  const gridDates = useFetch((sig) => api.gridDates(sig), "grid-dates", usable);
  const dates = gridDates.data ?? [];
  const date = dates.length ? snapTo(dates, rawDate) : rawDate;
  const latest = dates[dates.length - 1];
  const vrp = useFetch((sig) => api.vrp(sig), "vrp");

  const needsGrid = question === "surface" || question === "sim";
  const gridded = dates.includes(date);
  const grid = useFetch((sig) => api.grid(date, sig), `grid|${date}`, needsGrid && gridded);
  const setDate = useCallback((d: string) => setRawDate(d), []);
  const play = useGridPlayback(dates, date, setDate);

  useQuerySync(
    { q: question, date: date && date !== latest ? date : null, x: xMode, cmp, premium, sigma, quality: qualityOpen, section: null },
    { x: "k", cmp: "side", premium: 1, sigma: "config" },
  );
  useEffect(() => {
    if (q.params.get("section")) setQuery({ section: null, q: question, quality: qualityOpen });
  }, []);
  useSectionScroll(legacy === "quality" ? "quality" : "", Boolean(status.data));

  const h = vrp.data?.mapping?.headline;
  const teaser = useMemo(() => ({ premium: h ? `${signedPts(h.mean)} vol pts` : undefined, surface: dates.length ? `${dates.length.toLocaleString("en-US")} days` : undefined }), [h, dates.length]);
  const pick = (v: Question | null) => {
    if (play.playing) play.toggle();
    setQuestion(v);
  };
  const dateBar = question === "sim" || question === "seller" ? <DateBar dates={dates} date={date} onChange={setDate} /> : null;
  const title = QUESTIONS.find((x) => x.value === question);

  return (
    <>
      <PageHead eyebrow="Real data" title="What does the real SPX options market say?" lede="Pick a question; each view answers it in one sentence from the panel, with the charts to check it." />
      {status.error ? <ErrorBox what="the panel status" error={status.error} onRetry={status.reload} /> : null}
      {status.loading ? <Loading what="the panel status" /> : null}
      {status.data && !usable ? <PanelNotBuilt missing={missing} root={status.data.root} /> : null}
      <Questions value={question} onPick={pick} teaser={teaser} />
      {usable && question && title ? (
        <section className={s.view} aria-label={title.text}>
          <div className={s.viewHead}>
            <h2 className={s.viewTitle}>
              <Labelled term={`data.${title.term}`}>{title.text}</Labelled>
            </h2>
          </div>
          {gridDates.error && needsGrid ? <ErrorBox what="the gridded surface's dates" error={gridDates.error} onRetry={gridDates.reload} /> : null}
          {question === "premium" ? <PremiumView /> : null}
          {question === "surface" && dates.length ? (
            <SurfaceView
              dates={dates}
              date={date}
              setDate={setDate}
              grid={grid.data}
              gridError={grid.error ? <ErrorBox what="the surface" error={grid.error} onRetry={grid.reload} /> : null}
              play={play}
              xMode={xMode}
              setXMode={setXMode}
            />
          ) : null}
          {question === "sim" && dates.length ? (
            <>
              {dateBar}
              <RealVsSim t={t} date={date} grid={grid.data} gridded={gridded} mode={xMode} setMode={setXMode} view={cmp} setView={setCmp} premium={premium} setPremium={setPremium} sigmaFrom={sigma} setSigmaFrom={setSigma} />
            </>
          ) : null}
          {question === "seller" && dates.length ? (
            <>
              {dateBar}
              <SellerDay date={date} gridded={gridded} />
            </>
          ) : null}
          {needsGrid || question === "seller" ? !dates.length && gridDates.loading ? <Loading what="the trading days" /> : null : null}
        </section>
      ) : null}
      {usable && status.data ? (
        <Fold id="quality" title="Data quality" term="data.coverage" sub="coverage, the vendor seam, FRED and the build's version" open={qualityOpen} onToggle={setQualityOpen}>
          <QualityBody date={date || latest || ""} status={status.data} />
        </Fold>
      ) : null}
      {status.data && !usable && vrp.data?.mapping ? (
        <Card title="The real premium" info="data.premium_mean">
          <p>
            {signedPts(vrp.data.mapping.headline.mean)} vol pts, 95% [{signedPts(vrp.data.mapping.headline.mean_low)}, {signedPts(vrp.data.mapping.headline.mean_high)}] <InfoTip term="data.premium_interval" />
          </p>
        </Card>
      ) : null}
    </>
  );
}
