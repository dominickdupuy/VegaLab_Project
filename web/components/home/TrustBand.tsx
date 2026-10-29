"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import type { ReactNode } from "react";
import { InfoTip } from "@/components/InfoTip";
import ui from "@/components/ui.module.css";
import type { Check, Study } from "@/lib/api";
import { num } from "@/lib/format";
import { guide } from "@/lib/guide";
import { stepHref, WORKFLOWS } from "@/lib/tours";
import { setQuery } from "@/lib/url";

const REPRODUCE = ["git clone https://github.com/dominickdupuy/VegaLab_Project.git", "cd VegaLab_Project", "python scripts/reproduce.py"];
const PARTNER_SECTION = "https://github.com/dominickdupuy/VegaLab_Project#reproduce-our-results-partner-teams";

/** A check found by the words its name carries; undefined when the reports have none. */
function find(checks: Check[] | undefined, pattern: RegExp): Check | undefined {
  return checks?.find((c) => pattern.test(c.name) || pattern.test(c.label));
}

function Mark({ ok }: { ok: boolean | undefined }) {
  if (ok === undefined) return <span className={ui.trustMark} aria-label="not reported">–</span>;
  return (
    <span className={`${ui.trustMark} ${ok ? ui.trustPass : ui.trustFail}`} aria-label={ok ? "passes" : "fails"}>
      {ok ? "✓" : "✗"}
    </span>
  );
}

/** One line of evidence: a mark, what is claimed, the live figure, and its evidence on hover. */
function Line({ ok, name, claim, tip, toggle }: { ok: boolean | undefined; name: string; claim: ReactNode; tip: string; toggle?: { key: string; open: boolean; hook: string } }) {
  const label = toggle ? (
    <button type="button" className={ui.trustToggle} aria-expanded={toggle.open} aria-controls={toggle.key} onClick={() => setQuery({ [toggle.key]: toggle.open ? null : 1 })} {...guide(toggle.hook)}>
      {name}
    </button>
  ) : (
    <span className={ui.trustName}>{name}</span>
  );
  return (
    <li className={ui.trustLine}>
      <Mark ok={ok} />
      {label}
      <span className={ui.trustClaim}>{claim}</span>
      <InfoTip text={tip} label={`About ${name}`} />
    </li>
  );
}

function Shield() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}

/**
 * The evidence behind the numbers, at a glance: the checks, replay, the leak detector and
 * the pre-registration, each a line from the reports the API reads. Set apart from the
 * questions by its structure (a bordered band below them), not by its colour.
 */
export function TrustBand({ checks, study }: { checks?: Check[]; study?: Study }) {
  const q = useSearchParams();
  const reviewer = WORKFLOWS.find((w) => w.reviewer);
  const pass = checks?.filter((c) => c.verdict === "PASS").length ?? 0;
  const replay = find(checks, /replay/i);
  const leaks = find(checks, /leak detection suite/i);
  const prereg = find(checks, /pre-?registration/i);
  const evidence = (c: Check | undefined, fallback: string) => (c ? `${c.label}: ${c.verdict}. ${c.evidence}` : fallback);
  return (
    <section className={ui.trust} aria-labelledby="trust-title">
      <div className={ui.trustHead}>
        <span className={ui.trustIcon}>
          <Shield />
        </span>
        <h2 id="trust-title" className={ui.trustTitle}>
          Can you trust these numbers?
        </h2>
      </div>
      <ul className={ui.trustLines}>
        <Line
          ok={checks ? pass === checks.length && checks.length > 0 : undefined}
          name="Checks"
          claim={checks ? `${num(pass)} of ${num(checks.length)} pass` : "reading the reports…"}
          tip="Every check in the milestone reports, its verdict derived from the report's own criterion and evidence. Open the list for each one's evidence."
          toggle={{ key: "checks", open: q.get("checks") === "1", hook: "home-checks" }}
        />
        <Line ok={replay ? replay.verdict === "PASS" : undefined} name="Replay" claim="byte-identical on this machine; within tolerance across platforms" tip={evidence(replay, "No replay check in the reports on this server.")} />
        <Line ok={leaks ? leaks.verdict === "PASS" : undefined} name="Leak detector" claim="catches a planted leak in every field it tests" tip={evidence(leaks, "No leak-suite check in the reports on this server.")} />
        <Line
          ok={prereg ? prereg.verdict === "PASS" : undefined}
          name="Pre-registration"
          claim="the rule fixed before the data"
          tip={`${evidence(prereg, "The test, the size and the premiums, fixed in docs/PRE_REGISTRATION.md before the sweep ran.")}${study ? ` The rule: ${study.test}, α = ${study.alpha}, S = ${study.n_seeds} × N = ${num(study.n_episodes)}.` : ""}`}
          toggle={{ key: "study", open: q.get("study") === "1", hook: "home-study" }}
        />
      </ul>
      <div className={ui.trustActions}>
        {reviewer ? (
          <Link href={`${stepHref(reviewer, 0)}&start=1`} className={`${ui.btn} ${ui.btnPrimary}`}>
            Start the review
          </Link>
        ) : null}
        <details className={ui.trustRepro}>
          <summary>Reproduce it yourself</summary>
          <pre className={ui.trustCode}>{REPRODUCE.join("\n")}</pre>
          <p className={ui.trustNote}>
            A clone and Python 3.12: ruff, mypy, the tests, the gates and three sweep cells regenerated and compared, about 10 minutes.{" "}
            <a href={PARTNER_SECTION} target="_blank" rel="noreferrer">
              The README&apos;s partner section
            </a>
          </p>
        </details>
      </div>
    </section>
  );
}
