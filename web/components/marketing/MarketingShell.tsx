"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { docUrl } from "@/lib/api";
import { ThemeButton } from "../Shell";
import m from "./marketing.module.css";

/**
 * The chrome around the landing page and the tutorial: a slim bar with the product's name,
 * a few links and the way into the app, and a footer in columns. The app's ten tabs and the
 * workflow strip belong to the Workbench and the pages behind it, not here.
 */
export function MarketingShell({ children }: { children: ReactNode }) {
  return (
    <div className={m.shell}>
      <header className={m.bar}>
        <div className={m.barInner}>
          <Link href="/" className={m.logo} aria-label="Vega Lab, home">
            <Mark />
            <span>
              Vega Lab <span className={m.logoSub}>optspread</span>
            </span>
          </Link>
          <nav className={m.links} aria-label="Site">
            <Link href="/#features">The app</Link>
            <Link href="/tutorial/#loop">How it works</Link>
            <Link href="/#results">Results</Link>
            <Link href="/tutorial/">Tutorial</Link>
            <Link href="/#faq">FAQ</Link>
          </nav>
          <div className={m.barRight}>
            <ThemeButton />
            <Link href="/workbench/" className={m.cta}>
              Open the app
            </Link>
          </div>
        </div>
      </header>
      <main className={m.main}>{children}</main>
      <footer className={m.footer}>
        <div className={m.footerInner}>
          <div className={m.footerBrand}>
            <span className={m.logo}>
              <Mark />
              <span>Vega Lab</span>
            </span>
            <p>Test an options edge where the answer is known before you trust it on real SPX data.</p>
          </div>
          <div className={m.footerCol}>
            <h4>Product</h4>
            <Link href="/workbench/">Workbench</Link>
            <Link href="/alpha/">Synthetic alpha</Link>
            <Link href="/power/">Power lab</Link>
            <Link href="/leaks/">Leak detector</Link>
            <Link href="/runs/">Runs</Link>
          </div>
          <div className={m.footerCol}>
            <h4>Learn</h4>
            <Link href="/tutorial/">Tutorial</Link>
            <a href={docUrl("REPRODUCE.md")} target="_blank" rel="noreferrer">
              Reproduce the study
            </a>
            <a href={docUrl("PRE_REGISTRATION.md")} target="_blank" rel="noreferrer">
              Pre-registration
            </a>
          </div>
          <div className={m.footerCol}>
            <h4>Project</h4>
            <span>Vega Lab</span>
            <span>CIS 4914</span>
            <span>University of Florida</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

/** The mark: a short-vol payoff, a tent over a baseline. */
export function Mark() {
  return (
    <svg className={m.mark} width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
      <rect width="26" height="26" rx="7" fill="var(--accent)" />
      <path d="M5 17h4l4-8 4 8h4" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
