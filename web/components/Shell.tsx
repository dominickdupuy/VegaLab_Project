"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { type ReactNode, Suspense, useLayoutEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { short } from "@/lib/format";
import { flipTheme, useTheme } from "@/lib/theme";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";
import ui from "./ui.module.css";
import { CopyLink, WorkflowStrip } from "./Workflow";

const TABS: { href: string; label: string }[] = [
  { href: "/", label: "Home" },
  { href: "/workbench/", label: "Workbench" },
  { href: "/alpha/", label: "Synthetic alpha" },
  { href: "/runs/", label: "Runs" },
  { href: "/launch/", label: "Launch" },
  { href: "/synth/", label: "Synth data" },
  { href: "/power/", label: "Power lab" },
  { href: "/data/", label: "Real data" },
  { href: "/leaks/", label: "Leak detector" },
  { href: "/sweep/", label: "Sweep" },
  { href: "/attribution/", label: "Attribution" },
];

function active(path: string, href: string): boolean {
  if (href === "/") return path === "/";
  return path === href || path.startsWith(href) || `${path}/` === href;
}

function ThemeButton() {
  const [theme] = useTheme();
  const dark = theme === "dark";
  const ref = useRef<HTMLButtonElement>(null);
  return (
    <button
      ref={ref}
      type="button"
      className={ui.themeBtn}
      aria-label={dark ? "Switch to the light theme" : "Switch to the dark theme"}
      title={dark ? "Light theme" : "Dark theme"}
      onClick={() => flipTheme(dark ? "light" : "dark", ref.current)}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
        {dark ? (
          <>
            <circle cx="12" cy="12" r="4.2" />
            <path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6" />
          </>
        ) : (
          <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />
        )}
      </svg>
    </button>
  );
}

/** The server link: amber while connecting, green with the commit it serves, red when it is gone. */
function ServerStatus() {
  const health = useFetch((s) => api.health(s), "health");
  usePoll([health], true, 15000);
  const live = Boolean(health.data?.ok);
  const cls = live ? ui.dotLive : health.error ? ui.dotOff : "";
  const text = live ? `server code · ${short(health.data?.git_sha ?? "", 7)}` : health.error ? "server offline" : "connecting";
  // the commit the server runs, not a run's: a run names its own under Provenance
  const tip = health.error?.message ?? (health.data?.git_sha ? `The commit this server runs: ${health.data.git_sha}. Each run names the commit that made it under Provenance.` : undefined);
  // below 1400px only the dot shows: the words move to the tooltip and the accessible name
  return (
    <span className={ui.serverStatus} title={tip ? `${text}. ${tip}` : text} role="img" aria-label={text}>
      <span className={`${ui.dot} ${cls}`} aria-hidden="true" />
      <span className={ui.serverStatusText} aria-hidden="true">
        {text}
      </span>
    </span>
  );
}

/** The tabs, with one underline that slides to the current tab. */
function Nav({ path }: { path: string }) {
  const nav = useRef<HTMLElement>(null);
  const [track, setTrack] = useState<{ x: number; w: number } | null>(null);
  // on a phone the tabs scroll sideways: a chevron says more are off to the right
  const [more, setMore] = useState(false);
  useLayoutEffect(() => {
    const el = nav.current;
    if (!el) return;
    const measure = () => {
      const on = el.querySelector<HTMLElement>('[aria-current="page"]');
      setTrack(on ? { x: on.offsetLeft + 10, w: Math.max(0, on.offsetWidth - 20) } : null);
      setMore(el.scrollLeft + el.clientWidth < el.scrollWidth - 4);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    el.addEventListener("scroll", measure, { passive: true });
    return () => {
      ro.disconnect();
      el.removeEventListener("scroll", measure);
    };
  }, [path]);
  return (
    <div className={ui.navWrap}>
    <nav ref={nav} className={ui.nav} aria-label="Views" data-track={track ? "" : undefined}>
      {TABS.map((t) => (
        <Link
          key={t.href}
          href={t.href}
          className={`${ui.navLink} ${active(path, t.href) ? ui.navActive : ""}`}
          aria-current={active(path, t.href) ? "page" : undefined}
        >
          {t.label}
        </Link>
      ))}
      {track ? <span className={ui.navTrack} style={{ width: track.w, transform: `translateX(${track.x}px)` }} aria-hidden="true" /> : null}
    </nav>
      {more ? (
        <span className={ui.navMore} aria-hidden="true">
          ›
        </span>
      ) : null}
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname() ?? "/";
  return (
    <div className={ui.shell}>
      <header className={ui.header}>
        <div className={ui.headerInner}>
          <span className={ui.brand}>
            <Link href="/" className={ui.brandName} aria-label="optspread platform, home">
              optspread platform
            </Link>
            <Link href="/" className={ui.brandLab} title="Home" aria-label="Vega Lab: home">
              VEGA LAB
            </Link>
          </span>
          <Nav path={path} />
          <div className={ui.headerRight}>
            <ServerStatus />
            <CopyLink />
            <ThemeButton />
          </div>
        </div>
        <Suspense fallback={null}>
          <WorkflowStrip />
        </Suspense>
      </header>
      <main className={ui.main}>{children}</main>
      <footer className={ui.footer}>
        <span className={ui.footerCredits}>
          <span>Vega Lab</span>
          <span>CIS 4914</span>
          <span>University of Florida</span>
        </span>
        <Link href="/tutorial/">Tutorial</Link>
      </footer>
    </div>
  );
}
