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
  { href: "/runs/", label: "Runs" },
  { href: "/launch/", label: "Launch" },
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
  const text = live ? `server · ${short(health.data?.git_sha ?? "", 7)}` : health.error ? "server offline" : "connecting";
  return (
    <span className={ui.serverStatus} title={health.error?.message ?? health.data?.git_sha}>
      <span className={`${ui.dot} ${cls}`} aria-hidden="true" />
      <span className={ui.serverStatusText}>{text}</span>
    </span>
  );
}

/** The tabs, with one underline that slides to the current tab. */
function Nav({ path }: { path: string }) {
  const nav = useRef<HTMLElement>(null);
  const [track, setTrack] = useState<{ x: number; w: number } | null>(null);
  useLayoutEffect(() => {
    const el = nav.current;
    if (!el) return;
    const measure = () => {
      const on = el.querySelector<HTMLElement>('[aria-current="page"]');
      setTrack(on ? { x: on.offsetLeft + 10, w: Math.max(0, on.offsetWidth - 20) } : null);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [path]);
  return (
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
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const path = usePathname() ?? "/";
  return (
    <div className={ui.shell}>
      <header className={ui.header}>
        <div className={ui.headerInner}>
          <Link href="/" className={ui.brand} aria-label="optspread platform, Vega Lab">
            <span className={ui.brandName}>optspread platform</span>
            <span className={ui.brandLab}>VEGA LAB</span>
          </Link>
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
        <span>Vega Lab · CIS 4914 · University of Florida</span>
      </footer>
    </div>
  );
}
