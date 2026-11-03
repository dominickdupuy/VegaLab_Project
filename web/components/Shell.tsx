"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { api } from "@/lib/api";
import { short } from "@/lib/format";
import { useTheme } from "@/lib/theme";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";
import ui from "./ui.module.css";

const TABS: { href: string; label: string }[] = [
  { href: "/", label: "Overview" },
  { href: "/runs/", label: "Runs" },
  { href: "/launch/", label: "Launch" },
  { href: "/synth/", label: "Synth data" },
  { href: "/power/", label: "Power lab" },
  { href: "/data/", label: "Real data" },
  { href: "/leaks/", label: "Leak demo" },
  { href: "/sweep/", label: "Sweep" },
  { href: "/how-it-works/", label: "How it works" },
];

function active(path: string, href: string): boolean {
  if (href === "/") return path === "/";
  return path === href || path.startsWith(href) || `${path}/` === href;
}

function ThemeButton() {
  const [theme, setTheme] = useTheme();
  const dark = theme === "dark";
  return (
    <button
      type="button"
      className={ui.themeBtn}
      aria-label={dark ? "Switch to the light theme" : "Switch to the dark theme"}
      title={dark ? "Light theme" : "Dark theme"}
      onClick={() => setTheme(dark ? "light" : "dark")}
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
          <nav className={ui.nav} aria-label="Views">
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
          </nav>
          <div className={ui.headerRight}>
            <ServerStatus />
            <ThemeButton />
          </div>
        </div>
      </header>
      <main className={ui.main}>{children}</main>
      <footer className={ui.footer}>
        <span>Vega Lab · CIS 4914 · University of Florida</span>
      </footer>
    </div>
  );
}
