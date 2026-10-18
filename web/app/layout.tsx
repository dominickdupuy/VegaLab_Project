import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { Shell } from "@/components/Shell";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "optspread", template: "%s · optspread" },
  description: "Runs, gates, power and the real SPX panel for the optspread detection-floor harness (Vega Lab).",
  applicationName: "optspread",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#13233d",
  colorScheme: "light dark",
};

/**
 * The theme before the first paint, so a page never flashes the other one. Light is
 * the default; a choice made with the toggle is remembered under "optspread.theme",
 * the key lib/theme.ts writes.
 */
const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("optspread.theme");if(t!=="light"&&t!=="dark")t="light";document.documentElement.dataset.theme=t;}catch(e){document.documentElement.dataset.theme="light";}})();`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" data-theme="light" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
