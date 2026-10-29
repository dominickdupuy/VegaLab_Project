import type { ReactNode } from "react";
import ui from "@/components/ui.module.css";

/**
 * A page's entrance: its first blocks rise a few pixels into place, a little apart.
 * A template remounts on a route change only; a knob that writes the query string
 * (lib/url.ts) does not replay it. The wrapper is `display: contents`, so the page's
 * blocks stay the layout's own children.
 */
export default function Template({ children }: { children: ReactNode }) {
  return <div className={ui.page}>{children}</div>;
}
