import Link from "next/link";
import type { ReactNode } from "react";
import { stepHref, type Workflow } from "@/lib/tours";
import ui from "../ui.module.css";

const ICON: Record<Workflow["icon"], ReactNode> = {
  edge: <path d="M3 17l5-5 4 3 7-8M15 7h4v4" />,
  market: <path d="M3 18c3-1 4-9 7-9s3 5 6 5 3-7 5-9M3 21h18" />,
  pnl: <path d="M4 20V10M9 20V4M14 20v-7M19 20v-11M3 20h18" />,
  size: <path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z" />,
  leak: <path d="M12 3c3 4 6 7.5 6 11a6 6 0 0 1-12 0c0-3.5 3-7 6-11zM9.5 14.5l5 0" />,
  learner: <path d="M12 3a4 4 0 0 1 4 4v1a3 3 0 0 1 2 5 3 3 0 0 1-3 5h-6a3 3 0 0 1-3-5 3 3 0 0 1 2-5V7a4 4 0 0 1 4-4zM12 8v10" />,
  review: <path d="M5 4h10l4 4v12H5zM9 13l2 2 4-4" />,
  tune: <path d="M5 4v6M5 14v6M12 4v2M12 10v10M19 4v10M19 18v2M3 12h4M10 8h4M17 16h4" />,
  costs: <path d="M12 3v18M16.5 7.5c-.8-1.3-2.4-2-4.5-2-2.6 0-4.3 1.3-4.3 3.2 0 4.6 9.1 2.4 9.1 7 0 2-1.9 3.3-4.8 3.3-2.2 0-4-.8-4.9-2.3" />,
};

/** One workflow on the homepage: the question it answers and what you will do. */
export function WorkflowCard({ w, big }: { w: Workflow; big?: boolean }) {
  return (
    <Link href={`${stepHref(w, 0)}&start=1`} className={`${ui.wfCard} ${big ? ui.wfCardBig : ""}`}>
      <span className={ui.wfCardTop}>
        <span className={ui.wfIcon} aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            {ICON[w.icon]}
          </svg>
        </span>
      </span>
      <span className={ui.wfCardTitle}>{w.title}</span>
      <span className={ui.wfQuestion}>{w.question}</span>
      <ul className={ui.wfSteps}>
        {w.youWill.map((x) => (
          <li key={x}>{x}</li>
        ))}
      </ul>
      <span className={ui.wfGo}>Start →</span>
    </Link>
  );
}
