import type { ReactNode } from "react";
import type { ApiError } from "@/lib/api";
import ui from "./ui.module.css";

export function Loading({ what }: { what: string }) {
  return (
    <div className={ui.loading} role="status" aria-live="polite">
      <span className={ui.spinner} aria-hidden="true" />
      <span>Loading {what}…</span>
    </div>
  );
}

export function Skeleton({ height }: { height: number }) {
  return <div className={ui.skeleton} style={{ height }} aria-hidden="true" />;
}

/**
 * The shape of a chart before it draws: ghost axes and gridlines at the chart's own
 * height, so nothing on the page moves when the real one arrives.
 */
export function ChartSkeleton({ height, label = "Drawing the chart" }: { height: number; label?: string }) {
  return (
    <div className={ui.chartSkel} style={{ height }} role="status" aria-label={label}>
      <span className={ui.chartSkelPlot}>
        {[0, 1, 2, 3].map((i) => (
          <span key={i} className={ui.chartSkelLine} />
        ))}
      </span>
    </div>
  );
}

/**
 * A long computation in progress (a re-simulation, an attribution): what it is doing,
 * how far along it is, and how long it has taken. Determinate whenever the server says
 * how far it is; an honest "working" bar otherwise.
 */
export function ComputeProgress({ what, stage, fraction, seconds }: { what: string; stage?: string; fraction?: number | null; seconds?: number | null }) {
  const known = typeof fraction === "number" && Number.isFinite(fraction);
  return (
    <div className={ui.compute} role="status" aria-live="polite">
      <div className={ui.spread}>
        <span>
          <b>{what}</b>
          {stage ? <span className={ui.dim}> · {stage}</span> : null}
        </span>
        <span className={ui.dim}>
          {known ? `${Math.round((fraction ?? 0) * 100)}%` : "working"}
          {seconds ? ` · ${Math.round(seconds)} s` : ""}
        </span>
      </div>
      <div className={ui.progress} role="progressbar" aria-label={what} aria-valuemin={0} aria-valuemax={100} aria-valuenow={known ? Math.round((fraction ?? 0) * 100) : undefined}>
        <div className={`${ui.progressBar} ${known ? "" : ui.progressWorking}`} style={known ? { width: `${Math.max(2, Math.min(100, (fraction ?? 0) * 100))}%` } : undefined} />
      </div>
    </div>
  );
}

/** Nothing to show yet: say what would appear here, and what produces it. */
export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className={ui.state}>
      <p className={ui.stateTitle}>{title}</p>
      {children ? <div>{children}</div> : null}
    </div>
  );
}

/** A view whose input does not exist yet: the sweep, the panel, the attribution output. */
export function Pending({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className={`${ui.state} ${ui.statePending}`} role="status">
      <p className={ui.stateTitle}>{title}</p>
      {children ? <div>{children}</div> : null}
    </div>
  );
}

export function ErrorBox({ what, error, onRetry }: { what: string; error: ApiError; onRetry?: () => void }) {
  return (
    <div className={`${ui.state} ${ui.stateError}`} role="alert">
      <p className={ui.stateTitle}>Couldn&apos;t load {what}</p>
      <p className={ui.breakWords}>
        {error.message}
        {error.status ? ` (HTTP ${error.status})` : ""}
      </p>
      {onRetry ? (
        <div>
          <button type="button" className={`${ui.btn} ${ui.btnSmall}`} onClick={onRetry}>
            Try again
          </button>
        </div>
      ) : null}
    </div>
  );
}

/** The SPX panel is not built yet: a state with a reason, not an error. */
export function PanelNotBuilt({ missing, root }: { missing: string[]; root?: string }) {
  return (
    <Pending title="Panel not built">
      {root ? (
        <p>
          <code>{root}</code>
        </p>
      ) : null}
      <ul>
        {missing.map((m) => (
          <li key={m}>
            <code>{m}</code>
          </li>
        ))}
      </ul>
    </Pending>
  );
}
