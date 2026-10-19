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
