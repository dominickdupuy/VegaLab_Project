"use client";

import { type ReactNode, useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { define, type Glossary } from "@/lib/glossary";
import ui from "./ui.module.css";

/**
 * A small tooltip: what a term means, or where a number came from.
 *
 * It opens on hover, on keyboard focus and on a tap (a phone has no hover), and closes
 * on leaving, Escape, blur or a tap elsewhere. The trigger is a real button with an
 * accessible name, and the bubble is its `aria-describedby`, so a screen reader reads
 * it with the control. The bubble is portalled to <body> and placed beside the trigger
 * within the viewport, so a card or a scrolling table never clips it.
 *
 *   <InfoTip term="cvar" />                    the glossary's line (lib/glossary.ts)
 *   <InfoTip term="cvar" dict={RUNS} />        a page glossary first, then the glossary
 *   <InfoTip text="…" label="Where this comes from" />
 *   <InfoTip term="ppo">PPO</InfoTip>          the text itself is the trigger, underlined
 */
export function InfoTip({
  term,
  text,
  dict,
  label,
  title,
  children,
  icon = "info",
}: {
  term?: string;
  text?: ReactNode;
  dict?: Glossary;
  /** The trigger's accessible name; "About <term>" by default. */
  label?: string;
  /** A bold first line in the bubble. */
  title?: ReactNode;
  children?: ReactNode;
  icon?: "info" | "source";
}) {
  const body = text ?? define(term, dict);
  const id = useId();
  const btn = useRef<HTMLButtonElement | null>(null);
  const tip = useRef<HTMLSpanElement | null>(null);
  const [open, setOpen] = useState(false);
  const [pinned, setPinned] = useState(false); // opened by a tap or click: stays until dismissed
  const [pos, setPos] = useState<{ left: number; top: number; above: boolean } | null>(null);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const place = useCallback(() => {
    const b = btn.current?.getBoundingClientRect();
    const t = tip.current?.getBoundingClientRect();
    if (!b) return;
    const w = t?.width ?? 280;
    const h = t?.height ?? 60;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const left = Math.max(8, Math.min(vw - w - 8, b.left + b.width / 2 - w / 2));
    const above = b.bottom + 8 + h > vh && b.top - 8 - h > 0;
    setPos({ left, top: above ? b.top - 8 - h : b.bottom + 8, above });
  }, []);

  useLayoutEffect(() => {
    if (open) place();
  }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const away = (e: PointerEvent) => {
      const n = e.target as Node;
      if (btn.current?.contains(n) || tip.current?.contains(n)) return;
      setOpen(false);
      setPinned(false);
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        setPinned(false);
      }
    };
    window.addEventListener("pointerdown", away);
    window.addEventListener("keydown", key);
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("pointerdown", away);
      window.removeEventListener("keydown", key);
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open, place]);

  if (!body) return children ? <>{children}</> : null;
  const name = label ?? (term ? `About ${typeof children === "string" ? children : term.replaceAll("_", " ")}` : "More about this");

  return (
    <>
      <button
        ref={btn}
        type="button"
        className={children ? ui.tipTerm : ui.tipBtn}
        aria-label={children ? undefined : name}
        aria-describedby={id}
        aria-expanded={open}
        onPointerEnter={(e) => e.pointerType === "mouse" && setOpen(true)}
        onPointerLeave={(e) => e.pointerType === "mouse" && !pinned && setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => !pinned && setOpen(false)}
        onClick={(e) => {
          e.stopPropagation();
          e.preventDefault();
          const next = !(open && pinned);
          setPinned(next);
          setOpen(next);
        }}
      >
        {children ?? (
          <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true">
            {icon === "source" ? (
              <path d="M3 2.5h7l3 3v8H3z M10 2.5v3h3 M5.5 8h5 M5.5 10.5h5" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
            ) : (
              <>
                <circle cx="8" cy="8" r="6.6" fill="none" stroke="currentColor" strokeWidth="1.3" />
                <circle cx="8" cy="4.9" r="0.95" fill="currentColor" />
                <path d="M8 7.2v4.6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              </>
            )}
          </svg>
        )}
      </button>
      {mounted
        ? createPortal(
            <span
              ref={tip}
              id={id}
              role="tooltip"
              className={ui.tip}
              hidden={!open}
              style={pos ? { left: pos.left, top: pos.top } : { left: -9999, top: 0 }}
              onPointerEnter={() => setOpen(true)}
              onPointerLeave={(e) => e.pointerType === "mouse" && !pinned && setOpen(false)}
            >
              {title ? <b className={ui.tipTitle}>{title}</b> : null}
              {body}
            </span>,
            document.body,
          )
        : null}
    </>
  );
}

/** The label of a control or a tile, with the glossary's line beside it when there is one. */
export function Labelled({ children, term, dict }: { children: ReactNode; term?: string; dict?: Glossary }) {
  const key = term ?? (typeof children === "string" ? children : undefined);
  return (
    <span className={ui.labelled}>
      {children}
      {key && define(key, dict) ? <InfoTip term={key} dict={dict} label={`About ${typeof children === "string" ? children : key}`} /> : null}
    </span>
  );
}
