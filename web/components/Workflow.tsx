"use client";

import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type Activity, type Call, activity, calls, lastCall, useActivity } from "@/lib/activity";
import { GUIDE_ATTR, GUIDE_ON_ATTR } from "@/lib/guide";
import { type Ctx, type Step, type Target, stepHref, stepPath, type Workflow, workflow } from "@/lib/tours";
import { setQuery } from "@/lib/url";
import ui from "./ui.module.css";

/**
 * The workflow strip: one line under the header while a workflow runs (two on a phone).
 * It says what to do, rings the control to use, keeps Next locked until the step's
 * action is done ("Do this to continue"), then says what the result means.
 *
 * The workflow in progress lives in this tab's sessionStorage, so it survives the
 * reader following any link on a page; `wf` and `step` in the URL start or move it (a
 * link to a step works for anyone), `start=1` starts it afresh. × leaves it.
 */

const KEY = "optspread.guide";

interface Saved {
  wf: string;
  step: number; // 0-based
  entered: Record<string, number>;
  done: Record<string, { at: number; read?: string; kept?: Record<string, unknown> }>;
}

function load(): Saved | null {
  try {
    const raw = window.sessionStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Saved) : null;
  } catch {
    return null;
  }
}

function save(s: Saved | null): void {
  try {
    if (s) window.sessionStorage.setItem(KEY, JSON.stringify(s));
    else window.sessionStorage.removeItem(KEY);
  } catch {
    /* nowhere to keep it: the workflow lasts as long as the page */
  }
}

// reads the strip makes itself, shared by every step: key -> answer
const cache = new Map<string, { data?: unknown; error?: string }>();

function reducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

function textOf(el: Element): string {
  return (el.textContent ?? "").replace(/\s+/g, " ").trim();
}

/** The elements a target names: its data-guide hook, else its text, label, aria-label or selector. */
function resolve(t: Target): HTMLElement[] {
  const main = document.querySelector("main") ?? document.body;
  const hooked = [...main.querySelectorAll<HTMLElement>(`[${GUIDE_ATTR}="${CSS.escape(t.guide)}"]`)];
  if (hooked.length) return t.all ? hooked : hooked.slice(0, 1);
  let found: HTMLElement[] = [];
  if (t.text) found = [...main.querySelectorAll<HTMLElement>('button, [role="radio"], a')].filter((e) => t.text?.test(textOf(e)));
  if (!found.length && t.label) found = [...main.querySelectorAll<HTMLElement>("label, .field")].filter((e) => t.label?.test(textOf(e).slice(0, 80)));
  if (!found.length && t.aria) found = [...main.querySelectorAll<HTMLElement>("[aria-label]")].filter((e) => t.aria?.test(e.getAttribute("aria-label") ?? ""));
  if (!found.length && t.css) found = [...main.querySelectorAll<HTMLElement>(t.css)];
  // a label's ring goes on the label (it holds the control); keep only visible ones
  found = found.filter((e) => e.getClientRects().length > 0);
  if (t.label && !t.text) found.sort((a, b) => textOf(a).length - textOf(b).length);
  return t.all ? found : found.slice(0, 1);
}

function inView(el: HTMLElement): boolean {
  const r = el.getBoundingClientRect();
  return r.top >= 110 && r.bottom <= window.innerHeight - 12;
}

/** Ring the step's control, bring it into view once, and drop the ring when the step is done. */
function useRing(target: Target | Target[] | undefined, active: boolean, stepKey: string, nudge: number): boolean {
  const ringed = useRef<HTMLElement[]>([]);
  const scrolled = useRef("");
  const [found, setFound] = useState(false);
  const list = target === undefined ? [] : Array.isArray(target) ? target : [target];
  const spec = JSON.stringify(list.map((x) => ({ ...x, text: x.text?.source, label: x.label?.source, aria: x.aria?.source })));
  useEffect(() => {
    const clear = () => {
      for (const e of ringed.current) e.removeAttribute(GUIDE_ON_ATTR);
      ringed.current = [];
    };
    if (!list.length || !active) {
      clear();
      setFound(false);
      return;
    }
    const tick = () => {
      const els = list.flatMap(resolve);
      setFound(els.length > 0);
      const same = els.length === ringed.current.length && els.every((e, i) => e === ringed.current[i]);
      if (!same) {
        clear();
        for (const e of els) e.setAttribute(GUIDE_ON_ATTR, "");
        ringed.current = els;
      }
      const first = els[0];
      const want = `${stepKey}|${nudge}`;
      if (first && scrolled.current !== want) {
        scrolled.current = want;
        if (!inView(first)) first.scrollIntoView({ block: "center", behavior: reducedMotion() ? "auto" : "smooth" });
        if (nudge) {
          // "Show me": pulse again
          for (const e of els) {
            e.removeAttribute(GUIDE_ON_ATTR);
            void e.offsetWidth;
            e.setAttribute(GUIDE_ON_ATTR, "");
          }
        }
      }
    };
    tick();
    const id = window.setInterval(tick, 500);
    return () => {
      window.clearInterval(id);
      clear();
    };
  }, [spec, active, stepKey, nudge]);
  return found;
}

export function WorkflowStrip() {
  const params = useSearchParams();
  const path = usePathname() ?? "/";
  const router = useRouter();
  const version = useActivity();
  const [saved, setSaved] = useState<Saved | null>(null);
  const [, setTick] = useState(0);
  const [open, setOpen] = useState(false);
  const [nudge, setNudge] = useState(0);

  // the URL starts or moves a workflow; otherwise the one in progress carries on
  const urlWf = params.get("wf");
  const urlStep = Number(params.get("step")) || 1;
  const fresh = params.get("start") === "1";
  useEffect(() => {
    const prev = load();
    const w = workflow(urlWf);
    if (w) {
      const step = Math.min(w.steps.length, Math.max(1, urlStep)) - 1;
      const base: Saved = !fresh && prev?.wf === w.id ? prev : { wf: w.id, step, entered: {}, done: {} };
      const next = { ...base, step };
      save(next);
      setSaved(next);
      if (fresh) setQuery({ start: null });
    } else setSaved(prev && workflow(prev.wf) ? prev : null);
  }, [urlWf, urlStep, fresh]);

  const w: Workflow | undefined = workflow(saved?.wf);
  const i = saved ? Math.min((w?.steps.length ?? 1) - 1, saved.step) : 0;
  const step: Step | undefined = w?.steps[i];
  const stepKey = `${w?.id}:${step?.id}`;

  // when this step began: the first time the strip showed it
  useEffect(() => {
    if (!saved || !step || saved.entered[step.id]) return;
    const next = { ...saved, entered: { ...saved.entered, [step.id]: Date.now() } };
    save(next);
    setSaved(next);
  }, [saved, step]);

  const since = (step && saved?.entered[step.id]) || Date.now();
  const get = useCallback(<T,>(key: string, loadIt: () => Promise<T>): T | undefined => {
    const hit = cache.get(key);
    if (hit) return hit.data as T | undefined;
    cache.set(key, {});
    loadIt().then(
      (data) => {
        cache.set(key, { data });
        setTick((n) => n + 1);
      },
      (err: unknown) => {
        cache.set(key, { error: err instanceof Error ? err.message : String(err) });
        setTick((n) => n + 1);
      },
    );
    return undefined;
  }, []);

  const ctx: Ctx = useMemo(
    () => ({
      path,
      q: new URLSearchParams(params.toString()),
      since,
      call: (c: Call) => lastCall(c, since),
      calls: (c: Call) => calls(c, since),
      before: (c: Call) => calls(c, 0).filter((a) => a.at < since).pop(),
      did: (kind: string) => activity().filter((a: Activity) => a.kind === kind && a.at >= since).pop(),
      kept: (id: string) => saved?.done[id]?.kept,
      get,
    }),
    // version: a new answer from the server is a new context
    [path, params, since, saved, get, version],
  );

  let doneNow = false;
  let read: string | null = null;
  let blocked: string | null = null;
  let task = "";
  let target: Target | Target[] | undefined;
  if (step) {
    try {
      doneNow = step.done(ctx);
      read = doneNow || saved?.done[step.id] ? (step.read?.(ctx) ?? null) : null;
      blocked = doneNow ? null : (step.blocked?.(ctx) ?? null);
      task = typeof step.task === "function" ? step.task(ctx) : step.task;
      target = typeof step.target === "function" ? step.target(ctx) : step.target;
    } catch {
      // a page answering in a shape the step does not expect never breaks the strip
    }
  }
  const record = step ? saved?.done[step.id] : undefined;
  const done = doneNow || Boolean(record);
  const shownRead = read ?? record?.read ?? null;

  // a step done stays done, with what it said and what later steps need
  useEffect(() => {
    if (!saved || !step || !doneNow) return;
    const prev = saved.done[step.id];
    let kept: Record<string, unknown> | undefined;
    try {
      kept = step.keep?.(ctx) ?? prev?.kept;
    } catch {
      kept = prev?.kept;
    }
    const text = read ?? prev?.read;
    if (prev && prev.read === text && JSON.stringify(prev.kept) === JSON.stringify(kept)) return;
    const next = { ...saved, done: { ...saved.done, [step.id]: { at: prev?.at ?? Date.now(), read: text ?? undefined, kept } } };
    save(next);
    setSaved(next);
  });

  const here = step ? stepPath(step) === path || (stepPath(step) !== "/" && path.startsWith(stepPath(step))) : false;
  const found = useRing(target, Boolean(step && !done && here), stepKey, nudge);
  useEffect(() => setOpen(false), [stepKey]);

  if (!w || !step || !saved) return null;
  const n = w.steps.length;
  const last = i === n - 1;

  const go = (j: number) => {
    const s = w.steps[j];
    if (!s) return;
    const next = { ...saved, step: j };
    save(next);
    setSaved(next);
    // the next step on this same page carries on from where the reader is
    if (stepPath(s) === path && !s.merge && !s.open) setQuery({ wf: w.id, step: j + 1 });
    else if (stepPath(s) === path && s.merge) {
      const q = new URLSearchParams(window.location.search);
      for (const [k, v] of new URLSearchParams(s.href.split("?")[1] ?? "")) q.set(k, v);
      q.set("wf", w.id);
      q.set("step", String(j + 1));
      router.push(`${path}?${q.toString()}`);
    } else router.push(stepHref(w, j));
  };
  const leave = () => {
    save(null);
    setSaved(null);
    setQuery({ wf: null, step: null, start: null });
  };
  const finish = () => {
    leave();
    router.push("/");
  };

  const canNext = done || Boolean(blocked);
  const status = done ? (
    <span className={`${ui.wfStatus} ${ui.wfOk}`}>✓ Done</span>
  ) : blocked ? (
    <span className={`${ui.wfStatus} ${ui.wfSkip}`} title={blocked}>
      Not available here
    </span>
  ) : (
    <span className={`${ui.wfStatus} ${ui.wfWait}`}>Do this to continue</span>
  );

  return (
    <div className={ui.wfStrip} role="region" aria-label={`Workflow: ${w.title}`}>
      <div className={ui.wfInner}>
        <span className={ui.wfLabel}>
          <span className={ui.wfTitle}>{w.title}</span>
          <span className={ui.wfDots} aria-hidden="true">
            {w.steps.map((s, j) => (
              <span key={s.id} className={`${ui.wfDot} ${saved.done[s.id] ? ui.wfDotDone : ""} ${j === i ? ui.wfDotHere : ""}`} />
            ))}
          </span>
          <span className={ui.wfCount}>
            {i + 1}/{n}
          </span>
        </span>
        <span className={ui.wfBody}>
        <button type="button" className={`${ui.wfCaption} ${open ? ui.wfCaptionOpen : ""}`} aria-live="polite" title={shownRead ?? task} onClick={() => setOpen((o) => !o)} aria-expanded={open}>
          {done && shownRead ? (
            <span className={ui.wfRead}>
              <span className={ui.wfReadMark} aria-hidden="true">
                ✓
              </span>
              {shownRead}
            </span>
          ) : done ? (
            <span className={ui.wfRead}>Done. Reading the result…</span>
          ) : blocked ? (
            <span>
              <span className={ui.wfDo}>{task}</span> Not on this server: {blocked}.
            </span>
          ) : here ? (
            <span className={ui.wfDo}>{task}</span>
          ) : (
            <span>
              <span className={ui.wfDo}>{task}</span> This step is on another page.
            </span>
          )}
        </button>
        {status}
        </span>
        <span className={ui.wfNav}>
          {!done && !here ? (
            <Link href={stepHref(w, i)} className={`${ui.btn} ${ui.btnSmall}`}>
              Go to the step
            </Link>
          ) : !done && target && found ? (
            <button type="button" className={`${ui.btn} ${ui.btnSmall} ${ui.wfShow}`} onClick={() => setNudge((x) => x + 1)}>
              Show me
            </button>
          ) : null}
          <button type="button" className={`${ui.btn} ${ui.btnSmall}`} disabled={i === 0} onClick={() => go(i - 1)} aria-label="Previous step">
            ←
          </button>
          {last ? (
            <button type="button" className={`${ui.btn} ${ui.btnSmall} ${ui.btnPrimary}`} disabled={!canNext} onClick={finish} title={canNext ? undefined : "Do this step first"}>
              Finish
            </button>
          ) : (
            <button type="button" className={`${ui.btn} ${ui.btnSmall} ${ui.btnPrimary}`} disabled={!canNext} onClick={() => go(i + 1)} title={canNext ? undefined : "Do this step first"}>
              Next →
            </button>
          )}
          <button type="button" className={ui.wfClose} aria-label="Leave the workflow" title="Leave the workflow" onClick={leave}>
            ×
          </button>
        </span>
      </div>
    </div>
  );
}

/** Copy the address of the view as it stands: every page keeps its settings in it. */
export function CopyLink() {
  const copy = async () => {
    const url = window.location.href;
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      window.prompt("Copy this link", url);
    }
  };
  return (
    <button type="button" className={ui.themeBtn} aria-label="Copy a link to this view" title="Copy a link to this view" onClick={copy}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M10 13a5 5 0 0 0 7.07 0l3-3a5 5 0 0 0-7.07-7.07l-1.5 1.5" />
        <path d="M14 11a5 5 0 0 0-7.07 0l-3 3a5 5 0 0 0 7.07 7.07l1.5-1.5" />
      </svg>
    </button>
  );
}
