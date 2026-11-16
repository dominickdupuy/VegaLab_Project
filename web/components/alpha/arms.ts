/**
 * Plug-in arms on the Synthetic alpha page. The server lists them in its options
 * (`arm_details`, kind "plugin"); `nameArms` puts each one's label, "IV timer (plug-in)",
 * into the site's arm labels, so every picker, table and chart names it the same way. It is
 * idempotent and called while the page renders, before any child reads a label. A
 * plug-in is Python registered on the server (OPTSPREAD_ALPHA_PLUGINS); the page only
 * names it, it never sends code.
 */
import type { AlphaOptions } from "@/lib/api";
import { ARM_LABEL, armColor, type Tokens } from "@/lib/chart";

export function plugins(options: AlphaOptions): Set<string> {
  return new Set((options.arm_details ?? []).filter((a) => a.kind === "plugin").map((a) => a.name));
}

export function nameArms(options?: AlphaOptions): void {
  for (const a of options?.arm_details ?? []) if (a.kind === "plugin") ARM_LABEL[a.name] = a.label;
}

const PLUGIN_SLOTS: (keyof Tokens)[] = ["s3", "s4", "s5"];

/** A built-in arm's fixed colour; a plug-in's by its place in the server's list. */
export function alphaArmColor(t: Tokens, arm: string, options?: AlphaOptions): string {
  const listed = (options?.arm_details ?? []).filter((a) => a.kind === "plugin").map((a) => a.name);
  const i = listed.indexOf(arm);
  return i < 0 ? armColor(t, arm) : (t[PLUGIN_SLOTS[i % PLUGIN_SLOTS.length]!] as string);
}
