/**
 * The Runs, Compare and Launch payloads added on top of `api.ts`: each an alias onto the
 * schema generated from the server's OpenAPI document (./schema.d.ts), as in `api.ts`.
 */
import { type Capital, type Compare, get, type MixSummary, type RunFilters, request, type Spec } from "./api";
import type { components } from "./schema";

type Schemas = components["schemas"];

type Params = Parameters<typeof get>[1];

// ---------- P&L by premium per arm, and each arm against itself at zero ----------

export type World = Schemas["WorldOut"];

export type PremiumGroup = Schemas["PremiumGroupOut"];

export type PremiumContrast = Schemas["PremiumContrastOut"];

export type Premiums = Schemas["PremiumsOut"];

// ---------- the catalog: a designed matrix of runs ----------

export type CatalogRun = Schemas["CatalogRunOut"];

export type CatalogPair = Schemas["CatalogPairOut"];

export type CatalogGroup = Schemas["CatalogGroupOut"];

export type Catalog = Schemas["CatalogOut"];

// ---------- capital, with the re-simulated equity and margin ----------

export type Band = Schemas["BandOut"];

/** The capital payload: the generated Capital already carries the paths, bands and file. */
export type CapitalPaths = Capital;

// ---------- the action mix of one run, with its holdings ----------

export type Holding = Schemas["HoldingOut"];

export type MixDetail = Schemas["MixDetailOut"];

export type MixView = Schemas["MixViewOut"];

export type { MixSummary };

// ---------- compare, paired ----------

/** The compare payload: Compare already carries the paired interval, the seeds and the verdict. */
export type PairedCompare = Compare;

// ---------- the launch preview ----------

export type PreviewLeg = Schemas["PreviewLegOut"];

export type DayZero = Schemas["DayZeroOut"];

export type Peer = Schemas["PeerOut"];

export type Preview = Schemas["PreviewOut"];

export const runsApi = {
  premiums: (f: RunFilters, s?: AbortSignal) => get<Premiums>("/api/runs/premiums", f as Params, s),
  catalog: (s?: AbortSignal) => get<Catalog>("/api/runs/catalog", undefined, s),
  capital: (key: string, s?: AbortSignal) => get<CapitalPaths>("/api/runs/capital", { key }, s),
  mix: (key: string, s?: AbortSignal) => get<MixView>("/api/runs/mixview", { key }, s),
  computeMix: (key: string) => request<MixView>("POST", `/api/runs/mixview?key=${encodeURIComponent(key)}`),
  compare: (a: string, b: string, group?: string, s?: AbortSignal) => get<PairedCompare>("/api/runs/compare", { a, b, group }, s),
  preview: (spec: Spec, s?: AbortSignal) => request<Preview>("POST", "/api/launch/preview", spec, s),
};
