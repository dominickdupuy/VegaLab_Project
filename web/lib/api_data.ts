/**
 * The Real data page's question payloads: each view's numbers and its one-line reading.
 * Each an alias onto the schema generated from the server's OpenAPI document
 * (./schema.d.ts, PremiumOut, ShapeOut, QualityOut), as in `api.ts`.
 */

import { get, type Seller, type SigmaFrom, type Simulated } from "./api";
import type { components } from "./schema";

type Schemas = components["schemas"];

export type PremiumNow = Schemas["PremiumNowOut"];

/** The premium question: the VRP mapping's premium day by day, in vol points. */
export type Premium = Schemas["PremiumOut"];

/** The surface question: the surface's level, skew and term slope on one day. */
export type Shape = Schemas["ShapeOut"];

export type Region = Schemas["RegionOut"];

/** The simulator question: the simulated surface with where it is furthest from the market. */
export type SimulatedRead = Simulated;

/** The seller question: the seller's structure, read in words. */
export type SellerRead = Seller;

export type Quality = Schemas["QualityOut"];

export const dataApi = {
  premium: (s?: AbortSignal) => get<Premium>("/api/data/premium", undefined, s),
  shape: (date: string, s?: AbortSignal) => get<Shape>("/api/data/shape", { date }, s),
  simulated: (q: { date: string; premium: number; sigma_from: SigmaFrom }, s?: AbortSignal) => get<SimulatedRead>("/api/data/simulated", q, s),
  seller: (date: string, s?: AbortSignal) => get<SellerRead>("/api/data/seller", { date }, s),
  quality: (s?: AbortSignal) => get<Quality>("/api/data/quality", undefined, s),
};
