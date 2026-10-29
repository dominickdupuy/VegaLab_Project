/**
 * The Real data page's question payloads: each view's numbers and its one-line reading.
 * Every shape mirrors a pydantic model in optspread/webapi.py (PremiumOut, ShapeOut,
 * QualityOut) or the fields the page's questions added to SimulatedOut and SellerOut.
 */

import { get, type Seller, type Simulated, type SigmaFrom } from "./api";

export interface PremiumNow {
  date: string;
  iv30: number | null;
  rv_trailing: number | null;
  priced: number | null;
  measured_date: string | null;
  measured: number | null;
  recent: number | null;
  recent_share_below: number | null;
}

/** The premium question: the VRP mapping's premium day by day, in vol points. */
export interface Premium {
  date: string[];
  premium: (number | null)[];
  smooth: (number | null)[];
  iv30: (number | null)[];
  rv_fwd: (number | null)[];
  rv_trailing: (number | null)[];
  smooth_days: number;
  now: PremiumNow | null;
  mean: number | null;
  mean_low: number | null;
  mean_high: number | null;
  bracket: (number | null)[];
  sweep_axis: number[];
  reading: string;
  source: string;
}

/** The surface question: the surface's level, skew and term slope on one day. */
export interface Shape {
  date: string;
  atm: number | null;
  skew: number | null;
  call: number | null;
  term: number | null;
  atm_share_below: number | null;
  skew_share_below: number | null;
  term_share_below: number | null;
  share_inverted: number | null;
  skew_median: number | null;
  term_median: number | null;
  tenor: number;
  put_k: number;
  call_k: number;
  long_tenor: number;
  reading: string;
  source: string;
}

export interface Region {
  name: "put" | "atm" | "call" | string;
  label: string;
  mean_diff: number | null;
  n: number;
}

/** The simulator question: the simulated surface with where it is furthest from the market. */
export interface SimulatedRead extends Simulated {
  regions: Region[];
  widest: { strike: number; tenor_trading_days: number; moneyness: number; diff: number; real: number; sim: number } | null;
  reading: string;
}

/** The seller question: the seller's structure, read in words. */
export interface SellerRead extends Seller {
  reading: string;
}

export interface Quality {
  panel_version: number | null;
  manifest_files: number;
  checked: { path: string; recorded: string | null; actual: string | null; match: boolean | null }[];
  of_record: boolean | null;
  fred: { year: number; source: string; rate_minus_bill_bp: number | null; rate_minus_sofr_bp: number | null; spot_vs_sp500_bp: number | null }[];
  fred_available: boolean;
  reading: string;
  source: string;
}

export const dataApi = {
  premium: (s?: AbortSignal) => get<Premium>("/api/data/premium", undefined, s),
  shape: (date: string, s?: AbortSignal) => get<Shape>("/api/data/shape", { date }, s),
  simulated: (q: { date: string; premium: number; sigma_from: SigmaFrom }, s?: AbortSignal) => get<SimulatedRead>("/api/data/simulated", q, s),
  seller: (date: string, s?: AbortSignal) => get<SellerRead>("/api/data/seller", { date }, s),
  quality: (s?: AbortSignal) => get<Quality>("/api/data/quality", undefined, s),
};
