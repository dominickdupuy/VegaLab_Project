/**
 * The power of the study's one-sided t test, in the browser, for Home's "Can a backtest
 * see it?" widget, which redraws on every slider move.
 *
 * It is the same number the server computes (optspread.vrp.noncentral_power, scipy's
 * nct.sf at t_{1-alpha, S-1}): P(T > c) for T noncentral t with S - 1 degrees of freedom
 * and noncentrality lambda, written as T = (Z + lambda) / U with U = sqrt(V / df), V a
 * chi-square on df, and integrated over U:
 *
 *   P(T > c) = E_U[ Phi(lambda - c U) ]
 *
 * by Simpson's rule on U's density. It agrees with scipy to better than 1e-6 for S from 2 to
 * 100 seeds and lambda from 0 to 60.
 */

/** ln Gamma(x), x > 0 (Lanczos, g = 7, n = 9). */
function lgamma(x: number): number {
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  const z = x - 1;
  let a = c[0] ?? 1;
  const t = z + 7.5;
  for (let i = 1; i < 9; i++) a += (c[i] ?? 0) / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** The standard normal's upper tail, 1 - Phi(x): erfc by its Taylor series below 3, by its continued fraction above. */
function normSf(x: number): number {
  const z = Math.abs(x) / Math.SQRT2;
  let r: number;
  if (z < 3) {
    // erf by its Taylor series, converges fast for z < 3
    let term = z;
    let sum = z;
    for (let n = 1; n < 200; n++) {
      term *= (-z * z) / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    r = 1 - (2 / Math.sqrt(Math.PI)) * sum; // erfc(z)
  } else {
    // erfc by its continued fraction (Lentz), converges fast for z >= 3
    const tiny = 1e-300;
    let f = z;
    let C = z;
    let D = 0;
    for (let n = 1; n < 300; n++) {
      const an = n / 2;
      D = z + an * D;
      D = D === 0 ? tiny : 1 / D;
      C = z + an / C;
      if (C === 0) C = tiny;
      const delta = C * D;
      f *= delta;
      if (Math.abs(delta - 1) < 1e-16) break;
    }
    r = Math.exp(-z * z) / (f * Math.sqrt(Math.PI));
  }
  const upper = r / 2;
  return x >= 0 ? upper : 1 - upper;
}

const STEPS = 2000; // Simpson intervals over U; even

/** E_U[ Phi(lambda - c U) ] = P(T > c) for T noncentral t on df with noncentrality lambda. */
function sf(c: number, df: number, lambda: number): number {
  const lo = Math.max(0, 1 - 12 / Math.sqrt(df));
  const hi = 1 + 12 / Math.sqrt(df);
  const h = (hi - lo) / STEPS;
  const k = Math.log(2) + (df / 2) * Math.log(df / 2) - lgamma(df / 2);
  const f = (u: number) => {
    if (u <= 0) return df === 1 ? Math.exp(k) * normSf(-lambda) : 0;
    return Math.exp(k + (df - 1) * Math.log(u) - (df * u * u) / 2) * normSf(c * u - lambda);
  };
  let s = f(lo) + f(hi);
  for (let i = 1; i < STEPS; i++) s += (i % 2 ? 4 : 2) * f(lo + i * h);
  return Math.min(1, Math.max(0, (s * h) / 3));
}

const critical = new Map<string, number>();

/** t_{1-alpha, df}: the bar a one-sided test's t statistic must clear. */
export function tCritical(df: number, alpha: number): number {
  const key = `${df}|${alpha}`;
  const hit = critical.get(key);
  if (hit !== undefined) return hit;
  let a = 0;
  let b = 64;
  for (let i = 0; i < 60; i++) {
    const m = (a + b) / 2;
    if (sf(m, df, 0) > alpha) a = m;
    else b = m;
  }
  const c = (a + b) / 2;
  critical.set(key, c);
  return c;
}

/** P(nct(S - 1, lambda) > t_{1-alpha, S-1}): the chance a study of `seeds` seeds catches the edge. */
export function tPower(lambda: number, seeds: number, alpha: number): number {
  if (!Number.isFinite(lambda)) return lambda > 0 ? 1 : 0;
  if (lambda === 0) return alpha;
  const df = seeds - 1;
  return sf(tCritical(df, alpha), df, lambda);
}
