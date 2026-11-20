// gta6-trophy-forecast-v2 — a deterministic, data-only forecaster for the
// make.fun bounty gta6-trophy-rarity-forecast-01 (GTA VI PS5 trophy earn rates
// on Exophase). This is an independent implementation with a deliberately
// different trend model and a deliberately WIDER uncertainty envelope than the
// earlier entry.
//
// MODEL (everything is estimated from the case's own history; no network, no
// external priors, no randomness):
//
// 1. REFRESH COLLAPSE. Exophase republishes byte-identical pages between its
//    own refreshes, so consecutive reads are not independent samples. We keep
//    a read only when the tracked-player count or at least one rate differs
//    from the last kept read. Those are the independent snapshots.
//
// 2. TIME AXIS. Unlike a log-time clock, we use x = elapsed DAYS since the
//    first refresh point and fit a straight line to logit(rate) versus x.
//    Early in a game's life the earned share grows roughly exponentially in
//    odds, which is linear on this axis; the slope is a per-day logit drift.
//
// 3. RECENT-WEIGHTED FIT. Each trophy gets a weighted least-squares line whose
//    weights decay exponentially with age (half-life HALFLIFE_DAYS), so the
//    most recent refreshes dominate the estimated drift. This is a soft
//    recent-window trend rather than a whole-history average.
//
// 4. TIER SHRINKAGE. Per-trophy slopes are noisy (few points, 2-decimal
//    rounding). We learn a mean slope for each rarity tier (platinum/gold/
//    silver/bronze) from the same case and shrink each trophy's slope toward
//    its tier mean with empirical-Bayes reliability tau2/(tau2 + s2/Sxx).
//
// 5. MEDIAN. Anchor on the most recent observed rate and add the shrunk slope
//    times the day-gap from that refresh to the target. Clamp to 0.01..99.99.
//
// 6. SPREAD. Because a multi-day horizon is deeply uncertain, the total logit
//    sigma is built to be wide: fit residual, capped slope-extrapolation error
//    and a horizon term K*sqrt(days) combine in quadrature, with an extra
//    widening for rare trophies. Typical final-case sigma lands near 0.9-1.3,
//    far wider than a narrow random-walk band. Short practice horizons scale
//    the horizon term down but keep a visible spread.
//
// 7. Fallback for flat/short histories: persistence at the last rate with a
//    wide, horizon-aware spread, so every output stays valid.
//
// Pure JavaScript syntax (a Node 22 runnable subset of TypeScript), standard
// library only, fully deterministic — two runs produce identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed model constants — never tuned on any answer.
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const DAY_MS = 86_400_000;
const HALFLIFE_DAYS = 1.5;        // decay half-life for recent-weighted fit
const S2_FLOOR = 0.0025;          // floor on residual variance
const TAU2_FLOOR = 0.01;          // floor on between-trophy slope variance
const TREND_UNC_CAP = 0.80;       // cap on slope extrapolation uncertainty term
const HORIZON_K = 0.42;           // logit spread per sqrt(day) of horizon
const SIGMA_FLOOR = 0.30;
const SIGMA_CAP = 1.75;
const RARE_K = 0.06;              // extra spread per logit-unit of rarity
const RARE_CAP = 0.35;
const LOGIT_LO = -9.21, LOGIT_HI = 9.21; // logit of 0.01% and 99.99%

const clampRate = (x) => Math.min(99.99, Math.max(0.01, x));
const toLogit = (pct) => { const p = clampRate(pct) / 100; return Math.log(p / (1 - p)); };
const fromLogit = (l) => 100 / (1 + Math.exp(-l));
const round4 = (x) => Math.round(x * 10_000) / 10_000;
const average = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0);
function variance(a) {
  if (a.length < 2) return 0;
  const m = average(a);
  return a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - 1);
}

/** Keep only reads where players tracked or any rate actually changed. */
function collapseToRefreshes(history) {
  const kept = [];
  let prevPlayers = null, prevRates = null;
  for (const read of history) {
    const rates = read.rates ?? [];
    let changed = kept.length === 0 || read.players_tracked !== prevPlayers ||
      rates.length !== (prevRates ? prevRates.length : -1);
    if (!changed) {
      for (let i = 0; i < rates.length; i++) {
        if (rates[i] !== prevRates[i]) { changed = true; break; }
      }
    }
    if (changed) {
      kept.push({ t: Date.parse(read.at), players: read.players_tracked, rates: rates.slice() });
      prevPlayers = read.players_tracked;
      prevRates = rates.slice();
    }
  }
  return kept;
}

/** Exponentially decay-weighted least squares of y on x. */
function decayedFit(xs, ys, ws, lastT, times) {
  const halfMs = HALFLIFE_DAYS * DAY_MS;
  const w = times.map((t, i) => ws[i] * Math.exp(-(lastT - t) / halfMs));
  let sw = 0, sx = 0, sy = 0;
  for (let i = 0; i < xs.length; i++) { sw += w[i]; sx += w[i] * xs[i]; sy += w[i] * ys[i]; }
  if (sw <= 1e-9) return null;
  const xbar = sx / sw, ybar = sy / sw;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    const dx = xs[i] - xbar;
    sxx += w[i] * dx * dx;
    sxy += w[i] * dx * (ys[i] - ybar);
  }
  const slope = sxx > 1e-9 ? sxy / sxx : 0;
  const intercept = ybar - slope * xbar;
  let sse = 0;
  for (let i = 0; i < xs.length; i++) {
    const e = ys[i] - (intercept + slope * xs[i]);
    sse += w[i] * e * e;
  }
  let sumW2 = 0;
  for (let i = 0; i < xs.length; i++) sumW2 += w[i] * w[i];
  const neff = sumW2 > 1e-12 ? (sw * sw) / sumW2 : xs.length;
  const s2 = Math.max(S2_FLOOR, sse / Math.max(1.0, neff - 2));
  return { slope, intercept, s2, sxx, neff };
}

function forecastCase(c) {
  if (!Array.isArray(c.trophies) || c.trophies.length === 0) {
    throw new Error(`${c.id}: no trophies`);
  }
  const history = c.history ?? [];
  if (!history.length) throw new Error(`${c.id}: empty history`);

  const refreshes = collapseToRefreshes(history);
  const firstT = refreshes[0].t;
  const lastT = refreshes[refreshes.length - 1].t;
  const targetT = Date.parse(c.target);
  const horizonDays = Math.max(0, (targetT - lastT) / DAY_MS);
  const xOf = (t) => (t - firstT) / DAY_MS;
  const xTarget = xOf(targetT);
  const xLast = xOf(lastT);

  // --- Fit a recent-weighted per-day logit drift for each trophy. ---
  const tierOf = c.trophies.map((t) => t.tier ?? 'bronze');
  const records = c.trophies.map((trophy, i) => {
    const obs = refreshes.filter((p) => p.rates[i] != null && Number.isFinite(p.rates[i]));
    const lastObs = obs.length ? obs[obs.length - 1] : null;
    let lastRate = lastObs ? lastObs.rates[i] : null;
    if (lastRate == null) {
      for (let k = history.length - 1; k >= 0; k--) {
        const rv = history[k].rates;
        if (rv && rv[i] != null && Number.isFinite(rv[i])) { lastRate = rv[i]; break; }
      }
    }
    let fit = null;
    if (obs.length >= 2) {
      const xs = obs.map((p) => xOf(p.t));
      const ys = obs.map((p) => toLogit(p.rates[i]));
      const ts = obs.map((p) => p.t);
      const ones = obs.map(() => 1);
      fit = decayedFit(xs, ys, ones, lastT, ts);
    }
    return { i, tier: tierOf[i], fit, lastRate: lastRate ?? 0, xLastOfTrophy: lastObs ? xOf(lastObs.t) : xLast };
  });

  // --- Learn tier-level drift and pool between-trophy slope variance. ---
  const fitted = records.filter((r) => r.fit !== null);
  const slopeByTier = new Map();
  const tierNames = [...new Set(tierOf)];
  const globalSlope = fitted.length ? average(fitted.map((r) => r.fit.slope)) : 0;
  for (const tier of tierNames) {
    const slopes = fitted.filter((r) => r.tier === tier).map((r) => r.fit.slope);
    slopeByTier.set(tier, slopes.length ? average(slopes) : globalSlope);
  }
  const tierMeanOf = (r) => {
    const v = slopeByTier.get(r.tier);
    return Number.isFinite(v) ? v : globalSlope;
  };
  const slopeDeviations = fitted.map((r) => r.fit.slope - tierMeanOf(r));
  const tau2 = Math.max(TAU2_FLOOR, variance(slopeDeviations));

  // --- Build quantiles. ---
  const rows = records.map((r) => {
    const lLast = toLogit(r.lastRate);
    let centerLogit = lLast;
    let sigma;

    if (r.fit !== null) {
      const tm = tierMeanOf(r);
      const varSlope = r.fit.s2 / Math.max(r.fit.sxx, 1e-6);
      const reliability = tau2 / (tau2 + varSlope);
      const drift = reliability * r.fit.slope + (1 - reliability) * tm;
      const gapDays = Math.max(0, xTarget - r.xLastOfTrophy);
      centerLogit = lLast + drift * gapDays;

      const residSD = Math.sqrt(Math.max(r.fit.s2, S2_FLOOR));
      const trendUnc = Math.min(TREND_UNC_CAP, Math.sqrt(varSlope) * gapDays);
      const horizonTerm = HORIZON_K * Math.sqrt(horizonDays);
      sigma = Math.sqrt(residSD * residSD + trendUnc * trendUnc + horizonTerm * horizonTerm);
    } else {
      // Flat or too-short history: no drift can be learned. Keep persistence
      // but still reserve a visible, horizon-aware spread.
      const horizonTerm = HORIZON_K * Math.sqrt(horizonDays);
      sigma = Math.sqrt(0.50 * 0.50 + horizonTerm * horizonTerm);
    }

    // Rare trophy rates live further out on the logit tail and are noisier.
    const rarity = Math.max(0, -lLast - 2.0);
    sigma *= 1 + Math.min(RARE_CAP, rarity * RARE_K);
    sigma = Math.max(SIGMA_FLOOR, Math.min(SIGMA_CAP, sigma));
    centerLogit = Math.max(LOGIT_LO, Math.min(LOGIT_HI, centerLogit));

    const q = Z_SCORES.map((z) => round4(fromLogit(centerLogit + z * sigma)));
    for (let j = 1; j < q.length; j++) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    for (let j = 0; j < q.length; j++) q[j] = Math.min(100, Math.max(0, q[j]));
    return { id: c.trophies[r.i].id, q };
  });

  return { trophies: rows };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const data = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8'));
  const out = forecastCase(data);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify(out)}\n`);
}
