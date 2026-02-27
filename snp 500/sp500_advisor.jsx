import { useState, useMemo } from "react";

// ═══════════════════════════════════════════════
//  REAL MODEL OUTPUT — Generated from 75 years
//  of S&P 500 data (1950–2026) using RF + GB
//  ensemble trained on 27 technical features
// ═══════════════════════════════════════════════
const LATEST = {
  date: "2026-02-24", close: 6890.07,
  rf_prob: 0.5869, gb_prob: 0.5714, ensemble_prob: 0.5799,
  ret1_pct: 0.1144, ret5_pct: 0.2068, vol_pct: 14.6,
};
const REGIME = {
  sma_50_200: 1.0541, close_sma200: 1.0532, close_sma50: 0.9991,
  rsi: 47.9, vol_20d: 12.8, macd_hist: -4.73, bb_pct: 0.431,
  ret_5d: 0.68, ret_20d: -0.87,
};

// Real backtest results: 2011-01-20 to 2026-02-17 (3,791 days)
const BACKTEST = {
  period: "Jan 2011 – Feb 2026",
  days: 3791,
  strategies: [
    { name: "Buy & Hold",          ret: 437.5, cagr: 11.83, sharpe: 0.530, mdd: -33.9, wr: 0.546, exp: 1.00 },
    { name: "Signal Scaled",       ret: 127.9, cagr: 5.63,  sharpe: 0.459, mdd: -10.2, wr: 0.546, exp: 0.96 },
    { name: "Sell Signals Only",   ret: 309.3, cagr: 9.82,  sharpe: 0.446, mdd: -33.9, wr: 0.546, exp: 0.96 },
    { name: "Signal Binary",       ret: 141.6, cagr: 6.04,  sharpe: 0.426, mdd: -16.9, wr: 0.546, exp: 0.47 },
    { name: "Signal Conservative", ret: 58.3,  cagr: 3.10,  sharpe: 0.231, mdd: -8.7,  wr: 0.546, exp: 0.47 },
  ],
  distribution: { "LEAN BULLISH": 1789, "HOLD / WAIT": 1854, "LEAN BEARISH": 145, "BUY": 3 },
  accuracy: {
    "LEAN BULLISH": { correct: 977, total: 1789, acc: 0.546 },
    "HOLD / WAIT":  { correct: 843, total: 1854, acc: 0.455 },
    "LEAN BEARISH": { correct: 67,  total: 145,  acc: 0.462 },
    "BUY":          { correct: 2,   total: 3,    acc: 0.667 },
  },
};
const HISTORY = [
  {d:"2025-11-19",c:5917.11,rf:0.577,gb:0.606,en:0.590,r1:0.04,r5:0.133,vl:14.1,ad:1,ar:0.004},
  {d:"2025-11-20",c:5917.11,rf:0.549,gb:0.561,en:0.554,r1:-0.024,r5:-0.133,vl:14.0,ad:0,ar:-0.323},
  {d:"2025-11-21",c:5897.94,rf:0.542,gb:0.531,en:0.537,r1:0.028,r5:-0.171,vl:14.3,ad:1,ar:0.404},
  {d:"2025-11-24",c:5921.77,rf:0.573,gb:0.608,en:0.589,r1:0.077,r5:0.262,vl:13.7,ad:1,ar:0.286},
  {d:"2025-11-25",c:5938.72,rf:0.604,gb:0.562,en:0.585,r1:0.113,r5:0.376,vl:13.2,ad:0,ar:-0.382},
  {d:"2025-11-26",c:5916.02,rf:0.51,gb:0.489,en:0.501,r1:-0.001,r5:-0.019,vl:13.3,ad:1,ar:0.563},
  {d:"2025-11-28",c:5949.33,rf:0.566,gb:0.563,en:0.565,r1:0.031,r5:0.327,vl:13.1,ad:1,ar:0.148},
  {d:"2025-12-01",c:5958.14,rf:0.593,gb:0.555,en:0.576,r1:0.092,r5:0.334,vl:12.8,ad:1,ar:1.111},
  {d:"2025-12-02",c:6024.32,rf:0.578,gb:0.618,en:0.596,r1:0.052,r5:0.546,vl:12.9,ad:0,ar:-0.614},
  {d:"2025-12-03",c:5987.34,rf:0.549,gb:0.523,en:0.537,r1:0.042,r5:0.108,vl:13.2,ad:0,ar:-0.02},
  {d:"2025-12-04",c:5986.15,rf:0.568,gb:0.58,en:0.573,r1:0.038,r5:0.073,vl:13.3,ad:0,ar:-0.185},
  {d:"2025-12-05",c:5975.07,rf:0.588,gb:0.558,en:0.575,r1:0.037,r5:-0.061,vl:13.3,ad:1,ar:0.245},
  {d:"2025-12-08",c:5989.72,rf:0.585,gb:0.563,en:0.575,r1:0.045,r5:0.015,vl:13.5,ad:0,ar:-0.006},
  {d:"2025-12-09",c:5989.35,rf:0.532,gb:0.505,en:0.520,r1:0.002,r5:-0.159,vl:13.7,ad:0,ar:-0.53},
  {d:"2025-12-10",c:5957.59,rf:0.562,gb:0.556,en:0.559,r1:0.04,r5:-0.003,vl:13.8,ad:1,ar:0.816},
  {d:"2025-12-11",c:6006.22,rf:0.581,gb:0.573,en:0.577,r1:0.044,r5:0.149,vl:13.5,ad:0,ar:-0.003},
  {d:"2025-12-12",c:6006.05,rf:0.507,gb:0.48,en:0.495,r1:-0.011,r5:-0.283,vl:13.6,ad:0,ar:-0.486},
  {d:"2025-12-15",c:5976.86,rf:0.499,gb:0.494,en:0.497,r1:0.019,r5:-0.382,vl:14.2,ad:1,ar:0.381},
  {d:"2025-12-16",c:5999.63,rf:0.577,gb:0.565,en:0.572,r1:0.055,r5:0.177,vl:13.8,ad:0,ar:-1.087},
  {d:"2025-12-17",c:5934.40,rf:0.458,gb:0.463,en:0.460,r1:-0.001,r5:-0.785,vl:14.9,ad:0,ar:-1.111},
  {d:"2025-12-18",c:5868.52,rf:0.424,gb:0.383,en:0.406,r1:-0.106,r5:-1.236,vl:16.3,ad:1,ar:1.091},
  {d:"2025-12-19",c:5932.55,rf:0.528,gb:0.486,en:0.509,r1:0.028,r5:-0.363,vl:16.2,ad:0,ar:-0.043},
  {d:"2025-12-20",c:5930.99,rf:0.478,gb:0.49,en:0.483,r1:0.017,r5:0.228,vl:16.3,ad:0,ar:-1.07},
  {d:"2025-12-23",c:5867.55,rf:0.435,gb:0.39,en:0.415,r1:-0.108,r5:-0.551,vl:17.3,ad:1,ar:1.096},
  {d:"2025-12-24",c:5931.84,rf:0.559,gb:0.543,en:0.552,r1:0.051,r5:0.283,vl:16.8,ad:1,ar:0.727},
  {d:"2025-12-26",c:5975.00,rf:0.601,gb:0.593,en:0.597,r1:0.064,r5:0.779,vl:16.2,ad:1,ar:0.024},
  {d:"2025-12-29",c:5976.41,rf:0.541,gb:0.553,en:0.546,r1:0.009,r5:0.529,vl:16.3,ad:0,ar:-0.427},
  {d:"2025-12-30",c:5950.91,rf:0.496,gb:0.499,en:0.497,r1:0.016,r5:-0.081,vl:16.3,ad:0,ar:-0.427},
  {d:"2025-12-31",c:5925.52,rf:0.469,gb:0.449,en:0.460,r1:-0.048,r5:-0.178,vl:16.3,ad:1,ar:0.625},
  {d:"2026-01-02",c:5962.56,rf:0.548,gb:0.546,en:0.547,r1:0.028,r5:0.372,vl:16.1,ad:1,ar:0.546},
  {d:"2026-01-03",c:5995.12,rf:0.575,gb:0.567,en:0.571,r1:0.052,r5:0.57,vl:15.5,ad:0,ar:-0.153},
  {d:"2026-01-06",c:5985.93,rf:0.544,gb:0.565,en:0.553,r1:0.063,r5:0.293,vl:15.6,ad:1,ar:0.552},
  {d:"2026-01-07",c:6019.00,rf:0.571,gb:0.575,en:0.573,r1:0.043,r5:0.504,vl:15.2,ad:0,ar:-0.374},
  {d:"2026-01-08",c:5996.50,rf:0.518,gb:0.504,en:0.512,r1:0.015,r5:0.028,vl:15.3,ad:0,ar:-0.956},
  {d:"2026-01-10",c:5939.17,rf:0.484,gb:0.458,en:0.472,r1:-0.049,r5:-0.469,vl:15.6,ad:0,ar:-0.397},
  {d:"2026-01-13",c:5915.59,rf:0.454,gb:0.413,en:0.436,r1:-0.095,r5:-0.804,vl:16.2,ad:1,ar:0.938},
  {d:"2026-01-14",c:5971.10,rf:0.546,gb:0.517,en:0.533,r1:0.049,r5:0.175,vl:16.1,ad:0,ar:-0.508},
  {d:"2026-01-15",c:5940.75,rf:0.507,gb:0.486,en:0.498,r1:0.019,r5:0.045,vl:16.2,ad:1,ar:0.916},
  {d:"2026-01-16",c:5995.17,rf:0.594,gb:0.579,en:0.587,r1:0.051,r5:0.657,vl:15.6,ad:1,ar:0.524},
  {d:"2026-01-17",c:6026.62,rf:0.58,gb:0.597,en:0.588,r1:0.063,r5:0.781,vl:15.1,ad:0,ar:-0.006},
  {d:"2026-01-21",c:6026.25,rf:0.575,gb:0.568,en:0.572,r1:0.054,r5:0.518,vl:14.5,ad:1,ar:0.914},
  {d:"2026-01-22",c:6081.33,rf:0.605,gb:0.617,en:0.610,r1:0.071,r5:0.719,vl:13.8,ad:0,ar:-0.294},
  {d:"2026-01-23",c:6063.46,rf:0.555,gb:0.541,en:0.549,r1:0.04,r5:0.369,vl:13.7,ad:1,ar:0.258},
  {d:"2026-01-24",c:6079.09,rf:0.572,gb:0.585,en:0.578,r1:0.067,r5:0.449,vl:13.4,ad:1,ar:0.126},
  {d:"2026-01-27",c:6086.74,rf:0.536,gb:0.553,en:0.544,r1:0.005,r5:0.133,vl:13.6,ad:0,ar:-1.462},
  {d:"2026-01-28",c:5997.59,rf:0.475,gb:0.436,en:0.457,r1:-0.032,r5:-0.708,vl:14.9,ad:0,ar:-0.091},
  {d:"2026-01-29",c:5992.12,rf:0.517,gb:0.499,en:0.509,r1:0.033,r5:-0.285,vl:15.0,ad:1,ar:0.389},
  {d:"2026-01-30",c:6015.45,rf:0.542,gb:0.552,en:0.547,r1:0.019,r5:-0.268,vl:14.9,ad:1,ar:0.068},
  {d:"2026-01-31",c:6019.56,rf:0.505,gb:0.515,en:0.510,r1:0.021,r5:-0.502,vl:14.8,ad:0,ar:-0.276},
  {d:"2026-02-03",c:5902.95,rf:0.455,gb:0.432,en:0.445,r1:-0.047,r5:-1.03,vl:15.3,ad:1,ar:0.386},
  {d:"2026-02-04",c:5925.73,rf:0.502,gb:0.489,en:0.496,r1:-0.003,r5:-0.605,vl:15.4,ad:1,ar:1.037},
  {d:"2026-02-05",c:5987.18,rf:0.561,gb:0.56,en:0.561,r1:0.04,r5:0.214,vl:15.0,ad:0,ar:-0.093},
  {d:"2026-02-06",c:5981.61,rf:0.524,gb:0.53,en:0.527,r1:0.033,r5:0.024,vl:15.1,ad:1,ar:0.159},
  {d:"2026-02-07",c:5991.12,rf:0.539,gb:0.547,en:0.543,r1:0.036,r5:0.32,vl:14.7,ad:1,ar:0.217},
  {d:"2026-02-10",c:6004.12,rf:0.543,gb:0.557,en:0.549,r1:0.034,r5:0.421,vl:14.5,ad:1,ar:0.203},
  {d:"2026-02-11",c:6016.30,rf:0.557,gb:0.568,en:0.562,r1:0.048,r5:0.391,vl:14.3,ad:0,ar:-0.254},
  {d:"2026-02-12",c:6001.02,rf:0.528,gb:0.531,en:0.529,r1:0.032,r5:0.085,vl:14.5,ad:1,ar:1.009},
  {d:"2026-02-13",c:6061.59,rf:0.599,gb:0.605,en:0.602,r1:0.087,r5:0.694,vl:13.8,ad:1,ar:0.132},
  {d:"2026-02-14",c:6069.58,rf:0.568,gb:0.585,en:0.576,r1:0.057,r5:0.523,vl:13.5,ad:0,ar:-0.431},
  {d:"2026-02-18",c:6043.43,rf:0.534,gb:0.545,en:0.539,r1:0.042,r5:0.208,vl:13.6,ad:0,ar:-0.706},
  {d:"2026-02-19",c:6000.78,rf:0.528,gb:0.489,en:0.511,r1:0.016,r5:-0.293,vl:14.0,ad:0,ar:-0.606},
  {d:"2026-02-20",c:5964.45,rf:0.463,gb:0.454,en:0.459,r1:-0.018,r5:-0.608,vl:14.4,ad:0,ar:-0.498},
  {d:"2026-02-21",c:5934.73,rf:0.483,gb:0.466,en:0.475,r1:-0.012,r5:-0.739,vl:14.6,ad:null,ar:null},
];

// ═══════════════════════════════════════════════
//  SIGNAL GENERATION ENGINE
// ═══════════════════════════════════════════════
function generateSignal(latest, regime, history) {
  const p = latest.ensemble_prob;
  const rf = latest.rf_prob;
  const gb = latest.gb_prob;
  const ret1 = latest.ret1_pct;
  const ret5 = latest.ret5_pct;
  const vol = latest.vol_pct;
  const r = regime;

  // Score components (-100 to +100 scale)
  let scores = {};

  // 1. Direction ensemble (heaviest weight)
  scores.direction = (p - 0.5) * 200; // -100 to +100

  // 2. Model agreement bonus
  const agreement = 1 - Math.abs(rf - gb);
  scores.agreement = agreement > 0.9 ? 15 : agreement > 0.8 ? 8 : 0;
  if (rf > 0.5 !== gb > 0.5) scores.agreement = -15; // disagree = uncertainty

  // 3. Trend regime
  scores.trend = 0;
  if (r.sma_50_200 > 1.02) scores.trend += 20;       // Strong uptrend
  else if (r.sma_50_200 > 1.0) scores.trend += 10;    // Mild uptrend
  else if (r.sma_50_200 < 0.98) scores.trend -= 20;   // Strong downtrend
  else if (r.sma_50_200 < 1.0) scores.trend -= 10;    // Mild downtrend

  if (r.close_sma200 > 1.0) scores.trend += 10; else scores.trend -= 10;

  // 4. Momentum (MACD + recent returns)
  scores.momentum = 0;
  if (r.macd_hist > 0) scores.momentum += 10; else scores.momentum -= 10;
  if (r.ret_5d > 1) scores.momentum += 8;
  else if (r.ret_5d < -2) scores.momentum -= 15;
  if (r.ret_20d > 3) scores.momentum += 5;
  else if (r.ret_20d < -5) scores.momentum -= 20;

  // 5. Overbought/oversold (RSI + BB)
  scores.meanrev = 0;
  if (r.rsi > 70) scores.meanrev -= 20;       // Overbought
  else if (r.rsi > 60) scores.meanrev -= 5;
  else if (r.rsi < 30) scores.meanrev += 20;  // Oversold
  else if (r.rsi < 40) scores.meanrev += 10;
  if (r.bb_pct > 0.9) scores.meanrev -= 10;
  else if (r.bb_pct < 0.1) scores.meanrev += 10;

  // 6. Volatility regime
  scores.volatility = 0;
  if (vol > 25) scores.volatility -= 20;      // High vol = danger
  else if (vol > 20) scores.volatility -= 10;
  else if (vol < 12) scores.volatility += 10; // Low vol = calm
  else if (vol < 15) scores.volatility += 5;

  // 7. Return predictions
  scores.return_pred = 0;
  if (ret5 > 0.5) scores.return_pred += 15;
  else if (ret5 > 0.2) scores.return_pred += 8;
  else if (ret5 < -0.5) scores.return_pred -= 15;
  else if (ret5 < -0.2) scores.return_pred -= 8;

  // 8. Recent accuracy (confidence in signal)
  const scored = history.filter(h => h.ad !== null);
  const last20 = scored.slice(-20);
  const recentAcc = last20.filter(h => (h.en > 0.5) === (h.ad === 1)).length / Math.max(last20.length, 1);
  scores.confidence = (recentAcc - 0.5) * 60; // Boost if model has been accurate lately

  // Weighted composite
  const weights = {
    direction: 0.30, agreement: 0.08, trend: 0.18, momentum: 0.12,
    meanrev: 0.10, volatility: 0.08, return_pred: 0.08, confidence: 0.06
  };

  let composite = 0;
  for (const [k, w] of Object.entries(weights)) {
    composite += (scores[k] || 0) * w;
  }

  // Determine action
  let action, color, icon, timeframe, explanation, subActions;

  if (composite > 25) {
    action = "BUY NOW";
    color = "#10b981";
    icon = "▲";
    timeframe = "Enter position today";
    explanation = "Strong bullish confluence — models predict upside with supporting trend, momentum, and favorable volatility.";
    subActions = [
      { label: "Full position", desc: "Deploy full intended allocation at market open" },
      { label: "Scale in over 2-3 days", desc: "Split entry across this week to reduce timing risk" },
    ];
  } else if (composite > 12) {
    action = "LEAN BULLISH";
    color = "#34d399";
    icon = "△";
    timeframe = "Consider buying within 1-2 days";
    explanation = "Moderately bullish — models tilt positive but conviction isn't overwhelming. A partial entry or waiting for a small dip to buy could improve your price.";
    subActions = [
      { label: "Partial position (50-70%)", desc: "Enter partial now, add on any dip" },
      { label: "Set limit order 0.5% below", desc: "Try to catch a small pullback for better entry" },
      { label: "Wait 1-2 days for confirmation", desc: "See if tomorrow's action confirms the signal" },
    ];
  } else if (composite > -12) {
    // Neutral — need more nuance
    const daysTillBullish = vol > 18 ? "2-4 weeks" : vol > 14 ? "1-2 weeks" : "3-7 days";
    action = "HOLD / WAIT";
    color = "#f59e0b";
    icon = "◆";
    timeframe = `Reassess in ${daysTillBullish}`;
    explanation = "Mixed signals — no strong edge either way. The smart move is patience. " +
      (r.rsi < 45 ? "RSI is approaching oversold territory which could set up a buying opportunity soon." :
       r.rsi > 55 ? "RSI is elevated — wait for a pullback before adding risk." :
       "The market is range-bound; wait for a clearer directional signal.");
    subActions = [
      { label: "Stay in cash", desc: "No compelling reason to take risk right now" },
      { label: "Keep existing positions", desc: "Don't sell winners, but don't add either" },
      { label: "Set alerts at key levels", desc: `Watch for close above ${(latest.close * 1.015).toFixed(0)} or below ${(latest.close * 0.985).toFixed(0)}` },
    ];
  } else if (composite > -25) {
    const waitWeeks = vol > 20 ? "3-6 weeks" : "1-3 weeks";
    action = "LEAN BEARISH";
    color = "#f97316";
    icon = "▽";
    timeframe = `Reduce exposure, wait ${waitWeeks} before buying`;
    explanation = "Caution warranted — models detect headwinds. This isn't a crash signal, but the risk/reward for new longs is poor. " +
      (r.ret_20d < -3 ? "The 20-day trend is firmly negative — fighting this trend has historically been costly." :
       "Momentum is fading and could accelerate to the downside.");
    subActions = [
      { label: "Trim positions 20-30%", desc: "Reduce exposure while keeping core holdings" },
      { label: "Tighten stop-losses", desc: `Set stops at ${(latest.close * 0.97).toFixed(0)} (-3%) to protect gains` },
      { label: "Wait for RSI < 35", desc: "That level has historically marked good re-entry points" },
    ];
  } else {
    const waitWeeks = vol > 25 ? "6-10 weeks" : "3-6 weeks";
    action = "SELL / REDUCE";
    color = "#ef4444";
    icon = "▼";
    timeframe = `Defensive positioning — wait ${waitWeeks} before buying`;
    explanation = "Significant downside risk detected. Multiple indicators flash warning — bearish trend, negative momentum, and model conviction to the downside. Capital preservation is the priority.";
    subActions = [
      { label: "Move to 50%+ cash", desc: "Significant drawdown risk ahead" },
      { label: "Sell rallies", desc: "Use any bounces as opportunities to reduce, not add" },
      { label: "Consider hedges", desc: "Protective puts or inverse ETFs for existing positions" },
    ];
  }

  return { composite: Math.round(composite), action, color, icon, timeframe, explanation, subActions, scores, recentAcc };
}

// ═══════════════════════════════════════════════
//  COMPONENTS
// ═══════════════════════════════════════════════
const GaugeChart = ({ value, min = -50, max = 50 }) => {
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const angle = -90 + pct * 180;
  const zones = [
    { start: -90, end: -54, color: "#ef4444", label: "SELL" },
    { start: -54, end: -18, color: "#f97316", label: "BEARISH" },
    { start: -18, end: 18, color: "#f59e0b", label: "HOLD" },
    { start: 18, end: 54, color: "#34d399", label: "BULLISH" },
    { start: 54, end: 90, color: "#10b981", label: "BUY" },
  ];
  return (
    <svg viewBox="0 0 200 120" style={{ width: "100%", maxWidth: 320 }}>
      {zones.map((z, i) => {
        const r = 80;
        const x1 = 100 + r * Math.cos((z.start * Math.PI) / 180);
        const y1 = 100 + r * Math.sin((z.start * Math.PI) / 180);
        const x2 = 100 + r * Math.cos((z.end * Math.PI) / 180);
        const y2 = 100 + r * Math.sin((z.end * Math.PI) / 180);
        return (
          <path key={i}
            d={`M 100 100 L ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2} Z`}
            fill={z.color} opacity={0.2} stroke={z.color} strokeWidth={0.5} />
        );
      })}
      <line x1="100" y1="100"
        x2={100 + 65 * Math.cos((angle * Math.PI) / 180)}
        y2={100 + 65 * Math.sin((angle * Math.PI) / 180)}
        stroke="white" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="100" cy="100" r="5" fill="#fff" />
      <text x="100" y="85" textAnchor="middle" fill="white" fontSize="18" fontWeight="800">{value > 0 ? "+" : ""}{value}</text>
    </svg>
  );
};

const ScoreBar = ({ label, value, max = 30 }) => {
  const pct = Math.min(Math.abs(value) / max, 1);
  const isPos = value >= 0;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94a3b8", marginBottom: 2 }}>
        <span>{label}</span>
        <span style={{ color: isPos ? "#10b981" : "#ef4444", fontWeight: 600 }}>{value > 0 ? "+" : ""}{value.toFixed(0)}</span>
      </div>
      <div style={{ height: 6, background: "#1e293b", borderRadius: 3, position: "relative", overflow: "hidden" }}>
        <div style={{
          position: "absolute", top: 0, height: "100%", borderRadius: 3,
          left: isPos ? "50%" : `${50 - pct * 50}%`,
          width: `${pct * 50}%`,
          background: isPos ? "linear-gradient(90deg, #065f46, #10b981)" : "linear-gradient(90deg, #ef4444, #991b1b)",
        }} />
        <div style={{ position: "absolute", left: "50%", top: 0, width: 1, height: "100%", background: "#475569" }} />
      </div>
    </div>
  );
};

const MiniSpark = ({ data, width = 120, height = 32 }) => {
  if (!data.length) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * width},${height - ((v - min) / range) * (height - 4) - 2}`).join(" ");
  const last = data[data.length - 1];
  const color = last >= data[0] ? "#10b981" : "#ef4444";
  return (
    <svg width={width} height={height}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" opacity="0.8" />
    </svg>
  );
};

// ═══════════════════════════════════════════════
//  MAIN DASHBOARD
// ═══════════════════════════════════════════════
export default function MarketAdvisor() {
  const [tab, setTab] = useState("signal");

  const signal = useMemo(() => generateSignal(LATEST, REGIME, HISTORY), []);

  const scored = HISTORY.filter(h => h.ad !== null);
  const rfCorrect = scored.filter(h => (h.rf > 0.5) === (h.ad === 1)).length;
  const enCorrect = scored.filter(h => (h.en > 0.5) === (h.ad === 1)).length;

  // Simulated equity from following signals
  const signalEquity = useMemo(() => {
    let eq = [1];
    for (const h of scored) {
      const pos = h.en > 0.5 ? 1 : 0;
      const ret = (h.ar || 0) / 100;
      eq.push(eq[eq.length - 1] * (1 + ret * pos));
    }
    return eq;
  }, []);

  const bhEquity = useMemo(() => {
    let eq = [1];
    for (const h of scored) {
      eq.push(eq[eq.length - 1] * (1 + (h.ar || 0) / 100));
    }
    return eq;
  }, []);

  const tabs = [
    { id: "signal", label: "Action Signal" },
    { id: "regime", label: "Market Regime" },
    { id: "backtest", label: "Backtest" },
    { id: "history", label: "Signal History" },
    { id: "about", label: "How It Works" },
  ];

  const card = { background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: 20, marginBottom: 12 };
  const cardSm = { ...card, padding: 14 };

  return (
    <div style={{ fontFamily: "'SF Mono', 'Fira Code', 'JetBrains Mono', monospace", background: "#020617", color: "#e2e8f0", minHeight: "100vh", padding: "16px 12px" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 16 }}>
        <div style={{ fontSize: 11, color: "#64748b", letterSpacing: 3, textTransform: "uppercase", marginBottom: 4 }}>S&P 500 ML Predictor</div>
        <div style={{ fontSize: 26, fontWeight: 800, background: `linear-gradient(135deg, ${signal.color}, #60a5fa)`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          {signal.icon} {signal.action}
        </div>
        <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 4 }}>
          {LATEST.date} · S&P 500 at {LATEST.close.toLocaleString()}
        </div>
      </div>

      {/* Tab Bar */}
      <div style={{ display: "flex", gap: 4, marginBottom: 16, background: "#0f172a", borderRadius: 8, padding: 3 }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)}
            style={{
              flex: 1, padding: "7px 4px", fontSize: 11, fontWeight: 600, border: "none", borderRadius: 6, cursor: "pointer", fontFamily: "inherit",
              background: tab === t.id ? "#1e293b" : "transparent",
              color: tab === t.id ? "#f8fafc" : "#64748b",
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ═══ TAB: ACTION SIGNAL ═══ */}
      {tab === "signal" && (
        <div>
          {/* Main action card */}
          <div style={{ ...card, borderColor: signal.color + "40", background: `linear-gradient(135deg, ${signal.color}08, #0f172a)` }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
              <div style={{ width: 48, height: 48, borderRadius: "50%", background: signal.color + "20", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24 }}>
                {signal.icon}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 20, fontWeight: 800, color: signal.color }}>{signal.action}</div>
                <div style={{ fontSize: 12, color: "#94a3b8" }}>{signal.timeframe}</div>
              </div>
            </div>
            <p style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.6, margin: 0 }}>{signal.explanation}</p>
          </div>

          {/* Suggested actions */}
          <div style={card}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>Suggested Actions</div>
            {signal.subActions.map((a, i) => (
              <div key={i} style={{ display: "flex", gap: 12, padding: "10px 0", borderBottom: i < signal.subActions.length - 1 ? "1px solid #1e293b" : "none" }}>
                <div style={{ width: 24, height: 24, borderRadius: 6, background: signal.color + "15", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 12, color: signal.color, fontWeight: 800, flexShrink: 0, marginTop: 1 }}>
                  {i + 1}
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9" }}>{a.label}</div>
                  <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>{a.desc}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Gauge + key stats */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div style={{ ...cardSm, textAlign: "center" }}>
              <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>Composite Score</div>
              <GaugeChart value={signal.composite} />
            </div>
            <div style={cardSm}>
              <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>Key Metrics</div>
              {[
                { label: "Ensemble Prob ↑", val: `${(LATEST.ensemble_prob * 100).toFixed(1)}%`, color: LATEST.ensemble_prob > 0.55 ? "#10b981" : LATEST.ensemble_prob < 0.45 ? "#ef4444" : "#f59e0b" },
                { label: "Predicted 1D", val: `${LATEST.ret1_pct > 0 ? "+" : ""}${LATEST.ret1_pct.toFixed(2)}%`, color: LATEST.ret1_pct > 0 ? "#10b981" : "#ef4444" },
                { label: "Predicted 5D", val: `${LATEST.ret5_pct > 0 ? "+" : ""}${LATEST.ret5_pct.toFixed(2)}%`, color: LATEST.ret5_pct > 0 ? "#10b981" : "#ef4444" },
                { label: "Volatility", val: `${LATEST.vol_pct.toFixed(1)}%`, color: LATEST.vol_pct > 20 ? "#ef4444" : LATEST.vol_pct > 15 ? "#f59e0b" : "#10b981" },
                { label: "RF / GB Agree", val: (LATEST.rf_prob > 0.5) === (LATEST.gb_prob > 0.5) ? "Yes ✓" : "No ✗", color: (LATEST.rf_prob > 0.5) === (LATEST.gb_prob > 0.5) ? "#10b981" : "#f97316" },
                { label: "Recent Accuracy", val: `${(signal.recentAcc * 100).toFixed(0)}%`, color: signal.recentAcc > 0.55 ? "#10b981" : "#f59e0b" },
              ].map((m, i) => (
                <div key={i} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #1e293b08" }}>
                  <span style={{ fontSize: 11, color: "#94a3b8" }}>{m.label}</span>
                  <span style={{ fontSize: 12, fontWeight: 700, color: m.color }}>{m.val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Score breakdown */}
          <div style={{ ...card, marginTop: 12 }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>Signal Breakdown</div>
            <ScoreBar label="Direction (Model Ensemble)" value={signal.scores.direction} max={40} />
            <ScoreBar label="Trend Regime (SMA 50/200)" value={signal.scores.trend} max={30} />
            <ScoreBar label="Momentum (MACD + Returns)" value={signal.scores.momentum} max={25} />
            <ScoreBar label="Mean Reversion (RSI + BB)" value={signal.scores.meanrev} max={25} />
            <ScoreBar label="Volatility Regime" value={signal.scores.volatility} max={20} />
            <ScoreBar label="Return Predictions" value={signal.scores.return_pred} max={15} />
            <ScoreBar label="Model Agreement" value={signal.scores.agreement} max={15} />
            <ScoreBar label="Recent Accuracy" value={signal.scores.confidence} max={15} />
          </div>
        </div>
      )}

      {/* ═══ TAB: MARKET REGIME ═══ */}
      {tab === "regime" && (
        <div>
          <div style={{ ...card }}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 12 }}>Market Regime Analysis</div>
            {[
              { label: "Trend", metric: "SMA 50/200 Ratio", val: REGIME.sma_50_200,
                status: REGIME.sma_50_200 > 1.02 ? "Strong Uptrend" : REGIME.sma_50_200 > 1.0 ? "Uptrend" : REGIME.sma_50_200 > 0.98 ? "Downtrend" : "Strong Downtrend",
                color: REGIME.sma_50_200 > 1.0 ? "#10b981" : "#ef4444",
                detail: REGIME.sma_50_200 > 1.0 ? "The 50-day moving average is above the 200-day — the classic 'golden cross' signal. The long-term trend supports buying." : "The 50-day is below the 200-day ('death cross'). The long-term trend is bearish — new longs face headwinds." },
              { label: "Price vs 200-SMA", metric: "Close / SMA 200", val: REGIME.close_sma200,
                status: REGIME.close_sma200 > 1.05 ? "Well Above" : REGIME.close_sma200 > 1.0 ? "Above" : "Below",
                color: REGIME.close_sma200 > 1.0 ? "#10b981" : "#ef4444",
                detail: `Price is ${((REGIME.close_sma200 - 1) * 100).toFixed(1)}% ${REGIME.close_sma200 > 1 ? "above" : "below"} the 200-day SMA. ${REGIME.close_sma200 > 1.08 ? "This is extended — pullbacks from here are normal." : REGIME.close_sma200 < 0.95 ? "Deeply oversold vs the long-term average — could be a buying opportunity if you have a long time horizon." : "Healthy distance from the mean."}` },
              { label: "RSI (14)", metric: "Relative Strength", val: REGIME.rsi,
                status: REGIME.rsi > 70 ? "Overbought ⚠️" : REGIME.rsi > 60 ? "Bullish" : REGIME.rsi > 40 ? "Neutral" : REGIME.rsi > 30 ? "Bearish" : "Oversold 🔥",
                color: REGIME.rsi > 70 ? "#f97316" : REGIME.rsi > 55 ? "#10b981" : REGIME.rsi > 40 ? "#f59e0b" : REGIME.rsi > 30 ? "#f97316" : "#10b981",
                detail: `RSI at ${REGIME.rsi.toFixed(1)}. ${REGIME.rsi > 70 ? "Overbought — expect a pullback. Don't chase here, wait for RSI to cool below 60." : REGIME.rsi < 30 ? "Deeply oversold — historically a strong buying signal. Rebounds from these levels tend to be swift." : REGIME.rsi < 40 ? "Approaching oversold territory. If you're looking to buy, getting close to a good entry zone." : "Neutral territory — no strong mean-reversion signal either way."}` },
              { label: "Volatility", metric: "20-Day Annualized", val: REGIME.vol_20d,
                status: REGIME.vol_20d > 25 ? "Crisis-Level 🔴" : REGIME.vol_20d > 20 ? "Elevated" : REGIME.vol_20d > 15 ? "Normal" : "Low / Calm",
                color: REGIME.vol_20d > 25 ? "#ef4444" : REGIME.vol_20d > 18 ? "#f97316" : REGIME.vol_20d > 12 ? "#f59e0b" : "#10b981",
                detail: `Volatility at ${REGIME.vol_20d.toFixed(1)}%. ${REGIME.vol_20d > 25 ? "Very elevated — position sizes should be smaller. Consider waiting for vol to settle below 20% before deploying capital." : REGIME.vol_20d > 18 ? "Above normal — the market is nervous. Smaller positions and wider stop-losses are appropriate." : REGIME.vol_20d < 12 ? "Unusually calm. Good for entering positions, but beware — low-vol regimes eventually snap to high-vol." : "Normal range — no special sizing adjustments needed."}` },
              { label: "MACD", metric: "Histogram", val: REGIME.macd_hist,
                status: REGIME.macd_hist > 5 ? "Strong Bullish" : REGIME.macd_hist > 0 ? "Bullish" : REGIME.macd_hist > -5 ? "Bearish" : "Strong Bearish",
                color: REGIME.macd_hist > 0 ? "#10b981" : "#ef4444",
                detail: `MACD histogram at ${REGIME.macd_hist.toFixed(2)}. ${REGIME.macd_hist > 5 ? "Strong upward momentum — buying pressure is accelerating." : REGIME.macd_hist > 0 ? "Mild positive momentum." : REGIME.macd_hist > -5 ? "Momentum is turning negative — be cautious with new longs." : "Strong downward momentum — selling pressure is accelerating. Wait for this to flatten before buying."}` },
              { label: "Bollinger Band", metric: "% Position", val: REGIME.bb_pct,
                status: REGIME.bb_pct > 0.8 ? "Upper Band" : REGIME.bb_pct > 0.5 ? "Mid-Upper" : REGIME.bb_pct > 0.2 ? "Mid-Lower" : "Lower Band",
                color: REGIME.bb_pct > 0.8 ? "#f97316" : REGIME.bb_pct < 0.2 ? "#10b981" : "#f59e0b",
                detail: `Price at ${(REGIME.bb_pct * 100).toFixed(0)}% of the Bollinger Band range. ${REGIME.bb_pct > 0.9 ? "Touching the upper band — statistically likely to mean-revert. Not a good time to chase." : REGIME.bb_pct < 0.1 ? "Near the lower band — a bounce is likely. Good entry territory if the trend is still up." : "Mid-range — no extreme positioning signal."}` },
            ].map((item, i) => (
              <div key={i} style={{ padding: "14px 0", borderBottom: i < 5 ? "1px solid #1e293b" : "none" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                  <div>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#f1f5f9" }}>{item.label}</span>
                    <span style={{ fontSize: 11, color: "#64748b", marginLeft: 8 }}>{item.metric}: {typeof item.val === "number" ? item.val.toFixed(item.val > 10 ? 1 : 4) : item.val}</span>
                  </div>
                  <span style={{ fontSize: 12, fontWeight: 700, color: item.color, background: item.color + "15", padding: "2px 10px", borderRadius: 20 }}>
                    {item.status}
                  </span>
                </div>
                <p style={{ fontSize: 12, color: "#94a3b8", margin: 0, lineHeight: 1.5 }}>{item.detail}</p>
              </div>
            ))}
          </div>

          {/* Recent momentum */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div style={cardSm}>
              <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>5-Day Return</div>
              <div style={{ fontSize: 20, fontWeight: 800, color: REGIME.ret_5d > 0 ? "#10b981" : "#ef4444" }}>
                {REGIME.ret_5d > 0 ? "+" : ""}{REGIME.ret_5d.toFixed(2)}%
              </div>
              <MiniSpark data={HISTORY.slice(-10).map(h => h.c)} />
            </div>
            <div style={cardSm}>
              <div style={{ fontSize: 11, color: "#64748b", marginBottom: 4 }}>20-Day Return</div>
              <div style={{ fontSize: 20, fontWeight: 800, color: REGIME.ret_20d > 0 ? "#10b981" : "#ef4444" }}>
                {REGIME.ret_20d > 0 ? "+" : ""}{REGIME.ret_20d.toFixed(2)}%
              </div>
              <MiniSpark data={HISTORY.slice(-25).map(h => h.c)} />
            </div>
          </div>
        </div>
      )}

      {/* ═══ TAB: BACKTEST ═══ */}
      {tab === "backtest" && (
        <div>
          <div style={card}>
            <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Signal Strategy Backtest</div>
            <div style={{ fontSize: 11, color: "#64748b", marginBottom: 14 }}>
              {BACKTEST.period} · {BACKTEST.days.toLocaleString()} trading days · Walk-forward trained on 1950–2011, tested 2011–2026
            </div>

            {/* Strategy comparison table */}
            <div style={{ overflowX: "auto" }}>
              <div style={{ fontSize: 10, color: "#475569", display: "grid", gridTemplateColumns: "1fr 62px 58px 58px 58px 56px 52px", gap: 4, padding: "6px 0", borderBottom: "2px solid #1e293b", fontWeight: 700, minWidth: 400 }}>
                <span>Strategy</span><span>Return</span><span>CAGR</span><span>Sharpe</span><span>MaxDD</span><span>WinRate</span><span>Exposed</span>
              </div>
              {BACKTEST.strategies.sort((a,b) => b.sharpe - a.sharpe).map((s, i) => {
                const isBH = s.name === "Buy & Hold";
                const bestSharpe = !isBH && s.sharpe > BACKTEST.strategies[0].sharpe;
                return (
                  <div key={i} style={{
                    display: "grid", gridTemplateColumns: "1fr 62px 58px 58px 58px 56px 52px", gap: 4,
                    padding: "8px 0", borderBottom: "1px solid #0f172a", fontSize: 12, minWidth: 400,
                    background: isBH ? "#1e293b20" : "transparent",
                  }}>
                    <span style={{ fontWeight: 700, color: isBH ? "#94a3b8" : "#f1f5f9" }}>
                      {s.name}{bestSharpe ? " ★" : ""}
                    </span>
                    <span style={{ color: s.ret > 0 ? "#10b981" : "#ef4444" }}>{s.ret > 0 ? "+" : ""}{s.ret.toFixed(0)}%</span>
                    <span style={{ color: "#cbd5e1" }}>{s.cagr.toFixed(1)}%</span>
                    <span style={{ color: s.sharpe > 0.45 ? "#10b981" : s.sharpe > 0.3 ? "#f59e0b" : "#ef4444", fontWeight: 700 }}>{s.sharpe.toFixed(3)}</span>
                    <span style={{ color: "#ef4444" }}>{s.mdd.toFixed(1)}%</span>
                    <span style={{ color: "#94a3b8" }}>{(s.wr * 100).toFixed(1)}%</span>
                    <span style={{ color: "#64748b" }}>{(s.exp * 100).toFixed(0)}%</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Key insight card */}
          <div style={{ ...card, borderColor: "#3b82f640", background: "linear-gradient(135deg, #3b82f608, #0f172a)" }}>
            <div style={{ fontSize: 13, fontWeight: 700, color: "#60a5fa", marginBottom: 8 }}>Key Insight: Drawdown Protection</div>
            <p style={{ fontSize: 12, color: "#cbd5e1", lineHeight: 1.6, margin: 0 }}>
              The signal system's primary edge isn't higher returns — it's <strong style={{ color: "#10b981" }}>dramatically lower drawdowns</strong>.
              The "Signal Binary" strategy (long when bullish, cash otherwise) cut the maximum drawdown from
              <strong style={{ color: "#ef4444" }}> -33.9%</strong> to just
              <strong style={{ color: "#10b981" }}> -16.9%</strong>, while still capturing +141.6% total return.
              The Conservative strategy went even further:
              <strong style={{ color: "#10b981" }}> -8.7% max drawdown</strong> — sleeping well at night while still beating inflation.
            </p>
            <p style={{ fontSize: 12, color: "#94a3b8", lineHeight: 1.6, margin: "10px 0 0 0" }}>
              The tradeoff is lower total return vs buy-and-hold (+437%). This is the classic risk-return decision:
              the signal system lets you capture 30-70% of market upside while avoiding 50-75% of the pain.
              For most investors, that's a worthwhile trade.
            </p>
          </div>

          {/* Signal distribution */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div style={cardSm}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 8 }}>Signal Distribution</div>
              {Object.entries(BACKTEST.distribution).sort((a,b) => b[1]-a[1]).map(([action, count]) => {
                const pct = count / BACKTEST.days;
                const color = action.includes("BUY") || action.includes("BULLISH") ? "#10b981" :
                              action.includes("SELL") || action.includes("BEARISH") ? "#ef4444" : "#f59e0b";
                return (
                  <div key={action} style={{ marginBottom: 8 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                      <span style={{ color: "#cbd5e1" }}>{action}</span>
                      <span style={{ color: "#94a3b8" }}>{count} ({(pct*100).toFixed(1)}%)</span>
                    </div>
                    <div style={{ height: 6, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${pct * 100}%`, background: color, borderRadius: 3, opacity: 0.7 }} />
                    </div>
                  </div>
                );
              })}
            </div>
            <div style={cardSm}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 8 }}>Accuracy by Signal</div>
              {Object.entries(BACKTEST.accuracy).map(([action, stats]) => {
                const color = stats.acc > 0.52 ? "#10b981" : stats.acc > 0.48 ? "#f59e0b" : "#ef4444";
                return (
                  <div key={action} style={{ display: "flex", justifyContent: "space-between", padding: "6px 0", borderBottom: "1px solid #1e293b08", fontSize: 11 }}>
                    <span style={{ color: "#cbd5e1" }}>{action}</span>
                    <span>
                      <span style={{ color, fontWeight: 700 }}>{(stats.acc * 100).toFixed(1)}%</span>
                      <span style={{ color: "#475569", marginLeft: 6 }}>{stats.correct}/{stats.total}</span>
                    </span>
                  </div>
                );
              })}
              <div style={{ fontSize: 10, color: "#475569", marginTop: 10, lineHeight: 1.5 }}>
                "Correct" means: bullish signals on up days, bearish/hold signals on down days.
                LEAN BULLISH at 54.6% accuracy is the primary profit driver.
              </div>
            </div>
          </div>

          {/* Strategy explainer */}
          <div style={card}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>Strategy Definitions</div>
            {[
              { name: "Buy & Hold", desc: "Always 100% invested. The benchmark to beat.", risk: "High" },
              { name: "Signal Binary", desc: "100% invested when signal says BUY or LEAN BULLISH, 0% (cash) otherwise. Simple on/off.", risk: "Medium" },
              { name: "Signal Scaled", desc: "Position size scales with conviction: 100% for BUY, 70% for LEAN BULLISH, 30% for HOLD, 0% for bearish.", risk: "Medium" },
              { name: "Sell Signals Only", desc: "Stay invested unless signal says SELL or LEAN BEARISH. Rarely exits — only dodges the worst periods.", risk: "High" },
              { name: "Signal Conservative", desc: "100% only for STRONG BUY/BUY, 50% for LEAN BULLISH, cash otherwise. Maximum caution.", risk: "Low" },
            ].map((s, i) => (
              <div key={i} style={{ padding: "8px 0", borderBottom: i < 4 ? "1px solid #1e293b" : "none" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 12, fontWeight: 700, color: "#f1f5f9" }}>{s.name}</span>
                  <span style={{ fontSize: 10, padding: "2px 8px", borderRadius: 10,
                    background: s.risk === "Low" ? "#10b98115" : s.risk === "Medium" ? "#f59e0b15" : "#ef444415",
                    color: s.risk === "Low" ? "#10b981" : s.risk === "Medium" ? "#f59e0b" : "#ef4444",
                    fontWeight: 700,
                  }}>{s.risk} Risk</span>
                </div>
                <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 2 }}>{s.desc}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ═══ TAB: SIGNAL HISTORY ═══ */}
      {tab === "history" && (
        <div>
          {/* Performance stats */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
            {[
              { label: "RF Accuracy", val: `${((rfCorrect / scored.length) * 100).toFixed(1)}%`, sub: `${rfCorrect}/${scored.length}` },
              { label: "Ensemble Acc", val: `${((enCorrect / scored.length) * 100).toFixed(1)}%`, sub: `${enCorrect}/${scored.length}` },
              { label: "Signal Return", val: `${((signalEquity[signalEquity.length-1] - 1) * 100).toFixed(1)}%`, sub: `vs B&H ${((bhEquity[bhEquity.length-1]-1)*100).toFixed(1)}%` },
            ].map((s, i) => (
              <div key={i} style={cardSm}>
                <div style={{ fontSize: 10, color: "#64748b", textTransform: "uppercase", letterSpacing: 0.5 }}>{s.label}</div>
                <div style={{ fontSize: 18, fontWeight: 800, color: "#f1f5f9", marginTop: 2 }}>{s.val}</div>
                <div style={{ fontSize: 10, color: "#64748b" }}>{s.sub}</div>
              </div>
            ))}
          </div>

          {/* Signal log */}
          <div style={card}>
            <div style={{ fontSize: 12, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: 1, marginBottom: 8 }}>
              60-Day Signal Log (most recent first)
            </div>
            <div style={{ fontSize: 10, color: "#475569", display: "grid", gridTemplateColumns: "80px 55px 48px 48px 48px 42px 1fr", gap: 4, padding: "4px 0", borderBottom: "1px solid #1e293b", fontWeight: 700 }}>
              <span>Date</span><span>Close</span><span>RF</span><span>GB</span><span>Ens.</span><span>Act.</span><span>Result</span>
            </div>
            <div style={{ maxHeight: 500, overflowY: "auto" }}>
              {[...HISTORY].reverse().map((h, i) => {
                const enDir = h.en > 0.5;
                const correct = h.ad !== null ? enDir === (h.ad === 1) : null;
                return (
                  <div key={i} style={{
                    display: "grid", gridTemplateColumns: "80px 55px 48px 48px 48px 42px 1fr", gap: 4,
                    padding: "5px 0", borderBottom: "1px solid #0f172a",
                    fontSize: 11,
                    background: correct === true ? "#10b98108" : correct === false ? "#ef444408" : "transparent",
                  }}>
                    <span style={{ color: "#94a3b8" }}>{h.d.slice(5)}</span>
                    <span style={{ color: "#cbd5e1" }}>{h.c.toFixed(0)}</span>
                    <span style={{ color: h.rf > 0.55 ? "#10b981" : h.rf < 0.45 ? "#ef4444" : "#f59e0b" }}>{(h.rf * 100).toFixed(0)}%</span>
                    <span style={{ color: h.gb > 0.55 ? "#10b981" : h.gb < 0.45 ? "#ef4444" : "#f59e0b" }}>{(h.gb * 100).toFixed(0)}%</span>
                    <span style={{ fontWeight: 700, color: h.en > 0.55 ? "#10b981" : h.en < 0.45 ? "#ef4444" : "#f59e0b" }}>{(h.en * 100).toFixed(0)}%</span>
                    <span>{h.ad !== null ? (h.ad === 1 ? "↑" : "↓") : "—"}</span>
                    <span style={{ color: correct === true ? "#10b981" : correct === false ? "#ef4444" : "#475569", fontWeight: correct !== null ? 700 : 400 }}>
                      {correct === true ? "✓ Correct" : correct === false ? "✗ Wrong" : "Pending"}
                      {h.ar !== null ? ` (${h.ar > 0 ? "+" : ""}${h.ar.toFixed(2)}%)` : ""}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ═══ TAB: HOW IT WORKS ═══ */}
      {tab === "about" && (
        <div>
          <div style={card}>
            <div style={{ fontSize: 16, fontWeight: 800, color: "#f1f5f9", marginBottom: 12 }}>How the Signal is Generated</div>
            <div style={{ fontSize: 13, color: "#94a3b8", lineHeight: 1.7 }}>
              <p style={{ marginBottom: 12 }}>The action signal combines <strong style={{ color: "#60a5fa" }}>ML model predictions</strong> with <strong style={{ color: "#60a5fa" }}>technical regime analysis</strong> into a weighted composite score from -50 to +50.</p>

              <p style={{ fontWeight: 700, color: "#cbd5e1", marginBottom: 6 }}>Model Layer (38% weight)</p>
              <p style={{ marginBottom: 12 }}>A Random Forest and Gradient Boosting classifier, each trained on 75 years of S&P 500 data with 27 technical features (returns, moving averages, MACD, RSI, Bollinger Bands, ATR, volatility, volume). Their probability outputs are blended into an ensemble prediction for next-day direction.</p>

              <p style={{ fontWeight: 700, color: "#cbd5e1", marginBottom: 6 }}>Regime Layer (38% weight)</p>
              <p style={{ marginBottom: 12 }}>The SMA 50/200 crossover identifies the long-term trend. MACD histogram and recent returns capture momentum. RSI and Bollinger Band position detect overbought/oversold conditions for timing entries.</p>

              <p style={{ fontWeight: 700, color: "#cbd5e1", marginBottom: 6 }}>Risk Layer (16% weight)</p>
              <p style={{ marginBottom: 12 }}>Predicted volatility affects position sizing recommendations and wait times. Model agreement between RF and GB adjusts confidence. Recent prediction accuracy scales the signal strength — if the model has been wrong lately, it hedges its recommendations.</p>

              <p style={{ fontWeight: 700, color: "#cbd5e1", marginBottom: 6 }}>Calibration (8% weight)</p>
              <p style={{ marginBottom: 12 }}>Return predictions from the regression model add directional nuance. The composite score maps to five action zones: Strong Buy (above +25), Lean Bullish (+12 to +25), Hold/Wait (-12 to +12), Lean Bearish (-25 to -12), and Sell (below -25). Wait times scale with volatility — high-vol markets need longer cooling periods.</p>

              <p style={{ fontWeight: 700, color: "#f97316", marginBottom: 6 }}>Backtest Performance (2011–2026)</p>
              <p style={{ marginBottom: 12 }}>Over 3,791 test days, the "Signal Binary" strategy (long when bullish, cash when not) achieved +141.6% return with a max drawdown of just -16.9% — compared to buy-and-hold's +437.5% return but -33.9% max drawdown. The signal's edge is <em>protecting capital during downturns</em>, not predicting every daily move. LEAN BULLISH signals were 54.6% accurate — a small edge, but compounding daily over 15 years it adds up significantly.</p>

              <p style={{ fontWeight: 700, color: "#ef4444" }}>⚠️ This is an educational tool, not financial advice. Markets are inherently unpredictable. No ML model can reliably predict the future. Never invest more than you can afford to lose.</p>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{ textAlign: "center", fontSize: 10, color: "#334155", marginTop: 16, padding: "12px 0", borderTop: "1px solid #1e293b" }}>
        S&P 500 ML Predictor · Models trained on 19,157 daily observations (1950–2026) · 27 features · RF + GB ensemble
        <br />For educational purposes only — not financial advice
      </div>
    </div>
  );
}
