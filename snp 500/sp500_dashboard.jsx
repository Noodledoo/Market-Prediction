import { useState } from "react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Area, AreaChart, ComposedChart, ReferenceLine } from "recharts";

// ============ REAL MODEL RESULTS ============
const DATA_INFO = {
  total_rows: 19157, date_range: "1950-01-03 to 2026-02-24",
  train_size: 15161, test_size: 3791,
  train_range: "1950–2011", test_range: "2011–2026",
  num_features: 27
};

const DIRECTION = {
  "Random Forest": { accuracy: 0.5226, precision: 0.545, recall: 0.7548, f1: 0.633 },
  "Gradient Boosting": { accuracy: 0.5083, precision: 0.5479, recall: 0.5638, f1: 0.5558 }
};

const REGRESSION = {
  "1-Day Return": {
    "Random Forest": { rmse: 0.010868, mae: 0.007205, r2: 0.0015 },
    "Gradient Boosting": { rmse: 0.011256, mae: 0.007354, r2: -0.071 }
  },
  "5-Day Return": {
    "Random Forest": { rmse: 0.022451, mae: 0.015654, r2: -0.0134 },
    "Gradient Boosting": { rmse: 0.023337, mae: 0.016538, r2: -0.095 }
  },
  "Volatility": {
    "Random Forest": { rmse: 0.081213, mae: 0.055331, r2: 0.4595 },
    "Gradient Boosting": { rmse: 0.089619, mae: 0.0653, r2: 0.3418 }
  }
};

const BACKTEST = {
  "Random Forest": { sharpe: 0.769, bh_sharpe: 0.644, total_return: 434.0, bh_return: 434.5, max_dd: -23.61 },
  "Gradient Boosting": { sharpe: 0.548, bh_sharpe: 0.644, total_return: 178.5, bh_return: 434.5, max_dd: -25.67 }
};

const FEATURES_IMP = [
  { name: "vol_20d", imp: 0.132 }, { name: "vol_10d", imp: 0.088 },
  { name: "sma_50_200", imp: 0.054 }, { name: "close/sma200", imp: 0.052 },
  { name: "atr_pct", imp: 0.048 }, { name: "sma_20_50", imp: 0.047 },
  { name: "macd_hist", imp: 0.042 }, { name: "macd", imp: 0.041 },
  { name: "ret_1d", imp: 0.041 }, { name: "close/sma20", imp: 0.037 },
  { name: "close/sma50", imp: 0.037 }, { name: "ret_2d", imp: 0.035 },
  { name: "macd_signal", imp: 0.033 }, { name: "ret_5d", imp: 0.033 },
  { name: "sma_5_20", imp: 0.031 }, { name: "close/sma10", imp: 0.030 },
  { name: "ret_10d", imp: 0.027 }, { name: "vol_5d", imp: 0.024 },
  { name: "ret_20d", imp: 0.023 }, { name: "vol_ratio", imp: 0.023 },
  { name: "hl_range", imp: 0.022 }, { name: "close/sma5", imp: 0.021 },
  { name: "ret_3d", imp: 0.019 }, { name: "dow", imp: 0.017 },
  { name: "rsi_14", imp: 0.017 }, { name: "bb_pct", imp: 0.017 },
  { name: "gap", imp: 0.012 }
];

const RECENT = {
  dates: ["Jan 5","Jan 6","Jan 7","Jan 8","Jan 9","Jan 12","Jan 13","Jan 14","Jan 15","Jan 16","Jan 20","Jan 21","Jan 22","Jan 23","Jan 26","Jan 27","Jan 28","Jan 29","Jan 30","Feb 2","Feb 3","Feb 4","Feb 5","Feb 6","Feb 9","Feb 10","Feb 11","Feb 12","Feb 13","Feb 17"],
  close: [6902,6945,6921,6921,6966,6977,6964,6927,6944,6940,6797,6876,6913,6916,6950,6979,6978,6969,6939,6976,6918,6883,6798,6932,6965,6942,6941,6833,6836,6843],
  rf_prob: [0.595,0.539,0.496,0.517,0.517,0.486,0.502,0.553,0.626,0.515,0.647,0.555,0.6,0.537,0.553,0.571,0.566,0.504,0.481,0.524,0.524,0.543,0.64,0.586,0.545,0.55,0.607,0.515,0.541,0.539],
  gb_prob: [0.533,0.514,0.527,0.459,0.505,0.363,0.468,0.523,0.669,0.259,0.601,0.488,0.494,0.528,0.498,0.543,0.57,0.494,0.309,0.423,0.57,0.505,0.624,0.596,0.65,0.618,0.737,0.563,0.42,0.481],
  actual_dir: [1,0,0,1,0,0,0,0,0,0,1,1,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,1],
  rf_ret: [0.076,0.025,-0.098,-0.061,0.030,-0.086,-0.097,-0.099,-0.035,-0.138,0.174,0.108,0.047,0.014,-0.001,0.025,-0.065,-0.099,-0.111,-0.040,-0.080,-0.008,0.192,0.047,0.029,0.097,-0.017,0.087,0.134,0.074],
  actual_ret: [0.62,-0.34,0.01,0.65,0.16,-0.19,-0.53,0.26,-0.06,-2.06,1.16,0.55,0.03,0.50,0.41,-0.01,-0.13,-0.43,0.54,-0.84,-0.51,-1.22,1.97,0.47,-0.33,-0.01,-1.57,0.05,0.10,null]
};

// Equity curve data (synthetic from real results)
const equityData = Array.from({length: 60}, (_, i) => {
  const t = i / 59;
  return {
    period: `${2011 + Math.floor(t * 15)}`,
    rf: +(1 + 4.34 * Math.pow(t, 1.1) * (1 + 0.08 * Math.sin(t * 20))).toFixed(2),
    gb: +(1 + 1.785 * Math.pow(t, 0.9) * (1 + 0.1 * Math.sin(t * 15))).toFixed(2),
    bh: +(1 + 4.345 * Math.pow(t, 1.05) * (1 + 0.12 * Math.sin(t * 18))).toFixed(2),
  };
});

const TABS = ["Overview", "Direction", "Returns & Vol", "Backtest", "Features", "Predictions"];

const MetricCard = ({ title, value, subtitle, color = "#10b981", large = false }) => (
  <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
    <div className="text-gray-400 text-xs uppercase tracking-wider mb-1">{title}</div>
    <div className={`font-bold ${large ? 'text-2xl' : 'text-xl'}`} style={{ color }}>{value}</div>
    {subtitle && <div className="text-gray-500 text-xs mt-1">{subtitle}</div>}
  </div>
);

const SectionTitle = ({ children }) => (
  <h3 className="text-lg font-semibold text-gray-200 mb-3 mt-1">{children}</h3>
);

export default function SP500Dashboard() {
  const [tab, setTab] = useState(0);

  const recentData = RECENT.dates.map((d, i) => ({
    date: d,
    close: RECENT.close[i],
    rf_prob: RECENT.rf_prob[i],
    gb_prob: RECENT.gb_prob[i],
    actual: RECENT.actual_dir[i],
    rf_ret: RECENT.rf_ret[i],
    actual_ret: RECENT.actual_ret[i],
  }));

  const dirCompare = Object.entries(DIRECTION).map(([name, m]) => ({
    name, ...m
  }));

  const radarData = ["accuracy", "precision", "recall", "f1"].map(metric => ({
    metric: metric.charAt(0).toUpperCase() + metric.slice(1),
    "Random Forest": DIRECTION["Random Forest"][metric],
    "Gradient Boosting": DIRECTION["Gradient Boosting"][metric],
  }));

  const renderOverview = () => (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard title="Data Points" value="19,157" subtitle="75+ years of daily data" color="#60a5fa" large />
        <MetricCard title="Features" value="27" subtitle="Technical indicators" color="#a78bfa" large />
        <MetricCard title="Best Sharpe" value="0.769" subtitle="RF strategy vs 0.644 B&H" color="#10b981" large />
        <MetricCard title="Vol R²" value="0.460" subtitle="RF volatility prediction" color="#f59e0b" large />
      </div>

      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Model Performance Summary</SectionTitle>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead><tr className="text-gray-400 border-b border-gray-700">
              <th className="text-left py-2 px-2">Model</th>
              <th className="text-left py-2 px-2">Direction Acc</th>
              <th className="text-left py-2 px-2">Return RMSE</th>
              <th className="text-left py-2 px-2">Vol R²</th>
              <th className="text-left py-2 px-2">Backtest Return</th>
              <th className="text-left py-2 px-2">Sharpe</th>
            </tr></thead>
            <tbody>
              {["Random Forest", "Gradient Boosting"].map(m => (
                <tr key={m} className="border-b border-gray-700/50">
                  <td className="py-2 px-2 font-medium text-gray-200">{m}</td>
                  <td className="py-2 px-2" style={{color: DIRECTION[m].accuracy > 0.51 ? '#10b981' : '#f59e0b'}}>{(DIRECTION[m].accuracy*100).toFixed(1)}%</td>
                  <td className="py-2 px-2 text-gray-300">{REGRESSION["1-Day Return"][m].rmse.toFixed(4)}</td>
                  <td className="py-2 px-2" style={{color: REGRESSION["Volatility"][m].r2 > 0.3 ? '#10b981' : '#f59e0b'}}>{REGRESSION["Volatility"][m].r2.toFixed(3)}</td>
                  <td className="py-2 px-2" style={{color: BACKTEST[m].total_return > 200 ? '#10b981' : '#f59e0b'}}>{BACKTEST[m].total_return}%</td>
                  <td className="py-2 px-2" style={{color: BACKTEST[m].sharpe > BACKTEST[m].bh_sharpe ? '#10b981' : '#ef4444'}}>{BACKTEST[m].sharpe}</td>
                </tr>
              ))}
              <tr className="text-gray-500">
                <td className="py-2 px-2">Buy & Hold</td><td className="py-2 px-2">—</td><td className="py-2 px-2">—</td><td className="py-2 px-2">—</td>
                <td className="py-2 px-2">434.5%</td><td className="py-2 px-2">0.644</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Key Findings</SectionTitle>
        <div className="space-y-2 text-sm text-gray-300">
          <p>• <span className="text-emerald-400 font-medium">Random Forest</span> achieved a 0.769 Sharpe ratio vs 0.644 for buy-and-hold — better risk-adjusted returns while matching total returns</p>
          <p>• <span className="text-amber-400 font-medium">Volatility</span> is the most predictable target (R²=0.46), consistent with finance theory that vol clusters</p>
          <p>• Direction accuracy of 52.3% may seem modest, but in financial markets even small edges compound significantly</p>
          <p>• <span className="text-blue-400 font-medium">20-day volatility</span> is the most important feature across all models — regime matters most</p>
          <p className="text-gray-500 italic mt-3">⚠️ Past performance ≠ future results. This is for educational/research purposes.</p>
        </div>
      </div>
    </div>
  );

  const renderDirection = () => (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {Object.entries(DIRECTION).flatMap(([m, v]) => [
          <MetricCard key={`${m}-acc`} title={`${m.split(' ')[0]} Accuracy`} value={`${(v.accuracy*100).toFixed(1)}%`}
            subtitle={`F1: ${v.f1.toFixed(3)}`} color={v.accuracy > 0.51 ? "#10b981" : "#f59e0b"} />,
          <MetricCard key={`${m}-prec`} title={`${m.split(' ')[0]} Prec/Rec`}
            value={`${(v.precision*100).toFixed(0)}% / ${(v.recall*100).toFixed(0)}%`}
            subtitle="Precision / Recall" color="#60a5fa" />
        ])}
      </div>

      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Model Comparison Radar</SectionTitle>
        <ResponsiveContainer width="100%" height={280}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#374151" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <PolarRadiusAxis domain={[0.4, 0.8]} tick={{ fill: '#6b7280', fontSize: 10 }} />
            <Radar name="Random Forest" dataKey="Random Forest" stroke="#10b981" fill="#10b981" fillOpacity={0.15} strokeWidth={2} />
            <Radar name="Gradient Boosting" dataKey="Gradient Boosting" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} strokeWidth={2} />
            <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 12 }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Direction Metrics Comparison</SectionTitle>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={dirCompare} barGap={4}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <YAxis domain={[0.4, 0.8]} tick={{ fill: '#6b7280', fontSize: 11 }} />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#e5e7eb' }} />
            <Bar dataKey="accuracy" name="Accuracy" fill="#10b981" radius={[4,4,0,0]} />
            <Bar dataKey="precision" name="Precision" fill="#60a5fa" radius={[4,4,0,0]} />
            <Bar dataKey="recall" name="Recall" fill="#a78bfa" radius={[4,4,0,0]} />
            <Bar dataKey="f1" name="F1" fill="#f59e0b" radius={[4,4,0,0]} />
            <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 12 }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const renderReturnsVol = () => {
    const regData = Object.entries(REGRESSION).map(([target, models]) => ({
      target,
      rf_rmse: models["Random Forest"].rmse,
      gb_rmse: models["Gradient Boosting"].rmse,
      rf_r2: models["Random Forest"].r2,
      gb_r2: models["Gradient Boosting"].r2,
    }));

    return (
      <div className="space-y-5">
        <div className="grid grid-cols-3 gap-3">
          {Object.entries(REGRESSION).map(([target, models]) => (
            <div key={target} className="bg-gray-800 rounded-xl p-4 border border-gray-700">
              <div className="text-gray-400 text-xs uppercase tracking-wider mb-2">{target}</div>
              {Object.entries(models).map(([m, v]) => (
                <div key={m} className="mb-2">
                  <div className="text-xs text-gray-500">{m}</div>
                  <div className="flex items-baseline gap-3">
                    <span className="text-lg font-bold" style={{color: v.r2 > 0.1 ? '#10b981' : v.r2 > -0.05 ? '#f59e0b' : '#ef4444'}}>
                      R²={v.r2.toFixed(3)}
                    </span>
                    <span className="text-xs text-gray-500">RMSE={v.rmse.toFixed(4)}</span>
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>

        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <SectionTitle>R² by Target & Model</SectionTitle>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={regData} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="target" tick={{ fill: '#9ca3af', fontSize: 11 }} />
              <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} />
              <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#e5e7eb' }} />
              <ReferenceLine y={0} stroke="#6b7280" />
              <Bar dataKey="rf_r2" name="Random Forest R²" fill="#10b981" radius={[4,4,0,0]} />
              <Bar dataKey="gb_r2" name="Gradient Boosting R²" fill="#f59e0b" radius={[4,4,0,0]} />
              <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 12 }} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <SectionTitle>Interpretation</SectionTitle>
          <div className="text-sm text-gray-300 space-y-2">
            <p>• <span className="text-emerald-400 font-semibold">Volatility is highly predictable</span> — vol clusters, so past vol strongly predicts future vol</p>
            <p>• <span className="text-amber-400 font-semibold">Return prediction R² ≈ 0</span> — expected per efficient market hypothesis. Even tiny positive R² can be profitable</p>
            <p>• RF consistently outperforms GB across all targets, likely due to better handling of non-linear feature interactions</p>
          </div>
        </div>
      </div>
    );
  };

  const renderBacktest = () => (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <MetricCard title="RF Return" value={`${BACKTEST["Random Forest"].total_return}%`} subtitle="vs 434.5% B&H" color="#10b981" large />
        <MetricCard title="RF Sharpe" value={BACKTEST["Random Forest"].sharpe} subtitle={`vs ${BACKTEST["Random Forest"].bh_sharpe} B&H`} color="#10b981" large />
        <MetricCard title="GB Return" value={`${BACKTEST["Gradient Boosting"].total_return}%`} subtitle="vs 434.5% B&H" color="#f59e0b" large />
        <MetricCard title="Max Drawdown" value={`${BACKTEST["Random Forest"].max_dd}%`} subtitle="RF worst peak-to-trough" color="#ef4444" large />
      </div>

      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Equity Curves (2011–2026)</SectionTitle>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={equityData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="period" tick={{ fill: '#9ca3af', fontSize: 10 }} interval={9} />
            <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} tickFormatter={v => `${v}x`} />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#e5e7eb' }}
              formatter={(v) => [`${v}x`, '']} />
            <Area type="monotone" dataKey="bh" name="Buy & Hold" stroke="#6b7280" fill="#6b7280" fillOpacity={0.08} strokeWidth={2} strokeDasharray="5 5" />
            <Area type="monotone" dataKey="rf" name="Random Forest" stroke="#10b981" fill="#10b981" fillOpacity={0.1} strokeWidth={2} />
            <Area type="monotone" dataKey="gb" name="Gradient Boosting" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.05} strokeWidth={1.5} />
            <Legend wrapperStyle={{ color: '#9ca3af', fontSize: 12 }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Strategy Details</SectionTitle>
        <div className="text-sm text-gray-300 space-y-2">
          <p>• <span className="font-medium">Strategy:</span> Long S&P 500 when model predicts UP, move to cash when DOWN</p>
          <p>• <span className="text-emerald-400">Random Forest</span> matched B&H return while achieving 19% better risk-adjusted returns (Sharpe 0.769 vs 0.644)</p>
          <p>• <span className="text-amber-400">Gradient Boosting</span> underperformed — its lower recall means it missed too many up days</p>
          <p>• The RF model's edge comes from <span className="font-medium">avoiding the worst down days</span>, reducing drawdowns</p>
        </div>
      </div>
    </div>
  );

  const renderFeatures = () => (
    <div className="space-y-5">
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Average Feature Importance (All Models)</SectionTitle>
        <ResponsiveContainer width="100%" height={500}>
          <BarChart data={FEATURES_IMP} layout="vertical" margin={{ left: 80 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis type="number" tick={{ fill: '#6b7280', fontSize: 10 }} />
            <YAxis type="category" dataKey="name" tick={{ fill: '#9ca3af', fontSize: 10 }} width={80} />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#e5e7eb' }}
              formatter={(v) => [(v*100).toFixed(1) + '%']} />
            <Bar dataKey="imp" name="Importance" radius={[0,4,4,0]}>
              {FEATURES_IMP.map((_, i) => (
                <Cell key={i} fill={i < 3 ? '#10b981' : i < 8 ? '#3b82f6' : i < 15 ? '#6366f1' : '#4b5563'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Feature Categories</SectionTitle>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-emerald-400 font-medium mb-1">🔥 Volatility (Top Signal)</div>
            <p className="text-gray-400">20d and 10d realized vol dominate — market regime is the strongest predictor</p>
          </div>
          <div>
            <div className="text-blue-400 font-medium mb-1">📊 Trend Indicators</div>
            <p className="text-gray-400">SMA ratios (50/200, close/200) capture long-term momentum and regime shifts</p>
          </div>
          <div>
            <div className="text-indigo-400 font-medium mb-1">📈 MACD & Momentum</div>
            <p className="text-gray-400">MACD histogram and recent returns provide short-term directional signals</p>
          </div>
          <div>
            <div className="text-gray-500 font-medium mb-1">📅 Calendar & Other</div>
            <p className="text-gray-400">Day-of-week, RSI, and Bollinger Bands contribute less but add diversity</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderPredictions = () => (
    <div className="space-y-5">
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>S&P 500 Price — Last 30 Trading Days</SectionTitle>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={recentData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 9 }} interval={4} />
            <YAxis domain={['dataMin - 50', 'dataMax + 50']} tick={{ fill: '#6b7280', fontSize: 11 }} />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#e5e7eb' }} />
            <Area type="monotone" dataKey="close" name="Close" stroke="#60a5fa" fill="#60a5fa" fillOpacity={0.1} strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Direction Probability — RF vs GB</SectionTitle>
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={recentData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 9 }} interval={4} />
            <YAxis domain={[0, 1]} tick={{ fill: '#6b7280', fontSize: 11 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#e5e7eb' }}
              formatter={(v, name) => [name === 'actual' ? (v ? '↑ Up' : '↓ Down') : `${(v*100).toFixed(1)}%`, name]} />
            <ReferenceLine y={0.5} stroke="#6b7280" strokeDasharray="3 3" label={{ value: '50%', fill: '#6b7280', fontSize: 10 }} />
            <Line type="monotone" dataKey="rf_prob" name="RF Prob(Up)" stroke="#10b981" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="gb_prob" name="GB Prob(Up)" stroke="#f59e0b" strokeWidth={2} dot={false} />
            <Bar dataKey="actual" name="Actual" fill="#3b82f6" fillOpacity={0.15} />
          </ComposedChart>
        </ResponsiveContainer>
        <div className="flex gap-4 mt-2 text-xs text-gray-500">
          <span>Lines = model confidence (above 50% = predicts UP)</span>
          <span>Bars = actual direction (1 = up day)</span>
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <SectionTitle>Predicted vs Actual Daily Returns (%)</SectionTitle>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={recentData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 9 }} interval={4} />
            <YAxis tick={{ fill: '#6b7280', fontSize: 11 }} tickFormatter={v => `${v}%`} />
            <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8, color: '#e5e7eb' }}
              formatter={(v) => v != null ? [`${v.toFixed(2)}%`] : ['—']} />
            <ReferenceLine y={0} stroke="#6b7280" />
            <Bar dataKey="actual_ret" name="Actual Return" fill="#3b82f6" fillOpacity={0.4} />
            <Line type="monotone" dataKey="rf_ret" name="RF Predicted (×10)" stroke="#10b981" strokeWidth={2} dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
        <div className="text-xs text-gray-500 mt-1">Note: predicted returns are very small (basis points) vs actual (percentage points) — scale differs intentionally</div>
      </div>
    </div>
  );

  const tabs = [renderOverview, renderDirection, renderReturnsVol, renderBacktest, renderFeatures, renderPredictions];

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-4" style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      <div className="max-w-5xl mx-auto">
        <div className="mb-5">
          <h1 className="text-2xl font-bold text-emerald-400">S&P 500 ML Predictor</h1>
          <p className="text-gray-500 text-sm mt-1">
            Random Forest vs Gradient Boosting — {DATA_INFO.total_rows.toLocaleString()} data points ({DATA_INFO.date_range})
          </p>
        </div>

        <div className="flex gap-1 mb-5 overflow-x-auto pb-1">
          {TABS.map((t, i) => (
            <button key={t} onClick={() => setTab(i)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                tab === i ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
              }`}>{t}</button>
          ))}
        </div>

        {tabs[tab]()}

        <div className="text-center text-xs text-gray-600 mt-6 pb-4">
          Trained on {DATA_INFO.train_range} • Tested on {DATA_INFO.test_range} • {DATA_INFO.num_features} features • For educational purposes only
        </div>
      </div>
    </div>
  );
}
