import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  LineChart, Line, CartesianGrid, Legend
} from 'recharts'
import { Activity, TrendingUp, Users, Zap } from 'lucide-react'

// --- Mock training metrics (replace with real API data in production) ---
const CLASS_METRICS = [
  { class: 'Normal',      f1: 0.91, auc: 0.95, precision: 0.89, recall: 0.93 },
  { class: 'Scoliosis',   f1: 0.84, auc: 0.91, precision: 0.86, recall: 0.82 },
  { class: 'FHP',         f1: 0.88, auc: 0.93, precision: 0.90, recall: 0.86 },
  { class: 'Kyphosis',    f1: 0.82, auc: 0.89, precision: 0.80, recall: 0.84 },
  { class: 'Lordosis',    f1: 0.79, auc: 0.88, precision: 0.77, recall: 0.81 },
  { class: 'Pelvic Tilt', f1: 0.76, auc: 0.85, precision: 0.74, recall: 0.78 },
  { class: 'Genu Valgum', f1: 0.80, auc: 0.87, precision: 0.82, recall: 0.78 },
]

const TRAINING_CURVE = Array.from({ length: 20 }, (_, i) => ({
  epoch: (i + 1) * 5,
  'Train Loss': parseFloat((0.8 * Math.exp(-i * 0.18) + 0.05).toFixed(3)),
  'Val Loss':   parseFloat((0.85 * Math.exp(-i * 0.15) + 0.08).toFixed(3)),
  'Val F1':     parseFloat((0.95 * (1 - Math.exp(-i * 0.2))).toFixed(3)),
}))

const SEA_COMPARISON = [
  { ethnicity: 'East Asian',         'Without SEA': 0.74, 'With SEA': 0.86 },
  { ethnicity: 'South Asian',        'Without SEA': 0.72, 'With SEA': 0.85 },
  { ethnicity: 'Sub-Saharan African','Without SEA': 0.68, 'With SEA': 0.83 },
  { ethnicity: 'European',           'Without SEA': 0.88, 'With SEA': 0.91 },
  { ethnicity: 'Latin American',     'Without SEA': 0.71, 'With SEA': 0.84 },
  { ethnicity: 'Middle Eastern',     'Without SEA': 0.73, 'With SEA': 0.85 },
]

const RADAR_DATA = CLASS_METRICS.map(d => ({ class: d.class, F1: d.f1 * 100, AUC: d.auc * 100 }))

const STATS = [
  { icon: Activity, label: 'Macro F1',   value: '0.829', sub: 'across 7 classes',   color: 'text-spine-accent' },
  { icon: TrendingUp, label: 'Macro AUC', value: '0.897', sub: 'ROC across classes', color: 'text-spine-green' },
  { icon: Zap, label: 'PCKh@0.5',         value: '0.881', sub: '17 keypoints',       color: 'text-spine-amber' },
  { icon: Users, label: 'SEA Δ F1',       value: '+0.120', sub: 'avg equity lift',   color: 'text-spine-red' },
]

const TooltipStyle = {
  contentStyle: {
    background: '#161d27', border: '1px solid #1e2d3d',
    borderRadius: '12px', fontFamily: 'DM Sans', fontSize: 12, color: '#e2eaf4',
  }
}

export default function Dashboard() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <div className="mb-8">
        <p className="label mb-1">Model Performance</p>
        <h1 className="font-display text-4xl text-spine-text">Evaluation Dashboard</h1>
        <p className="text-spine-muted text-sm mt-1">Training metrics · SEA equity analysis · per-class breakdown</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8 stagger">
        {STATS.map(({ icon: Icon, label, value, sub, color }) => (
          <div key={label} className="card p-5">
            <div className="flex items-center gap-2 mb-3">
              <Icon size={15} className={color} />
              <p className="label text-[10px]">{label}</p>
            </div>
            <p className={`font-mono text-3xl font-bold ${color}`}>{value}</p>
            <p className="text-xs text-spine-muted mt-1">{sub}</p>
          </div>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        {/* Per-class F1 bar chart */}
        <div className="card p-5">
          <p className="label mb-4">Per-Class F1 Score</p>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={CLASS_METRICS} margin={{ left: -20 }}>
                <XAxis dataKey="class" tick={{ fill: '#6b8096', fontSize: 10 }} />
                <YAxis domain={[0.6, 1.0]} tick={{ fill: '#6b8096', fontSize: 10 }} />
                <Tooltip {...TooltipStyle} />
                <Bar dataKey="f1" fill="#00d4ff" radius={[4,4,0,0]} name="F1 Score" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Radar: F1 + AUC */}
        <div className="card p-5">
          <p className="label mb-4">F1 vs AUC per Class</p>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={RADAR_DATA}>
                <PolarGrid stroke="#1e2d3d" />
                <PolarAngleAxis dataKey="class" tick={{ fill: '#6b8096', fontSize: 9.5 }} />
                <Radar name="F1" dataKey="F1" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.15} strokeWidth={2} />
                <Radar name="AUC" dataKey="AUC" stroke="#00e5a0" fill="#00e5a0" fillOpacity={0.1} strokeWidth={1.5} />
                <Tooltip {...TooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#6b8096' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Training curves */}
        <div className="card p-5">
          <p className="label mb-4">Training Curves</p>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={TRAINING_CURVE} margin={{ left: -20 }}>
                <CartesianGrid stroke="#1e2d3d" strokeDasharray="3 3" />
                <XAxis dataKey="epoch" tick={{ fill: '#6b8096', fontSize: 10 }} />
                <YAxis tick={{ fill: '#6b8096', fontSize: 10 }} />
                <Tooltip {...TooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#6b8096' }} />
                <Line type="monotone" dataKey="Train Loss" stroke="#6b8096" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="Val Loss" stroke="#ffb84d" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="Val F1" stroke="#00d4ff" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* SEA Equity comparison */}
        <div className="card p-5">
          <p className="label mb-1">SEA Generalizer Equity Lift</p>
          <p className="text-xs text-spine-muted mb-4">F1 score with vs without SEA correction, per ethnicity</p>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={SEA_COMPARISON} margin={{ left: -20 }}>
                <XAxis dataKey="ethnicity" tick={{ fill: '#6b8096', fontSize: 9 }} />
                <YAxis domain={[0.6, 1.0]} tick={{ fill: '#6b8096', fontSize: 10 }} />
                <Tooltip {...TooltipStyle} />
                <Legend wrapperStyle={{ fontSize: 11, color: '#6b8096' }} />
                <Bar dataKey="Without SEA" fill="#6b8096" radius={[3,3,0,0]} />
                <Bar dataKey="With SEA" fill="#00d4ff" radius={[3,3,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Architecture summary */}
      <div className="mt-6 card p-6">
        <p className="label mb-4">Architecture Summary</p>
        <div className="grid md:grid-cols-3 gap-4 text-sm stagger">
          {[
            { title: 'PostureNet Backbone', desc: 'ResNet-34 + channel-wise SE attention gates. Dual output: global feature vector + spatial feature maps.' },
            { title: 'SEA Generalizer Layer ◆', desc: 'Ethnicity-aware normalization of 4 body ratios (THR, SHR, LBP, CLB) against 6 anthropometric baselines. Novel research contribution.' },
            { title: 'Heatmap + Classifier Head', desc: '17 Gaussian keypoint heatmaps via 3× deconvolution. Multi-label FC classifier with sigmoid activation for 7 deformity classes.' },
          ].map(({ title, desc }) => (
            <div key={title} className="bg-spine-surface rounded-xl p-4">
              <p className="text-spine-accent font-medium text-sm mb-2">{title}</p>
              <p className="text-spine-muted text-xs leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
