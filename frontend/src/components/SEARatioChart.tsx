import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, Tooltip } from 'recharts'
import { RatioResult } from '../api/client'

interface Props {
  rawRatios: RatioResult
  correctedRatios: RatioResult
  ethnicity: string
}

const RATIO_INFO: Record<string, { label: string; description: string }> = {
  THR: { label: 'Trunk/Height', description: 'Trunk-to-Height Ratio' },
  SHR: { label: 'Shoulder/Hip', description: 'Shoulder-to-Hip Ratio' },
  LBP: { label: 'Leg/Body', description: 'Leg-to-Body Proportion' },
  CLB: { label: 'Cerv/Lumb', description: 'Cervical-Lumbar Balance' },
}

export default function SEARatioChart({ rawRatios, correctedRatios, ethnicity }: Props) {
  const keys = ['THR', 'SHR', 'LBP', 'CLB'] as const

  const radarData = keys.map(k => ({
    ratio: RATIO_INFO[k].label,
    Raw: Number((rawRatios[k] * 100).toFixed(1)),
    Corrected: Number((correctedRatios[k] * 100).toFixed(1)),
  }))

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null
    return (
      <div className="bg-spine-card border border-spine-border rounded-xl px-3 py-2 text-xs">
        <p className="text-spine-text font-medium mb-1">{label}</p>
        {payload.map((p: any) => (
          <p key={p.name} style={{ color: p.color }}>
            {p.name}: {p.value}
          </p>
        ))}
      </div>
    )
  }

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="label mb-0.5">SEA Ratio Analysis</p>
          <p className="text-xs text-spine-muted">Corrected for <span className="text-spine-accent">{ethnicity}</span> anthropometric baseline</p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-px bg-spine-muted inline-block" />
            <span className="text-spine-muted">Raw</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-px bg-spine-accent inline-block" />
            <span className="text-spine-accent">SEA-Corrected</span>
          </span>
        </div>
      </div>

      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid stroke="#1e2d3d" />
            <PolarAngleAxis
              dataKey="ratio"
              tick={{ fill: '#6b8096', fontSize: 11, fontFamily: 'JetBrains Mono' }}
            />
            <Radar name="Raw" dataKey="Raw" stroke="#6b8096" fill="#6b8096" fillOpacity={0.1} strokeWidth={1.5} />
            <Radar name="Corrected" dataKey="Corrected" stroke="#00d4ff" fill="#00d4ff" fillOpacity={0.15} strokeWidth={2} />
            <Tooltip content={<CustomTooltip />} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Ratio table */}
      <div className="mt-4 grid grid-cols-2 gap-2">
        {keys.map(k => {
          const delta = correctedRatios[k] - rawRatios[k]
          const pct = Math.abs(delta / (rawRatios[k] || 1)) * 100
          return (
            <div key={k} className="bg-spine-surface rounded-xl px-3 py-2.5">
              <p className="text-[10px] font-mono text-spine-muted mb-0.5">{RATIO_INFO[k].description}</p>
              <div className="flex items-end gap-1.5">
                <span className="text-lg font-mono text-spine-text font-bold">
                  {correctedRatios[k].toFixed(3)}
                </span>
                <span className={`text-xs font-mono mb-0.5 ${delta >= 0 ? 'text-spine-green' : 'text-spine-red'}`}>
                  {delta >= 0 ? '+' : ''}{pct.toFixed(1)}%
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
