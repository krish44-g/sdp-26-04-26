import { DeformityResult } from '../api/client'
import { CheckCircle2, AlertTriangle, XCircle } from 'lucide-react'

const DESCRIPTIONS: Record<string, string> = {
  Normal:       'No significant postural deviation detected',
  Scoliosis:    'Lateral spinal curvature (C or S-shaped deviation)',
  FHP:          'Forward Head Posture — cervical lordosis compensation',
  Kyphosis:     'Excessive thoracic curvature (hunched back)',
  Lordosis:     'Exaggerated lumbar inward curve',
  'Pelvic Tilt':'Anterior or posterior pelvic rotation imbalance',
  'Genu Valgum':'Knee valgus — inward knee alignment (knock-knees)',
}

interface Props {
  deformities: DeformityResult[]
}

export default function DeformityBadges({ deformities }: Props) {
  const sorted = [...deformities].sort((a, b) => b.probability - a.probability)

  return (
    <div className="space-y-2 stagger">
      {sorted.map(d => {
        const pct = Math.round(d.probability * 100)
        const isHigh = d.probability >= 0.7
        const isMid = d.probability >= 0.4 && d.probability < 0.7
        const isDetected = d.detected

        return (
          <div
            key={d.name}
            className={`card p-4 transition-all duration-200
              ${isDetected ? 'border-spine-amber/50 bg-spine-amber/5' : 'opacity-60'}`}
          >
            <div className="flex items-start justify-between gap-3 mb-2">
              <div className="flex items-center gap-2">
                {isDetected ? (
                  isHigh ? (
                    <XCircle size={16} className="text-spine-red flex-shrink-0" />
                  ) : (
                    <AlertTriangle size={16} className="text-spine-amber flex-shrink-0" />
                  )
                ) : (
                  <CheckCircle2 size={16} className="text-spine-green flex-shrink-0" />
                )}
                <span className={`font-semibold text-sm ${isDetected ? 'text-spine-text' : 'text-spine-muted'}`}>
                  {d.name}
                </span>
              </div>
              <span className={`text-sm font-mono font-bold ${
                isHigh ? 'text-spine-red' : isMid ? 'text-spine-amber' : 'text-spine-green'
              }`}>
                {pct}%
              </span>
            </div>

            {/* Probability bar */}
            <div className="w-full h-1.5 bg-spine-surface rounded-full overflow-hidden mb-2">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isHigh ? 'bg-spine-red' : isMid ? 'bg-spine-amber' : 'bg-spine-green'
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>

            <p className="text-xs text-spine-muted leading-relaxed">
              {DESCRIPTIONS[d.name]}
            </p>
          </div>
        )
      })}
    </div>
  )
}
