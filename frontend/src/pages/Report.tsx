import { useLocation, useNavigate } from 'react-router-dom'
import { ReportResponse, AnalysisResponse } from '../api/client'
import {
  CheckCircle2, AlertTriangle, XCircle, ChevronLeft,
  Printer, Activity, Brain, ClipboardList, ArrowRight
} from 'lucide-react'

const SEVERITY_MAP: Record<string, { cls: string; icon: typeof CheckCircle2 }> = {
  low:      { cls: 'text-spine-green border-spine-green/30 bg-spine-green/5',  icon: CheckCircle2 },
  moderate: { cls: 'text-spine-amber border-spine-amber/30 bg-spine-amber/5',  icon: AlertTriangle },
  high:     { cls: 'text-spine-red   border-spine-red/30   bg-spine-red/5',    icon: XCircle },
}

export default function Report() {
  const { state } = useLocation()
  const navigate = useNavigate()
  const report: ReportResponse | undefined = state?.report
  const result: AnalysisResponse | undefined = state?.result

  if (!report) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-spine-muted">
        <p className="mb-4">No report found. Run an analysis first.</p>
        <button onClick={() => navigate('/')} className="btn-ghost">Go back</button>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="flex items-center justify-between mb-8 flex-wrap gap-4">
        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate(-1)}
            className="p-2 rounded-lg border border-spine-border text-spine-muted hover:text-spine-text hover:border-spine-accent/40 transition-all"
          >
            <ChevronLeft size={18} />
          </button>
          <div>
            <p className="label mb-0.5">AI Clinical Report</p>
            <p className="text-xs text-spine-muted font-mono">{report.report_id}</p>
          </div>
        </div>
        <button
          onClick={() => window.print()}
          className="btn-ghost flex items-center gap-2 text-sm"
        >
          <Printer size={14} />
          Print / Export
        </button>
      </div>

      {/* Summary banner */}
      <div className="card p-6 mb-6 border-spine-accent/30 bg-spine-accent/5 glow-accent">
        <div className="flex items-start gap-3">
          <Brain size={20} className="text-spine-accent flex-shrink-0 mt-0.5" />
          <div>
            <p className="label mb-2">Clinical Summary</p>
            <p className="text-spine-text leading-relaxed">{report.summary}</p>
          </div>
        </div>
      </div>

      {/* Detected conditions */}
      {report.detected_conditions.length > 0 && (
        <section className="mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Activity size={16} className="text-spine-accent" />
            <h2 className="font-display text-xl text-spine-text">Detected Conditions</h2>
          </div>
          <div className="space-y-3 stagger">
            {report.detected_conditions.map((cond, i) => {
              const sev = SEVERITY_MAP[cond.severity] || SEVERITY_MAP.moderate
              const Icon = sev.icon
              return (
                <div key={i} className={`card p-5 border ${sev.cls}`}>
                  <div className="flex items-start justify-between gap-4 mb-3">
                    <div className="flex items-center gap-2">
                      <Icon size={16} />
                      <h3 className="font-semibold text-spine-text">{cond.name}</h3>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className={`text-xs font-mono px-2 py-0.5 rounded-lg border ${sev.cls}`}>
                        {cond.severity} severity
                      </span>
                      <span className="text-sm font-mono text-spine-muted">
                        {Math.round(cond.probability * 100)}%
                      </span>
                    </div>
                  </div>
                  {/* Severity bar */}
                  <div className="w-full h-1 bg-spine-surface rounded-full mb-3">
                    <div
                      className="h-full rounded-full bg-current opacity-60 transition-all"
                      style={{ width: `${cond.severity_score * 100}%` }}
                    />
                  </div>
                  <p className="text-sm text-spine-muted mb-2 leading-relaxed">{cond.description}</p>
                  <p className="text-xs text-spine-text leading-relaxed border-t border-spine-border/50 pt-2 mt-2">
                    <span className="text-spine-accent font-medium">Clinical significance: </span>
                    {cond.clinical_significance}
                  </p>
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* Postural analysis */}
      <section className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <ClipboardList size={16} className="text-spine-accent" />
          <h2 className="font-display text-xl text-spine-text">Postural Analysis</h2>
        </div>
        <div className="grid md:grid-cols-2 gap-3">
          {Object.entries(report.postural_analysis).map(([key, value]) => (
            <div key={key} className="card p-4">
              <p className="label mb-1">{key.replace(/_/g, ' ')}</p>
              <p className="text-sm text-spine-text leading-relaxed">{value}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Recommendations */}
      <section className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <ArrowRight size={16} className="text-spine-accent" />
          <h2 className="font-display text-xl text-spine-text">Recommendations</h2>
        </div>
        <div className="card p-5 space-y-3 stagger">
          {report.recommendations.map((rec, i) => (
            <div key={i} className="flex items-start gap-3">
              <span className="w-5 h-5 rounded-full bg-spine-accent/20 text-spine-accent text-xs flex items-center justify-center flex-shrink-0 font-mono mt-0.5">
                {i + 1}
              </span>
              <p className="text-sm text-spine-text leading-relaxed">{rec}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Follow-up */}
      <div className="card p-5 mb-6 border-spine-green/30 bg-spine-green/5">
        <p className="label mb-2">Follow-up Plan</p>
        <p className="text-sm text-spine-text leading-relaxed">{report.follow_up}</p>
      </div>

      {/* Disclaimer */}
      <div className="bg-spine-amber/5 border border-spine-amber/20 rounded-xl p-4 text-xs text-spine-muted leading-relaxed">
        <span className="text-spine-amber font-semibold">Important Disclaimer — </span>
        {report.disclaimer}
      </div>
    </div>
  )
}
