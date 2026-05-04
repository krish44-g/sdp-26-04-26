import { useLocation, useParams, useNavigate } from 'react-router-dom'
import { useState } from 'react'
import { AnalysisResponse } from '../api/client'
import KeypointOverlay from '../components/KeypointOverlay'
import DeformityBadges from '../components/DeformityBadges'
import SEARatioChart from '../components/SEARatioChart'
import { Clock, FileText, Cpu, ChevronRight, Loader2 } from 'lucide-react'
import { generateReport } from '../api/client'
import toast from 'react-hot-toast'

export default function Results() {
  const { analysisId } = useParams()
  const { state } = useLocation()
  const navigate = useNavigate()
  const [genLoading, setGenLoading] = useState(false)

  const result: AnalysisResponse | undefined = state?.result

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] text-spine-muted">
        <p className="mb-4">No analysis data found.</p>
        <button onClick={() => navigate('/')} className="btn-ghost">Go back</button>
      </div>
    )
  }

  const detected = result.deformities.filter(d => d.detected && d.name !== 'Normal')

  const handleGenerateReport = async () => {
    setGenLoading(true)
    try {
      const report = await generateReport(result.analysis_id)
      toast.success('Clinical report generated!')
      navigate(`/report/${result.analysis_id}`, { state: { report, result } })
    } catch {
      toast.error("Report generation failed. Please try again.")
    } finally {
      setGenLoading(false)
    }
  }

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="flex items-start justify-between mb-8 flex-wrap gap-4">
        <div>
          <p className="label mb-1">Analysis Results</p>
          <h1 className="font-display text-3xl text-spine-text">
            {detected.length === 0
              ? 'No Deformities Detected'
              : `${detected.length} Condition${detected.length > 1 ? 's' : ''} Flagged`}
          </h1>
          <p className="text-spine-muted text-sm mt-1">
            Patient ethnicity: <span className="text-spine-accent">{result.ethnicity}</span>
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs font-mono text-spine-muted card px-3 py-2">
            <Clock size={12} />
            {result.processing_time_ms.toFixed(0)} ms
          </div>
          <div className="flex items-center gap-1.5 text-xs font-mono text-spine-muted card px-3 py-2">
            <Cpu size={12} />
            PostureNet + SEA
          </div>
          <button
            onClick={handleGenerateReport}
            disabled={genLoading}
            className="btn-primary flex items-center gap-2 py-2.5 text-sm"
          >
            {genLoading ? (
              <><Loader2 size={14} className="animate-spin" /> Generating...</>
            ) : (
              <><FileText size={14} /> AI Clinical Report <ChevronRight size={14} /></>
            )}
          </button>
        </div>
      </div>

      <div className="grid lg:grid-cols-5 gap-6">
        {/* Left: Image + keypoints — 3 cols */}
        <div className="lg:col-span-3 space-y-4">
          <KeypointOverlay
            imageUrl={result.image_url}
            keypoints={result.keypoints}
          />
          <SEARatioChart
            rawRatios={result.raw_ratios}
            correctedRatios={result.corrected_ratios}
            ethnicity={result.ethnicity}
          />
        </div>

        {/* Right: Classification — 2 cols */}
        <div className="lg:col-span-2 space-y-4">
          <div className="card p-5">
            <p className="label mb-4">Deformity Classification</p>
            <DeformityBadges deformities={result.deformities} />
          </div>

          {/* Keypoints summary */}
          <div className="card p-5">
            <p className="label mb-3">Detected Keypoints</p>
            <div className="grid grid-cols-2 gap-1.5 max-h-64 overflow-y-auto pr-1">
              {result.keypoints.map(kp => (
                <div key={kp.index} className="flex items-center justify-between bg-spine-surface rounded-lg px-2.5 py-1.5">
                  <span className="text-xs text-spine-muted truncate">{kp.name}</span>
                  <span className="text-[10px] font-mono text-spine-border ml-2 flex-shrink-0">
                    {kp.x.toFixed(2)},{kp.y.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Disclaimer */}
          <div className="bg-spine-amber/5 border border-spine-amber/20 rounded-xl p-4 text-xs text-spine-muted leading-relaxed">
            <span className="text-spine-amber font-semibold">Screening Tool Only — </span>
            This AI assessment is not a medical diagnosis. Always consult a qualified
            healthcare professional for clinical decisions.
          </div>
        </div>
      </div>
    </div>
  )
}
