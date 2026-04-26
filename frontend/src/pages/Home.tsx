import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon, ChevronRight, Loader2, AlertCircle } from 'lucide-react'
import { analyzeImage } from '../api/client'
import toast from 'react-hot-toast'

const ETHNICITIES = [
  'East Asian', 'South Asian', 'Sub-Saharan African',
  'European', 'Latin American', 'Middle Eastern',
]

const VIEWS = ['Frontal', 'Lateral', 'Posterior']

export default function Home() {
  const navigate = useNavigate()
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [ethnicity, setEthnicity] = useState('South Asian')
  const [age, setAge] = useState('')
  const [sex, setSex] = useState('')
  const [view, setView] = useState('Frontal')
  const [loading, setLoading] = useState(false)

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png'] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  })

  const handleAnalyze = async () => {
    if (!file) return toast.error('Please upload an image first')
    setLoading(true)
    try {
      const result = await analyzeImage(file, ethnicity, age ? Number(age) : undefined, sex || undefined)
      toast.success('Analysis complete!')
      navigate(`/results/${result.analysis_id}`, { state: { result } })
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || 'Analysis failed. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      {/* Hero */}
      <div className="text-center mb-12 stagger">
        <div className="inline-flex items-center gap-2 bg-spine-accent/10 border border-spine-accent/20 rounded-full px-4 py-1.5 text-xs font-mono text-spine-accent mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-spine-accent animate-pulse" />
          SEA Generalizer · Multi-label · 7-class detection
        </div>
        <h1 className="font-display text-5xl md:text-6xl text-spine-text mb-4 leading-tight">
          Posture &amp; Spine<br />
          <span className="text-spine-accent">Deformity Detection</span>
        </h1>
        <p className="text-spine-muted text-lg max-w-xl mx-auto leading-relaxed">
          Upload a posture image and receive an AI-powered clinical assessment corrected
          for ethnicity-specific anthropometric baselines.
        </p>
      </div>

      <div className="grid md:grid-cols-5 gap-6">
        {/* Upload zone — 3 cols */}
        <div className="md:col-span-3">
          <div
            {...getRootProps()}
            className={`relative card p-0 overflow-hidden cursor-pointer transition-all duration-300 h-80 flex flex-col items-center justify-center scan-overlay
              ${isDragActive ? 'border-spine-accent glow-accent' : 'hover:border-spine-accent/40'}
              ${preview ? 'border-spine-accent/30' : ''}`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <>
                <img src={preview} alt="Preview" className="w-full h-full object-contain" />
                <div className="absolute inset-0 bg-gradient-to-t from-spine-card/80 via-transparent to-transparent" />
                <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between">
                  <span className="text-xs font-mono text-spine-accent bg-spine-bg/80 px-2 py-1 rounded">
                    {file?.name}
                  </span>
                  <span className="text-xs text-spine-muted bg-spine-bg/80 px-2 py-1 rounded">
                    Click to replace
                  </span>
                </div>
              </>
            ) : (
              <div className="text-center p-8">
                <div className={`w-16 h-16 rounded-2xl mx-auto mb-4 flex items-center justify-center transition-colors
                  ${isDragActive ? 'bg-spine-accent/20' : 'bg-spine-surface'}`}>
                  {isDragActive ? (
                    <ImageIcon size={28} className="text-spine-accent" />
                  ) : (
                    <Upload size={28} className="text-spine-muted" />
                  )}
                </div>
                <p className="text-spine-text font-medium mb-1">
                  {isDragActive ? 'Drop to analyze' : 'Drop your posture image'}
                </p>
                <p className="text-spine-muted text-sm">JPEG or PNG · max 10MB</p>
              </div>
            )}
          </div>

          {/* View selector */}
          <div className="mt-3 flex gap-2">
            {VIEWS.map(v => (
              <button
                key={v}
                onClick={() => setView(v)}
                className={`flex-1 py-2 text-sm rounded-xl border transition-all duration-200 font-medium
                  ${view === v
                    ? 'bg-spine-accent/15 border-spine-accent text-spine-accent'
                    : 'border-spine-border text-spine-muted hover:border-spine-accent/40'}`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        {/* Config panel — 2 cols */}
        <div className="md:col-span-2 flex flex-col gap-4">
          <div className="card p-5">
            <p className="label mb-3">Patient Ethnicity</p>
            <div className="flex flex-col gap-2">
              {ETHNICITIES.map(e => (
                <button
                  key={e}
                  onClick={() => setEthnicity(e)}
                  className={`text-left px-3 py-2.5 rounded-xl text-sm transition-all duration-150 border
                    ${ethnicity === e
                      ? 'bg-spine-accent/15 border-spine-accent text-spine-accent'
                      : 'border-transparent text-spine-muted hover:bg-white/5 hover:text-spine-text'}`}
                >
                  {e}
                </button>
              ))}
            </div>
          </div>

          <div className="card p-5 flex flex-col gap-3">
            <p className="label mb-0">Patient Info (optional)</p>
            <input
              type="number"
              placeholder="Age"
              value={age}
              onChange={e => setAge(e.target.value)}
              className="bg-spine-surface border border-spine-border rounded-xl px-4 py-2.5 text-sm text-spine-text placeholder:text-spine-muted focus:outline-none focus:border-spine-accent transition-colors"
            />
            <select
              value={sex}
              onChange={e => setSex(e.target.value)}
              className="bg-spine-surface border border-spine-border rounded-xl px-4 py-2.5 text-sm text-spine-text focus:outline-none focus:border-spine-accent transition-colors appearance-none"
            >
              <option value="">Sex (select)</option>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>
        </div>
      </div>

      {/* SEA notice */}
      <div className="mt-4 flex items-start gap-3 bg-spine-accent/5 border border-spine-accent/20 rounded-xl px-4 py-3">
        <AlertCircle size={16} className="text-spine-accent mt-0.5 flex-shrink-0" />
        <p className="text-xs text-spine-muted leading-relaxed">
          <span className="text-spine-accent font-medium">SEA Correction active</span> — your selected ethnicity
          ({ethnicity}) will be used to normalize body proportion ratios (THR, SHR, LBP, CLB) against
          published anthropometric baselines before classification.
        </p>
      </div>

      {/* Analyze button */}
      <button
        onClick={handleAnalyze}
        disabled={!file || loading}
        className="btn-primary w-full mt-6 flex items-center justify-center gap-2 text-base py-4"
      >
        {loading ? (
          <>
            <Loader2 size={18} className="animate-spin" />
            Analyzing posture...
          </>
        ) : (
          <>
            Run Analysis
            <ChevronRight size={18} />
          </>
        )}
      </button>
    </div>
  )
}
