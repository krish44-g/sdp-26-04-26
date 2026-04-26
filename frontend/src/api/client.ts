import axios from 'axios'

const api = axios.create({ baseURL: '/api/v1' })

export interface DeformityResult {
  name: string
  probability: number
  detected: boolean
}

export interface KeypointResult {
  index: number
  name: string
  x: number
  y: number
}

export interface RatioResult {
  THR: number
  SHR: number
  LBP: number
  CLB: number
}

export interface AnalysisResponse {
  analysis_id: string
  image_url: string
  keypoints: KeypointResult[]
  deformities: DeformityResult[]
  raw_ratios: RatioResult
  corrected_ratios: RatioResult
  ethnicity: string
  processing_time_ms: number
}

export interface ReportCondition {
  name: string
  probability: number
  severity: string
  severity_score: number
  description: string
  clinical_significance: string
}

export interface ReportResponse {
  report_id: string
  analysis_id: string
  summary: string
  detected_conditions: ReportCondition[]
  postural_analysis: {
    trunk_ratio: string
    shoulder_hip_ratio: string
    leg_proportion: string
    cervical_balance: string
  }
  recommendations: string[]
  follow_up: string
  disclaimer: string
}

export const analyzeImage = async (
  file: File,
  ethnicity: string,
  age?: number,
  sex?: string,
  chiefComplaint?: string
): Promise<AnalysisResponse> => {
  const form = new FormData()
  form.append('file', file)
  form.append('ethnicity', ethnicity)
  if (age) form.append('age', String(age))
  if (sex) form.append('sex', sex)
  if (chiefComplaint) form.append('chief_complaint', chiefComplaint)
  const { data } = await api.post<AnalysisResponse>('/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export const generateReport = async (analysisId: string): Promise<ReportResponse> => {
  const { data } = await api.post<ReportResponse>('/report', { analysis_id: analysisId })
  return data
}

export const checkHealth = async () => {
  const { data } = await api.get('/health')
  return data
}
