"""
Report generation endpoint: calls Claude API to produce clinical report.
"""
import uuid
from fastapi import APIRouter, HTTPException
from api.schemas import ReportRequest, ReportResponse
from report.generator import generate_clinical_report
from config import settings

router = APIRouter()

# In-memory store for demo; replace with DB in production
from cache import store_analysis, get_analysis


@router.post("/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    print(f"DEBUG - Received analysis_id: {request.analysis_id}")  # ADD THIS
    
    analysis = get_analysis(request.analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found. Run /analyze first.")

    probabilities = {d["name"]: d["probability"] for d in analysis["deformities"]}
    corrected_ratios = analysis["corrected_ratios"]
    ethnicity = analysis["ethnicity"]
    patient_info = analysis.get("patient_info", {})

    try:
        report_data = await generate_clinical_report(
            probabilities=probabilities,
            corrected_ratios=corrected_ratios,
            ethnicity=ethnicity,
            patient_info=patient_info,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()  # ADD THIS
        raise HTTPException(500, f"Report generation failed: {str(e)}")

    return ReportResponse(
        report_id=str(uuid.uuid4()),
        analysis_id=request.analysis_id,
        summary=report_data.get("summary", ""),
        detected_conditions=report_data.get("detected_conditions", []),
        postural_analysis=report_data.get("postural_analysis", {}),
        recommendations=report_data.get("recommendations", []),
        follow_up=report_data.get("follow_up", ""),
        disclaimer=report_data.get("disclaimer", ""),
    )
