from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class EthnicityEnum(str, Enum):
    east_asian = "East Asian"
    south_asian = "South Asian"
    sub_saharan = "Sub-Saharan African"
    european = "European"
    latin_american = "Latin American"
    middle_eastern = "Middle Eastern"


class AnalysisRequest(BaseModel):
    ethnicity: EthnicityEnum = EthnicityEnum.european
    age: Optional[int] = None
    sex: Optional[str] = None
    chief_complaint: Optional[str] = None


class KeypointResult(BaseModel):
    index: int
    name: str
    x: float
    y: float


class DeformityResult(BaseModel):
    name: str
    probability: float
    detected: bool


class RatioResult(BaseModel):
    THR: float
    SHR: float
    LBP: float
    CLB: float


class AnalysisResponse(BaseModel):
    analysis_id: str
    image_url: str
    keypoints: list[KeypointResult]
    deformities: list[DeformityResult]
    raw_ratios: RatioResult
    corrected_ratios: RatioResult
    ethnicity: str
    processing_time_ms: float


class ReportRequest(BaseModel):
    analysis_id: str


class ReportCondition(BaseModel):
    name: str
    probability: float
    severity: str
    severity_score: float
    description: str
    clinical_significance: str


class PosturalAnalysis(BaseModel):
    trunk_ratio: str
    shoulder_hip_ratio: str
    leg_proportion: str
    cervical_balance: str


class ReportResponse(BaseModel):
    report_id: str
    analysis_id: str
    summary: str
    detected_conditions: list[ReportCondition]
    postural_analysis: PosturalAnalysis
    recommendations: list[str]
    follow_up: str
    disclaimer: str
