"""
AI Clinical Report Generator using Groq API (llama-3.3-70b-versatile).
"""
import json
from groq import AsyncGroq
from config import settings

DEFORMITY_CLASSES = settings.DEFORMITY_CLASSES

SYSTEM_PROMPT = """You are an expert clinical AI assistant specializing in musculoskeletal 
and spinal health assessment. You receive structured output from a deep learning model 
that analyzes patient posture images.

Your task is to generate a clear, professional, and actionable clinical report.

IMPORTANT RULES:
- Always include a disclaimer that this is an AI-assisted screening tool, NOT a medical diagnosis
- Use clinical terminology appropriate for a physiotherapist or orthopedic specialist
- Base all observations strictly on the data provided
- Never invent findings not supported by the classification data
- Keep severity language proportionate: low (<0.3), moderate (0.3-0.6), high (>0.6)
- Respond ONLY with valid JSON, no markdown, no preamble

Output JSON structure:
{
  "summary": "2-3 sentence clinical overview",
  "detected_conditions": [
    {
      "name": "condition name",
      "probability": 0.85,
      "severity": "moderate",
      "severity_score": 0.54,
      "description": "clinical description of this finding",
      "clinical_significance": "what this means for the patient"
    }
  ],
  "postural_analysis": {
    "trunk_ratio": "interpretation of THR",
    "shoulder_hip_ratio": "interpretation of SHR",
    "leg_proportion": "interpretation of LBP",
    "cervical_balance": "interpretation of CLB"
  },
  "recommendations": [
    "specific, actionable recommendation"
  ],
  "follow_up": "recommended follow-up actions",
  "disclaimer": "standard AI screening disclaimer"
}"""


async def generate_clinical_report(
    probabilities: dict,
    corrected_ratios: dict,
    ethnicity: str,
    patient_info: dict = None,
) -> dict:
    """
    Generate a structured clinical report from model outputs using Groq.
    """
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    # Build detected conditions list
    detected = [
        {"class": cls, "probability": round(prob, 3)}
        for cls, prob in probabilities.items()
        if prob > 0.35 and cls != "Normal"
    ]

    user_content = f"""Patient Analysis Results:

Ethnicity: {ethnicity}
Patient Info: {json.dumps(patient_info or {}, indent=2)}

Classification Probabilities:
{json.dumps(probabilities, indent=2)}

Detected Conditions (probability > 0.35):
{json.dumps(detected, indent=2)}

SEA-Corrected Anthropometric Ratios (z-scores relative to universal baseline):
- THR (Trunk-to-Height Ratio): {corrected_ratios.get('THR', 0):.3f}
- SHR (Shoulder-to-Hip Ratio): {corrected_ratios.get('SHR', 0):.3f}
- LBP (Leg-to-Body Proportion): {corrected_ratios.get('LBP', 0):.3f}
- CLB (Cervical-Lumbar Balance): {corrected_ratios.get('CLB', 0):.3f}

Generate a comprehensive clinical report in the specified JSON format."""

    response = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=1500,
    )

    response_text = response.choices[0].message.content.strip()

    # Strip any markdown fences if present
    if response_text.startswith("```"):
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:]

    return json.loads(response_text)
