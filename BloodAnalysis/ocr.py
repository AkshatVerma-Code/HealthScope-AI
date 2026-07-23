"""
OCR Pipeline — Mistral OCR API + Regex Parameter Extraction

Workflow:
  1. Upload PDF/image → encode to base64
  2. Send to Mistral OCR API
  3. Extract raw text
  4. Apply regex parsers per report type
  5. Return structured JSON parameters
"""

import os
import re
import base64
import logging
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Mistral OCR call
# ─────────────────────────────────────────────

def run_mistral_ocr(uploaded_file) -> str:
    """Send file to Mistral OCR and return extracted text."""
    api_key = getattr(settings, 'MISTRAL_API_KEY', '') or os.environ.get('MISTRAL_API_KEY', '')

    if not api_key:
        logger.warning("MISTRAL_API_KEY not set — returning empty OCR text.")
        return ""

    try:
        from mistralai.client import Mistral
        client = Mistral(api_key=api_key)

        file_bytes = uploaded_file.read()

        ext = Path(uploaded_file.name).suffix.lower()
        mime_map = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
        }
        mime_type = mime_map.get(ext, 'application/octet-stream')
        b64_data = base64.b64encode(file_bytes).decode('utf-8')
        data_url = f"data:{mime_type};base64,{b64_data}"

        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": data_url},
        )

        # Combine all page texts
        text = "\n".join(
            page.markdown for page in response.pages if hasattr(page, 'markdown')
        )
        return text

    except Exception as e:
        logger.error(f"Mistral OCR error: {e}")
        return ""


# ─────────────────────────────────────────────
# Generic numeric extractor
# ─────────────────────────────────────────────

def _extract_number(text: str, patterns: list) -> float | None:
    """Try multiple regex patterns and return the first matched float."""
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except (ValueError, IndexError):
                continue
    return None


def _extract_flag(text: str, patterns: list) -> int:
    """Return 1 if any pattern matches (presence indicator), else 0."""
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return 1
    return 0


# ─────────────────────────────────────────────
# Per-report type parameter parsers
# ─────────────────────────────────────────────

def _parse_diabetes_params(text: str) -> dict:
    params = {}

    params['HbA1c'] = _extract_number(text, [
        r'hba1c[^\d\n]*(\d+\.?\d*)',
        r'glycated\s+haemoglobin[^\d\n]*(\d+\.?\d*)',
        r'glycosylated[^\d\n]*(\d+\.?\d*)',
    ])
    params['Blood Glucose'] = _extract_number(text, [
        r'blood\s+glucose[^\d\n]*(\d+\.?\d*)',
        r'glucose[^\d\n]*(\d+\.?\d*)',
        r'fasting\s+glucose[^\d\n]*(\d+\.?\d*)',
        r'ppbs[^\d\n]*(\d+\.?\d*)',
        r'rbs[^\d\n]*(\d+\.?\d*)',
    ])
    params['BMI'] = _extract_number(text, [r'bmi[^\d\n]*(\d+\.?\d*)'])
    params['Age'] = _extract_number(text, [r'age[^\d\n]*(\d+)'])

    params['Hypertension'] = _extract_flag(text, [
        r'hypertension', r'high\s+blood\s+pressure', r'htn'
    ])
    params['Heart Disease'] = _extract_flag(text, [
        r'heart\s+disease', r'coronary', r'cardiac'
    ])

    # Smoking history flags
    smoking_text = text.lower()
    params['smoking_history_never'] = 1 if re.search(r'non[- ]smoker|never\s+smok', smoking_text) else 0
    params['smoking_history_current'] = 1 if re.search(r'current\s+smok|active\s+smok', smoking_text) else 0
    params['smoking_history_former'] = 1 if re.search(r'former\s+smok|ex[- ]smok', smoking_text) else 0
    params['smoking_history_not current'] = 1 if (
        params['smoking_history_former'] and not params['smoking_history_current']
    ) else 0

    return {k: v for k, v in params.items() if v is not None}


def _parse_kidney_params(text: str) -> dict:
    params = {}

    params['Blood Pressure'] = _extract_number(text, [
        r'blood\s+pressure[^\d\n]*(\d+\.?\d*)',
        r'bp[^\d\n]*(\d+)(?:/\d+)?',
    ])
    params['Specific Gravity'] = _extract_number(text, [
        r'specific\s+gravity[^\d\n]*(1\.\d+)',
        r'sp\.?\s*gr[^\d\n]*(1\.\d+)',
    ])
    params['Albumin'] = _extract_number(text, [
        r'albumin[^\d\n]*(\d+\.?\d*)',
        r'alb[^\d\n]*(\d+\.?\d*)',
    ])
    params['Sugar'] = _extract_number(text, [
        r'sugar[^\d\n]*(\d+\.?\d*)',
        r'glucose\s+urine[^\d\n]*(\d+\.?\d*)',
    ])
    params['Blood Urea'] = _extract_number(text, [
        r'blood\s+urea[^\d\n]*(\d+\.?\d*)',
        r'urea[^\d\n]*(\d+\.?\d*)',
        r'bun[^\d\n]*(\d+\.?\d*)',
    ])
    params['Serum Creatinine'] = _extract_number(text, [
        r'serum\s+creatinine[^\d\n]*(\d+\.?\d*)',
        r'creatinine[^\d\n]*(\d+\.?\d*)',
        r'scr[^\d\n]*(\d+\.?\d*)',
    ])
    params['Sodium'] = _extract_number(text, [
        r'sodium[^\d\n]*(\d+\.?\d*)',
        r'na\+?[^\d\n]*(\d+\.?\d*)',
    ])
    params['Potassium'] = _extract_number(text, [
        r'potassium[^\d\n]*(\d+\.?\d*)',
        r'k\+?[^\d\n]*(\d+\.?\d*)',
    ])
    params['Hemoglobin'] = _extract_number(text, [
        r'h[ae]moglobin[^\d\n]*(\d+\.?\d*)',
        r'hb[^\d\n]*(\d+\.?\d*)',
        r'hgb[^\d\n]*(\d+\.?\d*)',
    ])
    params['WBC Count'] = _extract_number(text, [
        r'wbc\s+count[^\d\n]*([\d,]+\.?\d*)',
        r'total\s+wbc[^\d\n]*([\d,]+\.?\d*)',
        r'leukocyte[^\d\n]*([\d,]+\.?\d*)',
    ])
    params['RBC Count'] = _extract_number(text, [
        r'rbc\s+count[^\d\n]*([\d,]+\.?\d*)',
        r'red\s+blood\s+cell[^\d\n]*([\d,]+\.?\d*)',
        r'erythrocyte[^\d\n]*([\d,]+\.?\d*)',
    ])

    # RBC = presence of abnormal RBCs (infection indicator)
    params['RBC'] = _extract_flag(text, [
        r'abnormal\s+rbc',
        r'rbc\s+present',
        r'rbc:\s*present',
        r'blood\s+infection',
        r'haematuria',
    ])
    params['Hypertension'] = _extract_flag(text, [
        r'hypertension', r'high\s+blood\s+pressure', r'htn'
    ])

    return {k: v for k, v in params.items() if v is not None}


def _parse_lft_params(text: str) -> dict:
    params = {}
    params['ALT'] = _extract_number(text, [r'alt[^\d\n]*(\d+\.?\d*)', r'sgpt[^\d\n]*(\d+\.?\d*)'])
    params['AST'] = _extract_number(text, [r'ast[^\d\n]*(\d+\.?\d*)', r'sgot[^\d\n]*(\d+\.?\d*)'])
    params['ALP'] = _extract_number(text, [r'alp[^\d\n]*(\d+\.?\d*)', r'alkaline\s+phosphatase[^\d\n]*(\d+\.?\d*)'])
    params['Total Bilirubin'] = _extract_number(text, [r'total\s+bilirubin[^\d\n]*(\d+\.?\d*)', r'bilirubin\s*\(?total\)?[^\d\n]*(\d+\.?\d*)'])
    params['Direct Bilirubin'] = _extract_number(text, [r'direct\s+bilirubin[^\d\n]*(\d+\.?\d*)', r'bilirubin\s*\(?direct\)?[^\d\n]*(\d+\.?\d*)'])
    params['Total Protein'] = _extract_number(text, [r'total\s+protein[^\d\n]*(\d+\.?\d*)', r'serum\s+protein[^\d\n]*(\d+\.?\d*)'])
    params['Albumin'] = _extract_number(text, [r'albumin[^\d\n]*(\d+\.?\d*)'])
    return {k: v for k, v in params.items() if v is not None}


def _parse_lipid_params(text: str) -> dict:
    params = {}
    params['Total Cholesterol'] = _extract_number(text, [r'total\s+cholesterol[^\d\n]*(\d+\.?\d*)'])
    params['HDL'] = _extract_number(text, [r'hdl[^\d\n]*(\d+\.?\d*)', r'good\s+cholesterol[^\d\n]*(\d+\.?\d*)'])
    params['LDL'] = _extract_number(text, [r'ldl[^\d\n]*(\d+\.?\d*)', r'bad\s+cholesterol[^\d\n]*(\d+\.?\d*)'])
    params['Triglycerides'] = _extract_number(text, [r'triglycerides[^\d\n]*(\d+\.?\d*)', r'tg[^\d\n]*(\d+\.?\d*)'])
    params['VLDL'] = _extract_number(text, [r'vldl[^\d\n]*(\d+\.?\d*)'])
    return {k: v for k, v in params.items() if v is not None}


def _parse_cbc_params(text: str) -> dict:
    params = {}
    params['Hemoglobin'] = _extract_number(text, [r'h[ae]moglobin[^\d\n]*(\d+\.?\d*)', r'hb[^\d\n]*(\d+\.?\d*)'])
    params['WBC'] = _extract_number(text, [r'wbc[^\d\n]*([\d,]+\.?\d*)', r'total\s+wbc[^\d\n]*([\d,]+\.?\d*)'])
    params['RBC Count'] = _extract_number(text, [r'rbc[^\d\n]*([\d,]+\.?\d*)'])
    params['Platelets'] = _extract_number(text, [r'platelet[^\d\n]*([\d,]+\.?\d*)', r'plt[^\d\n]*([\d,]+\.?\d*)'])
    params['PCV'] = _extract_number(text, [r'pcv[^\d\n]*(\d+\.?\d*)', r'haematocrit[^\d\n]*(\d+\.?\d*)'])
    params['MCV'] = _extract_number(text, [r'mcv[^\d\n]*(\d+\.?\d*)'])
    params['MCH'] = _extract_number(text, [r'mch[^\d\n]*(\d+\.?\d*)'])
    params['Neutrophils'] = _extract_number(text, [r'neutrophil[^\d\n]*(\d+\.?\d*)'])
    params['Lymphocytes'] = _extract_number(text, [r'lymphocyte[^\d\n]*(\d+\.?\d*)'])
    return {k: v for k, v in params.items() if v is not None}


# ─────────────────────────────────────────────
# Master dispatch
# ─────────────────────────────────────────────

REPORT_PARSERS = {
    'DIABETES': _parse_diabetes_params,
    'HBA1C': _parse_diabetes_params,
    'KFT': _parse_kidney_params,
    'LFT': _parse_lft_params,
    'LIPID': _parse_lipid_params,
    'CBC': _parse_cbc_params,
    'SUGAR': _parse_diabetes_params,
}


def extract_parameters(uploaded_file, report_type: str) -> dict:
    """
    Main entry point.
    Returns a dict of {parameter_name: value}.
    """
    text = run_mistral_ocr(uploaded_file)

    if not text.strip():
        logger.warning(f"No OCR text extracted from {uploaded_file.name}")
        return {}

    parser = REPORT_PARSERS.get(report_type.upper(), lambda t: {})
    params = parser(text)

    logger.info(f"Extracted {len(params)} parameters from {report_type} report")
    return params
