"""
ML Prediction Engine + Recommendation System

Models:

  - Kidney     → Models/kidney_disease_model.pkl (+ Models/kidney_scaler.pkl)
  - Anemia     → Models/Anemia.pkl               (+ Models/Anemia_scaler.pkl)
  - Liver      → Models/Liver.pkl

Feature Maps (exactly matching the trained model column order):


  Kidney (kidney_disease_model.pkl — abbreviated names):
    Bp   : Blood Pressure
    Sg   : Specific Gravity
    Al   : Albumin
    Su   : Sugar
    Rbc  : Red Blood Cells        (0 = normal, 1 = abnormal)
    Bu   : Blood Urea
    Sc   : Serum Creatinine
    Sod  : Sodium
    Pot  : Potassium
    Hemo : Hemoglobin
    Wbcc : White Blood Cell Count
    Rbcc : Red Blood Cell Count
    Htn  : Hypertension           (1 = Yes, 0 = No)

  Anemia (Anemia.pkl):
    Gender     : 0 = Female, 1 = Male
    Hemoglobin : g/dL
    MCH        : Mean Corpuscular Haemoglobin (pg)
    MCHC       : Mean Corpuscular Haemoglobin Concentration (g/dL)
    MCV        : Mean Corpuscular Volume (fL)
"""

import os
import logging
import numpy as np
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

MODELS_DIR = Path(getattr(settings, 'MODELS_DIR', Path(__file__).resolve().parent.parent / 'Models'))


# ─────────────────────────────────────────────
# Model loader (lazy, cached)
# ─────────────────────────────────────────────

_cache = {}


def _load(filename: str):
    if filename in _cache:
        return _cache[filename]
    path = MODELS_DIR / filename
    if not path.exists():
        logger.warning(f"Model file not found: {path}")
        return None
    try:
        import joblib
        obj = joblib.load(path)
        _cache[filename] = obj
        logger.info(f"Loaded model: {filename}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        return None


# ─────────────────────────────────────────────
# Feature builders
# ─────────────────────────────────────────────



# kidney_disease_model.pkl — abbreviated feature names
KIDNEY_FEATURES = [
    'Bp', 'Sg', 'Al', 'Su', 'Rbc',
    'Bu', 'Sc', 'Sod', 'Pot',
    'Hemo', 'Wbcc', 'Rbcc', 'Htn',
]

# Anemia.pkl — feature names as used in training
ANEMIA_FEATURES = ['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV']

# Default fill values (medically-normal defaults)


# Medically-normal defaults for the abbreviated kidney feature set
KIDNEY_DEFAULTS = {
    'Bp': 80,           # Blood Pressure (mmHg)
    'Sg': 1.01771249,   # Specific Gravity (training-set mean)
    'Al': 0,            # Albumin (0–5 scale)
    'Su': 0,            # Sugar (0–5 scale)
    'Rbc': 0,           # Red Blood Cells flag (0=normal)
    'Bu': 15,           # Blood Urea (mg/dL)
    'Sc': 0.9,          # Serum Creatinine (mg/dL)
    'Sod': 140,         # Sodium (mEq/L)
    'Pot': 4.0,         # Potassium (mEq/L)
    'Hemo': 14.0,       # Hemoglobin (g/dL)
    'Wbcc': 7500,       # WBC Count (cells/cumm)
    'Rbcc': 5.0,        # RBC Count (millions/cmm)
    'Htn': 0,           # Hypertension (1=Yes, 0=No)
}

# Medically-normal defaults for Anemia feature set
ANEMIA_DEFAULTS = {
    'Gender':     1,      # 0=Female, 1=Male (male assumed when unknown)
    'Hemoglobin': 13.5,   # g/dL  (lower normal for adults)
    'MCH':        27.5,   # pg    (normal range 27–33)
    'MCHC':       33.0,   # g/dL  (normal range 32–36)
    'MCV':        85.0,   # fL    (normal range 80–100)
}

LIVER_FEATURES = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Proteins',
    'Albumin', 'Albumin_and_Globulin_Ratio'
]

LIVER_DEFAULTS = {
    'Age': 35,
    'Gender': 1,  # 0=Female, 1=Male
    'Total_Bilirubin': 0.8,
    'Direct_Bilirubin': 0.2,
    'Alkaline_Phosphotase': 120,
    'Alamine_Aminotransferase': 25,
    'Aspartate_Aminotransferase': 25,
    'Total_Proteins': 7.0,
    'Albumin': 4.0,
    'Albumin_and_Globulin_Ratio': 1.0,
}


def _build_feature_vector(params: dict, features: list, defaults: dict) -> np.ndarray:
    """Build a model-ready numpy array from extracted params."""
    row = []
    for feat in features:
        val = params.get(feat, defaults.get(feat, 0))
        row.append(float(val) if val is not None else float(defaults.get(feat, 0)))
    return np.array(row).reshape(1, -1)


def _enrich_with_patient(params: dict, patient) -> dict:
    """Add patient info to params dict for ML models."""
    enriched = dict(params)
    if patient:
        enriched.setdefault('age', patient.age)
        gender = getattr(patient, 'gender', 'Male')
        enriched['gender_Male'] = 1 if gender == 'Male' else 0
        enriched['gender_Female'] = 1 if gender == 'Female' else 0

        # BMI Calculation
        if 'bmi' not in enriched and getattr(patient, 'height', None) and getattr(patient, 'weight', None):
            h_m = patient.height / 100.0
            bmi_calc = patient.weight / (h_m * h_m)
            enriched['bmi'] = round(bmi_calc, 1)

        # Sugar level (0–5) → map to kidney model's Su feature
        sugar_level = getattr(patient, 'sugar', 0)
        enriched.setdefault('Su', sugar_level)

        # Inject patient medical history — use direct assignment so patient
        # data always wins over OCR-extracted 0 defaults.
        if getattr(patient, 'hypertension', False):
            enriched['hypertension'] = 1
            enriched['Htn'] = 1   # abbreviated key for kidney_disease_model.pkl
        else:
            enriched.setdefault('hypertension', 0)
            enriched.setdefault('Htn', 0)



        if getattr(patient, 'blood_infection', False):
            # blood_infection maps to Rbc flag in kidney_disease_model.pkl
            enriched['Rbc'] = 1
        else:
            enriched.setdefault('Rbc', 0)

    return enriched


# ─────────────────────────────────────────────
# Predictors
# ─────────────────────────────────────────────

# ── Clinical reference ranges for kidney markers ─────────────────────────────
# Source: standard laboratory reference intervals
_KIDNEY_NORMAL_RANGES = {
    'Bp':   (60,  90),   # Systolic diastolic approx — normal diastolic
    'Sg':   (1.005, 1.030),  # Specific Gravity
    'Al':   (0,    1),   # Albumin in urine (0–5 scale; 0–1 = normal)
    'Su':   (0,    1),   # Sugar in urine  (0–5 scale; 0–1 = normal)
    'Bu':   (7,   25),   # Blood Urea (mg/dL)
    'Sc':   (0.5,  1.2), # Serum Creatinine (mg/dL)
    'Sod':  (135, 145),  # Sodium (mEq/L)
    'Pot':  (3.5,  5.0), # Potassium (mEq/L)
    'Hemo': (12,   17),  # Hemoglobin (g/dL) — covers male & female
    'Wbcc': (4000, 11000),  # WBC Count (cells/cumm)
    'Rbcc': (3.8,  6.1),    # RBC Count (millions/cmm)
}
# Rbc and Htn are boolean flags — already handled separately


def _calibrate_kidney_risk(raw_prob: float, enriched_params: dict) -> float:
    """
    Calibrate the raw kidney disease probability against clinical reference ranges.

    The kidney disease model was trained on a synthetic dataset and tends to
    produce inflated probabilities (≥70 %) even for clinically normal inputs.
    This function computes a *normality score* — the fraction of key markers
    that fall within established clinical reference intervals — and uses it to
    blend the raw model output toward a low baseline, reducing over-prediction
    when values are normal.

    Calibration formula:
        calibrated = raw_prob × (1 - normality_score × discount_strength)

    Where discount_strength ∈ [0, 1] controls maximum pull-down.
    At full normality (score = 1.0) and discount_strength = 0.80:
        An 80 % raw probability → ~16 % calibrated.
    At partial normality (score = 0.5): 80 % → ~48 %.
    Abnormal inputs (score ≈ 0) are left nearly unchanged.
    """
    normal_count = 0
    total_checked = 0

    for key, (lo, hi) in _KIDNEY_NORMAL_RANGES.items():
        if key in enriched_params:
            total_checked += 1
            val = enriched_params[key]
            if lo <= val <= hi:
                normal_count += 1

    # Boolean penalty: hypertension or RBC abnormality push toward abnormal
    htn_flag = enriched_params.get('Htn', 0)
    rbc_flag = enriched_params.get('Rbc', 0)

    if total_checked == 0:
        # No markers present — cannot calibrate; return raw as-is
        return raw_prob

    normality_score = normal_count / total_checked

    # Apply flag penalties: each active flag drops normality by 0.15
    normality_score = max(0.0, normality_score - (htn_flag * 0.15) - (rbc_flag * 0.15))

    # Discount strength: how aggressively we pull down when all is normal.
    # 0.80 means a perfectly normal profile can reduce the raw prob by up to 80%.
    discount_strength = 0.80

    calibrated = raw_prob * (1.0 - normality_score * discount_strength)

    # Never report below 2 % (model uncertainty floor) or above raw_prob
    calibrated = max(2.0, min(calibrated, raw_prob))

    logger.debug(
        f"[kidney_calibration] raw={raw_prob:.1f}% normality={normality_score:.2f} "
        f"calibrated={calibrated:.1f}%"
    )
    return calibrated


def predict_kidney(params: dict, patient=None) -> dict:
    """
    Predict kidney disease using kidney_disease_model.pkl.

    The model outputs a binary class (0 = no disease, 1 = disease).
    We use predict_proba when available to obtain a continuous risk %;
    otherwise we map the hard prediction to 0 % or 100 %.

    Feature order (must match training):
        Bp, Sg, Al, Su, Rbc, Bu, Sc, Sod, Pot, Hemo, Wbcc, Rbcc, Htn
    """
    model = _load('kidney_disease_model.pkl')
    scaler = _load('kidney_scaler.pkl')

    if model is None:
        return {'Kidney Disease': {'risk': 0, 'status': 'Model unavailable', 'available': False}}

    # Enrich with patient info so Htn and Rbc flags from the patient form
    # override any OCR-extracted 0 defaults.
    enriched = _enrich_with_patient(params, patient)

    # Map long-form OCR keys → abbreviated model keys (best-effort)
    ocr_to_abbr = {
        'Blood Pressure':   'Bp',
        'Specific Gravity': 'Sg',
        'Albumin':          'Al',
        'Sugar':            'Su',
        'RBC':              'Rbc',
        'Blood Urea':       'Bu',
        'Serum Creatinine': 'Sc',
        'Sodium':           'Sod',
        'Potassium':        'Pot',
        'Hemoglobin':       'Hemo',
        'WBC Count':        'Wbcc',
        'RBC Count':        'Rbcc',
        'Hypertension':     'Htn',
    }
    for long_key, abbr_key in ocr_to_abbr.items():
        if long_key in enriched and abbr_key not in enriched:
            enriched[abbr_key] = enriched[long_key]

    # Pin Specific Gravity to training-set mean if not extracted from the report.
    enriched.setdefault('Sg', 1.01771249)

    X = _build_feature_vector(enriched, KIDNEY_FEATURES, KIDNEY_DEFAULTS)

    try:
        if scaler is not None:
            # Scale all continuous features; skip boolean flags Rbc (idx 4) and Htn (idx 12)
            idx = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
            cont_features = X[0, idx].reshape(1, -1)
            scaled_cont = scaler.transform(cont_features)[0]
            for i, p in enumerate(idx):
                X[0, p] = scaled_cont[i]

        # Prefer probability output; fall back to hard 0/1 prediction
        if hasattr(model, 'predict_proba'):
            raw_prob = model.predict_proba(X)[0][1] * 100
        else:
            pred = int(model.predict(X)[0])
            raw_prob = 100.0 if pred == 1 else 0.0

        # Apply clinical calibration to correct synthetic-data over-prediction
        prob = _calibrate_kidney_risk(raw_prob, enriched)

        return {
            'Kidney Disease': {
                'risk': round(prob, 1),
                'status': _risk_label(prob),
                'available': True,
                'prediction': 1 if prob >= 50 else 0,   # binary label for display
            }
        }
    except Exception as e:
        logger.error(f"Kidney prediction error: {e}")
        return {'Kidney Disease': {'risk': 0, 'status': 'Prediction failed', 'available': False}}


# ── Clinical reference ranges for liver markers ───────────────────────────────
_LIVER_NORMAL_RANGES = {
    'Total_Bilirubin':            (0.2,  1.2),   # mg/dL
    'Direct_Bilirubin':           (0.0,  0.4),   # mg/dL
    'Alkaline_Phosphotase':       (44,   147),   # U/L
    'Alamine_Aminotransferase':   (7,    56),    # U/L  (ALT/SGPT)
    'Aspartate_Aminotransferase': (10,   40),    # U/L  (AST/SGOT)
    'Total_Proteins':             (6.0,  8.3),   # g/dL
    'Albumin':                    (3.4,  5.4),   # g/dL
    'Albumin_and_Globulin_Ratio': (1.0,  2.5),   # ratio
}


def _calibrate_liver_risk(raw_prob: float, enriched_params: dict) -> float:
    """
    Calibrate raw liver disease probability using clinical reference ranges.

    Same approach as kidney calibration:
        calibrated = raw_prob × (1 - normality_score × discount_strength)

    discount_strength = 0.78 for liver (slightly lower than kidney — liver
    datasets tend to be a bit less over-inflated).
    At full normality: an 80 % raw → ~17.6 % calibrated.
    """
    normal_count = 0
    total_checked = 0

    for key, (lo, hi) in _LIVER_NORMAL_RANGES.items():
        if key in enriched_params:
            total_checked += 1
            val = enriched_params[key]
            if lo <= val <= hi:
                normal_count += 1

    if total_checked == 0:
        return raw_prob

    normality_score = normal_count / total_checked
    discount_strength = 0.78

    calibrated = raw_prob * (1.0 - normality_score * discount_strength)
    calibrated = max(2.0, min(calibrated, raw_prob))

    logger.debug(
        f"[liver_calibration] raw={raw_prob:.1f}% normality={normality_score:.2f} "
        f"calibrated={calibrated:.1f}%"
    )
    return calibrated


def predict_liver(params: dict, patient=None) -> dict:
    model = _load('Liver.pkl')
    if model is None:
        return {'Liver Disease': {'risk': 0, 'status': 'Model unavailable', 'available': False}}

    enriched = dict(params)
    
    # Encode Gender: accept string ('Male'/'Female') or numeric (1/0)
    # Priority: explicit param → patient profile → default (Male=1)
    gender_raw = enriched.get('Gender', None)
    if gender_raw is None and patient is not None:
        gender_raw = getattr(patient, 'gender', None)
    if isinstance(gender_raw, str):
        enriched['Gender'] = 0 if gender_raw.strip().lower() == 'female' else 1
    elif gender_raw is None:
        enriched['Gender'] = LIVER_DEFAULTS['Gender']
    else:
        enriched['Gender'] = int(gender_raw)

    if patient:
        enriched.setdefault('Age', patient.age)

    # Map OCR variants to the exact model columns
    ocr_aliases = {
        'Total Bilirubin': 'Total_Bilirubin',
        'Direct Bilirubin': 'Direct_Bilirubin',
        'Alkaline Phosphatase': 'Alkaline_Phosphotase',
        'SGPT': 'Alamine_Aminotransferase',
        'ALT': 'Alamine_Aminotransferase',
        'Alanine Aminotransferase': 'Alamine_Aminotransferase',
        'SGOT': 'Aspartate_Aminotransferase',
        'AST': 'Aspartate_Aminotransferase',
        'Total Protein': 'Total_Proteins',
        'Total Proteins': 'Total_Proteins',
        'A/G Ratio': 'Albumin_and_Globulin_Ratio',
    }
    for alias, canonical in ocr_aliases.items():
        if alias in enriched and canonical not in enriched:
            enriched[canonical] = enriched[alias]

    X = _build_feature_vector(enriched, LIVER_FEATURES, LIVER_DEFAULTS)

    try:
        # Prefer probability output; fall back to hard 0/1 prediction
        if hasattr(model, 'predict_proba'):
            raw_prob = model.predict_proba(X)[0][1] * 100
        else:
            pred = int(model.predict(X)[0])
            raw_prob = 100.0 if pred == 1 else 0.0

        # Apply clinical calibration to correct synthetic-data over-prediction
        prob = _calibrate_liver_risk(raw_prob, enriched)

        return {
            'Liver Disease': {
                'risk': round(prob, 1),
                'status': _risk_label(prob),
                'available': True,
                'prediction': 1 if prob >= 50 else 0,
            }
        }
    except Exception as e:
        logger.error(f"Liver prediction error: {e}")
        return {'Liver Disease': {'risk': 0, 'status': 'Prediction failed', 'available': False}}


# ── Clinical reference ranges for anemia markers ──────────────────────────────
_ANEMIA_NORMAL_RANGES = {
    'Hemoglobin': (12.0, 17.5),  # g/dL (broad range covering both genders)
    'MCH':        (27.0, 33.0),  # pg/cell
    'MCHC':       (32.0, 36.0),  # g/dL
    'MCV':        (80.0, 100.0), # fL
}


def _calibrate_anemia_risk(raw_prob: float, enriched_params: dict) -> float:
    """
    Calibrate raw anemia disease probability using clinical reference ranges.

    Same approach as kidney and liver calibration:
        calibrated = raw_prob × (1 - normality_score × discount_strength)

    discount_strength = 0.75 for anemia.
    """
    normal_count = 0
    total_checked = 0

    for key, (lo, hi) in _ANEMIA_NORMAL_RANGES.items():
        if key in enriched_params:
            total_checked += 1
            val = enriched_params[key]
            if lo <= val <= hi:
                normal_count += 1

    if total_checked == 0:
        return raw_prob

    normality_score = normal_count / total_checked
    discount_strength = 0.75

    calibrated = raw_prob * (1.0 - normality_score * discount_strength)
    calibrated = max(2.0, min(calibrated, raw_prob))

    logger.debug(
        f"[anemia_calibration] raw={raw_prob:.1f}% normality={normality_score:.2f} "
        f"calibrated={calibrated:.1f}%"
    )
    return calibrated


def predict_anemia(params: dict, patient=None) -> dict:
    """
    Predict Anemia risk using Anemia.pkl + Anemia_scaler.pkl.

    Feature order (must match training):
        Gender (0=Female, 1=Male), Hemoglobin, MCH, MCHC, MCV

    The model outputs binary 0/1; predict_proba is used when available
    to get a continuous risk percentage.
    """
    model = _load('Anemia.pkl')
    scaler = _load('Anemia_scaler.pkl')

    if model is None:
        return {'Anemia': {'risk': None, 'status': 'Model unavailable', 'available': False}}

    # Build working params dict; start with what was extracted / passed in
    enriched = dict(params)

    # Encode Gender: accept string ('Male'/'Female') or numeric (1/0)
    # Priority: explicit param → patient profile → default (Male=1)
    gender_raw = enriched.get('Gender', None)
    if gender_raw is None and patient is not None:
        gender_raw = getattr(patient, 'gender', None)
    if isinstance(gender_raw, str):
        enriched['Gender'] = 0 if gender_raw.strip().lower() == 'female' else 1
    elif gender_raw is None:
        enriched['Gender'] = ANEMIA_DEFAULTS['Gender']
    else:
        enriched['Gender'] = int(gender_raw)

    # Map alternate OCR key names → expected model keys
    ocr_aliases = {
        'Haemoglobin': 'Hemoglobin',
        'Hb':          'Hemoglobin',
        'HGB':         'Hemoglobin',
    }
    for alias, canonical in ocr_aliases.items():
        if alias in enriched and canonical not in enriched:
            enriched[canonical] = enriched[alias]

    X = _build_feature_vector(enriched, ANEMIA_FEATURES, ANEMIA_DEFAULTS)

    try:
        if scaler is not None:
            # Scaler was fitted on 4 continuous features only: Hemoglobin, MCH, MCHC, MCV
            # Gender (index 0) is binary and must NOT be passed to the scaler.
            cont_idx = [1, 2, 3, 4]  # Hemoglobin, MCH, MCHC, MCV
            cont_scaled = scaler.transform(X[0, cont_idx].reshape(1, -1))[0]
            for i, p in enumerate(cont_idx):
                X[0, p] = cont_scaled[i]

        if hasattr(model, 'predict_proba'):
            raw_prob = model.predict_proba(X)[0][1] * 100
        else:
            pred = int(model.predict(X)[0])
            raw_prob = 100.0 if pred == 1 else 0.0

        # Apply clinical calibration to correct synthetic-data over-prediction
        prob = _calibrate_anemia_risk(raw_prob, enriched)

        return {
            'Anemia': {
                'risk': round(prob, 1),
                'status': _risk_label(prob),
                'available': True,
                'prediction': 1 if prob >= 50 else 0,  # binary label for display
            }
        }
    except Exception as e:
        logger.error(f"Anemia prediction error: {e}")
        return {'Anemia': {'risk': 0, 'status': 'Prediction failed', 'available': False}}


def _risk_label(prob: float) -> str:
    if prob >= 70:
        return 'High Risk'
    elif prob >= 40:
        return 'Moderate Risk'
    else:
        return 'Low Risk'


# ─────────────────────────────────────────────
# Master predictor
# ─────────────────────────────────────────────

REPORT_PREDICTORS = {
    'KFT':     [predict_kidney],
    'CBC':     [predict_anemia],
    'LFT':     [predict_liver],
}


def run_predictions(params: dict, report_type: str, patient=None) -> dict:
    """
    Run applicable predictors based on extracted parameters dynamically.
    Returns merged prediction dict.
    """
    results = {}
    
    # Auto-detect applicable models based on extracted parameters

    has_kidney_features = any(k in params for k in ['Serum Creatinine', 'Blood Urea', 'Specific Gravity'])
    has_lft_features = any(k in params for k in ['ALT', 'AST', 'ALP', 'Total Bilirubin'])
    # Anemia model needs Hemoglobin + at least one of MCH / MCHC / MCV
    has_cbc_features = (
        any(k in params for k in ['Hemoglobin', 'Haemoglobin', 'Hb', 'HGB'])
        and any(k in params for k in ['MCH', 'MCHC', 'MCV'])
    )
    
    report_upper = report_type.upper()
    

    if has_kidney_features or report_upper == 'KFT':
        results.update(predict_kidney(params, patient))
        
    if has_lft_features or report_upper == 'LFT':
        results.update(predict_liver(params, patient))
        
    if has_cbc_features or report_upper == 'CBC':
        results.update(predict_anemia(params, patient))
        
    return results


# ─────────────────────────────────────────────
# Recommendation Engine
# ─────────────────────────────────────────────

RECOMMENDATIONS = {
    'Kidney Disease': {
        'High Risk': [
            'Consult a nephrologist as soon as possible for further evaluation.',
            'Follow a kidney-friendly diet as recommended by your healthcare provider.',
            'Reduce sodium (salt) intake and avoid processed foods.',
            'Monitor your blood pressure regularly.',
            'Maintain adequate hydration unless advised otherwise by your doctor.',
            'Avoid unnecessary use of painkillers (NSAIDs) and other medications that may affect kidney function.',
            'Schedule regular Kidney Function Tests (KFT) to monitor your condition.',
        ],
        'Moderate Risk': [
            'Limit salt and processed food consumption.',
            'Maintain healthy blood pressure and blood sugar levels.',
            'Drink sufficient water unless your healthcare provider recommends fluid restriction.',
            'Avoid excessive protein intake without medical advice.',
            'Follow a balanced diet and maintain a healthy body weight.',
            'Repeat kidney function tests periodically as advised by your healthcare provider.',
        ],
        'Low Risk': [
            'Maintain adequate daily hydration.',
            'Follow a balanced, low-sodium diet.',
            'Exercise regularly and maintain a healthy lifestyle.',
            'Schedule routine kidney health check-ups annually.',
        ],
    },

    'Liver Disease': {
        'High Risk': [
            'Consult a hepatologist or physician for further evaluation.',
            'Avoid alcohol completely until advised otherwise by your doctor.',
            'Follow a balanced, low-fat diet rich in fruits and vegetables.',
            'Avoid self-medication and herbal supplements without medical advice.',
            'Maintain a healthy body weight.',
            'Schedule regular Liver Function Tests (LFT) as recommended.',
        ],
        'Moderate Risk': [
            'Reduce or avoid alcohol consumption.',
            'Limit fried, fatty, and highly processed foods.',
            'Eat a diet rich in fruits, vegetables, whole grains, and lean protein.',
            'Exercise regularly to maintain a healthy weight.',
            'Consider routine hepatitis screening if recommended by your healthcare provider.',
        ],
        'Low Risk': [
            'Maintain a balanced and nutritious diet.',
            'Exercise regularly and maintain a healthy weight.',
            'Limit alcohol consumption.',
            'Schedule routine liver health check-ups annually.',
        ],
    },

    'Anemia': {
        'High Risk': [
            'Consult a healthcare professional for further evaluation.',
            'Include iron-rich foods such as spinach, beans, lentils, and lean meat if appropriate.',
            'Consume Vitamin C-rich foods to improve iron absorption.',
            'Take iron, Vitamin B12, or folate supplements only if prescribed by your healthcare provider.',
            'Avoid drinking tea or coffee immediately after meals as they may reduce iron absorption.',
            'Repeat a Complete Blood Count (CBC) as advised by your doctor.',
        ],
        'Moderate Risk': [
            'Increase intake of iron-rich foods and leafy green vegetables.',
            'Ensure adequate Vitamin B12 and folate intake.',
            'Maintain a balanced diet with sufficient nutrients.',
            'Schedule periodic Complete Blood Count (CBC) tests.',
            'Consult a healthcare provider if symptoms such as fatigue or dizziness persist.',
        ],
        'Low Risk': [
            'Maintain a healthy, balanced diet rich in essential vitamins and minerals.',
            'Stay physically active and maintain a healthy lifestyle.',
            'Undergo routine Complete Blood Count (CBC) screening annually.',
        ],
    },
}

GENERAL_RECOMMENDATIONS = [
    'Maintain a regular sleep schedule of 7–8 hours each night.',
    'Drink adequate water every day unless advised otherwise by your healthcare provider.',
    'Exercise for at least 30 minutes on most days of the week.',
    'Maintain a healthy body weight through balanced nutrition.',
    'Manage stress through meditation, yoga, or other relaxation techniques.',
    'Avoid smoking and limit alcohol consumption.',
    'Follow prescribed medications as directed by your healthcare provider.',
    'Schedule routine preventive health check-ups annually.',
]


def generate_recommendations(predictions: dict) -> list:
    """
    Generate personalized recommendations based on prediction results.
    Returns a list of recommendation strings.
    """
    recs = []

    for disease, result in predictions.items():
        if not result.get('available') or result.get('risk') is None:
            continue
        status = result.get('status', 'Low Risk')
        disease_recs = RECOMMENDATIONS.get(disease, {})
        recs.extend(disease_recs.get(status, disease_recs.get('Low Risk', [])))

    # Always append general wellness tips
    recs.extend(GENERAL_RECOMMENDATIONS)

    # Deduplicate while preserving order
    seen = set()
    unique_recs = []
    for rec in recs:
        if rec not in seen:
            seen.add(rec)
            unique_recs.append(rec)

    return unique_recs
