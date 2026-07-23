"""
ML Prediction Engine + Recommendation System

Models:
  - Diabetes   → Models/Diabetes.pkl  (+ Models/scaler_diabetes.pkl)
  - Kidney     → Models/kidney.pkl    (+ Models/kidney_scaler.pkl)
  - Liver      → Models/liver.pkl  [STUB — add model file when ready]
  - Anemia     → Models/anemia.pkl [STUB — add model file when ready]

Feature Maps (exactly matching the trained model column order):
  Diabetes:
    age, hypertension, heart_disease, bmi, hbA1c_level,
    blood_glucose_level, gender_Female, gender_Male,
    smoking_history_No, smoking_history_Yes

  Kidney:
    Blood Pressure, Specific Gravity, Albumin, Sugar, RBC,
    Blood Urea, Serum Creatinine, Sodium, Potassium,
    Hemoglobin, WBC Count, RBC Count, Hypertension
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

DIABETES_FEATURES = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'hbA1c_level',
    'blood_glucose_level', 'gender_Female', 'gender_Male',
    'smoking_history_No', 'smoking_history_Yes',
]

KIDNEY_FEATURES = [
    'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar', 'RBC',
    'Blood Urea', 'Serum Creatinine', 'Sodium', 'Potassium',
    'Hemoglobin', 'WBC Count', 'RBC Count', 'Hypertension',
]

# Default fill values (medically-normal defaults)
DIABETES_DEFAULTS = {
    'age': 35, 'hypertension': 0, 'heart_disease': 0, 'bmi': 22.5,
    'hbA1c_level': 5.5, 'blood_glucose_level': 100,
    'gender_Female': 0, 'gender_Male': 1,
    'smoking_history_No': 1, 'smoking_history_Yes': 0,
}

KIDNEY_DEFAULTS = {
    'Blood Pressure': 80, 'Specific Gravity': 1.01771249, 'Albumin': 0,
    'Sugar': 0, 'RBC': 0, 'Blood Urea': 15, 'Serum Creatinine': 0.9,
    'Sodium': 140, 'Potassium': 4.0, 'Hemoglobin': 14.0,
    'WBC Count': 7500, 'RBC Count': 5.0, 'Hypertension': 0,
}


def _build_feature_vector(params: dict, features: list, defaults: dict) -> np.ndarray:
    """Build a model-ready numpy array from extracted params."""
    row = []
    for feat in features:
        val = params.get(feat, defaults.get(feat, 0))
        row.append(float(val) if val is not None else float(defaults.get(feat, 0)))
    return np.array(row).reshape(1, -1)


def _enrich_with_patient(params: dict, patient) -> dict:
    """Add patient info to params dict for Diabetes model."""
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

        # Smoking History → new model columns: smoking_history_No / smoking_history_Yes
        smoke = getattr(patient, 'smoking_history', 'Never')
        is_smoker = smoke == 'Current'
        enriched['smoking_history_Yes'] = 1 if is_smoker else 0
        enriched['smoking_history_No'] = 0 if is_smoker else 1

        # Inject patient medical history — use direct assignment so patient
        # data always wins over OCR-extracted 0 defaults.
        if getattr(patient, 'hypertension', False):
            enriched['hypertension'] = 1
            enriched['Hypertension'] = 1   # for Kidney model
        else:
            enriched.setdefault('hypertension', 0)
            enriched.setdefault('Hypertension', 0)

        if getattr(patient, 'heart_disease', False):
            enriched['heart_disease'] = 1
        else:
            enriched.setdefault('heart_disease', 0)

        if getattr(patient, 'diabetes', False):
            enriched.setdefault('diabetes_history', 1)

        if getattr(patient, 'blood_infection', False):
            # blood_infection maps to RBC flag in KFT model
            enriched['RBC'] = 1
        else:
            enriched.setdefault('RBC', 0)

    return enriched


# ─────────────────────────────────────────────
# Predictors
# ─────────────────────────────────────────────

def predict_diabetes(params: dict, patient=None) -> dict:
    model = _load('Diabetes.pkl')
    scaler = _load('scaler_diabetes.pkl')

    if model is None:
        return {'Diabetes': {'risk': 0, 'status': 'Model unavailable', 'available': False}}

    enriched = _enrich_with_patient(params, patient)
    X = _build_feature_vector(enriched, DIABETES_FEATURES, DIABETES_DEFAULTS)

    try:
        if scaler is not None:
            # Scaler expects only continuous features: age, bmi, hbA1c_level, blood_glucose_level
            # Indices in new DIABETES_FEATURES: age=0, bmi=3, hbA1c_level=4, blood_glucose_level=5
            idx = [0, 3, 4, 5]
            cont_features = X[0, idx].reshape(1, -1)
            scaled_cont = scaler.transform(cont_features)[0]
            for i, p in enumerate(idx):
                X[0, p] = scaled_cont[i]
                
        prob = model.predict_proba(X)[0][1] * 100
        return {
            'Diabetes': {
                'risk': round(prob, 1),
                'status': _risk_label(prob),
                'available': True,
            }
        }
    except Exception as e:
        logger.error(f"Diabetes prediction error: {e}")
        return {'Diabetes': {'risk': 0, 'status': 'Prediction failed', 'available': False}}


def predict_kidney(params: dict, patient=None) -> dict:
    model = _load('kidney.pkl')
    scaler = _load('kidney_scaler.pkl')

    if model is None:
        return {'Kidney Disease': {'risk': 0, 'status': 'Model unavailable', 'available': False}}

    # Enrich with patient info so Hypertension and RBC flags from the
    # patient form override any OCR-extracted 0 defaults.
    enriched = _enrich_with_patient(params, patient)

    # Pin Specific Gravity to the training-set mean (not present in most reports).
    enriched['Specific Gravity'] = 1.01771249

    X = _build_feature_vector(enriched, KIDNEY_FEATURES, KIDNEY_DEFAULTS)

    try:
        if scaler is not None:
            # Scaler expects all features EXCEPT 'RBC' (idx 4) and 'Hypertension' (idx 12)
            # which are booleans. Indices: 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11
            idx = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
            cont_features = X[0, idx].reshape(1, -1)
            scaled_cont = scaler.transform(cont_features)[0]
            for i, p in enumerate(idx):
                X[0, p] = scaled_cont[i]
                
        prob = model.predict_proba(X)[0][1] * 100
        return {
            'Kidney Disease': {
                'risk': round(prob, 1),
                'status': _risk_label(prob),
                'available': True,
            }
        }
    except Exception as e:
        logger.error(f"Kidney prediction error: {e}")
        return {'Kidney Disease': {'risk': 0, 'status': 'Prediction failed', 'available': False}}


def predict_liver(params: dict, patient=None) -> dict:
    model = _load('liver.pkl')
    if model is None:
        return {
            'Liver Disease': {
                'risk': None,
                'status': 'Model coming soon',
                'available': False,
            }
        }
    # Will be wired once model is available
    return {'Liver Disease': {'risk': None, 'status': 'Model coming soon', 'available': False}}


def predict_anemia(params: dict, patient=None) -> dict:
    model = _load('anemia.pkl')
    if model is None:
        return {
            'Anemia': {
                'risk': None,
                'status': 'Model coming soon',
                'available': False,
            }
        }
    return {'Anemia': {'risk': None, 'status': 'Model coming soon', 'available': False}}


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
    'DIABETES': [predict_diabetes],
    'HBA1C':   [predict_diabetes],
    'SUGAR':   [predict_diabetes],
    'KFT':     [predict_kidney],
    'CBC':     [predict_anemia],
    'LFT':     [predict_liver],
    'LIPID':   [predict_liver],
}


def run_predictions(params: dict, report_type: str, patient=None) -> dict:
    """
    Run applicable predictors based on extracted parameters dynamically.
    Returns merged prediction dict.
    """
    results = {}
    
    # Auto-detect applicable models based on extracted parameters
    has_diabetes_features = any(k in params for k in ['HbA1c', 'Blood Glucose'])
    has_kidney_features = any(k in params for k in ['Serum Creatinine', 'Blood Urea', 'Specific Gravity'])
    has_lft_features = any(k in params for k in ['ALT', 'AST', 'ALP', 'Total Bilirubin'])
    has_cbc_features = any(k in params for k in ['Hemoglobin', 'WBC', 'Platelets', 'PCV'])
    
    report_upper = report_type.upper()
    
    if has_diabetes_features or report_upper in ['DIABETES', 'HBA1C', 'SUGAR']:
        results.update(predict_diabetes(params, patient))
        
    if has_kidney_features or report_upper == 'KFT':
        results.update(predict_kidney(params, patient))
        
    if has_lft_features or report_upper in ['LFT', 'LIPID']:
        results.update(predict_liver(params, patient))
        
    if has_cbc_features or report_upper == 'CBC':
        results.update(predict_anemia(params, patient))
        
    return results


# ─────────────────────────────────────────────
# Recommendation Engine
# ─────────────────────────────────────────────

RECOMMENDATIONS = {
    'Diabetes': {
        'High Risk': [
            'Consult an endocrinologist immediately.',
            'Follow a strict low-glycemic index diet — avoid white rice, sugar, and refined flour.',
            'Engage in 30–45 minutes of moderate exercise daily (walking, cycling).',
            'Monitor blood glucose levels every morning.',
            'Avoid sugary drinks and processed snacks.',
            'Consider regular HbA1c monitoring every 3 months.',
        ],
        'Moderate Risk': [
            'Reduce sugar and carbohydrate intake in daily meals.',
            'Walk at least 30 minutes per day.',
            'Get fasting blood glucose tested regularly.',
            'Maintain a healthy body weight (BMI < 25).',
            'Limit alcohol and tobacco consumption.',
        ],
        'Low Risk': [
            'Maintain a balanced diet rich in vegetables and whole grains.',
            'Stay physically active with regular exercise.',
            'Get annual blood sugar check-ups.',
        ],
    },
    'Kidney Disease': {
        'High Risk': [
            'Consult a nephrologist immediately.',
            'Follow a low-protein, low-sodium diet.',
            'Restrict potassium and phosphorus intake.',
            'Monitor blood pressure daily.',
            'Stay well hydrated but within recommended fluid limits.',
            'Avoid NSAIDs and nephrotoxic medications.',
        ],
        'Moderate Risk': [
            'Reduce salt and processed food consumption.',
            'Control blood pressure and blood sugar levels.',
            'Stay hydrated with 8–10 glasses of water daily.',
            'Avoid excessive protein intake.',
            'Schedule follow-up kidney function tests every 3 months.',
        ],
        'Low Risk': [
            'Maintain adequate hydration daily.',
            'Eat a balanced, low-sodium diet.',
            'Annual kidney function screening recommended.',
        ],
    },
    'Liver Disease': {
        'High Risk': [
            'Consult a hepatologist immediately.',
            'Abstain completely from alcohol.',
            'Follow a low-fat, low-sodium diet.',
            'Avoid self-medication and herbal supplements without medical advice.',
            'Monitor liver enzymes every month.',
        ],
        'Moderate Risk': [
            'Significantly reduce or eliminate alcohol intake.',
            'Eat a diet rich in fruits, vegetables, and lean protein.',
            'Avoid fatty and fried foods.',
            'Maintain healthy body weight.',
            'Get hepatitis B & C screening.',
        ],
        'Low Risk': [
            'Limit alcohol to recommended safe levels.',
            'Eat a balanced, nutritious diet.',
            'Annual liver function tests recommended.',
        ],
    },
    'Anemia': {
        'High Risk': [
            'Consult a hematologist immediately.',
            'Increase iron-rich foods: red meat, spinach, lentils, fortified cereals.',
            'Include Vitamin C sources to enhance iron absorption.',
            'Consider iron supplementation under medical supervision.',
            'Avoid tea and coffee immediately after meals (inhibits iron absorption).',
        ],
        'Moderate Risk': [
            'Increase dietary iron intake through leafy greens and legumes.',
            'Ensure adequate Vitamin B12 and folate in diet.',
            'Get a complete blood count (CBC) every 3 months.',
            'Avoid excessive physical exertion.',
        ],
        'Low Risk': [
            'Maintain a diet rich in iron and vitamins.',
            'Annual CBC check-up recommended.',
        ],
    },
}

GENERAL_RECOMMENDATIONS = [
    'Maintain a regular sleep schedule of 7–8 hours per night.',
    'Stay hydrated — drink at least 8 glasses of water daily.',
    'Manage stress through meditation, yoga, or deep breathing.',
    'Schedule regular preventive health check-ups annually.',
    'Avoid smoking and limit alcohol consumption.',
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
