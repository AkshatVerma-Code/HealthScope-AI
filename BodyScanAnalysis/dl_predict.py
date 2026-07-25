"""
Deep Learning Inference Engine

Models (TensorFlow .keras format):
  - Brain MRI  → Models/brain_tumor.keras
    Classes: ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

  - Chest X-ray → Models/chest_xray.keras
    Classes: ['Normal', 'Pneumonia', 'Tuberculosis']

  - Alzheimer  → Models/alzheimer.keras
    Classes: ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

All models:
  - Input size: 224×224 (configurable per model below)
  - Normalization: pixel values / 255.0
  - Output: softmax probabilities

NOTE: If a model file doesn't exist, a STUB result is returned so the
      UI still works. Drop the .keras file in Models/ to activate real inference.
"""

import logging
import numpy as np
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

MODELS_DIR = Path(getattr(settings, 'MODELS_DIR', Path(__file__).resolve().parent.parent / 'Models'))

# ─────────────────────────────────────────────
# Model configurations
# ─────────────────────────────────────────────

MODEL_CONFIGS = {
    'BRAIN_MRI': {
        'file': 'brain_tumor.keras',
        'classes': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
        'input_size': (224, 224),
        'description': 'Brain Tumor Classification',
    },
    'CHEST_XRAY': {
        'file': 'chest_xray.keras',
        'classes': ['Normal', 'Pneumonia', 'Tuberculosis'],
        'input_size': (224, 224),
        'description': 'Lung Disease Classification',
    },
    'ALZHEIMER': {
        'file': 'alzheimer.keras',
        'config_file': 'config.json',
        'weights_file': 'model.weights.h5',
        'classes': ['Mild Impairment', 'Moderate Impairment', 'No Impairment'],
        'input_size': (224, 224),
        'description': "Alzheimer's Disease Staging",
    },
}

_model_cache = {}


def _load_keras_model(model_key: str):
    """Lazy load and cache a Keras model."""
    if model_key in _model_cache:
        return _model_cache[model_key]

    config = MODEL_CONFIGS.get(model_key)
    if not config:
        return None

    model_path = MODELS_DIR / config['file']
    json_path = MODELS_DIR / config.get('config_file', 'config.json')
    h5_path = MODELS_DIR / config.get('weights_file', 'model.weights.h5')

    if not model_path.exists() and not (json_path.exists() and h5_path.exists()):
        logger.warning(f"DL model not found for {model_key} — using stub prediction.")
        return None

    try:
        import tensorflow as tf
        if model_path.exists():
            model = tf.keras.models.load_model(str(model_path))
            logger.info(f"Loaded DL model: {config['file']}")
        else:
            with open(json_path, 'r') as f:
                config_json = f.read()
            model = tf.keras.models.model_from_json(config_json)
            model.load_weights(str(h5_path))
            logger.info(f"Loaded DL model from json & weights: {json_path.name}, {h5_path.name}")
            
        _model_cache[model_key] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load DL model for {model_key}: {e}")
        return None


def _preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    """Load, resize, and normalize an image for TF inference."""
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, H, W, 3)


# ─────────────────────────────────────────────
# Stub prediction (used when model unavailable)
# ─────────────────────────────────────────────

def _stub_result(image_type: str) -> dict:
    config = MODEL_CONFIGS.get(image_type, {})
    classes = config.get('classes', ['Unknown'])
    # Return uniform distribution stub
    n = len(classes)
    scores = {cls: round(100.0 / n, 1) for cls in classes}
    return {
        'prediction': classes[0],
        'confidence': round(100.0 / n, 1),
        'all_class_scores': scores,
        'model_available': False,
        'note': f"Model file '{config.get('file', '')}' not found. Drop it in Models/ to activate real predictions.",
    }


# ─────────────────────────────────────────────
# Main predictor
# ─────────────────────────────────────────────

def predict_image(image_path: str, image_type: str) -> dict:
    """
    Run DL inference on a medical image.

    Args:
        image_path: Absolute path to the uploaded image file.
        image_type: One of 'BRAIN_MRI', 'CHEST_XRAY', 'ALZHEIMER'

    Returns:
        dict with keys: prediction, confidence, all_class_scores, model_available
    """
    image_type = image_type.upper()
    config = MODEL_CONFIGS.get(image_type)

    if not config:
        return {
            'prediction': 'Unknown',
            'confidence': 0.0,
            'all_class_scores': {},
            'model_available': False,
            'note': f"Unknown image type: {image_type}",
        }

    model = _load_keras_model(image_type)

    if model is None:
        return _stub_result(image_type)

    try:
        target_size = config['input_size']
        classes = config['classes']

        X = _preprocess_image(image_path, target_size)
        preds = model.predict(X, verbose=0)[0]  # shape: (num_classes,)

        top_idx = int(np.argmax(preds))
        top_class = classes[top_idx]
        top_confidence = float(preds[top_idx]) * 100

        all_scores = {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(classes, preds)
        }

        return {
            'prediction': top_class,
            'confidence': round(top_confidence, 2),
            'all_class_scores': all_scores,
            'model_available': True,
        }

    except Exception as e:
        logger.error(f"DL prediction error for {image_type}: {e}")
        return {
            'prediction': 'Prediction failed',
            'confidence': 0.0,
            'all_class_scores': {},
            'model_available': False,
            'note': str(e),
        }
