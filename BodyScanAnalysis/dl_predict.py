"""
Deep Learning Inference Engine

Models (TensorFlow .keras format):
  - Brain MRI  → Models/brain_tumor.keras
    Classes: ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

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
import zipfile
import tempfile
import os
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
    'ALZHEIMER': {
        'file': 'alzheimer.keras',
        'classes': ['Mild Impairment', 'No Impairment', 'Very Mild Impairment'],
        'input_size': (224, 224),
        'description': "Alzheimer's Disease Staging",
    },
    'BRAIN_TUMOR_SEGMENTATION': {
        'file': 'brain_tumor_segmentation.keras',
        'classes': ['Background', 'Tumor'],
        'input_size': (256, 256),
        'description': 'Brain Tumor Segmentation',
    },
}

_model_cache = {}


def build_unet(input_shape=(256, 256, 3)):
    """Rebuild the U-Net architecture matching the trained model."""
    import tensorflow as tf

    def conv_block(inputs, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def encoder_block(inputs, filters):
        x = conv_block(inputs, filters)
        p = tf.keras.layers.MaxPooling2D((2, 2))(x)
        return x, p

    def decoder_block(inputs, skip, filters):
        x = tf.keras.layers.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(inputs)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = conv_block(x, filters)
        return x

    inputs = tf.keras.layers.Input(input_shape)
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)
    
    b1 = conv_block(p4, 512)
    
    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 64)
    d4 = decoder_block(d3, s1, 32)
    
    outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation="sigmoid", padding="same")(d4)
    return tf.keras.Model(inputs, outputs, name="U-Net")


def _load_keras_model(model_key: str):
    """Lazy load and cache a Keras model by rebuilding it and loading weights."""
    if model_key in _model_cache:
        return _model_cache[model_key]

    config = MODEL_CONFIGS.get(model_key)
    if not config:
        return None

    model_path = MODELS_DIR / config['file']

    if not model_path.exists():
        logger.warning(f"DL model not found for {model_key} at {model_path} — using stub prediction.")
        return None

    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
        from tensorflow.keras.applications import EfficientNetB0

        num_classes = len(config['classes'])
        
        if model_key == 'BRAIN_TUMOR_SEGMENTATION':
            logger.info(f"Rebuilding U-Net model for {model_key}...")
            model = build_unet()
        else:
            logger.info(f"Rebuilding Sequential model for {model_key} with {num_classes} classes...")
            # Rebuild classification architecture
            base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dense(128, activation="relu"),
                Dropout(0.3),
                Dense(num_classes, activation="linear")
            ])

        # Extract weights from the .keras zip archive
        temp_dir = tempfile.gettempdir()
        temp_weights_path = os.path.join(temp_dir, f"{model_key}_temp_weights.weights.h5")

        with zipfile.ZipFile(str(model_path), 'r') as zf:
            weights_name = None
            for name in zf.namelist():
                if "model.weights.h5" in name or "weights" in name:
                    weights_name = name
                    break
            
            if not weights_name:
                raise ValueError("Could not find weights file inside the .keras zip archive")
            
            with open(temp_weights_path, "wb") as f_out:
                f_out.write(zf.read(weights_name))

        # Load weights into reconstructed model
        model.load_weights(temp_weights_path)
        logger.info(f"Successfully loaded model weights for {model_key}")

        # Clean up temp file
        try:
            os.remove(temp_weights_path)
        except Exception:
            pass

        _model_cache[model_key] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load DL model for {model_key}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _preprocess_image(image_path: str, target_size: tuple, normalize: bool = False) -> np.ndarray:
    """Load, resize, and normalize an image for TF inference."""
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
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
        image_type: One of 'BRAIN_MRI', 'ALZHEIMER', 'BRAIN_TUMOR_SEGMENTATION'

    Returns:
        dict with prediction results
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

        if image_type == 'BRAIN_TUMOR_SEGMENTATION':
            X = _preprocess_image(image_path, target_size, normalize=True)
            preds = model.predict(X, verbose=0)[0]  # shape: (256, 256, 1)
            
            mask = (preds[:, :, 0] > 0.5).astype(np.float32)
            
            # Load original image for overlay
            from PIL import Image
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            
            # Resize mask to original size using NEAREST
            mask_resized = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST))
            
            img_arr = np.array(img)
            overlay = img_arr.copy()
            
            alpha = 0.45
            mask_indices = mask_resized > 0
            overlay[mask_indices, 0] = (1 - alpha) * img_arr[mask_indices, 0] + alpha * 255
            overlay[mask_indices, 1] = (1 - alpha) * img_arr[mask_indices, 1] + alpha * 0
            overlay[mask_indices, 2] = (1 - alpha) * img_arr[mask_indices, 2] + alpha * 0
            
            # Save overlay image next to original
            base, ext = os.path.splitext(image_path)
            overlay_path = f"{base}_overlay{ext}"
            Image.fromarray(overlay).save(overlay_path)
            
            # Construct relative URL for Django
            filename = os.path.basename(overlay_path)
            overlay_url = f"{settings.MEDIA_URL}medical_images/{filename}"
            
            # Percentage of tumor area
            tumor_percent = float((mask.sum() / (256 * 256)) * 100)
            
            return {
                'prediction': 'Tumor Segmented',
                'confidence': round(tumor_percent, 2),
                'all_class_scores': {
                    'overlay_url': overlay_url,
                    'tumor_area_percentage': round(tumor_percent, 2)
                },
                'model_available': True
            }
        else:
            X = _preprocess_image(image_path, target_size, normalize=False)
            preds = model.predict(X, verbose=0)[0]  # shape: (num_classes,)

            # Apply softmax to raw logits
            import tensorflow as tf
            probs = tf.nn.softmax(preds).numpy()

            top_idx = int(np.argmax(probs))
            top_class = classes[top_idx]
            top_confidence = float(probs[top_idx]) * 100

            all_scores = {
                cls: round(float(prob) * 100, 2)
                for cls, prob in zip(classes, probs)
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

