"""Capa Modelo: carga y cache del modelo Keras."""

from pathlib import Path

import tensorflow as tf

from src.model.exceptions import ModelLoadError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "conv_MLP_84.h5"

_model = None


def load_model():
    """Carga y cachea el modelo Keras.

    Retorna:
        Instancia de tf.keras.Model, cargada una sola vez (singleton).

    Raises:
        ModelLoadError: Si el archivo .h5 no existe o no se puede cargar.
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise ModelLoadError("Archivo del modelo no encontrado", str(MODEL_PATH))
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model
