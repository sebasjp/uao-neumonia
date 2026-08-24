#!/usr/bin/env python
"""Integrador: orquesta preprocesamiento, predicción y Grad-CAM.

Placeholder de contrato — ver docs/CONTRATOS_MODULOS.md. La implementación
real de este módulo es responsabilidad de Juan; estas firmas solo existen
para que src/detector_view.py sea importable mientras tanto.
"""

from typing import NamedTuple

import numpy as np


class PredictionResult(NamedTuple):
    """Resultado de una predicción completa sobre una radiografía.

    Attributes:
        label: Una de {"bacteriana", "normal", "viral"}.
        probability: Probabilidad de la predicción, en rango 0-100.
        heatmap: Mapa de calor RGB, shape (512, 512, 3), uint8.
    """

    label: str
    probability: float
    heatmap: np.ndarray


def predict(array: np.ndarray) -> PredictionResult:
    """Genera la predicción de clase y el mapa de calor para una radiografía.

    Es el único punto de entrada que la Vista necesita para obtener una
    predicción completa (orquesta preprocess + load_model + generate_gradcam
    internamente).

    Args:
        array: Imagen BGR sin procesar (alto x ancho x 3), uint8 — la misma
            salida cruda que entregan read_dicom_file/read_jpg_file.

    Returns:
        PredictionResult con la etiqueta, probabilidad y mapa de calor.
    """
    raise NotImplementedError
