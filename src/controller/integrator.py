#!/usr/bin/env python
"""Integrador: orquesta preprocesamiento, predicción y Grad-CAM.

Este módulo actúa como el controlador principal que une la capa Vista con
las capas Modelo y Procesamiento de Imágenes de manera desacoplada.
"""

from typing import NamedTuple

import numpy as np

from src.controller.preprocess_img import preprocess
from src.model.grad_cam import generate_gradcam
from src.model.load_model import load_model


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

    Raises:
        ValueError: Si la imagen es None o no cumple con las dimensiones de canales.
        TypeError: Si la imagen no es un arreglo de NumPy.
    """
    if array is None:
        raise ValueError("La imagen de entrada no puede ser None.")

    if not isinstance(array, np.ndarray):
        raise TypeError("La entrada debe ser un arreglo de NumPy.")

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(
            f"La imagen de entrada debe ser de 3 dimensiones con 3 canales. Se recibió ndim={array.ndim}."
        )

    # 1. Preprocesar la imagen
    preprocessed_array = preprocess(array)

    # 2. Cargar el modelo (Singleton)
    model = load_model()

    # 3. Realizar predicción con el modelo
    preds = model.predict(preprocessed_array)
    class_idx = np.argmax(preds[0])
    probability = float(preds[0][class_idx] * 100.0)

    # Mapeo de índices a clases textuales según el contrato de Sebas
    label_map = {0: "bacteriana", 1: "normal", 2: "viral"}
    label = label_map.get(class_idx, "desconocido")

    # 4. Generar el mapa de calor Grad-CAM
    heatmap = generate_gradcam(preprocessed_array, array, model)

    return PredictionResult(label=label, probability=probability, heatmap=heatmap)
