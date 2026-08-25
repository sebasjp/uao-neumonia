"""Módulo de preprocesamiento de imágenes radiográficas."""

import cv2
import numpy as np


def preprocess(array: np.ndarray) -> np.ndarray:
    """Preprocesa una imagen radiográfica BGR para la entrada del modelo.

    Realiza el cambio de tamaño a 512x512, la conversión a escala de grises,
    la ecualización de contraste adaptativa (CLAHE), la normalización a [0, 1]
    y la expansión de dimensiones para simular un lote (batch) de tamaño 1.

    Args:
        array: Imagen BGR sin procesar de forma (alto x ancho x 3) y tipo uint8.

    Returns:
        Un arreglo de NumPy preprocesado de forma (1, 512, 512, 1) y tipo float64.

    Raises:
        ValueError: Si la imagen es None, vacía, no tiene 3 dimensiones o no
            cuenta con exactamente 3 canales.
        TypeError: Si la entrada no es un arreglo de NumPy.
    """
    if array is None:
        raise ValueError("La imagen de entrada no puede ser None.")

    if not isinstance(array, np.ndarray):
        raise TypeError("La entrada debe ser un arreglo de NumPy (np.ndarray).")

    if array.ndim != 3:
        raise ValueError(
            f"La imagen debe tener exactamente 3 dimensiones (H, W, C). Se recibió ndim={array.ndim}."
        )

    if array.shape[2] != 3:
        raise ValueError(
            f"La imagen debe tener exactamente 3 canales (BGR). Se recibió shape={array.shape}."
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"La imagen no puede tener dimensiones vacías. Se recibió shape={array.shape}."
        )

    # Evitamos la mutación del array de entrada haciendo una copia si es necesario,
    # aunque las operaciones de OpenCV retornan nuevos arreglos.
    working_array = array.copy()

    # Si el dtype no es uint8, lo convertimos de forma segura para las funciones de OpenCV
    if working_array.dtype != np.uint8:
        working_array = working_array.astype(np.uint8)

    # 1. Redimensionar a 512x512
    resized = cv2.resize(working_array, (512, 512))

    # 2. Convertir de BGR a Escala de Grises
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(gray)

    # 4. Normalizar al rango [0.0, 1.0] con tipo float64
    normalized = equalized.astype(np.float64) / 255.0

    # 5. Expandir dimensiones a (1, 512, 512, 1)
    expanded = np.expand_dims(normalized, axis=-1)  # (512, 512, 1)
    batch_expanded = np.expand_dims(expanded, axis=0)  # (1, 512, 512, 1)

    return batch_expanded
