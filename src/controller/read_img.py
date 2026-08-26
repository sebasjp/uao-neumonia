"""Capa Controlador: lectura de imágenes radiográficas en formato DICOM y JPG."""

from typing import NamedTuple

import cv2
import numpy as np
import pydicom as dicom
from PIL import Image


class ImageReadResult(NamedTuple):
    """Resultado de leer una imagen radiográfica.

    Attributes:
        img_array: Arreglo RGB (uint8) de la imagen, en su tamaño original.
        img_display: Imagen en formato PIL, lista para mostrarse en la GUI.
    """

    img_array: np.ndarray
    img_display: Image.Image


def read_dicom_file(path: str) -> ImageReadResult:
    """Lee un archivo DICOM y lo convierte a un arreglo RGB normalizado.

    Args:
        path: Ruta al archivo DICOM (.dcm) a leer.

    Returns:
        ImageReadResult con el arreglo RGB (uint8) y la imagen para mostrar.
    """
    dataset = dicom.dcmread(path)
    pixel_array = dataset.pixel_array
    img_display = Image.fromarray(pixel_array)

    normalized = pixel_array.astype(float)
    normalized = (np.maximum(normalized, 0) / normalized.max()) * 255.0
    normalized = np.uint8(normalized)
    img_array = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

    return ImageReadResult(img_array=img_array, img_display=img_display)


def read_jpg_file(path: str) -> ImageReadResult:
    """Lee una imagen JPG/JPEG/PNG y la convierte a un arreglo RGB normalizado.

    Usa `cv2.imdecode` sobre los bytes leídos con `np.fromfile` en vez de
    `cv2.imread` directamente, porque `cv2.imread` falla silenciosamente con
    rutas que contienen caracteres Unicode (tildes, ñ) en Windows.

    Args:
        path: Ruta a la imagen (.jpg, .jpeg, .png) a leer.

    Returns:
        ImageReadResult con el arreglo RGB (uint8) y la imagen para mostrar.

    Raises:
        ValueError: Si el archivo no se pudo decodificar como imagen.
    """
    raw_bytes = np.fromfile(path, dtype=np.uint8)
    img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_display = Image.fromarray(img_rgb)

    normalized = img_rgb.astype(float)
    normalized = (np.maximum(normalized, 0) / normalized.max()) * 255.0
    img_array = np.uint8(normalized)

    return ImageReadResult(img_array=img_array, img_display=img_display)
