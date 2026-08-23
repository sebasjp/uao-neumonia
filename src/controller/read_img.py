#!/usr/bin/env python
"""Controlador: lectura de imágenes radiográficas (DICOM y JPG/PNG).

Placeholder de contrato — ver docs/CONTRATOS_MODULOS.md. La implementación
real de este módulo es responsabilidad de Cesar; estas firmas solo existen
para que src/detector_view.py sea importable mientras tanto.
"""

from typing import NamedTuple

import numpy as np
from PIL import Image


class ImageReadResult(NamedTuple):
    """Resultado de leer una imagen radiográfica.

    Attributes:
        img_array: Imagen RGB, uint8, tamaño original — insumo para preprocess().
        img_display: Imagen lista para mostrar en la GUI.
    """

    img_array: np.ndarray
    img_display: Image.Image


def read_dicom_file(path: str) -> ImageReadResult:
    """Lee un archivo DICOM (.dcm) y lo convierte a RGB.

    Args:
        path: Ruta absoluta o relativa a un archivo .dcm.

    Returns:
        ImageReadResult con la imagen leída y su versión para mostrar en la GUI.
    """
    raise NotImplementedError


def read_jpg_file(path: str) -> ImageReadResult:
    """Lee un archivo JPG/JPEG/PNG y lo convierte a RGB.

    Debe soportar rutas con caracteres no-ASCII (tildes, ñ).

    Args:
        path: Ruta absoluta o relativa a un archivo .jpg/.jpeg/.png.

    Returns:
        ImageReadResult con la imagen leída y su versión para mostrar en la
        GUI, misma estructura que read_dicom_file.

    Raises:
        ValueError: Si el archivo no pudo leerse/decodificarse.
    """
    raise NotImplementedError
