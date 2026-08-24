#!/usr/bin/env python
"""Helper interno de la Vista: generación de reportes JPG+PDF."""

from pathlib import Path

import pyautogui
from PIL import Image


class ReportePDFGenerator:
    """Genera reportes JPG+PDF a partir de una captura de pantalla."""

    def __init__(self) -> None:
        """Inicializa el generador con el contador de reportes en cero."""
        self.report_id = 0

    def generar(self, results_dir: Path, region: tuple[int, int, int, int]) -> Path:
        """Captura una región de pantalla y genera el reporte JPG+PDF.

        Incrementa el contador interno de reportes tras generar, así cada
        llamada produce un nombre distinto (Reporte0, Reporte1, ...).

        Args:
            results_dir: Directorio donde se guardan Reporte{N}.jpg/.pdf.
                Se crea si no existe.
            region: Región de pantalla a capturar, como (x, y, width, height).

        Returns:
            Ruta del PDF generado.
        """
        results_dir.mkdir(parents=True, exist_ok=True)
        jpg_path = results_dir / f"Reporte{self.report_id}.jpg"
        pdf_path = results_dir / f"Reporte{self.report_id}.pdf"
        screenshot = pyautogui.screenshot(region=region)
        screenshot.save(jpg_path)
        Image.open(jpg_path).convert("RGB").save(pdf_path)
        self.report_id += 1
        return pdf_path
