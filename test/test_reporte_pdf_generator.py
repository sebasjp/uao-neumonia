#!/usr/bin/env python
"""Pruebas para src/view/reporte_pdf_generator.py."""

from unittest.mock import patch

from PIL import Image

from src.view.reporte_pdf_generator import ReportePDFGenerator


class TestReportePDFGenerator:
    def test_generar_crea_jpg_y_pdf_en_results_dir(self, tmp_path):
        # Arrange
        generator = ReportePDFGenerator()
        captura_falsa = Image.new("RGB", (20, 20))

        # Act
        with patch(
            "src.view.reporte_pdf_generator.pyautogui.screenshot",
            return_value=captura_falsa,
        ):
            pdf_path = generator.generar(tmp_path, region=(0, 0, 20, 20))

        # Assert
        assert (tmp_path / "Reporte0.jpg").exists()
        assert (tmp_path / "Reporte0.pdf").exists()
        assert pdf_path == tmp_path / "Reporte0.pdf"

    def test_generar_incrementa_el_contador_en_llamadas_sucesivas(self, tmp_path):
        # Arrange
        generator = ReportePDFGenerator()
        captura_falsa = Image.new("RGB", (20, 20))

        # Act
        with patch(
            "src.view.reporte_pdf_generator.pyautogui.screenshot",
            return_value=captura_falsa,
        ):
            generator.generar(tmp_path, region=(0, 0, 20, 20))
            generator.generar(tmp_path, region=(0, 0, 20, 20))

        # Assert
        assert generator.report_id == 2
        assert (tmp_path / "Reporte0.pdf").exists()
        assert (tmp_path / "Reporte1.pdf").exists()

    def test_generar_crea_el_directorio_results_si_no_existe(self, tmp_path):
        # Arrange
        generator = ReportePDFGenerator()
        results_dir = tmp_path / "results"
        captura_falsa = Image.new("RGB", (10, 10))

        # Act
        with patch(
            "src.view.reporte_pdf_generator.pyautogui.screenshot",
            return_value=captura_falsa,
        ):
            generator.generar(results_dir, region=(0, 0, 10, 10))

        # Assert
        assert results_dir.exists()
