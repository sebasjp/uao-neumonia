#!/usr/bin/env python
"""Pruebas para src/view/detector_view.py."""

import csv
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.controller.integrator import PredictionResult
from src.controller.read_img import ImageReadResult, read_dicom_file, read_jpg_file
from src.view.detector_view import (
    elegir_lector,
    formatear_probabilidad,
    validar_cedula,
)

# Fixtures `tk_root` y `app`: ver test/conftest.py.


class TestValidarCedula:
    @pytest.mark.parametrize(
        "texto, esperado",
        [
            ("", False),
            ("   ", False),
            ("12345678", True),
            ("  12345678  ", True),
        ],
    )
    def test_valida_segun_contenido(self, texto, esperado):
        # Arrange: `texto` y `esperado` provienen del parametrize

        # Act
        resultado = validar_cedula(texto)

        # Assert
        assert resultado is esperado


class TestFormatearProbabilidad:
    @pytest.mark.parametrize(
        "proba, esperado",
        [
            (87.456, "87.46%"),
            (100.0, "100.00%"),
            (0.0, "0.00%"),
            (33.333, "33.33%"),
        ],
    )
    def test_formatea_con_dos_decimales_y_simbolo_porcentaje(self, proba, esperado):
        # Arrange: `proba` y `esperado` provienen del parametrize

        # Act
        resultado = formatear_probabilidad(proba)

        # Assert
        assert resultado == esperado


class TestElegirLector:
    @pytest.mark.parametrize(
        "path, esperado",
        [
            ("radiografia.dcm", read_dicom_file),
            ("radiografia.DCM", read_dicom_file),
            ("radiografia.jpg", read_jpg_file),
            ("radiografia.jpeg", read_jpg_file),
            ("radiografia.png", read_jpg_file),
        ],
    )
    def test_despacha_segun_extension(self, path, esperado):
        # Arrange: `path` y `esperado` provienen del parametrize

        # Act
        lector = elegir_lector(path)

        # Assert
        assert lector is esperado

    def test_extension_no_soportada_lanza_value_error(self):
        # Arrange
        path = "radiografia.bmp"

        # Act / Assert
        with pytest.raises(ValueError):
            elegir_lector(path)


class TestApp:
    def test_boton_predecir_deshabilitado_hasta_cargar_imagen(self, app):
        # Arrange: `app` recién construida, sin imagen cargada

        # Act: no hay acción, se evalúa el estado inicial

        # Assert
        assert str(app.button1["state"]) == "disabled"

    def test_cargar_imagen_habilita_predecir_y_limpia_cedula(self, app):
        # Arrange
        app.text1.insert(0, "99999999")
        resultado_lectura = ImageReadResult(
            img_array=np.zeros((10, 10, 3), dtype=np.uint8),
            img_display=Image.new("RGB", (10, 10)),
        )

        # Act
        with (
            patch(
                "src.view.detector_view.filedialog.askopenfilename",
                return_value="foto.jpg",
            ),
            patch(
                "src.view.detector_view.read_jpg_file", return_value=resultado_lectura
            ),
        ):
            app.load_img_file()

        # Assert
        assert app.text1.get() == ""
        assert str(app.button1["state"]) == "enabled"

    def test_cargar_imagen_cancelada_no_cambia_estado(self, app):
        # Arrange: el diálogo de selección se simula cancelado (retorna "")

        # Act
        with patch(
            "src.view.detector_view.filedialog.askopenfilename", return_value=""
        ):
            app.load_img_file()

        # Assert
        assert str(app.button1["state"]) == "disabled"

    def test_predecir_dos_veces_no_acumula_texto(self, app):
        # Arrange
        app.array = np.zeros((10, 10, 3), dtype=np.uint8)
        resultado = PredictionResult(
            label="normal",
            probability=87.5,
            heatmap=np.zeros((512, 512, 3), dtype=np.uint8),
        )

        # Act
        with patch("src.view.detector_view.predict", return_value=resultado):
            app.run_model()
            app.run_model()

        # Assert
        assert app.text2.get("1.0", "end-1c") == "normal"
        assert app.text3.get("1.0", "end-1c") == "87.50%"

    def test_borrar_no_lanza_tclerror_y_limpia_resultado(self, app):
        # Arrange
        app.array = np.zeros((10, 10, 3), dtype=np.uint8)
        resultado = PredictionResult(
            label="viral",
            probability=42.0,
            heatmap=np.zeros((512, 512, 3), dtype=np.uint8),
        )
        with patch("src.view.detector_view.predict", return_value=resultado):
            app.run_model()

        # Act
        with (
            patch("src.view.detector_view.askokcancel", return_value=True),
            patch("src.view.detector_view.showinfo"),
        ):
            app.delete()

        # Assert
        assert app.text2.get("1.0", "end-1c") == ""
        assert app.text3.get("1.0", "end-1c") == ""

    def test_guardar_sin_cedula_no_escribe(self, app, tmp_path, monkeypatch):
        # Arrange
        historial = tmp_path / "historial.csv"
        monkeypatch.setattr("src.view.detector_view.HISTORIAL_PATH", historial)

        # Act
        with patch("src.view.detector_view.showinfo") as mock_showinfo:
            app.save_results_csv()

        # Assert
        mock_showinfo.assert_called_once()
        assert not historial.exists()

    def test_guardar_con_cedula_escribe_fila_esperada(self, app, tmp_path, monkeypatch):
        # Arrange
        historial = tmp_path / "historial.csv"
        monkeypatch.setattr("src.view.detector_view.HISTORIAL_PATH", historial)
        app.text1.insert(0, "12345678")
        app.result = PredictionResult(
            label="bacteriana",
            probability=91.234,
            heatmap=np.zeros((512, 512, 3), dtype=np.uint8),
        )

        # Act
        with patch("src.view.detector_view.showinfo"):
            app.save_results_csv()

        # Assert
        with open(historial, newline="") as f:
            filas = list(csv.reader(f, delimiter="-"))
        assert filas == [["12345678", "bacteriana", "91.23%"]]

    def test_pdf_sin_cedula_no_genera_reporte(self, app):
        # Arrange: cédula vacía por defecto

        # Act
        with (
            patch("src.view.detector_view.showinfo") as mock_showinfo,
            patch.object(app.reporte_generator, "generar") as mock_generar,
        ):
            app.create_pdf()

        # Assert
        mock_showinfo.assert_called_once()
        mock_generar.assert_not_called()

    def test_pdf_con_cedula_genera_archivos_en_results_dir(
        self, app, tmp_path, monkeypatch
    ):
        # Arrange
        monkeypatch.setattr("src.view.detector_view.RESULTS_DIR", tmp_path)
        app.text1.insert(0, "12345678")
        captura_falsa = Image.new("RGB", (20, 20))

        # Act
        with (
            patch(
                "src.view.reporte_pdf_generator.pyautogui.screenshot",
                return_value=captura_falsa,
            ),
            patch("src.view.detector_view.showinfo"),
        ):
            app.create_pdf()

        # Assert
        assert (tmp_path / "Reporte0.jpg").exists()
        assert (tmp_path / "Reporte0.pdf").exists()
        assert app.reporte_generator.report_id == 1
