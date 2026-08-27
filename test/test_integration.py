"""Pruebas de integración end-to-end: Vista -> Integrador -> Controlador -> Modelo.

Cubre la sección "Integración end-to-end" de docs/PLAN_PRUEBAS.md. Solo se
mockea lo verdaderamente pesado o no reproducible en CI: el modelo Keras
real (el .h5 de producción se reemplaza por un modelo dummy, mínimo pero con
la misma arquitectura que espera grad_cam.py) y la captura de pantalla real
(pyautogui.screenshot). Todo lo demás -lectura de imagen, preprocesamiento,
orquestación del integrador, cálculo real de Grad-CAM y wiring de la Vista-
corre con el código real de cada módulo.
"""

import importlib
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.controller.integrator import PredictionResult, predict
from src.controller.read_img import read_dicom_file, read_jpg_file

# Fixtures `tk_root` y `app`: ver test/conftest.py.

lm = importlib.import_module("src.model.load_model")


def _build_dummy_model() -> tf.keras.Model:
    """Construye un modelo Keras mínimo pero diferenciable y compatible con grad_cam.py."""
    inputs = tf.keras.Input(shape=(512, 512, 1))
    x = tf.keras.layers.Conv2D(
        2, 3, padding="same", activation="relu", name="conv10_thisone"
    )(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _fake_dicom_dataset(pixel_array: np.ndarray):
    """Réplica del helper de test_read_img.py para simular un dataset de pydicom."""

    class _FakeDataset:
        pass

    dataset = _FakeDataset()
    dataset.pixel_array = pixel_array
    return dataset


@pytest.fixture(scope="module")
def dummy_model() -> tf.keras.Model:
    """Modelo dummy compartido entre pruebas — construirlo es lo más costoso."""
    return _build_dummy_model()


@pytest.fixture
def modelo_cargado(dummy_model):
    """Inyecta el modelo dummy como singleton de load_model() y limpia después."""
    lm._model = dummy_model
    yield dummy_model
    lm._model = None


@pytest.fixture
def imagen_jpg(tmp_path):
    """Crea un archivo JPG real en disco para ejercitar read_jpg_file real."""
    array = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    path = tmp_path / "radiografia.jpg"
    cv2.imwrite(str(path), array)
    return path


class TestPredictEndToEnd:
    """predict() con preprocess/load_model/grad_cam reales (solo el modelo es dummy)."""

    def test_predict_con_imagen_jpg_real_retorna_prediction_result(
        self, modelo_cargado, imagen_jpg
    ):
        # Arrange
        lectura = read_jpg_file(str(imagen_jpg))

        # Act
        resultado = predict(lectura.img_array)

        # Assert
        assert isinstance(resultado, PredictionResult)
        assert resultado.label in {"bacteriana", "normal", "viral"}
        assert 0.0 <= resultado.probability <= 100.0
        assert resultado.heatmap.shape == (512, 512, 3)
        assert resultado.heatmap.dtype == np.uint8

    def test_predict_con_imagen_dicom_real_retorna_prediction_result(
        self, modelo_cargado
    ):
        # Arrange
        pixel_array = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
        fake_dataset = _fake_dicom_dataset(pixel_array)
        with patch("src.controller.read_img.dicom.dcmread", return_value=fake_dataset):
            lectura = read_dicom_file("cualquier_ruta.dcm")

        # Act
        resultado = predict(lectura.img_array)

        # Assert
        assert isinstance(resultado, PredictionResult)
        assert resultado.heatmap.shape == (512, 512, 3)

    def test_predict_end_to_end_es_consistente_entre_llamadas(
        self, modelo_cargado, imagen_jpg
    ):
        # Arrange
        lectura = read_jpg_file(str(imagen_jpg))

        # Act
        resultado_1 = predict(lectura.img_array)
        resultado_2 = predict(lectura.img_array)

        # Assert
        assert resultado_1.label == resultado_2.label
        assert resultado_1.probability == pytest.approx(resultado_2.probability)

    def test_predict_end_to_end_reutiliza_el_modelo_cargado(
        self, modelo_cargado, imagen_jpg
    ):
        # Arrange
        lectura = read_jpg_file(str(imagen_jpg))

        # Act
        with patch("tensorflow.keras.models.load_model") as mock_tf_load:
            predict(lectura.img_array)
            predict(lectura.img_array)

        # Assert: el singleton ya estaba cacheado (fixture modelo_cargado),
        # así que el loader real de TensorFlow nunca debería tocarse.
        mock_tf_load.assert_not_called()

    def test_predict_end_to_end_heatmap_no_es_degenerado(
        self, modelo_cargado, imagen_jpg
    ):
        # Arrange
        lectura = read_jpg_file(str(imagen_jpg))

        # Act
        resultado = predict(lectura.img_array)

        # Assert: el heatmap real no debe quedar completamente vacío/negro
        assert resultado.heatmap.max() > 0

    def test_predict_end_to_end_propaga_value_error_con_imagen_invalida(self):
        # Arrange: imagen sin los 3 canales BGR que exige el contrato
        imagen_invalida = np.zeros((100, 100), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError):
            predict(imagen_invalida)


class TestVistaEndToEnd:
    """App usando read_img/integrator reales; solo se mockean diálogos de Tkinter y la captura de pantalla."""

    def test_cargar_y_predecir_imagen_jpg_real_puebla_la_gui(
        self, app, modelo_cargado, imagen_jpg
    ):
        # Arrange
        with patch(
            "src.view.detector_view.filedialog.askopenfilename",
            return_value=str(imagen_jpg),
        ):
            app.load_img_file()

        # Act
        app.run_model()

        # Assert
        assert app.text2.get("1.0", "end-1c") in {"bacteriana", "normal", "viral"}
        assert app.text3.get("1.0", "end-1c").endswith("%")

    def test_cargar_imagen_dicom_real_puebla_la_gui(self, app, modelo_cargado):
        # Arrange
        pixel_array = np.random.randint(0, 4096, (256, 256), dtype=np.uint16)
        fake_dataset = _fake_dicom_dataset(pixel_array)

        # Act
        with (
            patch(
                "src.view.detector_view.filedialog.askopenfilename",
                return_value="radiografia.dcm",
            ),
            patch("src.controller.read_img.dicom.dcmread", return_value=fake_dataset),
        ):
            app.load_img_file()
            app.run_model()

        # Assert
        assert app.text2.get("1.0", "end-1c") in {"bacteriana", "normal", "viral"}

    def test_guardar_despues_de_prediccion_real_escribe_fila_en_csv(
        self, app, modelo_cargado, imagen_jpg, tmp_path, monkeypatch
    ):
        # Arrange
        historial = tmp_path / "historial.csv"
        monkeypatch.setattr("src.view.detector_view.HISTORIAL_PATH", historial)
        with patch(
            "src.view.detector_view.filedialog.askopenfilename",
            return_value=str(imagen_jpg),
        ):
            app.load_img_file()
        app.run_model()
        app.text1.insert(0, "12345678")

        # Act
        with patch("src.view.detector_view.showinfo"):
            app.save_results_csv()

        # Assert
        contenido = historial.read_text()
        assert "12345678" in contenido
        assert app.result.label in contenido

    def test_generar_pdf_despues_de_prediccion_real_crea_archivos(
        self, app, modelo_cargado, imagen_jpg, tmp_path, monkeypatch
    ):
        # Arrange
        monkeypatch.setattr("src.view.detector_view.RESULTS_DIR", tmp_path)
        with patch(
            "src.view.detector_view.filedialog.askopenfilename",
            return_value=str(imagen_jpg),
        ):
            app.load_img_file()
        app.run_model()
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

    def test_cargar_imagen_no_decodificable_propaga_value_error_real(
        self, app, tmp_path
    ):
        # Arrange
        archivo_invalido = tmp_path / "no_es_una_imagen.jpg"
        archivo_invalido.write_bytes(b"esto no es una imagen valida")

        # Act & Assert
        with (
            patch(
                "src.view.detector_view.filedialog.askopenfilename",
                return_value=str(archivo_invalido),
            ),
            pytest.raises(ValueError),
        ):
            app.load_img_file()
