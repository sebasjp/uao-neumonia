"""Pruebas unitarias para src.controller.read_img."""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.controller.read_img import ImageReadResult, read_dicom_file, read_jpg_file


def _fake_dicom_dataset(pixel_array):
    """Crea un objeto simulado con el atributo `pixel_array` esperado por pydicom."""

    class _FakeDataset:
        pass

    dataset = _FakeDataset()
    dataset.pixel_array = pixel_array
    return dataset


class TestReadDicomFile:
    """Pruebas de read_dicom_file."""

    def test_retorna_image_read_result(self):
        # Arrange
        pixel_array = np.full((10, 10), 100, dtype=np.uint16)
        fake_dataset = _fake_dicom_dataset(pixel_array)

        # Act
        with patch("src.controller.read_img.dicom.dcmread", return_value=fake_dataset):
            result = read_dicom_file("cualquier_ruta.dcm")

        # Assert
        assert isinstance(result, ImageReadResult)

    def test_img_array_es_rgb_de_tres_canales(self):
        # Arrange
        pixel_array = np.full((10, 10), 100, dtype=np.uint16)
        fake_dataset = _fake_dicom_dataset(pixel_array)

        # Act
        with patch("src.controller.read_img.dicom.dcmread", return_value=fake_dataset):
            result = read_dicom_file("cualquier_ruta.dcm")

        # Assert
        assert result.img_array.shape == (10, 10, 3)

    def test_img_array_es_uint8(self):
        # Arrange
        pixel_array = np.full((10, 10), 100, dtype=np.uint16)
        fake_dataset = _fake_dicom_dataset(pixel_array)

        # Act
        with patch("src.controller.read_img.dicom.dcmread", return_value=fake_dataset):
            result = read_dicom_file("cualquier_ruta.dcm")

        # Assert
        assert result.img_array.dtype == np.uint8

    def test_shape_coincide_con_la_imagen_original(self):
        # Arrange
        pixel_array = np.random.randint(0, 4096, size=(64, 128), dtype=np.uint16)
        fake_dataset = _fake_dicom_dataset(pixel_array)

        # Act
        with patch("src.controller.read_img.dicom.dcmread", return_value=fake_dataset):
            result = read_dicom_file("cualquier_ruta.dcm")

        # Assert
        assert result.img_array.shape[:2] == (64, 128)

    def test_normaliza_pixel_data_de_16_bits_al_rango_0_255(self):
        # Arrange: valores típicos de un DICOM de 16 bits, muy por encima de 255
        pixel_array = np.array([[0, 4095], [2048, 4095]], dtype=np.uint16)
        fake_dataset = _fake_dicom_dataset(pixel_array)

        # Act
        with patch("src.controller.read_img.dicom.dcmread", return_value=fake_dataset):
            result = read_dicom_file("cualquier_ruta.dcm")

        # Assert: el valor máximo original debe mapear al máximo de 8 bits
        assert result.img_array.max() == 255
        assert result.img_array.min() == 0

    def test_ruta_inexistente_lanza_error_de_pydicom(self):
        # Arrange / Act / Assert: sin mock, pydicom.dcmread debe fallar solo
        with pytest.raises(FileNotFoundError):
            read_dicom_file("ruta/que/no/existe.dcm")


class TestReadJpgFile:
    """Pruebas de read_jpg_file."""

    def test_retorna_image_read_result(self, tmp_path):
        # Arrange
        path = tmp_path / "radiografia.jpg"
        Image.new("RGB", (20, 20), color=(120, 80, 200)).save(path)

        # Act
        result = read_jpg_file(str(path))

        # Assert
        assert isinstance(result, ImageReadResult)

    @pytest.mark.parametrize("extension", ["jpg", "jpeg", "png"])
    def test_lee_distintas_extensiones_de_imagen(self, tmp_path, extension):
        # Arrange
        path = tmp_path / f"radiografia.{extension}"
        Image.new("RGB", (16, 16), color=(50, 60, 70)).save(path)

        # Act
        result = read_jpg_file(str(path))

        # Assert
        assert result.img_array.shape == (16, 16, 3)

    @pytest.mark.parametrize("size", [(32, 32), (100, 200), (256, 128)])
    def test_parametrizado_sobre_resoluciones_de_entrada(self, tmp_path, size):
        # Arrange
        width, height = size
        path = tmp_path / "radiografia.jpg"
        Image.new("RGB", (width, height), color=(10, 20, 30)).save(path)

        # Act
        result = read_jpg_file(str(path))

        # Assert
        assert result.img_array.shape[:2] == (height, width)

    def test_ruta_con_tildes_y_enie_se_lee_correctamente(self, tmp_path):
        # Arrange — regresión bug #11 (DEBUGGING_MONOLITO.md): cv2.imread
        # fallaba silenciosamente con rutas Unicode en Windows.
        path = tmp_path / "radiografía_paño.jpg"
        Image.new("RGB", (10, 10), color=(200, 100, 50)).save(path)

        # Act
        result = read_jpg_file(str(path))

        # Assert
        assert isinstance(result, ImageReadResult)
        assert result.img_array.shape == (10, 10, 3)

    def test_imagen_no_decodificable_lanza_value_error(self, tmp_path):
        # Arrange — regresión bug #11: antes fallaba silenciosamente (None)
        path = tmp_path / "no_es_una_imagen.jpg"
        path.write_bytes(b"esto no son bytes de una imagen valida")

        # Act / Assert
        with pytest.raises(ValueError):
            read_jpg_file(str(path))

    def test_ruta_inexistente_lanza_error_claro(self, tmp_path):
        # Arrange
        path = tmp_path / "no_existe.jpg"

        # Act / Assert: np.fromfile falla antes de llegar a cv2.imdecode
        with pytest.raises(FileNotFoundError):
            read_jpg_file(str(path))

    def test_imagen_en_escala_de_grises_se_convierte_a_tres_canales(self, tmp_path):
        # Arrange
        path = tmp_path / "radiografia_gris.jpg"
        Image.new("L", (12, 12), color=128).save(path)

        # Act
        result = read_jpg_file(str(path))

        # Assert: cv2.IMREAD_COLOR fuerza siempre 3 canales
        assert result.img_array.shape == (12, 12, 3)

    def test_img_array_es_uint8(self, tmp_path):
        # Arrange
        path = tmp_path / "radiografia.jpg"
        Image.new("RGB", (10, 10), color=(30, 40, 50)).save(path)

        # Act
        result = read_jpg_file(str(path))

        # Assert
        assert result.img_array.dtype == np.uint8

    def test_canales_quedan_en_orden_rgb_no_bgr(self, tmp_path):
        # Arrange: color asimétrico para detectar si los canales quedaron invertidos
        path = tmp_path / "radiografia_color.jpg"
        Image.new("RGB", (5, 5), color=(200, 30, 10)).save(path)

        # Act
        result = read_jpg_file(str(path))

        # Assert: el canal rojo debe seguir siendo el dominante (no invertido a azul)
        avg_r, _avg_g, avg_b = result.img_array.mean(axis=(0, 1))
        assert avg_r > avg_b
