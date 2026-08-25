"""Pruebas unitarias para src/controller/preprocess_img.py."""

import numpy as np
import pytest

from src.controller.preprocess_img import preprocess


class TestPreprocessImg:
    """Suite de pruebas unitarias para la función preprocess."""

    def test_output_shape(self):
        """Verifica que la salida tenga el shape exacto (1, 512, 512, 1)."""
        # Arrange
        img_bgr = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        # Act
        result = preprocess(img_bgr)

        # Assert
        assert result.shape == (1, 512, 512, 1)

    def test_output_dtype(self):
        """Verifica que el tipo de datos retornado sea float64."""
        # Arrange
        img_bgr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

        # Act
        result = preprocess(img_bgr)

        # Assert
        assert result.dtype == np.float64

    def test_min_value_in_range(self):
        """Verifica que el valor mínimo de los píxeles procesados sea >= 0.0."""
        # Arrange
        img_bgr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        # Act
        result = preprocess(img_bgr)

        # Assert
        assert np.min(result) >= 0.0

    def test_max_value_in_range(self):
        """Verifica que el valor máximo de los píxeles procesados sea <= 1.0."""
        # Arrange
        img_bgr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        # Act
        result = preprocess(img_bgr)

        # Assert
        assert np.max(result) <= 1.0

    def test_resize_smaller_image(self):
        """Verifica el reescalado correcto de una imagen pequeña (100x100)."""
        # Arrange
        img_small = np.full((100, 100, 3), 128, dtype=np.uint8)

        # Act
        result = preprocess(img_small)

        # Assert
        assert result.shape == (1, 512, 512, 1)

    def test_resize_non_square_image(self):
        """Verifica el reescalado correcto de una imagen rectangular (800x600)."""
        # Arrange
        img_rect = np.full((800, 600, 3), 100, dtype=np.uint8)

        # Act
        result = preprocess(img_rect)

        # Assert
        assert result.shape == (1, 512, 512, 1)

    def test_black_image_processing(self):
        """Verifica que una imagen completamente negra no cause división por cero."""
        # Arrange
        img_black = np.zeros((512, 512, 3), dtype=np.uint8)

        # Act
        result = preprocess(img_black)

        # Assert
        assert result.shape == (1, 512, 512, 1)
        assert not np.isnan(result).any()

    def test_white_image_processing(self):
        """Verifica el procesamiento correcto de una imagen completamente blanca."""
        # Arrange
        img_white = np.full((512, 512, 3), 255, dtype=np.uint8)

        # Act
        result = preprocess(img_white)

        # Assert
        assert result.shape == (1, 512, 512, 1)
        assert not np.isnan(result).any()

    def test_grayscale_single_channel_expansion(self):
        """Verifica la reducción de 3 canales BGR a 1 canal en la última dimensión."""
        # Arrange
        img_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_bgr[:, :, 0] = 255  # Solo canal azul activo

        # Act
        result = preprocess(img_bgr)

        # Assert
        assert result.shape[-1] == 1

    def test_clahe_enhancement_effect(self):
        """Verifica que el ecualizador CLAHE altere la distribución del contraste."""
        # Arrange
        img_low_contrast = np.full((512, 512, 3), 100, dtype=np.uint8)
        img_low_contrast[200:300, 200:300, :] = 110  # Variación pequeña de contraste

        # Act
        result = preprocess(img_low_contrast)

        # Assert
        assert result.shape == (1, 512, 512, 1)
        assert np.max(result) > np.min(result)

    def test_raise_error_on_none_input(self):
        """Lanza ValueError o TypeError al recibir None como entrada."""
        # Arrange
        invalid_input = None

        # Act & Assert
        with pytest.raises((ValueError, TypeError)):
            preprocess(invalid_input)

    def test_raise_error_on_2d_input(self):
        """Lanza ValueError al recibir una imagen 2D sin canal de color."""
        # Arrange
        img_2d = np.zeros((512, 512), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError):
            preprocess(img_2d)

    def test_raise_error_on_4d_input(self):
        """Lanza ValueError al recibir una entrada de 4 dimensiones (batch)."""
        # Arrange
        img_4d = np.zeros((1, 512, 512, 3), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError):
            preprocess(img_4d)

    def test_raise_error_on_empty_image(self):
        """Lanza ValueError al recibir un array vacío de shape (0, 0, 3)."""
        # Arrange
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError):
            preprocess(empty_img)

    def test_input_immutability(self):
        """Verifica que la imagen original recibida no sea modificada por la función."""
        # Arrange
        img_original = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        img_copy = img_original.copy()

        # Act
        _ = preprocess(img_original)

        # Assert
        np.testing.assert_array_equal(img_original, img_copy)

    def test_non_uint8_dtype_conversion(self):
        """Soporta arreglos numéricos que no sean uint8 convirtiéndolos o procesándolos."""
        # Arrange
        img_float = np.random.randint(0, 256, (200, 200, 3)).astype(np.float32)

        # Act
        result = preprocess(img_float)

        # Assert
        assert result.shape == (1, 512, 512, 1)
        assert result.dtype == np.float64

    def test_no_nan_or_inf_in_output(self):
        """Garantiza que la salida no contenga valores NaN ni Infloats."""
        # Arrange
        img_bgr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

        # Act
        result = preprocess(img_bgr)

        # Assert
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_purity_and_consistency(self):
        """Garantiza idempotencia: dos ejecuciones con el mismo input entregan la misma salida."""
        # Arrange
        img_bgr = np.random.randint(0, 256, (250, 250, 3), dtype=np.uint8)

        # Act
        res1 = preprocess(img_bgr)
        res2 = preprocess(img_bgr)

        # Assert
        np.testing.assert_array_equal(res1, res2)

    def test_batch_dimension_is_one(self):
        """Verifica explícitamente que la primera dimensión (batch) sea 1."""
        # Arrange
        img_bgr = np.full((400, 400, 3), 150, dtype=np.uint8)

        # Act
        result = preprocess(img_bgr)

        # Assert
        assert result.shape[0] == 1

    def test_single_channel_dimension_is_one(self):
        """Verifica explícitamente que la última dimensión (canal) sea 1."""
        # Arrange
        img_bgr = np.full((400, 400, 3), 150, dtype=np.uint8)

        # Act
        result = preprocess(img_bgr)

        # Assert
        assert result.shape[3] == 1
