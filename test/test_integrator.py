"""Pruebas unitarias para src/controller/integrator.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.controller.integrator import PredictionResult, predict


class TestIntegrator:
    """Suite de 20 pruebas unitarias para el orquestador integrator.py."""

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture para mockear las dependencias externas del integrador."""
        with (
            patch("src.controller.integrator.preprocess") as mock_prep,
            patch("src.controller.integrator.load_model") as mock_load,
            patch("src.controller.integrator.generate_gradcam") as mock_gcam,
        ):
            # Setup mock model
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            # Setup mock preprocess result
            mock_prep.return_value = np.zeros((1, 512, 512, 1), dtype=np.float64)

            # Setup mock gradcam result
            mock_heatmap = np.ones((512, 512, 3), dtype=np.uint8) * 255
            mock_gcam.return_value = mock_heatmap

            yield mock_prep, mock_load, mock_model, mock_gcam

    def test_returns_prediction_result_namedtuple(self, mock_dependencies):
        """1. Verifica que predict retorne una instancia de PredictionResult."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        result = predict(img)

        # Assert
        assert isinstance(result, PredictionResult)

    def test_predict_calls_preprocess_exactly_once(self, mock_dependencies):
        """2. Verifica que preprocess sea llamado exactamente una vez con la imagen de entrada."""
        # Arrange
        mock_prep, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        _ = predict(img)

        # Assert
        mock_prep.assert_called_once_with(img)

    def test_predict_calls_load_model_exactly_once(self, mock_dependencies):
        """3. Verifica que load_model sea llamado exactamente una vez."""
        # Arrange
        _, mock_load, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        _ = predict(img)

        # Assert
        mock_load.assert_called_once()

    def test_predict_calls_generate_gradcam_exactly_once(self, mock_dependencies):
        """4. Verifica que generate_gradcam sea llamado exactamente una vez con los argumentos correctos."""
        # Arrange
        mock_prep, _, mock_model, mock_gcam = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        _ = predict(img)

        # Assert
        mock_gcam.assert_called_once()
        args, _ = mock_gcam.call_args
        assert np.array_equal(args[0], mock_prep.return_value)
        assert np.array_equal(args[1], img)
        assert args[2] is mock_model

    def test_predict_class_0_maps_to_bacteriana(self, mock_dependencies):
        """5. Verifica que el índice de clase 0 se traduzca como 'bacteriana'."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.9, 0.05, 0.05]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        result = predict(img)

        # Assert
        assert result.label == "bacteriana"

    def test_predict_class_1_maps_to_normal(self, mock_dependencies):
        """6. Verifica que el índice de clase 1 se traduzca como 'normal'."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.05, 0.9, 0.05]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        result = predict(img)

        # Assert
        assert result.label == "normal"

    def test_predict_class_2_maps_to_viral(self, mock_dependencies):
        """7. Verifica que el índice de clase 2 se traduzca como 'viral'."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.05, 0.05, 0.9]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        result = predict(img)

        # Assert
        assert result.label == "viral"

    def test_predict_returns_correct_probability(self, mock_dependencies):
        """8. Verifica que la probabilidad sea extraída de la predicción y multiplicada por 100."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.85, 0.05]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        result = predict(img)

        # Assert
        assert result.probability == pytest.approx(85.0)

    def test_predict_returns_correct_heatmap(self, mock_dependencies):
        """9. Verifica que el heatmap de salida sea el que retorna generate_gradcam."""
        # Arrange
        _, _, mock_model, mock_gcam = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        result = predict(img)

        # Assert
        assert np.array_equal(result.heatmap, mock_gcam.return_value)

    def test_predict_raises_value_error_on_none_input(self, mock_dependencies):
        """10. Verifica que predict lance ValueError si el input es None."""
        # Arrange
        img = None

        # Act & Assert
        with pytest.raises(ValueError):
            predict(img)

    def test_predict_raises_type_error_on_non_ndarray_input(self, mock_dependencies):
        """11. Verifica que predict lance TypeError si el input no es np.ndarray."""
        # Arrange
        img = "not an array"

        # Act & Assert
        with pytest.raises(TypeError):
            predict(img)

    def test_predict_raises_value_error_on_invalid_dimension(self, mock_dependencies):
        """12. Verifica que lance ValueError si la imagen no tiene 3 canales."""
        # Arrange
        img = np.zeros((100, 100), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError):
            predict(img)

    def test_predict_propagates_model_load_exception(self, mock_dependencies):
        """13. Verifica que se propaguen las excepciones al cargar el modelo."""
        # Arrange
        _, mock_load, _, _ = mock_dependencies
        mock_load.side_effect = RuntimeError("Error de carga del modelo")
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(RuntimeError, match="Error de carga del modelo"):
            predict(img)

    def test_predict_propagates_gradcam_exception(self, mock_dependencies):
        """14. Verifica que se propaguen las excepciones al generar Grad-CAM."""
        # Arrange
        _, _, mock_model, mock_gcam = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        mock_gcam.side_effect = ValueError("Error de Grad-CAM")
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError, match="Error de Grad-CAM"):
            predict(img)

    def test_predict_probability_is_float(self, mock_dependencies):
        """15. Verifica que el tipo de datos de probability sea un float nativo o compatible."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        result = predict(img)

        # Assert
        assert isinstance(result.probability, float)

    def test_predict_label_is_valid_category(self, mock_dependencies):
        """16. Verifica que la etiqueta predicha esté exactamente en {"bacteriana", "normal", "viral"}."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.0, 0.0, 1.0]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        result = predict(img)

        # Assert
        assert result.label in {"bacteriana", "normal", "viral"}

    def test_predict_handles_multiple_predictions_consistently(self, mock_dependencies):
        """17. Idempotencia: predicciones sucesivas con idéntico input son idénticas."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        res1 = predict(img)
        res2 = predict(img)

        # Assert
        assert res1.label == res2.label
        assert res1.probability == res2.probability
        np.testing.assert_array_equal(res1.heatmap, res2.heatmap)

    def test_predict_does_not_modify_original_image(self, mock_dependencies):
        """18. Inmutabilidad: la imagen original no es mutada por el integrador."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img_copy = img.copy()

        # Act
        _ = predict(img)

        # Assert
        np.testing.assert_array_equal(img, img_copy)

    def test_predict_handles_float32_model_output(self, mock_dependencies):
        """19. Robustez: maneja outputs del modelo de tipo float32 correctamente."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array(
            [[0.05, 0.05, 0.90]], dtype=np.float32
        )
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

        # Act
        result = predict(img)

        # Assert
        assert result.label == "viral"
        assert result.probability == pytest.approx(90.0)

    def test_prediction_result_fields_are_read_only(self, mock_dependencies):
        """20. Estructura de Datos: la NamedTuple PredictionResult es inmutable."""
        # Arrange
        _, _, mock_model, _ = mock_dependencies
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = predict(img)

        # Act & Assert
        with pytest.raises(AttributeError):
            result.label = "new_label"
