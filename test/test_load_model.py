"""Pruebas para src/model/load_model.py."""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from src.model.exceptions import ModelLoadError

# Importar el módulo real (no la función exportada por __init__.py)
lm = importlib.import_module("src.model.load_model")


class TestLoadModel:
    def setup_method(self):
        lm._model = None

    @patch("tensorflow.keras.models.load_model")
    @patch("src.model.load_model.MODEL_PATH")
    def test_returns_keras_model(self, mock_path, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_path.exists.return_value = True

        result = lm.load_model()

        assert result is mock_model
        mock_load_model.assert_called_once_with(mock_path, compile=False)

    @patch("tensorflow.keras.models.load_model")
    @patch("src.model.load_model.MODEL_PATH")
    def test_calls_load_model_with_compile_false(self, mock_path, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_path.exists.return_value = True

        lm.load_model()

        mock_load_model.assert_called_once()
        _, kwargs = mock_load_model.call_args
        assert kwargs.get("compile") is False

    @patch("tensorflow.keras.models.load_model")
    @patch("src.model.load_model.MODEL_PATH")
    def test_singleton_cache_returns_same_object(self, mock_path, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_path.exists.return_value = True

        result1 = lm.load_model()
        result2 = lm.load_model()

        assert result1 is result2
        assert mock_load_model.call_count == 1

    @patch("tensorflow.keras.models.load_model")
    @patch("src.model.load_model.MODEL_PATH")
    def test_load_model_called_once_on_multiple_calls(self, mock_path, mock_load_model):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_path.exists.return_value = True

        for _ in range(5):
            lm.load_model()

        assert mock_load_model.call_count == 1

    @patch("src.model.load_model.MODEL_PATH")
    def test_file_not_found_raises_model_load_error(self, mock_path):
        mock_path.exists.return_value = False

        with pytest.raises(ModelLoadError) as exc_info:
            lm.load_model()
        assert "no encontrado" in str(exc_info.value).lower()
        assert exc_info.value.path is not None

    @patch("tensorflow.keras.models.load_model")
    @patch("src.model.load_model.MODEL_PATH")
    def test_compile_false_avoids_keras3_reduction_error(
        self, mock_path, mock_load_model
    ):
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_path.exists.return_value = True

        lm.load_model()

        _, kwargs = mock_load_model.call_args
        assert kwargs.get("compile") is False
