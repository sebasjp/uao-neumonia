"""Pruebas para src/model/grad_cam.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.model.grad_cam as gc_module
from src.model.exceptions import GradCAMError


class MockTensor:
    """Mock que se comporta como un tensor de TF: soporta indexado y .numpy()."""

    def __init__(self, array):
        self._array = array

    def __getitem__(self, key):
        if isinstance(key, tuple):
            new_key = []
            for k in key:
                if isinstance(k, MockTensor):
                    new_key.append(k._array)
                else:
                    new_key.append(k)
            key = tuple(new_key)
        return MockTensor(self._array[key])

    def numpy(self):
        return self._array


def make_mock_model():
    model = MagicMock()
    mock_conv_layer = MagicMock()
    mock_conv_layer.output = MagicMock()
    model.get_layer.return_value = mock_conv_layer
    model.inputs = [MagicMock()]
    model.outputs = [MagicMock()]
    return model


@pytest.fixture
def mock_model():
    return make_mock_model()


@pytest.fixture
def preprocessed_array():
    return np.random.rand(1, 512, 512, 1).astype(np.float64)


@pytest.fixture
def original_array():
    return np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)


def setup_gradcam_mocks(
    mock_tf,
    mock_model,
    conv_output_shape=(1, 16, 16, 64),
    probs=None,
    grads_shape=(1, 16, 16, 64),
    argmax_value=1,
):
    if probs is None:
        probs = [0.1, 0.7, 0.2]

    conv_output_array = np.random.rand(*conv_output_shape).astype(np.float32)
    mock_conv_output = MockTensor(conv_output_array)

    preds_array = np.array([probs], dtype=np.float32)
    mock_preds = MockTensor(preds_array)

    grads_array = np.random.rand(*grads_shape).astype(np.float32)
    mock_grads = MockTensor(grads_array)

    mock_grad_model = MagicMock()
    mock_grad_model.return_value = (mock_conv_output, mock_preds)
    mock_tf.keras.models.Model.return_value = mock_grad_model

    # Configurar GradientTape con context manager
    mock_tape = MagicMock()
    mock_tape.gradient.return_value = mock_grads
    mock_tape.__enter__ = MagicMock(return_value=mock_tape)
    mock_tape.__exit__ = MagicMock(return_value=None)
    mock_tf.GradientTape.return_value = mock_tape

    mock_tf.argmax.return_value = MockTensor(np.array(argmax_value, dtype=np.int64))
    mock_tf.reduce_mean.return_value = MockTensor(np.ones(64, dtype=np.float32))
    mock_tf.convert_to_tensor.return_value = MagicMock()


@patch("src.model.grad_cam.tf")
class TestGenerateGradCAM:
    def test_output_shape_and_dtype(
        self, mock_tf, mock_model, preprocessed_array, original_array
    ):
        setup_gradcam_mocks(mock_tf, mock_model)
        result = gc_module.generate_gradcam(
            preprocessed_array, original_array, mock_model
        )
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8

    def test_output_values_in_range_0_255(
        self, mock_tf, mock_model, preprocessed_array, original_array
    ):
        setup_gradcam_mocks(mock_tf, mock_model)
        result = gc_module.generate_gradcam(
            preprocessed_array, original_array, mock_model
        )
        assert result.min() >= 0
        assert result.max() <= 255

    def test_uses_gradient_tape_not_k_gradients(
        self, mock_tf, mock_model, preprocessed_array, original_array
    ):
        setup_gradcam_mocks(mock_tf, mock_model)
        gc_module.generate_gradcam(preprocessed_array, original_array, mock_model)
        mock_tf.GradientTape.assert_called_once()
        mock_tf.GradientTape.return_value.gradient.assert_called_once()

    def test_uses_model_outputs_0_not_model_output(
        self, mock_tf, mock_model, preprocessed_array, original_array
    ):
        setup_gradcam_mocks(mock_tf, mock_model)
        gc_module.generate_gradcam(preprocessed_array, original_array, mock_model)
        call_args = mock_tf.keras.models.Model.call_args
        outputs_arg = call_args[1]["outputs"]
        assert mock_model.outputs[0] in outputs_arg

    def test_transparency_formula_heatmap_alpha(
        self, mock_tf, mock_model, preprocessed_array, original_array
    ):
        setup_gradcam_mocks(mock_tf, mock_model)
        result = gc_module.generate_gradcam(
            preprocessed_array, original_array, mock_model
        )
        assert result.shape == (512, 512, 3)
        import cv2

        assert not np.array_equal(result, cv2.resize(original_array, (512, 512)))

    @pytest.mark.parametrize("class_idx", [0, 1, 2])
    def test_parametrized_class_prediction(
        self, mock_tf, mock_model, preprocessed_array, original_array, class_idx
    ):
        probs = [0.1, 0.1, 0.1]
        probs[class_idx] = 0.8
        setup_gradcam_mocks(mock_tf, mock_model, probs=probs, argmax_value=class_idx)
        result = gc_module.generate_gradcam(
            preprocessed_array, original_array, mock_model
        )
        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize(
        "original_shape", [(512, 512, 3), (1024, 768, 3), (400, 800, 3), (800, 400, 3)]
    )
    def test_parametrized_original_sizes(
        self, mock_tf, mock_model, preprocessed_array, original_shape
    ):
        original_array = np.random.randint(0, 256, original_shape, dtype=np.uint8)
        setup_gradcam_mocks(mock_tf, mock_model)
        result = gc_module.generate_gradcam(
            preprocessed_array, original_array, mock_model
        )
        assert result.shape == (512, 512, 3)

    def test_zero_gradients_no_nan(
        self, mock_tf, mock_model, preprocessed_array, original_array
    ):
        grads_array = np.zeros((1, 16, 16, 64), dtype=np.float32)
        mock_grads = MockTensor(grads_array)
        mock_tf.GradientTape.return_value.gradient.return_value = mock_grads

        conv_output_array = np.random.rand(1, 16, 16, 64).astype(np.float32)
        mock_conv_output = MockTensor(conv_output_array)
        preds_array = np.array([[0.1, 0.7, 0.2]], dtype=np.float32)
        mock_preds = MockTensor(preds_array)

        mock_grad_model = MagicMock()
        mock_grad_model.return_value = (mock_conv_output, mock_preds)
        mock_tf.keras.models.Model.return_value = mock_grad_model

        mock_tf.argmax.return_value = MockTensor(np.array(1, dtype=np.int64))
        mock_tf.reduce_mean.return_value = MockTensor(np.ones(64, dtype=np.float32))
        mock_tf.convert_to_tensor.return_value = MagicMock()

        result = gc_module.generate_gradcam(
            preprocessed_array, original_array, mock_model
        )
        assert not np.any(np.isnan(result))
        assert result.shape == (512, 512, 3)

    def test_missing_target_layer_raises_gradcam_error(
        self, mock_tf, preprocessed_array, original_array
    ):
        mock_model = MagicMock()
        mock_model.get_layer.side_effect = ValueError("Layer not found")
        mock_model.inputs = [MagicMock()]
        mock_model.outputs = [MagicMock()]
        with pytest.raises(GradCAMError) as exc_info:
            gc_module.generate_gradcam(preprocessed_array, original_array, mock_model)
        assert gc_module.TARGET_LAYER_NAME in str(exc_info.value)
        assert exc_info.value.layer_name == gc_module.TARGET_LAYER_NAME

    def test_gradient_none_raises_gradcam_error(
        self, mock_tf, mock_model, preprocessed_array, original_array
    ):
        setup_gradcam_mocks(mock_tf, mock_model)
        mock_tf.GradientTape.return_value.gradient.return_value = None
        with pytest.raises(GradCAMError) as exc_info:
            gc_module.generate_gradcam(preprocessed_array, original_array, mock_model)
        assert "none" in str(exc_info.value).lower()
