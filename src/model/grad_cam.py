"""Capa Modelo: generación de mapas de calor Grad-CAM."""

import cv2
import numpy as np
import tensorflow as tf

from src.model.exceptions import GradCAMError

TARGET_LAYER_NAME = "conv10_thisone"
HEATMAP_ALPHA = 0.8
OUTPUT_SIZE = (512, 512)


def generate_gradcam(
    preprocessed_array: np.ndarray,
    original_array: np.ndarray,
    model,
) -> np.ndarray:
    """Genera heatmap Grad-CAM superpuesto sobre la imagen original.

    Args:
        preprocessed_array: Imagen preprocesada, shape (1, 512, 512, 1), float64 [0,1].
        original_array: Imagen BGR original, cualquier HxWx3, uint8.
        model: Instancia de tf.keras.Model ya cargada.

    Retorna:
        Imagen RGB con heatmap superpuesto, shape (512, 512, 3), uint8.

    Raises:
        GradCAMError: Si la capa objetivo no existe o falla el cálculo de gradientes.
    """
    try:
        last_conv_layer = model.get_layer(TARGET_LAYER_NAME)
    except ValueError as exc:
        raise GradCAMError(
            f"Capa '{TARGET_LAYER_NAME}' no encontrada en el modelo", TARGET_LAYER_NAME
        ) from exc

    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[last_conv_layer.output, model.outputs[0]]
    )

    img_tensor = tf.convert_to_tensor(preprocessed_array)

    with tf.GradientTape() as tape:
        conv_layer_output, preds = grad_model(img_tensor)
        argmax = tf.argmax(preds[0])
        output = preds[:, argmax]

    grads = tape.gradient(output, conv_layer_output)

    if grads is None:
        raise GradCAMError(
            "Gradientes son None - posible capa no diferenciable", TARGET_LAYER_NAME
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_layer_output_value = conv_layer_output[0].numpy()

    for filter_idx in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, :, filter_idx] *= pooled_grads[filter_idx]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    else:
        heatmap = np.zeros_like(heatmap)

    heatmap = cv2.resize(
        heatmap, (preprocessed_array.shape[2], preprocessed_array.shape[1])
    )
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_resized = cv2.resize(original_array, OUTPUT_SIZE)
    heatmap_resized = cv2.resize(heatmap, OUTPUT_SIZE)

    transparency = (heatmap_resized * HEATMAP_ALPHA).astype(np.uint8)
    superimposed = cv2.add(transparency, original_resized)
    superimposed = superimposed.astype(np.uint8)

    return superimposed[:, :, ::-1]
