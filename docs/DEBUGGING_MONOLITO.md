# Debugging log — uao-neumonia

Resumen de los problemas encontrados y corregidos para lograr que `detector_neumonia.py` corra con `uv run detector_neumonia.py`, incluyendo los ajustes hechos en respuesta a las revisiones del PR #6.

## Entorno

1. **`pyproject.toml` exigía `requires-python = ">=3.14"`**, pero `tensorflow` (última versión, 2.21) solo publica wheels hasta Python 3.13.
   **Fix:** `requires-python = ">=3.12,<3.14"`, `.python-version = 3.13`.

## Bugs en `detector_neumonia.py`

2. **Imports faltantes**: se usaban `tf` (tensorflow) y `dicom` (pydicom) sin importarlos.
   **Fix:** agregados `import tensorflow as tf` e `import pydicom as dicom`.
3. **`model_fun()` no existía**: el código la llamaba pero nunca se definió, y el `.h5` del modelo (`conv_MLP_84.h5`) nunca se cargaba.
   **Fix:** agregada `model_fun()` que carga y cachea `conv_MLP_84.h5` con `tf.keras.models.load_model(MODEL_PATH, compile=False)` (`compile=False` evita reconstruir el optimizer/loss legacy, que rompía con Keras 3 — ver punto 5).
4. **`dicom.read_file()`**: removido en `pydicom` 3.x.
   **Fix:** reemplazado por `dicom.dcmread()`.
5. **`Image.ANTIALIAS`**: removido en Pillow 10+.
   **Fix:** reemplazado por `Image.LANCZOS` (2 ocurrencias).
6. **Carga del modelo fallaba con `ValueError: Invalid value for argument 'reduction'... Received: reduction=auto`**: el `.h5` fue guardado con una versión antigua de Keras (2.x) cuyo default de `reduction` en las losses ya no es válido en Keras 3.
   **Fix:** cargar el modelo con `compile=False` (no necesitamos el optimizer/loss para inferencia).
7. **`grad_cam()` usaba `K.gradients()`** (API de grafos de TF1), lo que exigía `tf.compat.v1.disable_eager_execution()` al importar — pero `model.predict()` en Keras 3 requiere modo eager. Ambos requisitos eran incompatibles entre sí.
   **Fix:** reescrito `grad_cam()` con `tf.GradientTape()` (equivalente moderno en modo eager), removido `disable_eager_execution()`.
8. **`model.output` devolvía una lista anidada** (en vez de un tensor) al construir el `grad_model` para Grad-CAM, rompiendo el slicing `preds[:, argmax]`.
   **Fix:** usar `model.outputs[0]` en su lugar.
9. **`load_img_file()` siempre llamaba a `read_dicom_file()`**, sin importar la extensión elegida en el diálogo — un `.jpg`/`.png` fallaría al intentar leerse como DICOM.
   **Fix:** despachar según extensión: `.dcm` → `read_dicom_file()`, cualquier otra → `read_jpg_file()` (ya existía en el archivo, no se usaba).
10. **`ModuleNotFoundError: No module named 'tkinter.tix'`** al importar `tkcap` tras subir el proyecto a Python 3.13: `tkinter.tix` fue removido de la stdlib en 3.13 (deprecado desde 3.6, la lib Tix subyacente está sin mantenimiento), y `tkcap` lo importa internamente aunque no se use para nada.
    **Fix:** eliminada la dependencia `tkcap`. `create_pdf()` ahora captura la ventana con `pyautogui.screenshot(region=...)` usando la geometría de `self.root` (`winfo_rootx/rooty/width/height`), sin pasar por Tix.

## Revisión 1 del PR

11. **(Bloqueante) `read_jpg_file()` fallaba con rutas que contienen tildes/"ñ"**: `cv2.imread(path)` devuelve `None` silenciosamente en Windows cuando la ruta tiene caracteres no-ASCII, y el error real quedaba oculto hasta un `TypeError` confuso varias líneas después en `Image.fromarray()`.
    **Fix:** reemplazado por `cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)`, que sí soporta Unicode en Windows, más una validación explícita `if img is None: raise ValueError(...)`.
12. **Los campos "Resultado" y "Probabilidad" acumulaban texto en vez de reemplazarlo**: `self.text2.insert(END, ...)` y `self.text3.insert(END, ...)` en `run_model()` no borraban el contenido previo, así que predecir dos veces sin pasar por "Borrar" concatenaba el texto (ej. "normalnormal").
    **Fix:** agregado `self.text2.delete(1.0, "end")` / `self.text3.delete(1.0, "end")` antes de cada `insert` (mismo patrón que ya usa `delete_txt` para estos widgets `Text`).
13. **(Menor) No había validación de la Cédula del Paciente antes de guardar**: se podía guardar un registro en `historial.csv` con el campo vacío.
    **Fix:** en `save_results_csv()`, si `self.text1.get().strip()` está vacío se muestra un aviso y no se escribe la fila.
14. **`pyproject.toml` incluía `streamlit` y `plotly` sin ningún uso en el código**: paquetes pesados que solo alargaban `uv sync` (26 dependencias transitivas: streamlit, plotly, pyarrow, altair, uvicorn, starlette, etc.), sin aportar nada al alcance actual.
    **Fix:** eliminadas ambas de `dependencies`, `uv sync` confirmó la desinstalación de los 26 paquetes.

## Revisión 2 del PR

15. **(Bloqueante) Imágenes acumuladas**: `text_img1.image_create(...)` (en `load_img_file`) y `text_img2.image_create(...)` (en `run_model`) no borraban el contenido previo del widget, así que cargar/predecir dos veces seguidas sin pasar por "Borrar" insertaba otra imagen encima y el panel quedaba en blanco.
    **Fix:** agregado `self.text_img1.delete(1.0, "end")` / `self.text_img2.delete(1.0, "end")` antes de cada `image_create`, mismo patrón ya usado para `text2`/`text3`.
16. **`TclError: bad text index` en el botón "Borrar"**: `self.text_img1.delete(self.img1, "end")` y `self.text_img2.delete(self.img2, "end")` usaban el objeto `PhotoImage` como índice de `Text.delete()`, lo cual es inválido.
    **Fix:** reemplazado por `delete(1.0, "end")`.
17. **`create_pdf()` no exigía la Cédula del Paciente**, a diferencia de `save_results_csv()`, pese a ser también un registro formal del paciente.
    **Fix:** agregada la misma validación (`if not self.text1.get().strip(): ... return`) al inicio de `create_pdf()`.
18. **La Cédula del Paciente no se limpiaba al cargar una imagen nueva**, solo con "Borrar" — riesgo de guardar/generar un reporte nuevo con la cédula del paciente anterior aún en el campo.
    **Fix:** agregado `self.text1.delete(0, "end")` en `load_img_file()`, al cargar cada imagen nueva.
