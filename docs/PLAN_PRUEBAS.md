# Plan de pruebas unitarias

Objetivo: ~120 pruebas con `pytest`, repartidas según la complejidad real de cada módulo (no 20 parejo — ver `docs/CONTRATOS_MODULOS.md` para las firmas que estas pruebas deben validar).

Este documento es un **menú de ideas de partida**, no una checklist obligatoria exacta. Usen `@pytest.mark.parametrize` para cubrir varios casos con una sola función de prueba — cuenta como N pruebas en el reporte de pytest sin duplicar código (evita el problema de "código repetido").

Categorías, en cada módulo:
1. **Camino feliz** — cumple el contrato con inputs válidos típicos.
2. **Casos límite / inputs inválidos** — la mayoría del volumen suele salir de aquí.
3. **Regresión de bugs conocidos** — casos de `docs/DEBUGGING_MONOLITO.md`, para que no se reintroduzcan.
4. **Integración** — que los módulos conectados entre sí sigan cumpliendo el contrato.
5. **Errores/excepciones** — tipo de excepción correcto ante entradas inválidas.

## Patrón AAA con Pytest

Toda prueba (de cualquier módulo) se organiza en tres fases, marcadas con comentarios, y se agrupa en una clase `TestNombreDeLoQueSePrueba` — una clase por función o clase bajo prueba, no una clase gigante para todo el archivo:

- **Arrange** — preparar los datos/objetos de entrada.
- **Act** — ejecutar la acción bajo prueba (usualmente una sola línea).
- **Assert** — validar el resultado.

```python
# test/test_preprocess_img.py
import numpy as np
import pytest

from src.controller.preprocess_img import preprocess


class TestPreprocess:
    def test_shape_and_dimensions(self):
        # Arrange — datos de entrada ficticios
        raw_img = np.random.randint(0, 256, size=(1000, 1000, 3), dtype=np.uint8)

        # Act
        processed_img = preprocess(raw_img)

        # Assert
        assert processed_img.shape == (1, 512, 512, 1)

    def test_normalization_range(self):
        # Arrange
        raw_img = np.random.randint(0, 256, size=(800, 800, 3), dtype=np.uint8)

        # Act
        processed_img = preprocess(raw_img)

        # Assert
        assert 0.0 <= processed_img.min()
        assert processed_img.max() <= 1.0
```

Cuando el caso de prueba viene de `@pytest.mark.parametrize`, el Arrange es implícito (son los parámetros); igual se marcan las fases de Act y Assert.

---

## `load_model.py` — Julian (~5-8 pruebas)

- Retorna una instancia de `tf.keras.Model` (mockeado).
- Se invoca `tf.keras.models.load_model` con `MODEL_PATH` y `compile=False`.
- Llamar `load_model()` dos veces retorna el **mismo objeto** (cache/singleton).
- `tf.keras.models.load_model` se invoca **una sola vez** aunque `load_model()` se llame N veces.
- Si el archivo `.h5` no existe, lanza un error explícito y legible.
- *(Regresión)* `compile=False` evita el error de `reduction` de Keras 3 — mockear el escenario del bug.

## `grad_cam.py` — Julian (~15-20 pruebas)

- Retorna `np.ndarray` shape `(512, 512, 3)`, dtype `uint8`.
- Valores del array resultante en rango `[0, 255]`.
- El heatmap se normaliza (máximo = 1) antes de aplicar el colormap.
- *(Regresión)* usa `tf.GradientTape`, no `K.gradients()` — bug #7 de debugging.
- *(Regresión)* usa `model.outputs[0]`, no `model.output` — bug #8 de debugging.
- Combinación heatmap + imagen original replica la fórmula de transparencia (`heatmap * 0.8`, luego `cv2.add`).
- *Parametrizado* sobre 3 clases (`argmax` = 0, 1, 2 con predicciones mockeadas): el pipeline corre sin error para cada una.
- *Parametrizado* sobre distintos tamaños de `original_array` (cuadrada, panorámica, vertical): el resultado siempre es `512x512x3`.
- Gradientes todo cero no producen división por cero / `NaN` en la normalización del heatmap.
- Modelo sin la capa `conv10_thisone` lanza un error claro, no un `KeyError` críptico.

## `read_img.py` — Cesar (~15-20 pruebas)

`read_dicom_file`:
- Retorna `ImageReadResult(img_array, img_display)`.
- `img_array` es RGB (3 canales), uint8.
- Fixture de un `.dcm` válido produce el shape esperado.
- Ruta inexistente lanza un error apropiado.
- Pixel data de 16 bits se normaliza correctamente a 8 bits (0-255).

`read_jpg_file`:
- Retorna `ImageReadResult(img_array, img_display)`.
- *(Regresión)* ruta con tildes/ñ se lee correctamente — bug #11 de debugging.
- *(Regresión)* imagen no decodificable lanza `ValueError` explícito, no falla silenciosamente — bug #11.
- Ruta inexistente lanza error claro.
- *Parametrizado* sobre `.jpg`, `.jpeg`, `.png`: todas se leen igual.
- *Parametrizado* sobre varias resoluciones de entrada.
- Imagen de entrada ya en escala de grises (1 canal): comportamiento definido explícitamente (no un crash inesperado).

## `preprocess_img.py` — Juan (~10-15 pruebas)

- Shape de salida `(1, 512, 512, 1)`, dtype `float64`.
- Valores de salida en rango `[0, 1]`.
- *Parametrizado* sobre tamaños de entrada (200x200, 1024x768, cuadrada, panorámica): siempre resuelve a 512x512.
- Conversión a escala de grises deja un solo canal antes de CLAHE.
- CLAHE cambia el contraste respecto a la imagen sin ecualizar (comparar histogramas).
- Imagen completamente negra no produce `NaN` ni división por cero.
- Imagen completamente blanca se procesa sin error.
- `expand_dims` deja el orden correcto: batch primero, canal al final.

## `integrator.py` — Juan (~10-15 pruebas)

- Retorna `PredictionResult(label, probability, heatmap)`.
- *Parametrizado* mapeo de clase → label: `argmax=0` → `"bacteriana"`, `1` → `"normal"`, `2` → `"viral"` (con `model.predict` mockeado).
- `proba` siempre en rango `[0, 100]`.
- `preprocess()` se invoca **una sola vez** por llamada a `predict()` (el monolito original lo recalculaba dos veces — eficiencia a corregir, ver `CONTRATOS_MODULOS.md`).
- El array preprocesado se reutiliza tanto para `model.predict()` como para `generate_gradcam()`, no se recalcula.
- El modelo se obtiene vía `load_model()` (reutilizado), no se recarga en cada predicción.
- Verifica el orden de llamadas con mocks: `preprocess` → `model.predict` → `generate_gradcam`.
- Propaga la excepción si `preprocess()` falla con una imagen inválida.
- Propaga la excepción si la predicción del modelo falla.
- *Integración*: con `load_model`/`generate_gradcam` reales pero un modelo dummy pequeño (no el `.h5` de producción), `predict()` corre end-to-end sin mocks pesados.

## `detector_view.py` — Sebastian

Para que esto sea testeable sin abrir una ventana, conviene extraer la lógica de los callbacks a **funciones puras** dentro del mismo archivo (no cambia el contrato con otros módulos, es una decisión interna de la Vista):

```python
def validar_cedula(texto: str) -> bool: ...
def formatear_probabilidad(proba: float) -> str: ...
def elegir_lector(path: str) -> Callable: ...  # despacha a read_dicom_file o read_jpg_file
```

**Funciones puras (~15-20 pruebas, la mayoría vía parametrize):**
- `validar_cedula`: vacío → `False`, solo espacios → `False`, contenido válido → `True`, contenido con espacios alrededor → `True` (strip).
- `formatear_probabilidad`: *parametrizado* sobre varios floats (`87.456` → `"87.46%"`, `100.0` → `"100.00%"`, `0.0` → `"0.00%"`).
- `elegir_lector`: *parametrizado* sobre `.dcm`, `.jpg`, `.jpeg`, `.png`, `.DCM` (mayúsculas), y una extensión no soportada (debe lanzar error claro).

**Pruebas de integración/smoke de la GUI (~5, no forzar a que sean "unitarias puras"):**
- *(Regresión)* cargar una imagen limpia el campo de cédula — bug #18.
- *(Regresión)* "Borrar" no lanza `TclError` — bug #16.
- *(Regresión)* predecir dos veces seguidas no acumula texto en resultado/probabilidad — bug #12.
- *(Regresión)* "Guardar" y "PDF" sin cédula muestran aviso y no proceden — bugs #13 y #17.
- El botón "Predecir" permanece deshabilitado hasta que se carga una imagen.

## Integración end-to-end (aparte del conteo por módulo)

Unas 5-10 pruebas adicionales, a coordinar entre todos, que ejerciten la cadena completa Vista → Integrador → Controlador → Modelo con mocks solo en las partes pesadas (TensorFlow), para confirmar que los contratos realmente encajan entre sí y no solo en aislamiento.
