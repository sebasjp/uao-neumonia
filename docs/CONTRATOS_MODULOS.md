# Contratos entre módulos (MVC)

Este documento define la interfaz pública que debe exponer cada módulo del refactor, para que los 4 integrantes puedan trabajar en paralelo sin bloquearse entre sí ni generar dependencias implícitas.

Las firmas están extraídas del comportamiento real y ya funcional del monolito (`detector_neumonia.py`, en la raíz del repo, rama `main`), que es la fuente de verdad hasta que cada módulo lo reemplace.

**Qué fija un contrato:** nombre de la función, parámetros (nombre, tipo, shape/rango), y estructura del retorno (tipo, orden, shape).
**Qué NO fija:** nombres de variables internas, algoritmo o librería usada por dentro, funciones privadas auxiliares — eso es libertad de implementación de quien escribe el módulo.

**Regla de dependencia:** la Vista solo importa funciones del Controlador y del Integrador. Nunca importa `tensorflow` o `cv2` directamente.

**Retornos con más de un valor usan `NamedTuple`, no tuplas planas.** Una tupla plana (`(label, proba, heatmap)`) obliga a recordar el orden de memoria; si alguien lo invierte, el código sigue corriendo con datos mezclados y sin ningún error. Un `NamedTuple` se accede por nombre (`resultado.label`) además de por posición, así que un error de referencia falla de inmediato en vez de producir un bug silencioso. Cada `NamedTuple` se define en el módulo dueño del dato (no en un archivo de tipos compartido aparte) y se importa desde ahí.

```
Vista (view/detector_view.py)
   │  llama a
   ▼
Integrador (controller/integrator.py) ──► Controlador (controller/read_img.py, controller/preprocess_img.py)
   │
   └──────────────► Modelo (model/load_model.py, model/grad_cam.py)
```

Todas las rutas de arriba son relativas a `src/`.

---

## Modelo — Julian

### `src/model/load_model.py`

`MODEL_PATH` apunta a `model/conv_MLP_84.h5`, resuelto relativo a la raíz del proyecto (no al `cwd`) — ver convención de rutas en [`AGENTS.md`](../AGENTS.md#rutas-de-archivos).

```python
def load_model() -> tf.keras.Model:
    """
    Sin parámetros.

    Retorna: instancia de tf.keras.Model, cargada y cacheada en
    memoria (no debe recargar el .h5 en cada llamada).
    """
```

### `src/model/grad_cam.py`

```python
def generate_gradcam(
    preprocessed_array: np.ndarray,  # shape (1, 512, 512, 1), salida de preprocess()
    original_array: np.ndarray,      # imagen BGR original, cualquier alto x ancho x 3, uint8
    model: tf.keras.Model,
) -> np.ndarray:
    """
    Retorna: imagen RGB, shape (512, 512, 3), dtype uint8 — el mapa
    de calor superpuesto sobre la imagen original.
    """
```

---

## Controlador

### `src/controller/read_img.py` — Cesar

```python
class ImageReadResult(NamedTuple):
    img_array: np.ndarray        # RGB, uint8, tamaño original — insumo para preprocess()
    img_display: PIL.Image.Image # insumo para mostrar en la GUI


def read_dicom_file(path: str) -> ImageReadResult:
    """
    Parámetro: path, ruta absoluta o relativa a un archivo .dcm.

    Retorna: ImageReadResult(img_array, img_display)
    """

def read_jpg_file(path: str) -> ImageReadResult:
    """
    Parámetro: path, ruta absoluta o relativa a un archivo .jpg/.jpeg/.png.
    Debe soportar rutas con caracteres no-ASCII (tildes, ñ).

    Retorna: ImageReadResult(img_array, img_display), misma estructura
    que read_dicom_file.

    Lanza: ValueError si el archivo no pudo leerse/decodificarse.
    """
```

- La decisión de cuál función llamar según la extensión del archivo es responsabilidad de quien orquesta la carga (Vista), no de este módulo.

### `src/controller/preprocess_img.py` — Juan

```python
def preprocess(array: np.ndarray) -> np.ndarray:
    """
    Parámetro: array, imagen BGR sin procesar (alto x ancho x 3), uint8.

    Retorna: np.ndarray shape (1, 512, 512, 1), dtype float64,
    valores normalizados en el rango [0, 1].
    """
```

### `src/controller/integrator.py` — Juan

```python
class PredictionResult(NamedTuple):
    label: str            # uno de {"bacteriana", "normal", "viral"}
    probability: float    # rango 0-100
    heatmap: np.ndarray    # RGB, shape (512, 512, 3), uint8


def predict(array: np.ndarray) -> PredictionResult:
    """
    Parámetro: array, imagen BGR sin procesar (alto x ancho x 3), uint8
    — la misma salida cruda que entregan read_dicom_file/read_jpg_file.

    Retorna: PredictionResult(label, probability, heatmap)

    Es el único punto de entrada que la Vista necesita para obtener
    una predicción completa (orquesta preprocess + load_model +
    generate_gradcam internamente).
    """
```

---

## Vista / Cliente — Sebastian

### `src/view/detector_view.py`

No expone funciones públicas para otros módulos — es la capa más externa (punto de entrada de la app). Solo importa:

- `read_dicom_file`, `read_jpg_file` de `src.controller.read_img`
- `predict` de `src.controller.integrator`

Rutas de archivos que le corresponden (ver [`AGENTS.md`](../AGENTS.md#rutas-de-archivos)):
- El diálogo de carga de imagen abre por defecto en `images/`.
- `historial.csv` y los reportes (`ReporteN.jpg` / `ReporteN.pdf`) se escriben en `results/`, nunca en la raíz del repo ni en el `cwd`.

## Cómo desarrollar en paralelo sin bloquearse

Mientras Julian/Cesar/Juan implementan sus módulos, Sebastian puede avanzar la Vista contra **mocks** que cumplan estas firmas exactas (mismo nombre, mismos parámetros, misma estructura de retorno), devolviendo datos falsos pero con la forma correcta:

```python
def predict_mock(array: np.ndarray) -> PredictionResult:
    return PredictionResult(
        label="normal",
        probability=87.5,
        heatmap=np.zeros((512, 512, 3), dtype=np.uint8),
    )
```

Cuando el módulo real esté listo, se reemplaza el import del mock por el import real (`from src.controller.integrator import predict`). Si el contrato se respetó, `detector_view.py` no debería necesitar cambios.
