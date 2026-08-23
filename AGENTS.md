# AGENTS.md

Lineamientos para cualquier persona o agente de código (Claude Code, Cursor, Opencode, etc.) que trabaje en este repositorio. Léelo antes de modificar cualquier archivo.

## Contexto del proyecto

Proyecto Neumonía: herramienta de apoyo al diagnóstico médico que clasifica radiografías (bacteriana / normal / viral) y genera un mapa de calor Grad-CAM. Está en refactor hacia el patrón MVC — ver [`docs/CONTRATOS_MODULOS.md`](docs/CONTRATOS_MODULOS.md) para la interfaz exacta que debe exponer cada módulo.

## Estructura

- `src/` — todo el código fuente de la aplicación, organizado por capa MVC.
  - `model/load_model.py`, `model/grad_cam.py` — Modelo
  - `controller/read_img.py`, `controller/preprocess_img.py`, `controller/integrator.py` — Controlador
  - `view/detector_view.py` — Vista / Cliente (más sus helpers internos, ej. `view/historial_csv_writer.py`, `view/reporte_pdf_generator.py`)
- `test/` — pruebas unitarias con `pytest`, un archivo de test por módulo de `src/`.
- `docs/` — documentación del proyecto (contratos, plan de pruebas, debugging log, este archivo referenciado).

No crear módulos ni archivos fuera de esta estructura sin acordarlo con el equipo.

## Gestor de paquetes: uv, exclusivamente

Prohibido usar `pip install` directamente. Toda instalación de dependencias pasa por `uv`:

```bash
uv add <paquete>          # dependencia de producción
uv add --dev <paquete>    # dependencia de desarrollo (linter, tests, etc.)
uv sync                   # instalar/actualizar el entorno desde pyproject.toml + uv.lock
```

Los comandos del día a día están centralizados en el `Makefile` — ver [`docs/MAKEFILE.md`](docs/MAKEFILE.md) para el detalle de cada target (`make install`, `make lint`, `make format`, `make test`, `make run`, `make docker-build`, `make docker-run`, etc.).

## Estilo de código

- PEP 8, verificado con `ruff` (`make lint`) y autoformateado con `make format`.
- Docstrings obligatorios en todo módulo, clase, función y método público, en formato **Google style** (`Args:`, `Returns:`, `Raises:`, `Attributes:`) — cada parámetro documentado por su nombre, qué retorna, qué excepción lanza y cuándo; no cómo funciona por dentro. Verificado automáticamente por `ruff` (`make lint`, reglas `D100-D107` y `D417` en `pyproject.toml`); no aplica a `test/` (el nombre del test ya documenta el caso).
- Cero warnings: si algo emite un warning (de TensorFlow, Pillow, pydicom, etc.), se corrige la causa, no se silencia con `warnings.filterwarnings`.
- Cero código deprecado (APIs marcadas deprecated en las versiones usadas).
- Sin abstracciones ni manejo de errores para casos que no pueden ocurrir. No hacer refactors o limpiezas fuera del alcance de lo que se está trabajando.
- **Cohesión**: cada clase resuelve un solo propósito. No agrupar funciones inconexas en clases/archivos tipo `utils.py` o `Manager.py`. Si un método hace algo claramente distinto al resto de la clase (ej. persistencia, generación de reportes, acceso a I/O externo), extraerlo a su propia clase con nombre específico (`HistorialCSVWriter`, `ReportePDFGenerator`, no `Helper`/`Utils`).
- **Acoplamiento**: exponer el contrato de un método (qué hace, qué recibe, qué retorna) sin filtrar cómo lo hace por dentro. Preferir que un objeto reciba sus colaboradores (o los cree él mismo de forma aislada) en vez de que el resto del código dependa de su implementación interna — esto es lo que permite mockear en los tests sin tocar la lógica real.

Antes de cualquier commit: `make lint` y `make format-check` deben pasar sin errores.

## Pruebas

- Toda función nueva o modificada en `src/` necesita su prueba correspondiente en `test/`.
- `make test` (o `make test-cov` para ver cobertura) debe pasar en verde antes de abrir un Pull Request.
- Cada prueba se estructura en las tres fases del patrón **AAA** (Arrange / Act / Assert), marcadas con comentarios, y se agrupa en una clase `TestNombreDeLoQueSePrueba` — ver el ejemplo en [`docs/PLAN_PRUEBAS.md`](docs/PLAN_PRUEBAS.md#patrón-aaa-con-pytest).
- Ideas de casos de prueba por módulo, incluyendo regresión de bugs ya corregidos: [`docs/PLAN_PRUEBAS.md`](docs/PLAN_PRUEBAS.md).

## Contratos entre módulos

Las firmas de función documentadas en [`docs/CONTRATOS_MODULOS.md`](docs/CONTRATOS_MODULOS.md) son la interfaz que el resto del equipo asume para integrar su propio módulo. Si necesitas cambiar el nombre, los parámetros o la estructura de retorno de una función ya contratada:

1. Actualiza `docs/CONTRATOS_MODULOS.md` en el mismo PR.
2. Avisa al equipo — quien depende de esa función puede estar trabajando con un mock basado en la firma anterior.

## Errores ya corregidos, no reintroducir

[`docs/DEBUGGING_MONOLITO.md`](docs/DEBUGGING_MONOLITO.md) documenta bugs reales ya resueltos en la versión monolítica del proyecto (rutas Unicode, acumulación de texto/imágenes en la GUI, APIs removidas en Pillow/pydicom/Keras 3, etc.). Revísalo antes de reimplementar la lógica equivalente en el módulo correspondiente, para no reintroducir el mismo problema.

## Rutas de archivos

Tres carpetas fijas en la raíz desacoplan el código de dónde viven los archivos. Ojo: `model/` (raíz, datos) es distinta de `src/model/` (código del Modelo) — mismo nombre, propósito distinto.

- **`model/`** — el `.h5` del modelo. `load_model.py` resuelve `MODEL_PATH` apuntando a `model/conv_MLP_84.h5` (una sola constante, no hardcodeada en más de un lugar). Nunca se commitea (ya está en `.gitignore`); cada integrante debe tener su propia copia local ahí.
- **`images/`** — carpeta por defecto (`initialdir`) del diálogo de carga de imagen en `detector_view.py`. No es obligatorio que las imágenes del usuario vivan ahí — es solo el punto de partida del explorador de archivos, no una restricción de origen.
- **`results/`** — todo lo que la Vista genera: `historial.csv` (botón "Guardar") y los reportes (`ReporteN.jpg` / `ReporteN.pdf`, botón "PDF"). `detector_view.py` nunca escribe en la raíz del repo ni asume el directorio de trabajo actual.

Estas rutas se resuelven relativas a la raíz del proyecto con `pathlib`, no al directorio de trabajo actual (`cwd`) — así funcionan igual sin importar desde dónde se invoque `python`/`uv run` (ej. el `WORKDIR` dentro de Docker puede ser distinto). Como el código vive en `src/<capa>/archivo.py` (dos niveles bajo la raíz), hacen falta tres `.parent`:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/<capa>/ -> raíz del repo
MODEL_PATH = PROJECT_ROOT / "model" / "conv_MLP_84.h5"
```

## Flujo de Git

- Todo cambio se integra mediante Pull Request — nunca push directo a `main`.
- Usar la plantilla oficial de PR del curso.
- El nombre de la rama de trabajo queda a criterio de cada desarrollador.
- Un PR no se mergea si `make lint`, `make format-check` o `make test` fallan.
