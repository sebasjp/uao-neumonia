# AGENTS.md

Lineamientos para cualquier persona o agente de código (Claude Code, Copilot, Cursor, Codex, etc.) que trabaje en este repositorio. Léelo antes de modificar cualquier archivo.

## Contexto del proyecto

Proyecto Neumonía: herramienta de apoyo al diagnóstico médico que clasifica radiografías (bacteriana / normal / viral) y genera un mapa de calor Grad-CAM. Está en refactor hacia el patrón MVC — ver [`docs/CONTRATOS_MODULOS.md`](docs/CONTRATOS_MODULOS.md) para la interfaz exacta que debe exponer cada módulo.

## Estructura

- `src/` — todo el código fuente de la aplicación.
  - `load_model.py`, `grad_cam.py` — Modelo
  - `read_img.py`, `preprocess_img.py`, `integrator.py` — Controlador
  - `detector_view.py` — Vista / Cliente
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
- Docstrings obligatorios en toda función y método público — qué recibe, qué retorna, no cómo funciona por dentro.
- Cero warnings: si algo emite un warning (de TensorFlow, Pillow, pydicom, etc.), se corrige la causa, no se silencia con `warnings.filterwarnings`.
- Cero código deprecado (APIs marcadas deprecated en las versiones usadas).
- Sin abstracciones ni manejo de errores para casos que no pueden ocurrir. No hacer refactors o limpiezas fuera del alcance de lo que se está trabajando.

Antes de cualquier commit: `make lint` y `make format-check` deben pasar sin errores.

## Pruebas

- Toda función nueva o modificada en `src/` necesita su prueba correspondiente en `test/`.
- `make test` (o `make test-cov` para ver cobertura) debe pasar en verde antes de abrir un Pull Request.
- Ideas de casos de prueba por módulo, incluyendo regresión de bugs ya corregidos: [`docs/PLAN_PRUEBAS.md`](docs/PLAN_PRUEBAS.md).

## Contratos entre módulos

Las firmas de función documentadas en [`docs/CONTRATOS_MODULOS.md`](docs/CONTRATOS_MODULOS.md) son la interfaz que el resto del equipo asume para integrar su propio módulo. Si necesitas cambiar el nombre, los parámetros o la estructura de retorno de una función ya contratada:

1. Actualiza `docs/CONTRATOS_MODULOS.md` en el mismo PR.
2. Avisa al equipo — quien depende de esa función puede estar trabajando con un mock basado en la firma anterior.

## Errores ya corregidos, no reintroducir

[`docs/DEBUGGING_MONOLITO.md`](docs/DEBUGGING_MONOLITO.md) documenta bugs reales ya resueltos en la versión monolítica del proyecto (rutas Unicode, acumulación de texto/imágenes en la GUI, APIs removidas en Pillow/pydicom/Keras 3, etc.). Revísalo antes de reimplementar la lógica equivalente en el módulo correspondiente, para no reintroducir el mismo problema.

## Modelo (`.h5`)

`conv_MLP_84.h5` y `WilhemNet86.h5` nunca se commitean (ya están en `.gitignore`). Cada integrante debe tener su propia copia local en la ruta que espera `load_model.py`.

## Flujo de Git

- Todo cambio se integra mediante Pull Request — nunca push directo a `main`.
- Usar la plantilla oficial de PR del curso.
- El nombre de la rama de trabajo queda a criterio de cada desarrollador.
- Un PR no se mergea si `make lint`, `make format-check` o `make test` fallan.
