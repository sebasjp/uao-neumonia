# Makefile

Este archivo centraliza los comandos del proyecto para que todo el equipo use exactamente las mismas instrucciones, sin depender de recordar flags de `uv`, `ruff`, `pytest` o `docker`. Cada bloque del `Makefile` se llama *target*, y se ejecuta con `make <target>` desde la raíz del repositorio.

> **Requisitos previos:** tener `make` y [`uv`](https://docs.astral.sh/uv/) instalados. `make` no viene preinstalado en Windows — instálalo con `winget install GnuWin32.Make` (y agrega `C:\Program Files (x86)\GnuWin32\bin` al `PATH` si `make --version` no lo reconoce) o mediante Chocolatey/Scoop. En Linux/macOS suele venir preinstalado o disponible vía el gestor de paquetes del sistema (`apt install make`, `brew install make`).

No se usa `pip` directamente en ningún target, según lo exigido por el proyecto.

## Targets disponibles

### `make install`
Sincroniza el entorno virtual con lo declarado en `pyproject.toml` / `uv.lock` (dependencias de producción y de desarrollo).

**Cuándo usarlo:** al clonar el repo por primera vez, o después de que alguien agregue/actualice una dependencia (`uv add`, `uv add --dev`) y necesitas traer esos cambios a tu entorno local.

```bash
make install
```

### `make lint`
Corre `ruff check` sobre `src/` y `test/`. Detecta errores reales y violaciones de estilo PEP 8: imports sin usar, variables no definidas, código muerto, complejidad, etc. **No modifica archivos**, solo reporta.

**Cuándo usarlo:** antes de hacer commit o abrir un Pull Request, para asegurarte de que no quedan warnings ni código deprecado (requisito del curso).

```bash
make lint
```

### `make format`
Corre `ruff format` sobre `src/` y `test/`. Reescribe automáticamente el código para que cumpla el estilo estándar (indentación, comillas, longitud de línea, espacios). Sí modifica archivos.

**Cuándo usarlo:** justo antes de un commit, para no perder tiempo formateando a mano.

```bash
make format
```

### `make format-check`
Igual que `format`, pero solo valida — no reescribe nada. Falla (exit code distinto de 0) si algún archivo no está formateado correctamente.

**Cuándo usarlo:** en integración continua (CI) o como paso previo a un Pull Request, para confirmar que alguien corrió `make format` antes de subir sus cambios.

```bash
make format-check
```

### `make test`
Ejecuta la suite de pruebas unitarias con `pytest` en modo verboso (`-v`), mostrando el resultado de cada prueba individual.

**Cuándo usarlo:** después de cualquier cambio en `src/`, y obligatoriamente antes de abrir un Pull Request. También se ejecuta en vivo durante la sustentación del módulo.

```bash
make test
```

### `make test-cov`
Igual que `test`, pero además calcula el porcentaje de cobertura de `src/` y muestra qué líneas específicas no están cubiertas por ninguna prueba (`--cov-report=term-missing`).

**Cuándo usarlo:** para verificar avance hacia el objetivo de 120 pruebas unitarias y detectar módulos con baja cobertura antes de la entrega.

```bash
make test-cov
```

### `make run`
Levanta la aplicación de vista/cliente (`src/detector_view.py`) usando el entorno gestionado por `uv`.

**Cuándo usarlo:** para probar manualmente la interfaz Tkinter mientras se desarrolla, o en la demo en vivo de la sustentación.

```bash
make run
```

> Nota: el nombre `detector_view.py` (en vez de `detector_neumonia.py`) es intencional durante el refactor: el monolito original sigue viviendo en la raíz del repo (`detector_neumonia.py`) mientras el resto del equipo trabaja sobre él. Usar un nombre distinto en `src/` evita confundir cuál archivo se está ejecutando o importando. 

### `make clean`
Elimina carpetas de caché generadas por Python, `pytest` y `ruff` (`__pycache__`, `.pytest_cache`, `.ruff_cache`, `htmlcov`, `.coverage`).

**Cuándo usarlo:** si el repo se siente "sucio" con archivos de caché, o antes de hacer `git status` para confirmar que no se está por commitear nada generado automáticamente.

```bash
make clean
```

### `make docker-build`
Construye la imagen Docker del proyecto (`uao-neumonia`) a partir del `Dockerfile`.

**Cuándo usarlo:** para validar que el contenedor sigue construyéndose correctamente después de cambios en dependencias o en el `Dockerfile`.

```bash
make docker-build
```

### `make docker-run`
Primero reconstruye la imagen (depende de `docker-build`, así nunca se ejecuta una versión desactualizada; gracias al cache de capas de Docker, si no hay cambios la reconstrucción es casi instantánea) y luego levanta el contenedor, propagando la variable de entorno `DISPLAY` y montando el socket de X11 — necesario para que la interfaz Tkinter pueda mostrarse fuera del contenedor.

**Cuándo usarlo:** para probar el aplicativo tal como correrá en el contenedor final, no solo en el entorno local.

```bash
make docker-run
```

> En Windows, mostrar una ventana Tkinter desde un contenedor Docker requiere un servidor X corriendo en el host (por ejemplo [VcXsrv](https://sourceforge.net/projects/vcxsrv/) o Xming) y configurar `DISPLAY` apuntando a él. En Linux esto suele funcionar de forma nativa.

## Dependencias de desarrollo

Los targets `lint`, `format`, `format-check`, `test` y `test-cov` requieren que `ruff`, `pytest` y `pytest-cov` estén declarados como dependencias de desarrollo. Si no están instaladas aún:

```bash
uv add --dev ruff pytest pytest-cov
```

Esto actualiza `pyproject.toml` y `uv.lock`. Después, cualquiera del equipo solo necesita correr `make install` para tenerlas disponibles.
