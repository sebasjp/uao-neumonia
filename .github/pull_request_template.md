### 📝 Convención del Título de la PR
Aplica un [prefijo convencional](https://www.conventionalcommits.org/) en el título para clasificar el trabajo:

| Prefijo | Cuándo usar | Ejemplo |
|--------|-------------|---------|
| `feat:` | Nueva funcionalidad o módulo | `feat: agregar preprocesamiento CLAHE en preprocess_img.py` |
| `fix:` | Corrección de errores o eliminación de warnings | `fix: corregir advertencia de deprecación en TensorFlow` |
| `test:` | Adición o refactorización de pruebas unitarias | `test: agregar 15 pruebas unitarias para load_model.py` |
| `chore:` | Tareas de mantenimiento, Docker, docs o config | `chore: actualizar Dockerfile y README.md conbadges` |

---

### 🧱 Módulo o Componente Afectado
Selecciona las áreas principales en las que trabaja esta PR:
- [ ] **Preprocesamiento / Imágenes** (`src/read_img.py`, `src/preprocess_img.py`)
- [ ] **Modelo e Inferencia** (`src/load_model.py`, `src/grad_cam.py`)
- [ ] **Integración & CLI** (`src/integrator.py`, `src/detector_neumonia.py`)
- [ ] **Pruebas Unitarias** (`test/`)
- [ ] **Infraestructura & Contenedores** (`Dockerfile`, `.gitignore`, `Makefile`, `uv.lock`)
- [ ] **Documentación & README** (`README.md`, Licencia MIT)

---

### 📚 Descripción de los Cambios
<!--- 
Describe qué logra esta PR, la motivación y los detalles técnicos relevantes.
-->

**Módulo / Historia / Tarea asociada:** <!-- Ej. Implementación de Grad-CAM o Ajuste de Preprocesamiento -->

#### Resumen de Cambios:
- 

#### Justificación del Diseño (Clean Code / Principios SOLID):
<!--- 
Explica brevemente cómo se mantuvo la Alta Cohesión y Bajo Acoplamiento (Clean Code) en las clases/funciones modificadas.
-->
- 

---

### ✅ Lista de Chequeo Pre-PR (Estándares del Curso UAO)
Antes de solicitar revisión, verifica que tu código cumpla con los lineamientos imperativos del curso:

- [ ] **Entorno de Ejecución:** El código se ejecutó y probó exitosamente usando exclusivamente **`uv`** (Python 3.13).
- [ ] **Sin Warnings:** La ejecución de `uv run integrator.py` o los tests corre limpia **sin advertencias (warnings)**.
- [ ] **Control de Exclusiones (`.gitignore`):** Los modelos pesados (`conv_MLP_84.h5`) **NO** están rastreados por Git.
- [ ] **Estructura del Código:** El código fuente está dentro de `src/` y las pruebas dentro de `test/`.
- [ ] **Pruebas Unitarias (`pytest`):** Se ejecutó `uv run pytest` y todas las pruebas pasaron correctamente.
- [ ] **Clean Code & Refactorización:** Las funciones son cohesivas, tienen responsabilidades únicas y usan nombres descriptivos.

---

### 🧪 Evidencia de Pruebas Ejecutadas
<!--- 
Adjunta capturas de pantalla, logs o salidas del comando `uv run pytest` o ejecución en Docker demostrando que el código funciona.
-->

```bash
# Salida de pruebas unitarias ejecutadas con UV:
uv run pytest
```