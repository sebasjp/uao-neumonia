![Python](https://img.shields.io/badge/python-3.13-blue)
![uv](https://img.shields.io/badge/deps-uv-purple)
![License](https://img.shields.io/badge/license-MIT-green)

## Hola! Bienvenido a la herramienta para la detección rápida de neumonía

Deep Learning aplicado en el procesamiento de imágenes radiográficas de tórax en formato DICOM con el fin de clasificarlas en 3 categorías diferentes:

1. Neumonía Bacteriana

2. Neumonía Viral

3. Sin Neumonía

Aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.

---

## Uso de la herramienta

El proyecto usa [`uv`](https://docs.astral.sh/uv/) como gestor de dependencias y **Python 3.13** (versión exigida por el profesor). No se usa `pip` ni `conda` directamente — todos los comandos del día a día están centralizados en el `Makefile` (ver [`docs/MAKEFILE.md`](docs/MAKEFILE.md) para el detalle de cada uno).

### Opción 1: correr localmente

Requerimientos: tener [`uv`](https://docs.astral.sh/uv/getting-started/installation/) instalado.

```bash
git clone https://github.com/sebasjp/uao-neumonia.git
cd uao-neumonia
make install   # instala las dependencias (equivalente a `uv sync`)
make run       # corre la aplicación
```

Coloca el archivo del modelo entrenado (`conv_MLP_84.h5`) dentro de la carpeta `model/` antes de correr la aplicación — no se versiona en el repositorio por su tamaño.

### Opción 2: correr con Docker

Requerimientos: tener [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado, y (para ver la interfaz gráfica) un servidor X — en Windows, [VcXsrv](https://sourceforge.net/projects/vcxsrv/) o Xming; en Linux suele funcionar de forma nativa.

```bash
make docker-build   # construye la imagen
make docker-run     # levanta el contenedor con la interfaz gráfica
```

`docker-run` monta tu carpeta local `model/` dentro del contenedor, así que el modelo entrenado debe estar ahí igual que en la opción local.

### Pruebas y calidad de código

El proyecto exige cero warnings y pruebas unitarias sobre cada módulo. Antes de abrir un Pull Request, corre:

```bash
make lint            # revisa estilo y docstrings (ruff)
make format-check     # verifica formateo sin modificar archivos
make test             # corre la suite de pruebas unitarias (pytest)
```

Ver [`docs/MAKEFILE.md`](docs/MAKEFILE.md) para el detalle de cada comando, y [`docs/PLAN_PRUEBAS.md`](docs/PLAN_PRUEBAS.md) para el criterio de pruebas de cada módulo.

### Uso de la Interfaz Gráfica

- Ingrese la cédula del paciente en la caja de texto
- Presione el botón 'Cargar Imagen', seleccione la imagen del explorador de archivos del computador (Imagenes de prueba en https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link)
- Presione el botón 'Predecir' y espere unos segundos hasta que observe los resultados
- Presione el botón 'Guardar' para almacenar la información del paciente en un archivo excel con extensión .csv
- Presione el botón 'PDF' para descargar un archivo PDF con la información desplegada en la interfaz
- Presión el botón 'Borrar' si desea cargar una nueva imagen

---

## Arquitectura de archivos

El proyecto sigue una arquitectura MVC (Modelo-Vista-Controlador), organizada dentro de `src/`:

```
src/
├── model/
│   ├── load_model.py     # Carga y cachea el modelo Keras entrenado
│   └── grad_cam.py        # Genera el mapa de calor Grad-CAM
├── controller/
│   ├── read_img.py         # Lee imágenes DICOM/JPG y las convierte a arreglo
│   ├── preprocess_img.py   # Preprocesa el arreglo para el modelo
│   └── integrator.py       # Integra lectura + preprocesamiento + modelo
└── view/
    └── detector_view.py    # Interfaz gráfica (Tkinter)
```

### src/model/load_model.py

Carga el archivo binario del modelo de red neuronal convolucional previamente entrenado (`conv_MLP_84.h5`), cacheándolo para no recargarlo en cada predicción.

### src/model/grad_cam.py

Recibe la imagen preprocesada y el modelo, calcula el gradiente respecto a la capa convolucional de interés, y genera el mapa de calor Grad-CAM.

### src/controller/read_img.py

Lee la imagen en formato DICOM o JPG, la prepara para visualizarla en la interfaz gráfica y la convierte a arreglo para su preprocesamiento posterior.

### src/controller/preprocess_img.py

Recibe el arreglo proveniente de `read_img.py` y realiza las siguientes modificaciones:

- resize a 512x512
- conversión a escala de grises
- ecualización del histograma con CLAHE
- normalización de la imagen entre 0 y 1
- conversión del arreglo de imagen a formato de batch (tensor)

### src/controller/integrator.py

Integra los demás módulos y retorna solamente lo necesario para ser visualizado en la interfaz gráfica: la clase, la probabilidad, y la imagen del mapa de calor generado por Grad-CAM.

### src/view/detector_view.py

Contiene el diseño de la interfaz gráfica utilizando Tkinter. Los botones llaman métodos contenidos en los módulos del controlador y el modelo.

---

## Acerca del Modelo

La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer
en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad.
Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.

Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.

## Acerca de Grad-CAM

Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.

---

## Documentación adicional

- [`AGENTS.md`](AGENTS.md) — convenciones de código y estructura del proyecto
- [`docs/CONTRATOS_MODULOS.md`](docs/CONTRATOS_MODULOS.md) — contratos (firmas de funciones) de cada módulo
- [`docs/PLAN_PRUEBAS.md`](docs/PLAN_PRUEBAS.md) — plan de pruebas unitarias
- [`docs/DEBUGGING_MONOLITO.md`](docs/DEBUGGING_MONOLITO.md) — historial de bugs corregidos durante el refactor
- [`docs/MAKEFILE.md`](docs/MAKEFILE.md) — referencia de comandos del `Makefile`

## Licencia

Este proyecto se distribuye bajo la licencia [MIT](LICENSE).

---

## Equipo

Refactor a arquitectura MVC, migración a `uv`, Dockerización y suite de pruebas realizados por:

- Cesar Carabali ([`@CesarCR14`](https://github.com/CesarCR14)) — [`src/controller/read_img.py`](src/controller/read_img.py), revisión de Pull Requests, Docker y documentación (Issue #12)
- Sebastian Jimenez Parra ([`@sebasjp`](https://github.com/sebasjp)) — [`src/view/`](src/view/), documentación inicial y configuración del proyecto
- Julian David Correa ([`@jdcg5299`](https://github.com/jdcg5299)) — [`src/model/`](src/model/) (carga del modelo y Grad-CAM)
- Juan Plata ([`@Juanxo17`](https://github.com/Juanxo17)) — [`src/controller/preprocess_img.py`](src/controller/preprocess_img.py) y [`src/controller/integrator.py`](src/controller/integrator.py)

## Proyecto original realizado por:

Isabella Torres Revelo - https://github.com/isa-tr
Nicolas Diaz Salazar - https://github.com/nicolasdiazsalazar
