#!/usr/bin/env python
"""Helper interno de la Vista: persistencia del historial CSV de resultados."""

import csv
from pathlib import Path


class HistorialCSVWriter:
    """Persiste registros de predicción en el historial CSV de resultados."""

    def __init__(self, path: Path) -> None:
        """Inicializa el escritor.

        Args:
            path: Ruta del archivo CSV donde se agregarán las filas. No se
                crea ni se valida en este punto, solo al llamar a `guardar`.
        """
        self.path = path

    def guardar(self, cedula: str, label: str, probabilidad: str) -> None:
        """Agrega una fila al CSV, creando el directorio padre si no existe.

        Args:
            cedula: Cédula del paciente.
            label: Etiqueta predicha ("bacteriana", "normal" o "viral").
            probabilidad: Probabilidad de la predicción, ya formateada
                como texto (ej. "87.46%").
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter="-")
            writer.writerow([cedula, label, probabilidad])
