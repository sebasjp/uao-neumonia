#!/usr/bin/env python
"""Pruebas para src/view/historial_csv_writer.py."""

import csv

from src.view.historial_csv_writer import HistorialCSVWriter


class TestHistorialCSVWriter:
    def test_guardar_agrega_fila_con_los_valores_dados(self, tmp_path):
        # Arrange
        historial_path = tmp_path / "historial.csv"
        writer = HistorialCSVWriter(historial_path)

        # Act
        writer.guardar(cedula="12345678", label="bacteriana", probabilidad="91.23%")

        # Assert
        with open(historial_path, newline="") as f:
            filas = list(csv.reader(f, delimiter="-"))
        assert filas == [["12345678", "bacteriana", "91.23%"]]

    def test_guardar_crea_el_directorio_padre_si_no_existe(self, tmp_path):
        # Arrange
        historial_path = tmp_path / "subdir" / "historial.csv"
        writer = HistorialCSVWriter(historial_path)

        # Act
        writer.guardar(cedula="1", label="normal", probabilidad="50.00%")

        # Assert
        assert historial_path.exists()

    def test_guardar_dos_veces_agrega_dos_filas_sin_sobrescribir(self, tmp_path):
        # Arrange
        historial_path = tmp_path / "historial.csv"
        writer = HistorialCSVWriter(historial_path)

        # Act
        writer.guardar(cedula="1", label="normal", probabilidad="10.00%")
        writer.guardar(cedula="2", label="viral", probabilidad="20.00%")

        # Assert
        with open(historial_path, newline="") as f:
            filas = list(csv.reader(f, delimiter="-"))
        assert filas == [
            ["1", "normal", "10.00%"],
            ["2", "viral", "20.00%"],
        ]
