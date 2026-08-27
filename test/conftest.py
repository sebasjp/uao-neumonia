"""Fixtures compartidas entre archivos de test.

Un único intérprete Tcl (`Tk()`) para toda la sesión de pytest: crear y
destruir múltiples `Tk()` independientes en el mismo proceso es inestable en
Windows (errores intermitentes de Tcl). Cada prueba que necesite una `App`
abre su propio `Toplevel` sobre este mismo intérprete.
"""

from tkinter import Tk, Toplevel

import pytest

from src.view.detector_view import App


@pytest.fixture(scope="session")
def tk_root():
    root = Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(tk_root):
    toplevel = Toplevel(tk_root)
    instance = App(master=toplevel)
    yield instance
    toplevel.destroy()
