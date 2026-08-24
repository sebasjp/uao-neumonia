#!/usr/bin/env python
"""Vista/Cliente: interfaz gráfica para el detector de neumonía.

Único punto de entrada de la aplicación. Solo importa del Controlador
(src.controller.read_img) y del Integrador (src.controller.integrator) —
nunca tensorflow ni cv2 directamente (ver docs/CONTRATOS_MODULOS.md).
"""

from collections.abc import Callable
from pathlib import Path
from tkinter import END, Misc, StringVar, Text, Tk, filedialog, font, ttk
from tkinter.messagebox import WARNING, askokcancel, showinfo

from PIL import Image, ImageTk

from src.controller.integrator import PredictionResult, predict
from src.controller.read_img import ImageReadResult, read_dicom_file, read_jpg_file
from src.view.historial_csv_writer import HistorialCSVWriter
from src.view.reporte_pdf_generator import ReportePDFGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
RESULTS_DIR = PROJECT_ROOT / "results"
HISTORIAL_PATH = RESULTS_DIR / "historial.csv"

THUMBNAIL_SIZE = (250, 250)


def validar_cedula(texto: str) -> bool:
    """Indica si un texto de cédula es válido.

    Args:
        texto: Contenido del campo de cédula, tal cual lo escribió el usuario.

    Returns:
        True si `texto` (tras strip) no queda vacío, False en caso contrario.
    """
    return bool(texto.strip())


def formatear_probabilidad(proba: float) -> str:
    """Formatea una probabilidad para mostrarla en la GUI.

    Args:
        proba: Probabilidad en el rango 0-100.

    Returns:
        String con dos decimales y símbolo '%', ej. "87.46%".
    """
    return f"{proba:.2f}%"


def elegir_lector(path: str) -> Callable[[str], ImageReadResult]:
    """Elige la función de lectura adecuada según la extensión del archivo.

    Args:
        path: Ruta del archivo a cargar (no distingue mayúsculas/minúsculas
            en la extensión).

    Returns:
        read_dicom_file para .dcm, read_jpg_file para .jpg/.jpeg/.png.

    Raises:
        ValueError: Si la extensión no está soportada.
    """
    extension = Path(path).suffix.lower()
    if extension == ".dcm":
        return read_dicom_file
    if extension in (".jpg", ".jpeg", ".png"):
        return read_jpg_file
    raise ValueError(f"Extensión de archivo no soportada: {extension}")


class App:
    """Ventana principal de la herramienta de detección de neumonía."""

    def __init__(self, master: Misc | None = None) -> None:
        """Construye la ventana y todos sus widgets, sin iniciar el loop de eventos.

        Args:
            master: Ventana padre a usar como raíz (por defecto crea un
                `Tk()` nuevo). Permite en pruebas reutilizar un único
                intérprete Tcl entre instancias de `App`, en vez de
                crear/destruir uno por prueba.
        """
        self.root = master if master is not None else Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")

        fonti = font.Font(weight="bold")

        self.root.geometry("815x560")
        self.root.resizable(False, False)

        self.lab1 = ttk.Label(self.root, text="Imagen Radiográfica", font=fonti)
        self.lab2 = ttk.Label(self.root, text="Imagen con Heatmap", font=fonti)
        self.lab3 = ttk.Label(self.root, text="Resultado:", font=fonti)
        self.lab4 = ttk.Label(self.root, text="Cédula Paciente:", font=fonti)
        self.lab5 = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=fonti,
        )
        self.lab6 = ttk.Label(self.root, text="Probabilidad:", font=fonti)

        self.cedula = StringVar()
        self.text1 = ttk.Entry(self.root, textvariable=self.cedula, width=10)

        self.text_img1 = Text(self.root, width=31, height=15)
        self.text_img2 = Text(self.root, width=31, height=15)
        self.text2 = Text(self.root)
        self.text3 = Text(self.root)

        self.button1 = ttk.Button(
            self.root, text="Predecir", state="disabled", command=self.run_model
        )
        self.button2 = ttk.Button(
            self.root, text="Cargar Imagen", command=self.load_img_file
        )
        self.button3 = ttk.Button(self.root, text="Borrar", command=self.delete)
        self.button4 = ttk.Button(self.root, text="PDF", command=self.create_pdf)
        self.button6 = ttk.Button(
            self.root, text="Guardar", command=self.save_results_csv
        )

        self.lab1.place(x=110, y=65)
        self.lab2.place(x=545, y=65)
        self.lab3.place(x=500, y=350)
        self.lab4.place(x=65, y=350)
        self.lab5.place(x=122, y=25)
        self.lab6.place(x=500, y=400)
        self.button1.place(x=220, y=460)
        self.button2.place(x=70, y=460)
        self.button3.place(x=670, y=460)
        self.button4.place(x=520, y=460)
        self.button6.place(x=370, y=460)
        self.text1.place(x=200, y=350)
        self.text2.place(x=610, y=350, width=90, height=30)
        self.text3.place(x=610, y=400, width=90, height=30)
        self.text_img1.place(x=65, y=90)
        self.text_img2.place(x=500, y=90)

        self.text1.focus_set()

        self.array = None
        self.result: PredictionResult | None = None
        self.reporte_generator = ReportePDFGenerator()

    def run(self) -> None:
        """Inicia el loop de eventos de Tkinter (bloqueante)."""
        self.root.mainloop()

    def load_img_file(self) -> None:
        """Abre el diálogo de selección de imagen y muestra la imagen cargada."""
        filepath = filedialog.askopenfilename(
            initialdir=str(IMAGES_DIR),
            title="Select image",
            filetypes=(
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpeg"),
                ("jpg files", "*.jpg"),
                ("png files", "*.png"),
            ),
        )
        if not filepath:
            return
        lector = elegir_lector(filepath)
        self.array, img2show = lector(filepath)
        self.img1 = img2show.resize(THUMBNAIL_SIZE, Image.LANCZOS)
        self.img1 = ImageTk.PhotoImage(self.img1)
        self.text_img1.delete(1.0, "end")
        self.text_img1.image_create(END, image=self.img1)
        self.text1.delete(0, "end")
        self.button1["state"] = "enabled"

    def run_model(self) -> None:
        """Ejecuta la predicción sobre la imagen cargada y muestra el resultado."""
        self.result = predict(self.array)
        self.img2 = Image.fromarray(self.result.heatmap)
        self.img2 = self.img2.resize(THUMBNAIL_SIZE, Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(self.img2)
        self.text_img2.delete(1.0, "end")
        self.text_img2.image_create(END, image=self.img2)
        self.text2.delete(1.0, "end")
        self.text3.delete(1.0, "end")
        self.text2.insert(END, self.result.label)
        self.text3.insert(END, formatear_probabilidad(self.result.probability))

    def save_results_csv(self) -> None:
        """Guarda cédula, etiqueta y probabilidad en el historial CSV.

        Muestra un aviso y no guarda nada si la cédula está vacía.
        """
        if not validar_cedula(self.text1.get()):
            showinfo(title="Guardar", message="Debe ingresar la cédula del paciente.")
            return
        HistorialCSVWriter(HISTORIAL_PATH).guardar(
            cedula=self.text1.get(),
            label=self.result.label,
            probabilidad=formatear_probabilidad(self.result.probability),
        )
        showinfo(title="Guardar", message="Los datos se guardaron con éxito.")

    def create_pdf(self) -> None:
        """Captura la ventana actual y genera un reporte JPG y PDF en results/.

        Muestra un aviso y no genera nada si la cédula está vacía.
        """
        if not validar_cedula(self.text1.get()):
            showinfo(title="PDF", message="Debe ingresar la cédula del paciente.")
            return
        self.root.update()
        region = (
            self.root.winfo_rootx(),
            self.root.winfo_rooty(),
            self.root.winfo_width(),
            self.root.winfo_height(),
        )
        self.reporte_generator.generar(RESULTS_DIR, region)
        showinfo(title="PDF", message="El PDF fue generado con éxito.")

    def delete(self) -> None:
        """Borra todos los campos de la GUI, previa confirmación del usuario."""
        answer = askokcancel(
            title="Confirmación", message="Se borrarán todos los datos.", icon=WARNING
        )
        if answer:
            self.text1.delete(0, "end")
            self.text2.delete(1.0, "end")
            self.text3.delete(1.0, "end")
            self.text_img1.delete(1.0, "end")
            self.text_img2.delete(1.0, "end")
            showinfo(title="Borrar", message="Los datos se borraron con éxito")


def main() -> int:
    """Punto de entrada de la aplicación: construye la ventana y la ejecuta.

    Returns:
        Código de salida del proceso (siempre 0).
    """
    app = App()
    app.run()
    return 0


if __name__ == "__main__":
    main()
