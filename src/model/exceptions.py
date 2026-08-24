"""Excepciones personalizadas para la capa Modelo."""


class ModelLoadError(RuntimeError):
    """Error al cargar el modelo Keras desde disco."""

    def __init__(self, message: str, path: str | None = None) -> None:
        """Inicializa el error.

        Args:
            message: Mensaje de error descriptivo.
            path: Ruta al archivo del modelo (opcional).
        """
        if path:
            message = f"{message} (ruta: {path})"
        super().__init__(message)
        self.path = path


class GradCAMError(RuntimeError):
    """Error durante la generación del mapa de calor Grad-CAM."""

    def __init__(self, message: str, layer_name: str | None = None) -> None:
        """Inicializa el error.

        Args:
            message: Mensaje de error descriptivo.
            layer_name: Nombre de la capa objetivo (opcional).
        """
        if layer_name:
            message = f"{message} (capa: {layer_name})"
        super().__init__(message)
        self.layer_name = layer_name
