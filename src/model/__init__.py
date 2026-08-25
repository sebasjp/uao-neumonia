"""Capa Modelo: carga de modelo y generación de Grad-CAM."""

from src.model.grad_cam import generate_gradcam
from src.model.load_model import load_model

__all__ = ["generate_gradcam", "load_model"]
