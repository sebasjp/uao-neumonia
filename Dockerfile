FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
    libxext6 \
    libxrender1 \
    tk \
    libgl1 \
    libglib2.0-0 \
    gnome-screenshot \
    && rm -rf /var/lib/apt/lists/*

# Archivo vacío requerido por python-xlib (usada por pyautogui) para no
# fallar al buscar credenciales de X11, incluso cuando el servidor X del
# host (VcXsrv) tiene deshabilitado el control de acceso.
RUN touch /root/.Xauthority

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "-m", "src.view.detector_view"]
