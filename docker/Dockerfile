# Базовый образ зафиксирован по digest (python 3.12.14, Debian bookworm)
FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 dxaqc
USER dxaqc

ENTRYPOINT ["dxaqc"]
CMD ["--help"]
