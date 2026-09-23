FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install --no-install-recommends -y build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a"
RUN uv sync --locked --no-dev --no-install-project

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY decision_service ./decision_service
RUN mkdir -p /app/.cache/huggingface && chown -R app:app /app

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface \
    PORT=50051
USER app
EXPOSE 50051
CMD ["python", "-m", "decision_service.server"]
