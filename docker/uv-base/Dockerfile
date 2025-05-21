FROM python:3.10-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /uvx /bin/
WORKDIR /app
RUN uv python install 3.10 \
    && uv venv .venv
ENV PATH="/.venv/bin:${PATH}"
COPY pyproject.toml .
RUN uv sync
