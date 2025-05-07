FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        curl \
        g++ \
        vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.6.9 /uv /uvx /bin/

WORKDIR /app

RUN uv python install 3.10 \
    && uv venv .venv
ENV PATH="/.venv/bin:${PATH}"

COPY pyproject.toml .
RUN uv sync

# COPY main.py .

ENV PORT=8080
EXPOSE 8080

# CMD [ ]
