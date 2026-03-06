# ── Training image ───────────────────────────────────────────
FROM python:3.11-slim AS training

WORKDIR /app

# Install system dependencies for PyBullet
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY env/ env/
COPY agents/ agents/
COPY safety/ safety/
COPY utils/ utils/
COPY dashboard/ dashboard/
COPY train.py evaluate.py config.yaml ./

ENTRYPOINT ["uv", "run"]
CMD ["train.py"]

# ── Dashboard image ──────────────────────────────────────────
FROM node:18-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM training AS dashboard
COPY --from=frontend-build /frontend/dist /app/frontend/dist
CMD ["uvicorn", "evaluate:app", "--host", "0.0.0.0", "--port", "8000"]
