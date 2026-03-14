# ---- Stage 1: Build the Next.js frontend ----
FROM node:22-alpine AS frontend-builder
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ .
RUN npm run build

# ---- Stage 2: Final image (Python + Node runtime) ----
FROM python:3.12-slim

WORKDIR /app

# System deps: libmagic (unstructured), libpq (psycopg2), Node.js runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libmagic1 \
    libpq-dev \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Backend source
COPY backend/ ./backend/

# Frontend standalone build
COPY --from=frontend-builder /build/.next/standalone ./frontend/
COPY --from=frontend-builder /build/.next/static ./frontend/.next/static

# Entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 8000 3000

CMD ["./entrypoint.sh"]
