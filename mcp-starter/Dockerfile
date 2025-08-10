# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8086

WORKDIR /app

# Install minimal OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY mcp-bearer-token ./mcp-bearer-token
COPY pyproject.toml ./

EXPOSE 8086

# Start the MCP server
CMD ["python", "-u", "mcp-bearer-token/mcp_starter.py"]


