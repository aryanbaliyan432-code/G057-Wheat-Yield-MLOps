# Production multi-stage or lean single-stage Dockerfile
# Using official lightweight Python 3.12 image as required
FROM python:3.12-slim

# Set environment variables for Python in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Layer caching: copy requirements first and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root application user for container security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy application code and model artifacts
COPY app/ /app/app/
COPY src/ /app/src/
COPY models/ /app/models/
COPY params.yaml /app/params.yaml
COPY findings.yaml /app/findings.yaml

# Set ownership to appuser
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check configuration
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
