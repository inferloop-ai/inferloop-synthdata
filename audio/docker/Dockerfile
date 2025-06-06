# docker/Dockerfile
"""
Docker configuration for Audio Synthetic Data Framework
"""

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV AUDIO_SYNTH_HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR $AUDIO_SYNTH_HOME

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY examples/ ./examples/

# Install the package
RUN pip install -e .

# Create directories for models and output
RUN mkdir -p /app/models /app/output /app/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "audio_synth.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

