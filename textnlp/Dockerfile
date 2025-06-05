# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install -e .

# Copy application code
COPY . .

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "api/app.py"]