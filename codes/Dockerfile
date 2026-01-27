# ============================================================
# DocuVision Pro - Dockerfile
# Multi-stage build for optimized image size
# ============================================================

FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu && \
    pip install -r requirements.txt

# ============================================================
# Production Stage
# ============================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="DocuVision Pro" \
      version="1.0" \
      description="AI-Powered Document Image Processing"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Gradio settings
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    # App settings
    APP_HOME=/app \
    MODEL_DIR=/app/models \
    OUTPUT_DIR=/app/output \
    TEMP_DIR=/tmp/docuvision

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # For HEIC support
    libheif-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r docuvision && useradd -r -g docuvision docuvision

# Create application directory
WORKDIR ${APP_HOME}

# Create necessary directories
RUN mkdir -p ${MODEL_DIR} ${OUTPUT_DIR} ${TEMP_DIR} \
    && chown -R docuvision:docuvision ${APP_HOME} ${TEMP_DIR}

# Copy application code
COPY --chown=docuvision:docuvision . .

# Set permissions
RUN chmod -R 755 ${APP_HOME}

# Switch to non-root user
USER docuvision

# Expose Gradio port
EXPOSE 7861

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7861')" || exit 1

# Default command
CMD ["python", "frontend_docker_com.py"]