# Base image using official Python 3.11 slim
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies for pygame/headless operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    libsdl2-mixer-2.0-0 \
    libsdl2-ttf-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and repository structure
COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/
COPY screenshots/ ./screenshots/
COPY README.md LICENSE ./

# Default entrypoint runs training, but can be overridden
CMD ["python", "-m", "src.train"]
