# Use a lightweight python:3.11-slim base image
FROM python:3.11-slim

# Set environment variables to optimize Python performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Set the working directory inside the container
WORKDIR /app

# Install basic compiler tools if needed and clear package cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user

# Copy requirements file first to maximize Docker layer caching
COPY requirements.txt .

# Install python packages with no caching to keep the image compact
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the remaining project files and assign ownership to the user
COPY --chown=user:user . .

# Create log/model folders and set proper permissions
RUN mkdir -p /app/logs /app/models && \
    chown -R user:user /app/logs /app/models

# Switch to the non-root user for security (Hugging Face standard)
USER user

# Expose the default port routed by Hugging Face Spaces
EXPOSE 7860

# Execute uvicorn FastAPI serving layer
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
