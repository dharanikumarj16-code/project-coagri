# Use official Python lightweight base image
FROM python:3.11-slim

# Set environment variables to prevent pyc files and unbuffer logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set root working directory
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . /app

# Set working directory to smart-farm subfolder where main.py and index.html are located
WORKDIR /app/smart-farm

# Expose backend port
EXPOSE 8000

# Healthcheck to monitor container status
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/crops || exit 1

# Launch FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
