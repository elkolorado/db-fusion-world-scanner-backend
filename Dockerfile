FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libgl1 \
       libglib2.0-0 \
       libsm6 \
       libxrender1 \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt && \
    pip install faiss-cpu

# Copy application
COPY . /app

# Ensure directories exist for mounted volumes
RUN mkdir -p /app/indexes /app/card_images

EXPOSE 8002

# Run the initializer which will generate indexes if missing and start uvicorn
CMD ["python", "initialize.py"]
