FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-dev libgl1-mesa-glx libglib2.0-0 && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "initialize.py"]