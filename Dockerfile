FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

LABEL maintainer="vijaypn <vijaypn.in@gmail.com>"
LABEL description="VisionGuard AI - video anomaly detection"

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

# System dependencies (libgl/libglib are needed by OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-base.txt requirements-gpu.txt ./
RUN pip3 install --no-cache-dir -r requirements-gpu.txt

COPY . .
RUN mkdir -p outputs results weights data

# Mount a trained head at /app/weights/anomaly_head.pt to enable anomaly
# scoring; without it the app runs detection + tracking only.
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Headless CLI on a mounted video instead:
#   docker run --gpus all -v $PWD/data:/app/data <image> \
#     python3 -m src.pipeline --source data/video.mp4 --no-display
