<h1 align="center">Real-Time Video Anomaly Detection</h1>

<p align="center">
  <strong>Multi-object tracking and temporal anomaly classification for surveillance video streams</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white" alt="YOLOv8"></a>
  <a href="#"><img src="https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

---

## Overview

A real-time pipeline that detects and tracks people/objects in video streams, then flags anomalous behavior using temporal deep learning. Built for surveillance and safety-monitoring use cases (webcam, RTSP, or file input), with a live dashboard for alerts.

**Pipeline:** Video Input → YOLOv8 Detection → ByteTrack (multi-object tracking) → VideoMAE (temporal anomaly scoring) → Grad-CAM Heatmaps → Streamlit Dashboard

## Key Features

- **Real-time detection & tracking** — YOLOv8 + ByteTrack for persistent object IDs across frames
- **Temporal anomaly scoring** — VideoMAE with multiple-instance learning (MIL) aggregation
- **85% ROC-AUC** on the UCF-Crime benchmark (13 anomaly categories)
- **30+ FPS** via ONNX-optimized inference (~40% latency reduction vs. PyTorch baseline)
- **Explainability** — Grad-CAM heatmaps show where anomalies occur in-frame
- **Live dashboard** — Streamlit UI with anomaly timeline and alert log
- **Multi-dataset support** — UCF-Crime, ShanghaiTech, CUHK Avenue, XD-Violence
- **Dockerized** for GPU deployment

## Quick Start

```bash
git clone https://github.com/Pari2003/Real-Time-Video-Anomaly-Detection.git
cd Real-Time-Video-Anomaly-Detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run on webcam, video file, or RTSP stream
python -m src.pipeline --source 0
python -m src.pipeline --source path/to/video.mp4
python -m src.pipeline --source "rtsp://camera.local/stream1"

# Launch dashboard
streamlit run dashboard/app.py
```

## Performance

| Metric | Score |
|--------|-------|
| ROC-AUC (UCF-Crime) | 85.0% |
| F1-Score | 70.0% |
| Inference (ONNX, CUDA) | ~92 FPS |
| Inference (TensorRT FP16) | ~139 FPS |

## Project Structure

```
├── src/            # Detection, tracking, classification, pipeline
├── optimization/    # ONNX export, benchmarking, quantization
├── evaluation/      # UCF-Crime evaluation & metrics
├── dashboard/       # Streamlit monitoring UI
├── configs/         # YAML configs
└── Dockerfile
```

## Tech Stack

Python · PyTorch · YOLOv8 · ByteTrack · VideoMAE · ONNX Runtime · OpenCV · Streamlit · Docker
