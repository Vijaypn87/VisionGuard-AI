<h1 align="center">VisionGuard AI — Video Anomaly Detection</h1>

<p align="center">
  <strong>Multi-object tracking and clip-level anomaly scoring for surveillance video</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white" alt="YOLOv8"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

---

## Overview

A pipeline that detects and tracks people and vehicles in video (webcam, RTSP or file) and scores short clips for anomalous activity, with a Streamlit dashboard.

**Pipeline:** Video input → YOLOv8 detection → ByteTrack tracking → VideoMAE clip features → MIL anomaly head → overlays → display / saved video

## Project status

| Component | Status |
|---|---|
| Detection + tracking (YOLOv8 + ByteTrack) | Works out of the box with the pretrained COCO weights |
| Anomaly scoring | **Requires training.** You must train the MIL head on UCF-Crime (below). Until `weights/anomaly_head.pt` exists, the pipeline runs detection + tracking only and shows "Anomaly scoring OFF" |
| Benchmark results | None published yet. Run the evaluation and benchmark scripts on your own hardware; they write to `results/` |

## Features

- **Detection & tracking** — YOLOv8 + ByteTrack with persistent object IDs
- **Anomaly scoring** — frozen VideoMAE (Kinetics-400) clip embeddings + a small MLP head trained with the MIL ranking loss of [Sultani et al., CVPR 2018](https://arxiv.org/abs/1801.04264) on UCF-Crime video-level labels
- **Two scoring modes** — `frame` (whole-frame clips; matches training, default) and `track` (per-object crops; experimental)
- **Score heatmap overlay** — highlights objects by anomaly score (track mode). This is a visual cue, not Grad-CAM or model attention
- **Dashboard** — Streamlit UI for uploaded videos. Its "Simulated demo" mode shows random data and is labelled as such
- **ONNX export & latency benchmark** for the detector

## Quick start

```bash
git clone https://github.com/Vijaypn87/VisionGuard-AI.git
cd VisionGuard-AI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # use requirements-gpu.txt on CUDA machines

# Detection + tracking (anomaly scoring stays off until you train a head)
python -m src.pipeline --source path/to/video.mp4
python -m src.pipeline --source 0                         # webcam
python -m src.pipeline --source "rtsp://camera.local/stream1"

streamlit run dashboard/app.py
```

## Training the anomaly head

Download [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) into `data/UCF-Crime/` (videos plus `Anomaly_Train.txt` and `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`).

```bash
# 1. Extract frozen VideoMAE features: 32 segments per video (GPU strongly recommended)
python -m training.train_mil extract --dataset-path data/UCF-Crime \
    --list Anomaly_Train.txt --out data/features/train

# 2. Train the MIL head → weights/anomaly_head.pt
python -m training.train_mil train --features data/features/train

# 3. Frame-level ROC-AUC on the official test split → results/metrics.json
python -m evaluation.evaluate_ucf_crime --dataset-path data/UCF-Crime
#    Optical-flow baseline for comparison:
python -m evaluation.evaluate_ucf_crime --dataset-path data/UCF-Crime --baseline motion
```

After training, tune `anomaly.anomaly_threshold` in `configs/default.yaml` on held-out data.

## Benchmarking the detector

```bash
python -m optimization.export_onnx          # optional, for the ONNX row
python -m optimization.benchmark            # writes results/benchmark_table.md
```

The table records the hardware it ran on and covers the detector only.

## Project structure

```
├── src/            # Detection, tracking, anomaly scoring, pipeline, overlays
├── training/       # Feature extraction + MIL head training
├── evaluation/     # UCF-Crime evaluation & metrics
├── optimization/   # ONNX export, benchmarking, quantization
├── dashboard/      # Streamlit UI
├── configs/        # YAML configs
└── Dockerfile
```

## Tech stack

Python · PyTorch · YOLOv8 · ByteTrack · VideoMAE · ONNX Runtime · OpenCV · Streamlit · Docker
