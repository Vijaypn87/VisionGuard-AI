"""Real-Time Anomaly Detection Dashboard.

A Streamlit-based monitoring dashboard that provides:
  - Live video feed with detection/tracking overlays
  - Real-time anomaly score timeline
  - Alert log with timestamps
  - System performance metrics (FPS, latency)
  - Model configuration panel

Launch:
    streamlit run dashboard/app.py
"""

import time
import tempfile
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import AnomalyDetectionPipeline
from src.utils import load_config

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="Video Anomaly Detection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2196F3, #FF5722);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .alert-critical {
        background: rgba(244, 67, 54, 0.1);
        border-left: 4px solid #F44336;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin: 0.25rem 0;
    }
    .alert-normal {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin: 0.25rem 0;
    }
    .stMetric > div {
        background: rgba(33, 150, 243, 0.05);
        border-radius: 8px;
        padding: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main dashboard application."""

    # ── Header ─────────────────────────────────────────────
    st.markdown('<p class="main-header">🎯 Real-Time Video Anomaly Detection</p>', unsafe_allow_html=True)
    st.markdown("*Multi-object tracking + temporal anomaly classification powered by YOLOv8, ByteTrack & VideoMAE*")

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")

        source_type = st.selectbox(
            "Video Source",
            ["Webcam", "Video File", "RTSP Stream", "Demo Mode"],
            index=3,
        )

        if source_type == "Video File":
            _video_file = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mkv"])
            if _video_file is not None:
                st.video(_video_file)
        elif source_type == "RTSP Stream":
                _rtsp_url = st.text_input("RTSP URL", "rtsp://camera.local/stream")
        elif source_type == "Webcam":
                _cam_index = st.number_input("Camera Index", 0, 10, 0)

        st.divider()

        st.subheader("Detection Settings")
        _conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
        anomaly_threshold = st.slider("Anomaly Threshold", 0.3, 0.9, 0.65, 0.05)

        st.divider()

        st.subheader("Display Options")
        _show_heatmap = st.checkbox("Show Anomaly Heatmap", True)
        _show_tracks = st.checkbox("Show Tracking Trails", True)
        _show_scores = st.checkbox("Show Anomaly Scores", True)

        st.divider()

        st.subheader("Model Info")
        st.info(
            "**Detector:** YOLOv8s\n"
            "**Tracker:** ByteTrack\n"
            "**Classifier:** VideoMAE-Base\n"
            "**Dataset:** UCF-Crime"
        )

    # ── Main Content ───────────────────────────────────────
    col_video, col_metrics = st.columns([3, 1])

    with col_video:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()

    with col_metrics:
        st.subheader("📊 Metrics")
        fps_metric = st.empty()
        tracks_metric = st.empty()
        alerts_metric = st.empty()
        latency_metric = st.empty()

    st.divider()

    # ── Charts Row ─────────────────────────────────────────
    col_timeline, col_distribution = st.columns([2, 1])

    with col_timeline:
        st.subheader("📈 Anomaly Score Timeline")
        timeline_placeholder = st.empty()

    with col_distribution:
        st.subheader("🎯 Score Distribution")
        dist_placeholder = st.empty()

    st.divider()

    # ── Alert Log ──────────────────────────────────────────
    st.subheader("🚨 Alert Log")
    alert_placeholder = st.empty()

    # ── Run Demo Mode ──────────────────────────────────────
        # ── Run Demo Mode ──────────────────────────────────────
    if source_type == "Demo Mode":
        _run_demo_mode(
            video_placeholder=video_placeholder,
            fps_metric=fps_metric,
            tracks_metric=tracks_metric,
            alerts_metric=alerts_metric,
            latency_metric=latency_metric,
            timeline_placeholder=timeline_placeholder,
            dist_placeholder=dist_placeholder,
            alert_placeholder=alert_placeholder,
            anomaly_threshold=anomaly_threshold,
        )

    elif source_type == "Video File":

        if _video_file is not None:

            if st.button("▶️ Start Analysis", type="primary"):

                with st.spinner("Running AI video analysis..."):
                    # Save uploaded video temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(_video_file.name).suffix,
                    ) as temp_video:
                        temp_video.write(_video_file.getbuffer())
                        input_path = temp_video.name

                    # Load project configuration
                    config = load_config("configs/default.yaml")

                    # Use settings from dashboard
                    config.setdefault("detector", {})[
                        "conf_threshold"
                    ] = _conf_threshold

                    config.setdefault("anomaly", {})[
                        "anomaly_threshold"
                    ] = anomaly_threshold

                    # CPU configuration
                    config.setdefault("detector", {})["device"] = "cpu"
                    config.setdefault("detector", {})["half"] = False

                    config.setdefault("anomaly", {})["device"] = "cpu"

                    # Don't open an OpenCV window from Streamlit
                    config.setdefault("pipeline", {})["show_display"] = False
                    config.setdefault("pipeline", {})["save_output"] = True
                    config.setdefault("pipeline", {})[
                        "output_path"
                    ] = "outputs"

                    # Create and run the actual pipeline
                    pipeline = AnomalyDetectionPipeline(config)

                    stats = pipeline.run(source=input_path)

                st.success("✅ Video analysis completed!")

                # Show processed video
                output_video = Path("outputs/output.mp4")

                if output_video.exists():
                    st.subheader("🎥 Processed Video")

                    with open(output_video, "rb") as video:
                        video_bytes = video.read()

                    st.video(video_bytes)

                # Show actual pipeline statistics
                fps_metric.metric(
                    "FPS",
                    f"{stats['avg_fps']:.1f}",
                )

                tracks_metric.metric(
                    "Total Tracks",
                    stats["total_tracks"],
                )

                alerts_metric.metric(
                    "Total Alerts",
                    stats["total_alerts"],
                )

                st.success(
                    f"Processed {stats['total_frames']} frames."
                )

        else:
            st.info("Please upload a video first.")

    else:
        st.info(
            "Webcam and RTSP processing will be connected next."
        )


def _run_demo_mode(
    video_placeholder,
    fps_metric,
    tracks_metric,
    alerts_metric,
    latency_metric,
    timeline_placeholder,
    dist_placeholder,
    alert_placeholder,
    anomaly_threshold: float = 0.65,
):
    """Run a demonstration with simulated data."""

    # Simulated timeline data
    np.random.seed(42)
    timeline_scores = []
    alerts = []
    frame_count = 0

    for i in range(200):
        frame_count += 1

        # Simulate anomaly scores with periodic spikes
        base_score = 0.2 + 0.1 * np.sin(i / 20)
        if 60 < i < 90 or 140 < i < 160:
            score = min(base_score + np.random.uniform(0.3, 0.6), 1.0)
        else:
            score = max(base_score + np.random.normal(0, 0.05), 0.0)

        timeline_scores.append(score)

        # Generate demo frame
        demo_frame = _generate_demo_frame(i, score, anomaly_threshold)

        # Update video
        video_placeholder.image(demo_frame, channels="BGR", use_container_width=True)

        # Update metrics
        fps_val = 28 + np.random.uniform(-2, 4)
        num_tracks = np.random.randint(2, 8)
        det_latency = 8 + np.random.uniform(-1, 3)

        fps_metric.metric("FPS", f"{fps_val:.1f}", f"{fps_val - 30:.1f}")
        tracks_metric.metric("Active Tracks", num_tracks)
        latency_metric.metric("Detection Latency", f"{det_latency:.1f}ms")

        # Check for alerts
        if score >= anomaly_threshold:
            alert = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "frame": frame_count,
                "score": score,
                "type": np.random.choice(["Fighting", "Robbery", "Vandalism", "Assault"]),
            }
            alerts.append(alert)

        alerts_metric.metric("Total Alerts", len(alerts), f"+{1 if score >= anomaly_threshold else 0}")

        # Update timeline chart
        if i % 5 == 0:
            fig = _create_timeline_chart(timeline_scores, anomaly_threshold)
            timeline_placeholder.plotly_chart(fig, use_container_width=True)

        # Update distribution
        if i % 10 == 0 and len(timeline_scores) > 10:
            fig = _create_distribution_chart(timeline_scores, anomaly_threshold)
            dist_placeholder.plotly_chart(fig, use_container_width=True)

        # Update alert log
        if alerts:
            alert_html = ""
            for a in reversed(alerts[-10:]):
                css_class = "alert-critical" if a["score"] >= anomaly_threshold else "alert-normal"
                alert_html += (
                    f'<div class="{css_class}">'
                    f'<strong>{a["time"]}</strong> | Frame #{a["frame"]} | '
                    f'{a["type"]} | Score: {a["score"]:.3f}'
                    f'</div>'
                )
            alert_placeholder.markdown(alert_html, unsafe_allow_html=True)

        time.sleep(0.05)


def _generate_demo_frame(
    frame_idx: int,
    anomaly_score: float,
    threshold: float,
) -> np.ndarray:
    """Generate a synthetic demo frame with visualizations."""
    frame = np.zeros((480, 854, 3), dtype=np.uint8)

    # Gradient background
    for y in range(480):
        intensity = int(20 + 10 * (y / 480))
        frame[y, :] = [intensity, intensity + 5, intensity + 10]

    # Simulated bounding boxes
    np.random.seed(frame_idx)
    num_objects = np.random.randint(2, 5)

    for obj_idx in range(num_objects):
        x1 = np.random.randint(50, 600)
        y1 = np.random.randint(50, 350)
        w = np.random.randint(60, 150)
        h = np.random.randint(80, 200)

        obj_score = anomaly_score * np.random.uniform(0.5, 1.2)
        obj_score = min(obj_score, 1.0)

        if obj_score >= threshold:
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        label = f"ID:{obj_idx + 1} Score:{obj_score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # FPS overlay
    cv2.putText(
        frame,
        f"FPS: {28 + np.random.uniform(-2, 4):.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Alert banner
    if anomaly_score >= threshold:
        cv2.rectangle(frame, (0, 440), (854, 480), (0, 0, 200), -1)
        cv2.putText(
            frame,
            f"ANOMALY DETECTED - Score: {anomaly_score:.3f}",
            (10, 465),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return frame


def _create_timeline_chart(
    scores: list[float],
    threshold: float,
) -> go.Figure:
    """Create an anomaly score timeline chart."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=scores,
            mode="lines",
            name="Anomaly Score",
            line=dict(color="#2196F3", width=2),
            fill="tozeroy",
            fillcolor="rgba(33, 150, 243, 0.1)",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#F44336",
        annotation_text=f"Threshold ({threshold})",
    )

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        yaxis=dict(range=[0, 1], title="Score"),
        xaxis=dict(title="Frame"),
        template="plotly_dark",
        showlegend=False,
    )

    return fig


def _create_distribution_chart(
    scores: list[float],
    threshold: float,
) -> go.Figure:
    """Create anomaly score distribution chart."""
    scores_arr = np.array(scores)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=scores_arr,
            nbinsx=30,
            marker_color="#2196F3",
            opacity=0.7,
            name="Score Distribution",
        )
    )

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#F44336",
        annotation_text="Threshold",
    )

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(title="Score", range=[0, 1]),
        yaxis=dict(title="Count"),
        template="plotly_dark",
        showlegend=False,
    )

    return fig


if __name__ == "__main__":
    main()
