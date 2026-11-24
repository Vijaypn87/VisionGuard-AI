"""Setup script for VisionGuard AI."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent


def read_requirements(name: str) -> list[str]:
    reqs = []
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-r")):
            reqs.append(line)
    return reqs


setup(
    name="visionguard-ai",
    version="1.0.0",
    author="vijaypn",
    author_email="Vijaypn.in@gmail.com",
    description="Video multi-object tracking (YOLOv8 + ByteTrack) and clip-level anomaly scoring (VideoMAE + MIL head)",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/Vijaypn87/VisionGuard-AI",
    # Ship only the runtime packages; never install a top-level `tests` package.
    packages=find_packages(include=["src", "src.*", "training", "training.*", "evaluation", "evaluation.*", "optimization", "optimization.*"]),
    python_requires=">=3.10",
    install_requires=read_requirements("requirements-base.txt"),
    extras_require={
        "cpu": ["onnxruntime>=1.16.0"],
        "gpu": ["onnxruntime-gpu>=1.16.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "ruff>=0.4.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        "console_scripts": [
            "vad-detect=src.pipeline:main",
            "vad-train=training.train_mil:main",
            "vad-benchmark=optimization.benchmark:main",
            "vad-evaluate=evaluation.evaluate_ucf_crime:main",
        ],
    },
)
