#!/usr/bin/env bash
# Download the YOLOv11 weights used by YOLOAFI.
# Usage: bash scripts/download_weights.sh
set -euo pipefail

BASE="https://github.com/LaloCornejo/YOLOAFI/releases/download/v1.0.0"
DEST="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$DEST/weights"

echo "Downloading model weights from GitHub Releases..."
curl -L -o "$DEST/weights/person_detector_yolo11m.pt" "$BASE/person_detector_yolo11m.pt"
echo "✅ person_detector_yolo11m.pt"
