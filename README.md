# YOLOAFI — Data Center Security Monitoring with YOLOv11

Proactive physical-security system for data centers (AFI — Administración de la Función Informática) using two complementary YOLOv11 models: a custom-trained person detector and a COCO-pretrained model that flags unauthorized objects (bottles, cell phones, laptops, backpacks).

## Features

- **Dual-model detection** — custom fine-tuned `yolo11m` person detector + COCO model for 80-class object detection
- **Custom IoU tracking** — tracks people across frames, confirms a persistent threat over 10 consecutive frames before alerting (false positives < 5%)
- **Real-time pipeline** — camera to YOLO (CUDA) to annotated overlay streamed via ffplay
- **JSON activity logging** — confirmed person counts timestamped and appended to `people_log.json`
- **Configurable thresholds** — detection at 0.65 confidence, visualization at 0.50

## Performance (validated, RTX 3060)

| Metric | Value |
|---|---|
| mAP50 (global) | 79.7% |
| mAP50-95 (global) | 60.2% |
| Person class precision | 78.7% |
| Person class recall | 55.1% |
| Inference latency | 4.0 ms |
| False positive rate | < 5% |
| Threat confirmation | 10 consecutive frames |

## Architecture

```
Camera → YOLOv11 (CUDA) → IoU tracker (10-frame confirm) → Overlay + alert → ffplay
                                                              └→ JSON log
```

- **Person model:** `yolo11m` fine-tuned on a Roboflow people-detection dataset (10 epochs, imgsz 640, batch 16)
- **Object model:** COCO-pretrained YOLOv11 (80 classes)
- **Hardware:** NVIDIA RTX 3060 12GB (edge/on-premise inference)

## Getting Started

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (or switch `device` to `cpu`)
- ffplay (from FFmpeg)

### Install
```bash
pip install ultralytics opencv-python
# ffplay must be on your PATH (install FFmpeg)
```

### Run
```bash
python test.py            # live camera detection (camera index 1)
```

### Training (reproduce the custom person model)
```bash
# 1. Set your Roboflow API key (never hardcode it)
export ROBOFLOW_API_KEY="your-key-here"

# 2. Open main.ipynb — cells download datasets via Roboflow and train yolo11m/yolo11n
# (dataset sources are the "people-detection" and "pendrive-detection" Roboflow projects)
```

## Repository Layout

```
├── test.py               # live inference + IoU tracking + ffplay overlay
├── main.ipynb            # dataset download + training notebooks
├── docs/doc.tex          # full technical report (Spanish) — architecture, ROI, ISO 27001/ITIL 4
├── scripts/
│   └── download_weights.sh  # fetch pretrained weights from GitHub Releases
└── people_log.json       # runtime activity log (gitignored)
```

## Documentation

The full technical report (`docs/doc.tex`) covers:
- On-premise / edge computing architecture rationale
- Training methodology (Roboflow MLOps pipeline, phases 1-4)
- 718% year-one ROI analysis
- Alignment with ISO/IEC 27001:2022, ITIL 4, and COBIT 2019

## License

MIT
