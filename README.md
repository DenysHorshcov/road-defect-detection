# Road Defect Detection — End-to-End MLOps Project

Detecting road surface defects (potholes, cracks, faded lane markings) from street-level photos, built as a complete MLOps pipeline: data collection, model training, experiment tracking, API serving, containerization, and cloud deployment.

## Overview

This project demonstrates a full ML lifecycle around a computer vision task, using data collected from real sources rather than a ready-made dataset:

1. **Data collection** — street-level photos sourced from [Mapillary](https://www.mapillary.com/) (CC-BY-SA), optionally supplemented with self-collected dashcam footage.
2. **Model training** — object detection with YOLOv8 (Ultralytics), fine-tuned from COCO-pretrained weights.
3. **Experiment tracking** — MLflow for logging parameters, metrics, and artifacts, with the Model Registry for staging/production versioning.
4. **API serving** — FastAPI service for image upload → inference → bounding box predictions.
5. **Containerization** — Docker for a reproducible runtime environment.
6. **Cloud deployment** — AWS (ECS/Fargate).

## Task

Object detection (bounding boxes) across 3 defect classes:

- `pothole`
- `crack`
- `faded_lane_marking`

## Model

YOLOv8n/s (Ultralytics), transfer-learned from COCO-pretrained weights — chosen for fast inference, strong built-in augmentations, and native MLflow integration, given a small self-labeled dataset.

## Status

Project scaffolding in progress. See planned pipeline stages above; each will be built out incrementally (data pipeline → training/MLflow → API → Docker → AWS).
