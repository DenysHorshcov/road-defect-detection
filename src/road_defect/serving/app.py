"""FastAPI inference service for road defect detection.

The model is resolved once at startup, not per request. Two sources are
supported: a weights file named by MODEL_WEIGHTS, which is how the container
runs, and otherwise the MLflow registry alias, which is how a developer runs it
against whatever is currently promoted.

Usage:
    uv run uvicorn road_defect.serving.app:app --reload
"""

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from road_defect.classes import CLASS_NAMES

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_CONFIDENCE = 0.25


class Box(BaseModel):
    class_name: str
    confidence: float
    xyxy: list[float]


class Prediction(BaseModel):
    boxes: list[Box]
    image_size: list[int]
    inference_ms: float


class Health(BaseModel):
    status: str
    model_source: str
    classes: list[str]


def load_model() -> tuple[object, str]:
    """Load from an explicit weights file, else from the MLflow registry."""
    from ultralytics import YOLO

    weights = os.environ.get("MODEL_WEIGHTS")
    if weights:
        path = Path(weights)
        if not path.exists():
            raise RuntimeError(f"MODEL_WEIGHTS={path} does not exist")
        return YOLO(path), str(path)

    from road_defect.registry import MODEL_NAME, download_weights

    alias = os.environ.get("MODEL_ALIAS", "champion")
    path = download_weights(alias=alias)
    return YOLO(path), f"{MODEL_NAME}@{alias}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.model_source = load_model()
    yield
    app.state.model = None


app = FastAPI(
    title="Road Defect Detection",
    description="Detects potholes, cracks and faded lane markings in street-level photos.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok", model_source=app.state.model_source, classes=CLASS_NAMES)


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile, confidence: float = DEFAULT_CONFIDENCE) -> Prediction:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(415, f"Expected an image, got {file.content_type!r}")
    if not 0 < confidence <= 1:
        raise HTTPException(422, "confidence must be in (0, 1]")

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(400, "Could not decode the uploaded image") from None

    started = time.perf_counter()
    results = app.state.model.predict(image, conf=confidence, verbose=False)
    elapsed_ms = (time.perf_counter() - started) * 1000

    boxes = [
        Box(
            class_name=CLASS_NAMES[int(cls)],
            confidence=round(float(conf), 4),
            xyxy=[round(float(v), 2) for v in xyxy],
        )
        for result in results
        for cls, conf, xyxy in zip(
            result.boxes.cls, result.boxes.conf, result.boxes.xyxy, strict=True
        )
    ]

    return Prediction(
        boxes=boxes,
        image_size=list(image.size),
        inference_ms=round(elapsed_ms, 2),
    )
