"""FastAPI website for real-time facial-expression estimation."""
from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from emotion_engine import analyze_bgr

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="OG Detector", description="Live facial-expression demo")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class FramePayload(BaseModel):
    image: str = Field(..., description="Base64-encoded JPEG frame")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
def analyze(payload: FramePayload):
    encoded = payload.image
    if "," in encoded:
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image encoding.")
    if len(raw) > 2_500_000:
        raise HTTPException(status_code=413, detail="Frame is too large.")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode camera frame.")
    try:
        predictions = analyze_bgr(image)
    except ValueError as exc:
        # No face is a normal live-camera state, not a server error.
        if "face" in str(exc).lower():
            return {"faces": []}
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    faces = []
    for item in predictions:
        region = item.get("region", {})
        emotion = item.get("emotion", "unknown")
        scores = item.get("scores", {})
        faces.append({
            "x": int(region.get("x", 0)),
            "y": int(region.get("y", 0)),
            "w": int(region.get("w", 0)),
            "h": int(region.get("h", 0)),
            "emotion": str(emotion),
            "confidence": round(float(scores.get(emotion, 0)), 1),
            "scores": scores,
        })
    return {"faces": faces}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
