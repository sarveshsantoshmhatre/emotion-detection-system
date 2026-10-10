"""Pre-trained facial-expression analysis helpers.

This project estimates visible facial-expression categories; it does not
determine or validate a person's internal emotional state.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np


@lru_cache(maxsize=1)
def _deepface():
    """Import DeepFace lazily so non-model tests do not load TensorFlow."""
    from deepface import DeepFace
    return DeepFace


def analyze_bgr(image: np.ndarray) -> list[dict[str, Any]]:
    """Analyze a BGR image and return normalized results for detected faces."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("A valid image is required.")
    if image.size == 0:
        raise ValueError("The image is empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel color image.")

    height, width = image.shape[:2]
    if height < 32 or width < 32:
        raise ValueError("Image is too small. Use an image at least 32x32 pixels.")
    if max(height, width) > 4000:
        scale = 4000 / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    try:
        raw = _deepface().analyze(
            img_path=image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True,
        )
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Facial-expression analysis failed. Check installed model "
            "dependencies and try a clear, front-facing image."
        ) from exc

    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError("No face was detected in the image.")

    normalized: list[dict[str, Any]] = []
    for result in raw:
        if not isinstance(result, dict):
            continue
        scores = result.get("emotion")
        if not isinstance(scores, dict) or not scores:
            continue

        clean_scores: dict[str, float] = {}
        for label, score in scores.items():
            try:
                number = float(score)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                clean_scores[str(label)] = max(0.0, min(100.0, number))
        if not clean_scores:
            continue

        region = result.get("region") or result.get("facial_area") or {}
        if not isinstance(region, dict):
            region = {}
        box = {}
        for key in ("x", "y", "w", "h"):
            try:
                box[key] = int(region.get(key, 0))
            except (TypeError, ValueError):
                box[key] = 0

        dominant = result.get("dominant_emotion")
        if dominant not in clean_scores:
            dominant = max(clean_scores, key=clean_scores.get)

        normalized.append({
            "emotion": str(dominant),
            "scores": clean_scores,
            "region": box,
        })

    if not normalized:
        raise ValueError("The model returned no usable face predictions.")
    return normalized


def annotate_image(image_bgr: np.ndarray,
                   predictions: list[dict[str, Any]]) -> np.ndarray:
    """Draw a face rectangle and predicted expression label for each face."""
    if image_bgr is None or not isinstance(image_bgr, np.ndarray):
        raise ValueError("A valid image is required.")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel color image.")

    output = image_bgr.copy()
    height, width = output.shape[:2]
    for prediction in predictions:
        area = prediction.get("region", {})
        x = max(0, min(width - 1, int(area.get("x", 0))))
        y = max(0, min(height - 1, int(area.get("y", 0))))
        x2 = max(x, min(width - 1, x + max(0, int(area.get("w", 0)))))
        y2 = max(y, min(height - 1, y + max(0, int(area.get("h", 0)))))
        label = str(prediction.get("emotion", "unknown")).title()
        cv2.rectangle(output, (x, y), (x2, y2), (40, 200, 80), 2)
        cv2.putText(
            output, label, (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 200, 80), 2, cv2.LINE_AA
        )
    return output
