import numpy as np
import pytest

import emotion_engine


def test_rejects_missing_image():
    with pytest.raises(ValueError):
        emotion_engine.analyze_bgr(None)


def test_rejects_empty_image():
    with pytest.raises(ValueError):
        emotion_engine.analyze_bgr(np.array([]))


def test_rejects_grayscale_image():
    with pytest.raises(ValueError):
        emotion_engine.analyze_bgr(np.zeros((64, 64), dtype=np.uint8))


def test_rejects_too_small_image():
    with pytest.raises(ValueError, match="too small"):
        emotion_engine.analyze_bgr(np.zeros((16, 16, 3), dtype=np.uint8))


def test_normalizes_model_output(monkeypatch):
    fake_result = [{
        "dominant_emotion": "happy",
        "emotion": {"happy": 95.0, "sad": 5.0},
        "region": {"x": 10, "y": 15, "w": 50, "h": 60},
    }]

    class FakeDeepFace:
        @staticmethod
        def analyze(**kwargs):
            return fake_result

    monkeypatch.setattr(emotion_engine, "_deepface", lambda: FakeDeepFace)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = emotion_engine.analyze_bgr(image)

    assert len(result) == 1
    assert result[0]["emotion"] == "happy"
    assert result[0]["scores"]["happy"] == 95.0
    assert result[0]["region"]["x"] == 10


def test_handles_multiple_faces(monkeypatch):
    fake_result = [
        {
            "dominant_emotion": "happy",
            "emotion": {"happy": 90.0, "sad": 10.0},
            "region": {"x": 5, "y": 5, "w": 20, "h": 20},
        },
        {
            "dominant_emotion": "neutral",
            "emotion": {"neutral": 80.0, "happy": 20.0},
            "region": {"x": 40, "y": 5, "w": 20, "h": 20},
        },
    ]

    class FakeDeepFace:
        @staticmethod
        def analyze(**kwargs):
            return fake_result

    monkeypatch.setattr(emotion_engine, "_deepface", lambda: FakeDeepFace)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    result = emotion_engine.analyze_bgr(image)

    assert len(result) == 2
    assert result[0]["emotion"] == "happy"
    assert result[1]["emotion"] == "neutral"


def test_annotation_preserves_image_shape():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    prediction = [{
        "emotion": "happy",
        "scores": {"happy": 90.0},
        "region": {"x": 10, "y": 10, "w": 40, "h": 40},
    }]
    output = emotion_engine.annotate_image(image, prediction)
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert not np.shares_memory(output, image)
