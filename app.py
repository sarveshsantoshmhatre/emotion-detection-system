"""Streamlit dashboard for live facial-expression estimation and image analysis."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from emotion_engine import analyze_bgr, annotate_image

st.set_page_config(
    page_title="OG Detector — Live Expression Detection",
    page_icon="🎥",
    layout="wide",
)

# Match the standalone web_app.py dashboard while keeping the UI native to Streamlit.
st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp {
        background: radial-gradient(ellipse at top, #202638 0, #0c0e13 55%);
        color: #f4f5f9;
      }
      .block-container { max-width: 1420px; padding-top: 1.5rem; padding-bottom: 2.4rem; }
      header[data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"] { right: 1rem; }
      h1, h2, h3 { letter-spacing: -0.035em; }
      .brand-row { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:24px; }
      .brand { font-weight:850; font-size:24px; letter-spacing:-.6px; color:#f4f5f9; }
      .brand span { color:#f5d547; }
      .pill { border:1px solid #2b3040; border-radius:999px; padding:7px 12px; color:#a5adbf; font-size:11px; letter-spacing:.5px; }
      .eyebrow { color:#f5d547; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:800; }
      .hero-title { font-size:clamp(29px,4vw,44px); font-weight:850; letter-spacing:-1.4px; line-height:1.05; margin:8px 0 10px; }
      .hero-copy { color:#a5adbf; max-width:760px; line-height:1.6; margin-bottom:20px; }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color:#2b3040 !important; border-radius:18px !important;
        background:linear-gradient(145deg,rgba(26,30,41,.94),rgba(20,23,32,.94));
      }
      div[data-testid="stTabs"] button[role="tab"] { font-weight:700; }
      div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color:#f5d547; border-color:#f5d547;
      }
      .section-title { font-size:18px; font-weight:800; margin:0 0 3px; }
      .muted { color:#a5adbf; font-size:12px; }
      .notice { border:1px solid #4a4426; background:#262315; color:#e7dca1; padding:12px 14px; border-radius:12px; font-size:12px; margin-top:18px; }
      .stButton > button[kind="primary"] { background:#f5d547; border-color:#f5d547; color:#17170c; font-weight:800; }
      .stButton > button { border-radius:10px; }
      [data-testid="stMetric"] { background:#11141d; border:1px solid #2b3040; border-radius:12px; padding:12px 14px; }
      [data-testid="stMetricLabel"] { color:#a5adbf; }
      div[data-testid="stProgressBar"] > div { background:#2b3040; }
      div[data-testid="stProgressBar"] > div > div { background:#f5d547; }
      .footer-note { color:#777f92; font-size:11px; margin-top:18px; display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; }
      @media (max-width: 800px) { .block-container { padding-top:1rem; } .pill { font-size:9px; } }
    </style>
    <div class="brand-row">
      <div class="brand">OG <span>Detector</span></div>
      <div class="pill">LIVE FACIAL EXPRESSION DEMO</div>
    </div>
    <div class="eyebrow">Computer vision · Real-time</div>
    <div class="hero-title">See expression predictions<br>as they happen.</div>
    <div class="hero-copy">Turn on your camera to estimate visible facial-expression categories. The video stays live while analysis runs in the background, with face boxes and score summaries refreshed as results arrive.</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice">For demonstration only. Facial-expression predictions do not establish a person’s actual feelings. Do not use results for medical, employment, educational, security, or other high-impact decisions.</div>',
    unsafe_allow_html=True,
)


def safe_error_message(exc: Exception) -> str:
    """Show actionable errors without dumping library internals into the UI."""
    if isinstance(exc, (ValueError, RuntimeError)):
        return str(exc)
    return (
        "Analysis could not be completed. Check your internet connection for the "
        "initial model download, verify the installation, and try a clear image."
    )


def _smooth_predictions(
    previous: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Smooth face boxes and score changes by matching detections between frames."""
    old = list(previous)
    used: set[int] = set()
    output: list[dict[str, Any]] = []
    mix = 0.40

    # Larger faces are a useful and stable primary display order.
    ordered = sorted(incoming, key=lambda f: int(f.get("region", {}).get("w", 0)) *
                     int(f.get("region", {}).get("h", 0)), reverse=True)

    for face in ordered:
        region = face.get("region", {})
        cx = float(region.get("x", 0)) + float(region.get("w", 0)) / 2
        cy = float(region.get("y", 0)) + float(region.get("h", 0)) / 2
        best_idx = -1
        best_distance = float("inf")
        for idx, prior in enumerate(old):
            if idx in used:
                continue
            p = prior.get("region", {})
            px = float(p.get("x", 0)) + float(p.get("w", 0)) / 2
            py = float(p.get("y", 0)) + float(p.get("h", 0)) / 2
            distance = float(np.hypot(px - cx, py - cy))
            if distance < best_distance:
                best_idx, best_distance = idx, distance

        if best_idx < 0:
            output.append(face)
            continue

        prior = old[best_idx]
        p = prior.get("region", {})
        max_distance = max(90.0, float(np.hypot(
            float(p.get("w", 0)), float(p.get("h", 0))
        )) * 0.8)
        if best_distance > max_distance:
            output.append(face)
            continue
        used.add(best_idx)

        scores = {
            label: (
                (1 - mix) * float(prior.get("scores", {}).get(label, 0.0))
                + mix * float(face.get("scores", {}).get(label, 0.0))
            )
            for label in set(prior.get("scores", {})) | set(face.get("scores", {}))
        }
        emotion = max(scores, key=scores.get) if scores else face.get("emotion", "unknown")
        previous_emotion = str(prior.get("emotion", "")).lower()
        previous_key = next((k for k in scores if k.lower() == previous_emotion), None)
        # Require a meaningful score gap before switching the visible label.
        if previous_key and scores[previous_key] + 7 >= scores.get(emotion, 0):
            emotion = previous_key

        box = {}
        for key in ("x", "y", "w", "h"):
            box[key] = int(
                (1 - mix) * float(p.get(key, region.get(key, 0)))
                + mix * float(region.get(key, 0))
            )
        output.append({
            "emotion": emotion,
            "scores": scores,
            "region": box,
        })
    return output


class EmotionVideoProcessor(VideoProcessorBase):
    """Keep WebRTC video responsive by running model inference on a worker thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._predictions: list[dict[str, Any]] = []
        self._status = "Camera started. Waiting for the first analysis."
        self._latency: float | None = None
        self._last_detected_at = 0.0
        self._last_submit_at = 0.0
        self._frame_count = 0
        self._worker = threading.Thread(
            target=self._analysis_loop, name="emotion-analysis", daemon=True
        )
        self._worker.start()

    def _offer_latest_frame(self, image: np.ndarray) -> None:
        now = time.monotonic()
        # Limit capture work while the queue itself always keeps the newest frame.
        if now - self._last_submit_at < 0.22:
            return
        self._last_submit_at = now
        try:
            self._queue.put_nowait(image.copy())
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(image.copy())
            except queue.Full:
                pass

    def _analysis_loop(self) -> None:
        while not self._closed.is_set():
            try:
                image = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if image is None:
                return

            started = time.perf_counter()
            try:
                incoming = analyze_bgr(image)
                with self._lock:
                    self._predictions = _smooth_predictions(self._predictions, incoming)
                    self._last_detected_at = time.monotonic()
                    self._status = f"Detected {len(self._predictions)} face(s)"
                    self._latency = time.perf_counter() - started
            except Exception as exc:
                message = safe_error_message(exc)
                no_face = isinstance(exc, ValueError) and "face" in str(exc).lower()
                with self._lock:
                    elapsed = time.monotonic() - self._last_detected_at
                    if no_face and self._predictions and elapsed < 1.4:
                        self._status = "Face briefly lost · holding the last result"
                    elif no_face:
                        self._predictions = []
                        self._status = "No face detected · improve lighting or face the camera"
                    else:
                        # Keep the last good overlay during temporary model errors.
                        self._status = message
                    self._latency = time.perf_counter() - started

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "predictions": [
                    {
                        "emotion": p.get("emotion", "unknown"),
                        "scores": dict(p.get("scores", {})),
                        "region": dict(p.get("region", {})),
                    }
                    for p in self._predictions
                ],
                "status": self._status,
                "latency": self._latency,
                "last_detected_at": self._last_detected_at,
            }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self._frame_count += 1
        self._offer_latest_frame(image)

        snapshot = self.get_snapshot()
        predictions = snapshot["predictions"]
        output = annotate_image(image, predictions) if predictions else image.copy()

        color = (80, 220, 120) if predictions else (0, 190, 240)
        cv2.putText(
            output,
            str(snapshot["status"])[:90],
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(output, format="bgr24")

    def __del__(self) -> None:
        self._closed.set()
        try:
            self._queue.put_nowait(None)
        except (queue.Full, AttributeError):
            pass


@st.fragment(run_every=1)
def render_analysis_panel(context: Any) -> None:
    """Refresh the analysis card independently while the WebRTC stream stays live."""
    st.markdown('<div class="section-title">Expression analysis</div>', unsafe_allow_html=True)
    st.caption("Predictions update as each background analysis finishes.")
    processor = context.video_processor
    if not context.state.playing or processor is None:
        st.info("Start the camera to see expression labels and scores here.")
        return

    snapshot = processor.get_snapshot()
    predictions = snapshot["predictions"]
    left, right = st.columns(2)
    with left:
        st.metric("Faces detected", len(predictions))
    with right:
        latency = snapshot["latency"]
        st.metric("Last analysis", f"{latency:.2f} s" if latency is not None else "Loading…")
    st.caption(snapshot["status"])

    if not predictions:
        st.markdown(
            '<div class="muted">No result yet. Keep your face centered and use even lighting.</div>',
            unsafe_allow_html=True,
        )
        return

    # The website presents the main face first; follow the same behavior here.
    primary = max(
        predictions,
        key=lambda p: int(p.get("region", {}).get("w", 0)) *
        int(p.get("region", {}).get("h", 0)),
    )
    emotion = str(primary.get("emotion", "unknown"))
    confidence = float(primary.get("scores", {}).get(emotion, 0))
    st.markdown("**TOP PREDICTION**")
    st.markdown(f"### {emotion.title()}  ·  {confidence:.0f}%")
    scores = sorted(
        primary.get("scores", {}).items(), key=lambda item: item[1], reverse=True
    )
    for label, value in scores:
        amount = max(0.0, min(100.0, float(value))) / 100.0
        st.markdown(f"**{label.title()}** <span class='muted'>· {value:.0f}%</span>", unsafe_allow_html=True)
        st.progress(amount)
    st.caption("Model scores are not calibrated probabilities of a person's internal emotional state.")


tab_camera, tab_image, tab_about = st.tabs(
    ["Live Camera", "Image Analysis", "About the Project"]
)

with tab_camera:
    camera_col, analysis_col = st.columns([1.65, 0.85], gap="large")
    with camera_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">Live camera</div>', unsafe_allow_html=True)
            st.caption("Start the camera and allow permission in your browser. Select STOP to end the live feed.")
            try:
                context = webrtc_streamer(
                    key="emotion-webcam-dashboard",
                    video_processor_factory=EmotionVideoProcessor,
                    media_stream_constraints={
                        "video": {
                            "facingMode": "user",
                            "width": {"ideal": 640},
                            "height": {"ideal": 480},
                        },
                        "audio": False,
                    },
                    rtc_configuration=RTCConfiguration({
                        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                    }),
                    async_processing=False,
                )
                if context.state.playing:
                    st.success("Camera running · video remains live while the model analyzes frames in the background.")
                else:
                    st.markdown(
                        '<div class="muted">Your live preview and face labels appear here after you start the camera.</div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                context = None
                st.error(
                    "The webcam could not start. Check browser camera permissions and "
                    "ensure the app is served from localhost or HTTPS."
                )

    with analysis_col:
        with st.container(border=True):
            if context is not None and context.state.playing:
                render_analysis_panel(context)
            else:
                st.markdown('<div class="section-title">Expression analysis</div>', unsafe_allow_html=True)
                st.caption("Live prediction summary")
                st.info("Start the camera to see expression labels and scores.")

with tab_image:
    with st.container(border=True):
        st.markdown('<div class="section-title">Image analysis</div>', unsafe_allow_html=True)
        st.caption("Upload a clear, well-lit image containing one or more visible faces.")
        uploaded = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
            help="Image contents are processed by this app.",
        )

    if uploaded is not None:
        try:
            pil_image = Image.open(uploaded).convert("RGB")
            rgb_image = np.asarray(pil_image)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            preview_col, action_col = st.columns([1.2, 0.8], gap="large")
            with preview_col:
                st.image(pil_image, caption="Uploaded image", use_container_width=True)
            with action_col:
                st.markdown("#### Ready to analyze")
                st.caption(f"Image size: {pil_image.width} × {pil_image.height}")
                if st.button("Analyze Image", type="primary", use_container_width=True):
                    started = time.perf_counter()
                    with st.spinner("Loading the model if needed and analyzing the image…"):
                        predictions = analyze_bgr(bgr_image)
                        elapsed = time.perf_counter() - started
                        st.session_state["image_predictions"] = predictions
                        st.session_state["image_elapsed"] = elapsed
                        st.session_state["image_annotated"] = annotate_image(
                            bgr_image, predictions
                        )
        except Exception as exc:
            st.error(safe_error_message(exc))

    if "image_predictions" in st.session_state:
        predictions = st.session_state["image_predictions"]
        annotated = st.session_state["image_annotated"]
        result_left, result_right = st.columns([1, 1], gap="large")
        with result_left:
            with st.container(border=True):
                st.markdown('<div class="section-title">Detected faces</div>', unsafe_allow_html=True)
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Face boxes and predicted expression categories",
                    use_container_width=True,
                )
        with result_right:
            with st.container(border=True):
                st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
                metric_left, metric_right = st.columns(2)
                metric_left.metric("Faces detected", len(predictions))
                metric_right.metric("Analysis time", f"{st.session_state['image_elapsed']:.2f} s")
                selected = st.selectbox(
                    "Select a detected face",
                    options=list(range(len(predictions))),
                    format_func=lambda index: (
                        f"Face {index + 1}: {predictions[index]['emotion'].title()}"
                    ),
                )
                prediction = predictions[selected]
                emotion = prediction["emotion"]
                confidence = float(prediction["scores"].get(emotion, 0))
                st.markdown(f"### {emotion.title()} · {confidence:.0f}%")
                score_df = pd.DataFrame(
                    [{"Expression": label.title(), "Score (%)": score}
                     for label, score in prediction["scores"].items()]
                ).sort_values("Score (%)", ascending=False)
                st.bar_chart(score_df.set_index("Expression"))
                st.dataframe(score_df, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download selected face scores (CSV)",
                    data=score_df.to_csv(index=False).encode("utf-8"),
                    file_name="emotion_scores.csv",
                    mime="text/csv",
                )

with tab_about:
    about_left, about_right = st.columns(2, gap="large")
    with about_left:
        with st.container(border=True):
            st.markdown('<div class="section-title">Project summary</div>', unsafe_allow_html=True)
            st.markdown(
                """
                **Objective:** Estimate visible facial-expression categories in an image or live video.

                **Approach:** Pre-trained model inference through DeepFace. This application does not train a custom model.

                **Expression labels:** Common outputs include happy, sad, angry, fearful, surprised, disgusted, and neutral.
                """
            )
    with about_right:
        with st.container(border=True):
            st.markdown('<div class="section-title">Privacy and limitations</div>', unsafe_allow_html=True)
            st.markdown(
                """
                - Use images only with appropriate consent.
                - The app does not intentionally record a camera video or save uploaded photographs.
                - Model scores are not calibrated probabilities.
                - Lighting, head pose, occlusion, and camera quality can affect results.
                - Facial expressions are not a reliable or complete measure of internal feelings.
                """
            )

st.markdown(
    """
    <div class="footer-note">
      <span>DeepFace expression model · OpenCV face detection</span>
      <span>Camera processing runs while you choose to keep it on.</span>
    </div>
    """,
    unsafe_allow_html=True,
)
