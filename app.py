"""Streamlit UI for the AI-Based Facial Expression Detection System."""
from __future__ import annotations

import time

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from emotion_engine import analyze_bgr, annotate_image

st.set_page_config(
    page_title="AI-Based Emotion Detection",
    page_icon=None,
    layout="wide",
)

st.title("AI-Based Emotion Detection System")
st.caption("Live camera face boxes and real-time facial-expression estimates, plus optional photo analysis")
st.warning(
    "This is an educational demonstration. It estimates visible facial-expression "
    "categories and does not establish a person's actual internal emotional state. "
    "Do not use results for medical, employment, educational, security, or other "
    "high-impact decisions."
)


def safe_error_message(exc: Exception) -> str:
    if isinstance(exc, (ValueError, RuntimeError)):
        return str(exc)
    return (
        "Analysis could not be completed. Check your internet connection for the "
        "initial model download, verify the installation, and try a clear image."
    )


class EmotionVideoProcessor(VideoProcessorBase):
    """Video processor; only selected frames run expensive model inference."""

    def __init__(self) -> None:
        self.frame_count = 0
        self.predictions: list[dict] = []
        self.status = "Camera started. Waiting for the first analysis."

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 8 == 1:
            try:
                self.predictions = analyze_bgr(image)
                self.status = f"Detected {len(self.predictions)} face(s)."
            except Exception:
                self.predictions = []
                self.status = (
                    "No face found or analysis failed. Improve lighting and face "
                    "position, then wait for the next analysis."
                )

        output = annotate_image(image, self.predictions) if self.predictions else image
        cv2.putText(
            output, self.status[:95], (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (40, 200, 80) if self.predictions else (0, 165, 255),
            2, cv2.LINE_AA
        )
        return av.VideoFrame.from_ndarray(output, format="bgr24")


tab_image, tab_camera, tab_about = st.tabs(
    ["Live Camera", "Image Analysis", "About the Project"]
)

with tab_image:
    st.subheader("Analyze a photograph")
    uploaded = st.file_uploader(
        "Upload an image containing a face",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear, well-lit image. Image contents are processed by this app.",
    )

    if uploaded is not None:
        try:
            pil_image = Image.open(uploaded).convert("RGB")
            rgb_image = np.asarray(pil_image)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            st.image(pil_image, caption="Uploaded image", use_container_width=True)

            if st.button("Analyze Image", type="primary"):
                started = time.perf_counter()
                with st.spinner(
                    "Loading the pre-trained model if needed and analyzing the image..."
                ):
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
        result_left, result_right = st.columns([1, 1])

        with result_left:
            st.subheader("Detected faces")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Face boxes and predicted expression categories",
                use_container_width=True,
            )

        with result_right:
            st.subheader("Results")
            st.metric("Faces detected", len(predictions))
            st.metric("Analysis time", f"{st.session_state['image_elapsed']:.2f} s")

            selected = st.selectbox(
                "Select a detected face",
                options=list(range(len(predictions))),
                format_func=lambda index: (
                    f"Face {index + 1}: {predictions[index]['emotion'].title()}"
                ),
            )
            prediction = predictions[selected]
            st.markdown(f"**Predicted expression:** {prediction['emotion'].title()}")
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

with tab_camera:
    st.subheader("Live camera emotion detection")
    st.write(
        "Start the camera to see a live feed. A rectangle is drawn around each "
        "detected face, with the top expression and score shown above it. "
        "This does not record or save a video; select STOP to end the live feed."
    )
    st.caption(
        "The camera feed stays live. Expression detection runs on selected frames to "
        "reduce CPU load, so labels may update a little slower than the video. "
        "The first prediction may take longer while model weights load."
    )
    try:
        context = webrtc_streamer(
            key="emotion-webcam",
            video_processor_factory=EmotionVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }),
            async_processing=True,
        )
        if context.state.playing:
            st.success("Live camera is running. Face rectangles and expression labels update as frames are analyzed.")
    except Exception:
        st.error(
            "The webcam could not start. Check browser camera permissions and "
            "ensure the app is served from localhost or HTTPS."
        )

with tab_about:
    st.subheader("Project summary")
    st.markdown(
        """
        **Objective:** Estimate visible facial-expression categories in an image
        or live video stream.

        **Approach:** Pre-trained model inference through DeepFace. No custom
        model training is performed by this application.

        **Expression labels:** Common model outputs include happy, sad, angry,
        fearful, surprised, disgusted, and neutral. Available labels depend on
        the installed model implementation.

        **Modules:** User interface, image preprocessing, face/expression inference,
        video frame processing, result visualization, CSV export, and testing.

        **Limitations:** Prediction scores are not necessarily calibrated
        probabilities. Performance can vary with lighting, pose, occlusion,
        camera quality, and demographic differences. Facial expressions are not a
        reliable or complete measure of internal feelings.
        """
    )
    st.subheader("Privacy")
    st.write(
        "Do not upload images of other people without appropriate permission. "
        "This app does not intentionally save uploaded photographs or create a "
        "prediction history. The model may download its model weights on first use."
    )
