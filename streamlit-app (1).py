# streamlit_app.py
import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time
import altair as alt
import traceback

# streamlit-webrtc imports wrapped in try/except (import errors will be shown)
webrtc_available = True
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
except Exception as e:
    webrtc_available = False
    webrtc_import_exception = e
    webrtc_import_traceback = traceback.format_exc()

# -----------------------------
# Sidebar: Page Selection
# -----------------------------
page = st.sidebar.selectbox("Navigate", ["Overview", "App", "Metrics"])

# -----------------------------
# PAGE: OVERVIEW (restored exactly)
# -----------------------------
if page == "Overview":
    # Show DEPI logo from URL
    st.image("https://mohamedhassan2004.github.io/My-Portfolio/assets/imgs/logos/DEPI%20logo.png", width=150)
    st.markdown("---")

    st.markdown("""
# DEPI – Real-Time Object Detection

**Overview**  
This application demonstrates a real-time object detection system designed for traffic scenes. The system identifies road objects such as vehicles, pedestrians, cyclists, and more from images, videos, or live webcam input.  
The goal is to simulate a lightweight perception module that could be used in autonomous driving pipelines.

**Team Members (DEPI – AI & ML Track):-**  
**Team Leader:** George Badea Anwar  
**Team Members:**  
- Mohamed Taha Ahmed Mohamed  
- Hussien Hamdy Hussien Habeeb  
- Philopateer Gawdat Habib Ibrahim  
- Abdelrahman Osama Mohamed Mekhemer

Part of the Digital Egypt Pioneers Initiative (DEPI).

**How to Use**  
- Go to the **App** tab to upload an image, upload a short video, or use your webcam.  
- Adjust confidence threshold in the sidebar to tune detection sensitivity.  
- Use the **Metrics** tab to view example performance numbers.
""")

# -----------------------------
# PAGE: APP
# -----------------------------
elif page == "App":
    # -----------------------------
    # Sidebar: Settings
    # -----------------------------
    st.sidebar.title("Settings")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.1, 1.0, 0.25, 0.05
    )

    upload_option = st.sidebar.radio(
        "Choose Input Type",
        ["Upload Image/Video", "Use Webcam"]
    )

    # -----------------------------
    # Main App UI
    # -----------------------------
    st.title("Real-time Object Detection ")
    st.write(
        "Upload an image or video, or use your webcam to perform real-time object detection "
        "using Ultralytics YOLO."
    )

    # -----------------------------
    # Load YOLO model (fixed single model)
    # -----------------------------
    @st.cache_resource
    def load_yolo():
        return YOLO("Team5.pt")  # fixed model

    model = load_yolo()

    # -----------------------------
    # Helper: Run YOLO on Frames
    # -----------------------------
    def process_frame(frame, conf):
        results = model(frame, conf=conf)
        annotated = results[0].plot()
        return annotated

    # -----------------------------
    # Image / Video Upload Option
    # -----------------------------
    if upload_option == "Upload Image/Video":
        uploaded_file = st.file_uploader(
            "Upload an image or video file",
            type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"]
        )

        if uploaded_file is not None:
            file_type = uploaded_file.type

            # Image Processing
            if file_type.startswith("image"):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                processed = process_frame(img_rgb, confidence_threshold)

                st.subheader("Processed Image")
                st.image(processed, channels="RGB")

            # Video Processing
            elif file_type.startswith("video"):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                vid_path = tfile.name

                st.video(vid_path)
                run = st.button("Run Detection")

                if run:
                    st.subheader("Processing Video...")
                    cap = cv2.VideoCapture(vid_path)
                    frame_container = st.empty()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed = process_frame(frame, confidence_threshold)

                        frame_container.image(processed, channels="RGB")
                        time.sleep(0.02)

                    cap.release()
                    st.success("Video processing completed.")

    # -----------------------------
    # Webcam Option
    # -----------------------------
    elif upload_option == "Use Webcam":
        st.subheader("Webcam Detection (continuous)")

        # If webrtc wasn't importable, show the import error and stop here
        if not webrtc_available:
            st.error("streamlit-webrtc (or its dependencies) failed to import on the server.")
            st.write("Exception:")
            st.code(str(webrtc_import_exception))
            st.write("Full traceback:")
            st.text(webrtc_import_traceback)
            st.info(
                "Common causes: PyAV (`av`) failed to build on the cloud builder or aiortc is missing. "
                "See requirements.txt suggestions in the project and consider deploying to a host that provides prebuilt binary wheels or conda-forge."
            )
            st.stop()

        # RTC configuration (Google STUN as a starter; TURN may be required on some networks)
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        class SimpleProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = model
                self.conf = confidence_threshold

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img_bgr = frame.to_ndarray(format="bgr24")
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                try:
                    results = self.model(img_rgb, conf=self.conf)
                    annotated = results[0].plot()
                    if annotated.ndim == 3:
                        out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    else:
                        out_bgr = annotated
                    return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")
                except Exception as e:
                    # On inference error, draw the message and return original frame
                    cv2.putText(img_bgr, "Inference error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

        # Wrap webrtc_streamer in try/except to capture runtime exceptions and show traceback
        try:
            webrtc_ctx = webrtc_streamer(
                key="yolo-webcam",
                rtc_configuration=rtc_configuration,
                video_processor_factory=SimpleProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                desired_playing_fps=15,
            )
        except Exception as e:
            st.error("webrtc_streamer raised an exception during initialization.")
            st.write("Exception:")
            st.code(str(e))
            st.write("Full traceback:")
            st.text(traceback.format_exc())
            st.info(
                "If this is an ImportError/BuildError for `av` or `aiortc`, inspect build logs in Streamlit Cloud. "
                "If WebRTC initializes but shows a black screen, check browser console for ICE/getUserMedia errors."
            )

# -----------------------------
# PAGE: METRICS (Altair robust implementation)
# -----------------------------
elif page == "Metrics":
    st.title("Model Evaluation — Summary Bar Chart")

    st.markdown("""
    Below are the current summary metrics for the model (single snapshot). Hover each bar for the exact value.
    """)

    metrics = {
        "Precision": 0.891,
        "Recall": 0.854,
        "mAP@50": 0.908,
        "mAP@50-95": 0.703
    }

    df_metrics = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": list(metrics.values())
    })

    selection = alt.selection_single(fields=["Metric"], on="mouseover", empty="none")

    chart = (
        alt.Chart(df_metrics)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", sort=None, title=None),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title="Value"),
            tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Value:Q", format=".3f")],
            color=alt.value("#4C78A8"),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.85))
        )
        .add_selection(selection)
        .properties(height=360)
    )

    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Per-Class mAP (Static)")
    per_class_map = {
        "Car": 0.840,
        "Truck": 0.867,
        "Van": 0.822,
        "Tram": 0.768,
        "Misc": 0.736,
        "Cyclist": 0.640,
        "Pedestrian": 0.531,
        "Person Sitting": 0.528,
    }

    df_map = pd.DataFrame({
        "Class": list(per_class_map.keys()),
        "mAP50-95": list(per_class_map.values())
    })

    st.table(df_map)
