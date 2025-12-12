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

# Try to import streamlit-webrtc and av. If this fails we'll use a camera_input fallback.
webrtc_available = True
webrtc_import_exception = None
webrtc_import_traceback = None
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
        # Change path if necessary; ensure Team5.pt is in your repo or adjust to a public path
        return YOLO("Team5.pt")

    model = load_yolo()

    # -----------------------------
    # Helper: Run YOLO on Frames
    # -----------------------------
    def process_frame(frame_rgb, conf):
        """
        frame_rgb: HxWx3 RGB numpy array
        returns: annotated RGB numpy array
        """
        results = model(frame_rgb, conf=conf)
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

        # If webrtc import failed earlier, show error and automatically use fallback below.
        if not webrtc_available:
            st.error("streamlit-webrtc (or its dependencies) failed to import on the server; using camera_input fallback.")
            st.write("Import exception:")
            st.code(str(webrtc_import_exception))
            st.write("Full traceback:")
            st.text(webrtc_import_traceback)

            # FALLBACK: camera_input snapshot loop (reliable on Streamlit Cloud)
            st.info("Fallback: Please allow camera access in the browser. The app will capture snapshots and process them.")
            file = st.camera_input("Allow camera and take a snapshot")

            frame_placeholder = st.empty()
            if file is not None:
                # Read bytes into numpy and process
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                processed = process_frame(rgb, confidence_threshold)
                frame_placeholder.image(processed, channels="RGB")
            else:
                st.info("Waiting for camera permission / initial snapshot. Click the camera widget to take a photo.")

        else:
            # RTC configuration (use Google STUN server for NAT traversal; add TURN if required)
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )

            # Processor class: internal frame-skipping to control inference rate without depending on
            # webrtc_streamer optional parameters that may not exist in all versions.
            class YoloProcessor(VideoProcessorBase):
                def __init__(self):
                    self.model = model
                    self.conf = confidence_threshold
                    self.frame_count = 0
                    # process 1 of every N frames to reduce CPU load / inference rate
                    # tune this value depending on model size / CPU power
                    self.process_every_n = 2  # change to 3 or 4 to reduce load further

                def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                    self.frame_count += 1
                    img_bgr = frame.to_ndarray(format="bgr24")

                    # Skip heavy inference on most frames: return original frame unchanged.
                    if (self.frame_count % self.process_every_n) != 0:
                        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

                    # Do inference on selected frames
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    try:
                        results = self.model(img_rgb, conf=self.conf)
                        annotated = results[0].plot()
                        # convert annotated (RGB) back to BGR for returning
                        out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                        return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")
                    except Exception as e:
                        # On inference error, overlay message and return original frame
                        cv2.putText(img_bgr, "Inference error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

            # Initialize the stream. Note: we do NOT pass 'desired_playing_fps' to avoid
            # API mismatch errors on older/newer streamlit-webrtc versions.
            try:
                webrtc_ctx = webrtc_streamer(
                    key="yolo-webcam",
                    rtc_configuration=rtc_configuration,
                    video_processor_factory=YoloProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
            except Exception as e:
                # Show full initialization traceback in the UI for diagnosis
                st.error("webrtc_streamer raised an exception during initialization. Falling back to camera_input.")
                st.write("Exception:")
                st.code(str(e))
                st.write("Full traceback:")
                st.text(traceback.format_exc())

                # FALLBACK to camera_input (same as above)
                st.info("Fallback: Please allow camera access in the browser. The app will capture snapshots and process them.")
                file = st.camera_input("Allow camera and take a snapshot")

                frame_placeholder = st.empty()
                if file is not None:
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    processed = process_frame(rgb, confidence_threshold)
                    frame_placeholder.image(processed, channels="RGB")
                else:
                    st.info("Waiting for camera permission / initial snapshot. Click the camera widget to take a photo.")
            else:
                # keep the processor's confidence synced with the sidebar slider while running
                if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                    webrtc_ctx.video_processor.conf = confidence_threshold
                    st.sidebar.success("Webcam streaming: running")
                else:
                    st.sidebar.info("Webcam not running yet. Allow camera permissions and click 'Start' in the stream.")

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
