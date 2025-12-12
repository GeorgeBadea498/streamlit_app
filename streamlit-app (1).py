# app_streamlit_webrtc.py
import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import pandas as pd
import altair as alt

# -----------------------------
# Sidebar: Page Selection
# -----------------------------
page = st.sidebar.selectbox("Navigate", ["Overview", "App", "Metrics"])

# ----------------------------- Overview (same as your original) -----------------------------
if page == "Overview":
    st.image("https://mohamedhassan2004.github.io/My-Portfolio/assets/imgs/logos/DEPI%20logo.png", width=150)
    st.markdown("---")
    st.markdown("""
# DEPI – Real-Time Object Detection
... (overview text unchanged; keep your original content) ...
""")

# ----------------------------- APP -----------------------------
elif page == "App":
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    upload_option = st.sidebar.radio("Choose Input Type", ["Upload Image/Video", "Use Webcam"])

    st.title("Real-time Object Detection")
    st.write("Upload an image or video, or use your webcam to perform real-time object detection using Ultralytics YOLO.")

    # Load YOLO model once per session (cached)
    @st.cache_resource
    def load_yolo(path="Team5.pt"):
        # This will be cached by Streamlit and reused across reruns
        model = YOLO(path)
        return model

    model = load_yolo()  # cached load

    def process_frame_numpy(frame_rgb, conf):
        """
        Run YOLO on a single RGB numpy frame and return an annotated RGB image (numpy).
        frame_rgb: HxWx3 RGB numpy array
        """
        # Run model (Ultralytics accepts numpy arrays)
        results = model(frame_rgb, conf=conf)
        # results[0].plot() returns an annotated image (RGB)
        annotated = results[0].plot()
        # Ensure numpy array and dtype
        if isinstance(annotated, np.ndarray):
            return annotated
        else:
            # Fallback: convert to numpy if necessary
            return np.array(annotated)

    # IMAGE / VIDEO upload handling (same as your original, unchanged) --------------------------------
    if upload_option == "Upload Image/Video":
        uploaded_file = st.file_uploader("Upload an image or video file", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])
        if uploaded_file is not None:
            file_type = uploaded_file.type
            if file_type.startswith("image"):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed = process_frame_numpy(img_rgb, confidence_threshold)
                st.subheader("Processed Image")
                st.image(processed, channels="RGB")
            elif file_type.startswith("video"):
                tfile = None
                try:
                    import tempfile
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
                            processed = process_frame_numpy(frame, confidence_threshold)
                            frame_container.image(processed, channels="RGB")
                            time.sleep(0.02)
                        cap.release()
                        st.success("Video processing completed.")
                finally:
                    if tfile is not None:
                        tfile.close()

    # WEBCAM using streamlit-webrtc (continuous)
    elif upload_option == "Use Webcam":
        st.subheader("Webcam Detection (continuous)")

        # RTC configuration (use Google STUN by default for better NAT traversal)
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        # Processor class: perform YOLO inference in recv()
        class YoloProcessor(VideoProcessorBase):
            def __init__(self):
                # Access cached model
                # NOTE: it's safe to refer to the cached model object here
                self.model = model
                # local copy of confidence threshold (will be refreshed in loop below)
                self.conf = confidence_threshold

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                # Convert frame to ndarray (BGR)
                img_bgr = frame.to_ndarray(format="bgr24")
                # Convert to RGB for the model / consistent plotting
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Run model inference
                try:
                    results = self.model(img_rgb, conf=self.conf)
                    annotated = results[0].plot()  # typically returns RGB ndarray
                    # If annotated is RGB, convert back to BGR for WebRTC (bgr24 expected)
                    if annotated.ndim == 3:
                        out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                    else:
                        out_bgr = annotated
                    return av.VideoFrame.from_ndarray(out_bgr, format="bgr24")
                except Exception as e:
                    # On any inference error, return the original frame to avoid breaking the stream
                    # (you can also draw the error text on the frame)
                    img = img_bgr
                    cv2.putText(img, f"Inference error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Launch the WebRTC streamer. video_processor_factory will create an instance of YoloProcessor.
        webrtc_ctx = webrtc_streamer(
            key="yolo-webcam",
            rtc_configuration=rtc_configuration,
            video_processor_factory=YoloProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            desired_playing_fps=15,
        )

        # Allow runtime updates of confidence slider: write the latest value into the processor if it exists.
        if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
            # update the processor's confidence threshold to the current slider value
            webrtc_ctx.video_processor.conf = confidence_threshold
            st.sidebar.success("Webcam streaming: running")
        else:
            st.sidebar.info("Webcam not running yet. Allow camera permissions and click 'Start' in the stream.")

# ----------------------------- METRICS (unchanged) -----------------------------
elif page == "Metrics":
    st.title("Model Evaluation — Summary Bar Chart")
    st.markdown("Below are the current summary metrics for the model (single snapshot). Hover each bar for the exact value.")
    metrics = {"Precision": 0.891, "Recall": 0.854, "mAP@50": 0.908, "mAP@50-95": 0.703}
    df_metrics = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    selection = alt.selection_single(fields=["Metric"], on="mouseover", empty="none")
    chart = (
        alt.Chart(df_metrics)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", sort=None, title=None),
            y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1]), title="Value"),
            tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Value:Q", format=".3f")],
            color=alt.value("#4C78A8"),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.85)),
        )
        .add_selection(selection)
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("### Per-Class mAP (Static)")
    per_class_map = {"Car": 0.840, "Truck": 0.867, "Van": 0.822, "Tram": 0.768, "Misc": 0.736, "Cyclist": 0.640, "Pedestrian": 0.531, "Person Sitting": 0.528}
    df_map = pd.DataFrame({"Class": list(per_class_map.keys()), "mAP50-95": list(per_class_map.values())})
    st.table(df_map)
