# streamlit_app.py
import streamlit as st
import tempfile
import numpy as np
import pandas as pd
import time
import traceback

# Optional imports that may fail in some environments; fail gracefully
try:
    import cv2
except Exception as e:
    cv2 = None
    # we still proceed; camera_input and imageio workflows do not strictly need cv2 to import
    st = st  # keep linter happy

from ultralytics import YOLO  # ultralytics YOLO python API
import imageio

# streamlit-webrtc (for continuous browser -> python streaming)
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# -----------------------------
# Utility: Safe log + import check
# -----------------------------
def check_cv2():
    if cv2 is None:
        st.error(
            "OpenCV failed to import in this environment. "
            "If you're on Streamlit Cloud, ensure `opencv-python-headless` is listed in requirements.txt "
            "and consider adding `libgl1` to packages.txt if logs show missing libGL."
        )
        st.stop()

# -----------------------------
# Sidebar: Page Selection
# -----------------------------
page = st.sidebar.selectbox("Navigate", ["Overview", "App", "Live (WebRTC)", "Metrics"])

# -----------------------------
# PAGE: OVERVIEW
# -----------------------------
if page == "Overview":
    st.image("https://mohamedhassan2004.github.io/My-Portfolio/assets/imgs/logos/DEPI%20logo.png", width=150)
    st.markdown("---")
    st.markdown("""
# DEPI – Real-Time Object Detection

**Overview**  
This application demonstrates a real-time object detection system designed for traffic scenes. The system identifies road objects such as vehicles, pedestrians, cyclists, and more from images, videos, or live webcam input.

**How to use**  
- **App**: Upload an image or video, or use the browser camera for snapshots.
- **Live (WebRTC)**: Use for continuous live preview (browser streaming -> Python). This requires `streamlit-webrtc`.
    """)

# -----------------------------
# Helper: Model Loading
# -----------------------------
@st.cache_resource
def load_yolo_model(model_path: str = "yolov8n.pt"):
    """Load the Ultralytics YOLO model once per process."""
    model = YOLO(model_path)
    return model

# Expose a small default model name to use for live preview; keep a fallback if user has Team5.pt
DEFAULT_MODEL = "yolov8n.pt"  # change to "Team5.pt" if you upload your custom model into repo

# -----------------------------
# PAGE: APP (uploads + camera snapshots)
# -----------------------------
elif page == "App":
    st.sidebar.title("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    resize_w = st.sidebar.selectbox("Inference Width", [320, 480, 640, 768, 1024], index=2)
    skip_frames = st.sidebar.slider("Frame-skip (process every Nth frame)", 1, 10, 2)
    upload_option = st.sidebar.radio("Choose Input Type", ["Upload Image/Video", "Use Browser Camera"])

    st.title("Real-time Object Detection (App)")
    st.write("Upload an image/video or use the browser camera for snapshots.")

    # Load model (small default — replace with your model filename if committed)
    model = load_yolo_model(DEFAULT_MODEL)

    # Helper: process one frame (numpy RGB)
    def process_frame(frame_rgb: np.ndarray, conf: float):
        """
        frame_rgb: HxWx3 numpy array in RGB (uint8)
        returns: annotated RGB image (numpy uint8)
        """
        # Ultralytics accepts numpy images directly
        results = model(frame_rgb, conf=conf)
        annotated = results[0].plot()  # returns RGB
        return annotated

    # Upload path
    if upload_option == "Upload Image/Video":
        uploaded_file = st.file_uploader("Upload an image or video", type=["jpg","jpeg","png","mp4","mov","avi","mkv"])

        if uploaded_file is not None:
            file_type = uploaded_file.type
            if file_type.startswith("image"):
                # IMAGE flow
                if cv2 is None:
                    check_cv2()  # fail with friendly message if cv2 is required and absent
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # Resize for inference (preserve aspect via width)
                h, w = img_rgb.shape[:2]
                new_h = int(h * (resize_w / w))
                img_small = cv2.resize(img_rgb, (resize_w, new_h))
                processed = process_frame(img_small, confidence_threshold)
                st.subheader("Processed Image")
                st.image(processed, channels="RGB", use_column_width=True)

            elif file_type.startswith("video"):
                # VIDEO flow (server-side): write to temp file and iterate frames with imageio
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split(".")[-1])
                tfile.write(uploaded_file.read())
                vid_path = tfile.name

                st.video(vid_path)
                run = st.button("Run Detection on Video")
                if run:
                    st.subheader("Processing Video...")
                    frame_container = st.empty()
                    reader = imageio.get_reader(vid_path)
                    frame_idx = 0
                    for frame in reader:
                        frame_idx += 1
                        if frame_idx % skip_frames != 0:
                            # show previous annotated frame while skipping heavy inference
                            time.sleep(0.01)
                            continue
                        # imageio frames are RGB by default
                        h, w = frame.shape[:2]
                        new_h = int(h * (resize_w / w))
                        small = imageio.core.util.Array(frame)
                        # Use numpy resizing via cv2 if available (faster)
                        if cv2 is not None:
                            small = cv2.resize(frame, (resize_w, new_h))
                        processed = process_frame(small, confidence_threshold)
                        frame_container.image(processed, channels="RGB")
                        time.sleep(0.01)  # small sleep to yield UI
                    reader.close()
                    st.success("Video processing completed.")

    # Browser camera (single-shot) flow
    elif upload_option == "Use Browser Camera":
        st.subheader("Browser Camera (snapshot)")
        st.markdown("Use the camera widget to take a snapshot (works on Streamlit Cloud and local).")
        cam_img = st.camera_input("Take a picture")
        if cam_img is not None:
            # camera_input returns bytes; decode via numpy + cv2 if available
            file_bytes = cam_img.read()
            arr = np.frombuffer(file_bytes, np.uint8)
            if cv2 is None:
                # If cv2 import failed, try PIL route (less common)
                from PIL import Image
                img = Image.open(cam_img)
                img_rgb = np.array(img.convert("RGB"))
            else:
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            h, w = img_rgb.shape[:2]
            new_h = int(h * (resize_w / w))
            if cv2 is not None:
                img_small = cv2.resize(img_rgb, (resize_w, new_h))
            else:
                import numpy as _np
                img_small = _np.array(Image.fromarray(img_rgb).resize((resize_w, new_h)))

            processed = process_frame(img_small, confidence_threshold)
            st.image(processed, channels="RGB")

# -----------------------------
# PAGE: Live (WebRTC) - continuous preview with streamlit-webrtc
# -----------------------------
elif page == "Live (WebRTC)":
    st.title("Live Preview (streamlit-webrtc)")
    st.markdown("""
    This mode uses `streamlit-webrtc` to stream the browser camera to the Python backend for continuous processing.
    It works in the browser and provides lower-latency continuous preview compared with server-side cv2.VideoCapture.
    """)

    # Provide settings for live mode
    live_conf = st.sidebar.slider("Confidence Threshold (live)", 0.1, 1.0, 0.25, 0.05)
    live_resize_w = st.sidebar.selectbox("Live inference width", [320, 480, 640], index=1)
    live_skip = st.sidebar.slider("Live frame-skip (process every Nth frame)", 1, 6, 2)
    use_custom = st.sidebar.checkbox("Use custom model file in repo (Team5.pt)", value=False)
    live_model_path = "Team5.pt" if use_custom else DEFAULT_MODEL

    st.info("Press the Start button in the video frame to begin streaming. Allow camera permissions in your browser.")

    # Transformer class used by streamlit-webrtc
    class YoloTransformer(VideoTransformerBase):
        def __init__(self):
            # load model once per transformer worker
            try:
                self.model = YOLO(live_model_path)
            except Exception:
                # fallback to default model if load fails
                self.model = YOLO(DEFAULT_MODEL)
            self.i = 0
            self.last_frame = None

        def transform(self, frame):
            """
            frame: av.VideoFrame from webrtc -> convert to ndarray
            Return: bgr24 ndarray
            """
            self.i += 1
            img_bgr = frame.to_ndarray(format="bgr24")
            if self.i % live_skip != 0 and self.last_frame is not None:
                return self.last_frame

            # convert to RGB, resize for inference
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if cv2 is not None else img_bgr[:, :, ::-1]
            h, w = img_rgb.shape[:2]
            new_h = int(h * (live_resize_w / w))
            if cv2 is not None:
                small = cv2.resize(img_rgb, (live_resize_w, new_h))
            else:
                from PIL import Image
                small = np.array(Image.fromarray(img_rgb).resize((live_resize_w, new_h)))

            # inference
            try:
                results = self.model(small, conf=live_conf)
                annotated = results[0].plot()  # RGB
                out_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            except Exception:
                # on any inference error, return original frame
                out_bgr = img_bgr

            self.last_frame = out_bgr
            return out_bgr

    # Start the webrtc streamer - displays start/stop control in the UI
    webrtc_ctx = webrtc_streamer(
        key="yolo-live",
        video_transformer_factory=YoloTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=None,
        async_transform=True,
        video_frame_callback=None,
        in_recorder=False
    )

# -----------------------------
# PAGE: METRICS
# -----------------------------
elif page == "Metrics":
    st.title("Model Evaluation Over Time")
    timesteps = list(range(1, 11))
    metric_history = pd.DataFrame({
        "Time Step": timesteps,
        "Precision": [0.90, 0.91, 0.92, 0.93, 0.934, 0.935, 0.936, 0.937, 0.938, 0.934],
        "Recall": [0.85, 0.86, 0.865, 0.867, 0.867, 0.868, 0.869, 0.870, 0.867, 0.867],
        "mAP@50": [0.91, 0.915, 0.920, 0.925, 0.928, 0.929, 0.930, 0.931, 0.928, 0.928],
        "mAP@50-95": [0.70, 0.705, 0.71, 0.715, 0.717, 0.718, 0.716, 0.717, 0.717, 0.717],
    }).set_index("Time Step")
    st.line_chart(metric_history)

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
