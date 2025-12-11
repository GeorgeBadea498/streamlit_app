import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
import time

# -----------------------------
# Sidebar: Page Selection
# -----------------------------
page = st.sidebar.selectbox("Navigate", ["Overview", "App", "Metrics"])

# -----------------------------
# PAGE: OVERVIEW
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

**Team Members (DEPI – AI & Data Science Track)**  
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
        st.subheader("Webcam Detection")
        run_webcam = st.checkbox("Start Webcam")

        FRAME_WINDOW = st.image([])

        if run_webcam:
            cap = cv2.VideoCapture(0)

            while cap.isOpened() and run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Could not access webcam.")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = process_frame(frame, confidence_threshold)
                FRAME_WINDOW.image(processed, channels="RGB")
                time.sleep(0.03)

            cap.release()

# -----------------------------
# PAGE: METRICS
# -----------------------------
elif page == "Metrics":
    st.title("Model Evaluation Over Time")

    st.markdown("""
    This section shows a time series visualization of model metrics (simulated updates). 
    In a real scenario, these could be collected from batch evaluations or live performance logs.
    """)

    # Simulate metrics over time
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
