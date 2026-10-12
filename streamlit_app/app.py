import streamlit as st
import pandas as pd
import plotly.express as px
import json
import av
from pathlib import Path

from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Welding Defect Detection",
    page_icon="🏭",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
@st.cache_resource
def load_model():
    return YOLO(str(ROOT_DIR / "models" / "best.pt"))

model = load_model()

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#00E5FF;
}

.sub-title{
    text-align:center;
    font-size:18px;
    color:white;
}

.metric-card{
    padding:20px;
    border-radius:12px;
    background:#111827;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<div class='main-title'>🏭 Welding Defect Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>YOLOv11 Industrial Inspection Dashboard</div>",
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙ Model Information")

    st.write("Model : YOLOv11n")
    st.write("Framework : PyTorch")
    st.write("Frontend : Streamlit")
    st.write("Computer Vision : OpenCV")
    st.write("Use Case : Welding Quality Inspection")

# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Welding Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file:

    image = Image.open(uploaded_file)

    results = model(image)

    annotated_image = results[0].plot()

    detections = []

    total_defects = 0

    max_confidence = 0

    DEFECT_CLASSES = ["Defect", "Bad Weld"]

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        confidence = float(box.conf[0])

        max_confidence = max(
            max_confidence,
            confidence
        )

        bbox = box.xyxy[0].tolist()

# Count only defect classes
        if (
            class_name in DEFECT_CLASSES
            and confidence >= 0.20
            ):
            total_defects += 1

            detections.append({
                "Class": class_name,
                "Confidence (%)":
                        round(confidence * 100, 2),
                "Bounding Box": bbox
            })

    # =================================================
    # IMAGES
    # =================================================

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("Prediction Result")
        st.image(
            annotated_image,
            use_container_width=True
        )

    st.divider()

    # =================================================
    # METRICS
    # =================================================

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
        "Total Defects",
        total_defects
    )

    with m2:
        st.metric(
            "Highest Confidence",
            f"{round(max_confidence*100,2)}%"
        )

    with m3:

        if len(detections) == 0 or detections[0]["Class"] == "Good Weld" and detections[0]["Confidence (%)"] >= 25:

            st.metric(
                "Inspection Status",
                "PASS ✅"
        )

        else:

            st.metric(
            "Inspection Status",
            "DEFECT ⚠"
            )

    st.divider()

    # =================================================
    # TABLE
    # =================================================

    st.subheader("📋 Detection Details")

    if len(detections) > 0:

        df = pd.DataFrame(detections)

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.success(
            "No welding defects detected."
        )

    # =================================================
    # BAR CHART
    # =================================================

    if len(detections) > 0:

        st.subheader("📊 Confidence Scores")

        chart_df = pd.DataFrame(detections)

        fig = px.bar(
            chart_df,
            x="Class",
            y="Confidence (%)",
            color="Class",
            text="Confidence (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =================================================
    # JSON OUTPUT
    # =================================================

    output_data = {

    "total_defects":
        total_defects,

    "highest_confidence":
        round(max_confidence*100,2),

    "detections":
        detections
}

    st.divider()

    st.subheader("📦 Prediction JSON")

    st.json(output_data)

    # =================================================
    # DOWNLOAD REPORT
    # =================================================

    st.download_button(
        label="⬇ Download JSON Report",
        data=json.dumps(
            output_data,
            indent=4
        ),
        file_name="welding_report.json",
        mime="application/json"
    )

# =====================================================
# WEBCAM SECTION
# =====================================================

st.divider()

st.subheader("📹 Live Webcam Detection")

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        results = model(img)

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24"
        )

webrtc_streamer(
    key="welding-webcam",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)