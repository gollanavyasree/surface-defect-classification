
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from datetime import datetime

st.set_page_config(
    page_title="SteelVision AI",
    page_icon="🔬",
    layout="wide"
)

MODEL_PATH = "MobileNetV2_Surface_Defect_Model (2).keras"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "scratches",
    "pitted_surface",
    "rolled-in_scale"
]

DISPLAY_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "scratches": "Scratches",
    "pitted_surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale"
}

COLORS = [
    "#38BDF8", "#FBBF24", "#34D399",
    "#A78BFA", "#FB7185", "#22D3EE"
]

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071326, #10284b);
    color: #F1F5F9;
}
h1, h2, h3, p, label {
    color: #F1F5F9 !important;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(110deg, #263c91, #132d53, #432477);
    border: 1px solid #4365b0;
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #D7E5FF !important;
    margin-bottom: 0;
}
.card {
    background: #102747;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #28518a;
    margin-bottom: 16px;
}
.result-card {
    background: linear-gradient(110deg, #087F5B, #065F73);
    border: 1px solid #34D399;
    border-radius: 16px;
    padding: 22px;
    margin: 12px 0 20px 0;
}
.result-label {
    color: #D1FAE5 !important;
    font-size: 15px;
}
.result-value {
    color: white !important;
    font-size: 29px;
    font-weight: 750;
}
.small-note {
    color: #B9CBE5 !important;
    font-size: 13px;
}
div.stButton > button {
    background: linear-gradient(90deg, #8B5CF6, #06B6D4);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-weight: 700;
    min-height: 48px;
}
div.stButton > button:hover {
    border: 1px solid white;
    color: white;
}
div[data-testid="stFileUploader"] {
    background: #102747;
    padding: 14px;
    border-radius: 14px;
    border: 1px dashed #6388C5;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
# Sidebar Menu
st.sidebar.title("🔬 SteelVision AI")

page = st.sidebar.radio(
    "Choose a page",
    ["Defect Prediction", "Model Performance"]
)
st.markdown("""
<div class="hero">
    <h1>🔬 Industrial Surface Defect Classification</h1>
    <p>SteelVision AI · Intelligent inspection powered by MobileNetV2</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📤 Upload Steel Surface Image")
st.write("Choose a clear JPG, JPEG or PNG image to identify its defect.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, use_container_width=True)
        st.markdown(
            f'<p class="small-note">{uploaded_file.name}</p>',
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        predict_clicked = st.button(
            "✨ Predict Defect",
            use_container_width=True
        )

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🧠 Prediction Results")

        if predict_clicked:
            with st.spinner("Analyzing the steel surface..."):
                img = np.array(image)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                gray = cv2.resize(gray, (128, 128))
                gray = cv2.equalizeHist(gray)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                gray = gray.astype("float32") / 255.0

                rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                rgb = rgb * 255.0
                rgb = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)
                rgb = np.expand_dims(rgb, axis=0)

                model = load_model()
                scores = model.predict(rgb, verbose=0)[0]

                # Display scores as percentages.
                # This assumes the model output is six class probabilities.
                scores = np.asarray(scores, dtype=float)
                if len(scores) != len(CLASSES):
                    st.error("The model output does not match the six classes.")
                    st.stop()

                scores_percent = scores * 100
                index = int(np.argmax(scores))
                defect = DISPLAY_NAMES[CLASSES[index]]
                confidence = float(scores_percent[index])

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-label">✅ Predicted Defect</div>
                    <div class="result-value">{defect}</div>
                    <br>
                    <div class="result-label">📊 Model Confidence</div>
                    <div class="result-value">{confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### 🎯 Confidence by Class")

                for i, name in enumerate(CLASSES):
                    value = float(scores_percent[i])
                    st.markdown(
                        f"**{DISPLAY_NAMES[name]}** — {value:.2f}%"
                    )
                    st.progress(max(0.0, min(value / 100.0, 1.0)))

                report = (
                    "Industrial Surface Defect Classification\n"
                    f"Predicted Defect: {defect}\n"
                    f"Model Confidence: {confidence:.2f}%\n"
                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )

                st.download_button(
                    "📥 Download Prediction Report",
                    data=report,
                    file_name="defect_prediction_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

                st.caption(
                    "Confidence is the model's output score, "
                    "not a guarantee that the prediction is correct."
                )
        else:
            st.info(
                "Upload an image, then click **Predict Defect** "
                "to view the results and confidence chart."
            )

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="card">
        <h3>✨ What this app can identify</h3>
        <p>🔹 Crazing &nbsp; 🔹 Inclusion &nbsp; 🔹 Patches</p>
        <p>🔹 Scratches &nbsp; 🔹 Pitted Surface</p>
        <p>🔹 Rolled-in Scale</p>
        <p class="small-note">
            Upload a steel surface image to begin the analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#B9CBE5'>"
    "SteelVision AI · Industrial Surface Inspection · MobileNetV2"
    "</p>",
    unsafe_allow_html=True
)
