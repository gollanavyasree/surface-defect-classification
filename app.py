
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

# ---------------- PAGE SETTINGS ----------------
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

# ---------------- DESIGN AND COLORS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #102b46);
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #0b1b30 !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background-color: #183653;
    border-radius: 8px;
    padding: 8px;
    margin-bottom: 6px;
}

.stApp h1, .stApp h2, .stApp h3,
.stApp p, .stApp label {
    color: #ffffff;
}

div[data-testid="stMetric"] {
    background-color: #163653;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #285477;
}

.stButton > button,
.stDownloadButton > button {
    background-color: #00c8e0;
    color: #071426 !important;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    width: 100%;
}

[data-testid="stFileUploader"] {
    background-color: #102b46;
    border-radius: 10px;
    padding: 12px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔬 SteelVision AI")
st.sidebar.write("Industrial Surface Defect Classification")

page = st.sidebar.radio(
    "Choose a page",
    ["Defect Prediction", "Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.write("Dataset: NEU Surface Defect Dataset")
st.sidebar.write("Model: MobileNetV2")

# ==================================================
# PAGE 1: DEFECT PREDICTION
# ==================================================
if page == "Defect Prediction":

    st.title("🔬 Industrial Surface Defect Classification")

    st.write(
        "Upload a steel surface image to identify possible defects "
        "using our trained deep learning model."
    )

    uploaded_file = st.file_uploader(
        "Upload a steel surface image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        original = np.array(image)

        # Image preprocessing
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))
        equalized = cv2.equalizeHist(gray)
        processed = cv2.GaussianBlur(equalized, (5, 5), 0)

        # Original and processed image previews
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("⚙️ Processed Image")
            st.image(
                processed,
                caption="Grayscale + Histogram Equalization + Gaussian Blur",
                use_container_width=True
            )

        st.markdown("---")

        if st.button("🔍 Predict Surface Defect"):

            with st.spinner("Analyzing image..."):

                normalized = processed.astype("float32") / 255.0

                rgb = cv2.cvtColor(
                    normalized,
                    cv2.COLOR_GRAY2RGB
                )

                rgb = cv2.resize(rgb, (224, 224))
                rgb = rgb * 255.0

                model_input = (
                    tf.keras.applications.mobilenet_v2
                    .preprocess_input(rgb)
                )

                model_input = np.expand_dims(model_input, axis=0)

                model = load_model()
                scores = model.predict(model_input, verbose=0)[0]

                scores = np.asarray(scores, dtype=float)

                # Convert outputs to probabilities if necessary
                if (
                    np.any(scores < 0)
                    or np.any(scores > 1)
                    or not np.isclose(np.sum(scores), 1.0, atol=0.05)
                ):
                    scores = tf.nn.softmax(scores).numpy()

                index = int(np.argmax(scores))
                defect = CLASSES[index]
                confidence = float(scores[index] * 100)

            st.success("Analysis completed!")

            st.subheader("🎯 Prediction Result")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Defect",
                    DISPLAY_NAMES[defect]
                )

            with col2:
                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            st.subheader("📊 Class-wise Confidence")

            for i, class_name in enumerate(CLASSES):
                percentage = float(scores[i] * 100)

                st.write(
                    f"**{DISPLAY_NAMES[class_name]}:** "
                    f"{percentage:.2f}%"
                )

                st.progress(
                    min(1.0, max(0.0, percentage / 100))
                )

            # Detailed downloadable report
            report = f"""
STEELVISION AI
Industrial Surface Defect Classification
=========================================

Image: {uploaded_file.name}
Date and Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION RESULT
-----------------
Predicted Defect: {DISPLAY_NAMES[defect]}
Model: MobileNetV2
Model Confidence: {confidence:.2f}%

CLASS-WISE CONFIDENCE
---------------------
"""

            for i, class_name in enumerate(CLASSES):
                report += (
                    f"{DISPLAY_NAMES[class_name]}: "
                    f"{scores[i] * 100:.2f}%\n"
                )

            report += """
PREPROCESSING STEPS
-------------------
1. Convert image to grayscale
2. Resize to 128 x 128
3. Apply histogram equalization
4. Apply Gaussian blur
5. Normalize pixel values
6. Resize model input to 224 x 224
7. Apply MobileNetV2 preprocessing

Note: This AI prediction should be reviewed by a
qualified person before making industrial decisions.
"""

            st.download_button(
                label="⬇️ Download Detailed Report",
                data=report,
                file_name="surface_defect_report.txt",
                mime="text/plain"
            )

    else:
        st.info("Upload a JPG, JPEG, or PNG image to get started.")

# ==================================================
# PAGE 2: MODEL PERFORMANCE
# ==================================================
else:

    st.title("📊 Model Performance Comparison")

    st.write(
        "Compare the reported test accuracy of the three models "
        "evaluated in this project."
    )

    model_names = [
        "Baseline CNN",
        "Improved CNN",
        "MobileNetV2"
    ]

    accuracies = [66.94, 69.44, 95.56]

    col1, col2, col3 = st.columns(3)

    col1.metric("Baseline CNN", "66.94%")
    col2.metric("Improved CNN", "69.44%")
    col3.metric("MobileNetV2", "95.56%")

    st.subheader("📈 Accuracy Comparison")

    chart_data = {
        "Model": model_names,
        "Accuracy (%)": accuracies
    }

    st.bar_chart(
        chart_data,
        x="Model",
        y="Accuracy (%)",
        color="#00c8e0"
    )

    st.subheader("🧠 Project Summary")

    st.markdown("""
    **Baseline CNN**
    
    The initial convolutional neural network used as a baseline.

    **Improved CNN**
    
    The updated CNN model with improved reported accuracy.

    **MobileNetV2**
    
    A transfer learning model that achieved the highest reported
    test accuracy among these three models.
    """)

    st.info(
        "Reported MobileNetV2 test accuracy: 95.56% on 360 test images. "
        "For a fair comparison, all models should be evaluated using "
        "the same test dataset and evaluation method."
    )

    st.subheader("🔬 Image Processing Pipeline")

    st.markdown("""
    1. Upload the steel surface image.
    2. Convert the image to grayscale.
    3. Resize to 128 × 128 pixels.
    4. Apply histogram equalization.
    5. Apply Gaussian blur.
    6. Normalize and prepare the image for MobileNetV2.
    7. Predict the defect class.
    8. Display confidence scores and generate a report.
    """)
