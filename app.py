
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

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #102b46);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div[data-testid="stMetric"] {
    background: #163653;
    padding: 18px;
    border-radius: 12px;
}
.stButton > button {
    background-color: #00bcd4;
    color: #071426;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    width: 100%;
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
        "Upload a steel surface image to identify possible surface defects "
        "using our trained deep learning model."
    )

    uploaded_file = st.file_uploader(
        "Upload a steel surface image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        original = np.array(image)

        # Image preprocessing for preview
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (128, 128))
        equalized = cv2.equalizeHist(gray)
        processed = cv2.GaussianBlur(equalized, (5, 5), 0)

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

            with st.spinner("Analyzing image... Please wait."):

                # Keep preprocessing consistent with model training
                normalized = processed.astype("float32") / 255.0
                rgb = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                rgb = rgb * 255.0

                model_input = tf.keras.applications.mobilenet_v2.preprocess_input(rgb)
                model_input = np.expand_dims(model_input, axis=0)

                model = load_model()
                scores = model.predict(model_input, verbose=0)[0]

                # Ensure scores are normalized for displaying percentages
                scores = np.asarray(scores, dtype=float)
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
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Predicted Defect", DISPLAY_NAMES[defect])

            with result_col2:
                st.metric("Model Confidence", f"{confidence:.2f}%")

            st.subheader("📊 Class-wise Confidence")

            for i, class_name in enumerate(CLASSES):
                percentage = float(scores[i] * 100)
                st.write(f"**{DISPLAY_NAMES[class_name]}:** {percentage:.2f}%")
                st.progress(min(1.0, max(0.0, percentage / 100)))

            # Detailed downloadable report
            report = f"""
STEELVISION AI
Industrial Surface Defect Classification
----------------------------------------

Image: {uploaded_file.name}
Date and Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION RESULT
Predicted Defect: {DISPLAY_NAMES[defect]}
Model: MobileNetV2
Model Confidence: {confidence:.2f}%

CLASS-WISE CONFIDENCE
"""

            for i, class_name in enumerate(CLASSES):
                report += (
                    f"\n{DISPLAY_NAMES[class_name]}: "
                    f"{scores[i] * 100:.2f}%"
                )

            report += """

PREPROCESSING
1. Convert image to grayscale
2. Resize to 128 x 128
3. Apply histogram equalization
4. Apply Gaussian blur
5. Normalize pixel values
6. Resize model input to 224 x 224

Note: This is an AI prediction and should be reviewed
by a qualified person before making industrial decisions.
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
        "Compare the accuracy of the models evaluated in this project."
    )

    model_names = ["Baseline CNN", "Improved CNN", "MobileNetV2"]
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
        color="#00bcd4"
    )

    st.subheader("🧠 Project Summary")

    st.write("""
    - **Baseline CNN:** Initial convolutional neural network model.
    - **Improved CNN:** Updated CNN model with improved accuracy.
    - **MobileNetV2:** Transfer learning model with the highest
      reported test accuracy in this project.
    """)

    st.info(
        "Reported MobileNetV2 test accuracy: 95.56% on 360 test images. "
        "Accuracy comparisons are meaningful when models use the same "
        "test dataset and evaluation method."
    )

    st.subheader("🔬 Main Processing Steps")

    st.write("""
    1. Image upload
    2. Grayscale conversion
    3. Histogram equalization
    4. Gaussian blur
    5. Image normalization
    6. MobileNetV2 prediction
    7. Defect classification and confidence display
    """)
