
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Steel Defect Detection",
    page_icon="🔍",
    layout="centered"
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

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

st.title("Industrial Surface Defect Classification")
st.caption("AI-powered steel surface inspection using MobileNetV2")

st.markdown(
    "Upload a steel surface image to identify its defect."
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload steel surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict Defect", type="primary",
                 use_container_width=True):
        with st.spinner("Analyzing image..."):
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
            predictions = model.predict(rgb, verbose=0)

            index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][index]) * 100

        st.divider()
        st.subheader("Prediction Result")

        st.markdown(f"**Predicted Defect:** {CLASSES[index]}")
        st.markdown(f"**Model Confidence:** {confidence:.2f}%")
