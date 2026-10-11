
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Industrial Surface Defect Classification",
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
st.write("Upload a steel surface image to predict its defect class.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Defect"):
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

        predictions = model.predict(rgb, verbose=0)
        index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][index]) * 100

        st.success(f"Predicted Defect: {CLASSES[index]}")
        st.write(f"Model confidence: {confidence:.2f}%")
