
import streamlit as st

st.set_page_config(
    page_title="Surface Defect Classification",
    page_icon="🔍",
    layout="centered"
)

st.title("Industrial Surface Defect Classification")

st.write(
    "Upload a steel surface image to classify its defect "
    "using our trained MobileNetV2 model."
)

uploaded_file = st.file_uploader(
    "Upload a surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Surface Image",
        use_container_width=True
    )

    st.info("Image uploaded successfully. Prediction will be added next.")