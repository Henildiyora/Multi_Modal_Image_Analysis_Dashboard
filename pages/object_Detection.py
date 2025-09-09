# pages/2_📦_Object_Detection.py
import streamlit as st
from PIL import Image
from src import analysis_functions, ui_utils

st.set_page_config(page_title="Object Detection", layout="wide")
st.title("📦 Object Detection")
st.info("Upload an image to identify and locate objects within it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="object_uploader")

if uploaded_file is not None:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")

if 'image' in st.session_state:
    st.image(st.session_state.image, caption='Your Image.', width=400)
    
    if st.button("Detect Objects", type="primary"):
        models = st.session_state.models
        image = st.session_state.image.copy()

        with st.spinner("Detecting objects..."):
            detected_objects = analysis_functions.detect_objects(image, models)
            if detected_objects:
                st.success(f"Detected {len(detected_objects)} objects.")
                image_with_boxes = ui_utils.draw_object_boxes(image, detected_objects)
                st.image(image_with_boxes, caption="Detected Objects", use_column_width=True)
            else:
                st.warning("No objects were detected.")