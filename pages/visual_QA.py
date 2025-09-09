# pages/3_❓_Visual_Q&A.py
import streamlit as st
from PIL import Image
from src import analysis_functions

st.set_page_config(page_title="Visual Q&A", layout="wide")
st.title("❓ Visual Question Answering (VQA)")
st.info("Upload an image and ask a question about its content.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="vqa_uploader")

if uploaded_file is not None:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")

if 'image' in st.session_state:
    st.image(st.session_state.image, caption='Your Image.', width=400)
    
    question = st.text_input("Ask a question about the image:", "What is the main color of the object?")
    
    if st.button("Get Answer", type="primary"):
        if question:
            models = st.session_state.models
            image = st.session_state.image

            with st.spinner("Thinking..."):
                answer = analysis_functions.answer_question(image, question, models)
                st.markdown(f"### Question: *{question}*")
                st.success(f"### Answer: **{answer}**")
        else:
            st.warning("Please enter a question.")