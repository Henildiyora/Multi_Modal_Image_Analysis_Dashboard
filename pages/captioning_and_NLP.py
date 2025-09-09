# pages/1_📝_Captioning_and_NLP.py
import streamlit as st
from PIL import Image
from src import analysis_functions

st.set_page_config(page_title="Captioning & NLP", layout="wide")
st.title("📝 Image Captioning & NLP Analysis")

st.info("Upload an image to generate a caption, then perform further text analysis on that caption.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="caption_uploader")

if uploaded_file is not None:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")

if 'image' in st.session_state:
    st.image(st.session_state.image, caption='Your Image.', width=400)
    
    if st.button("Generate Caption & Analyze", type="primary"):
        models = st.session_state.models
        image = st.session_state.image
        
        with st.spinner("Generating caption..."):
            caption = analysis_functions.generate_caption(image, models)
            st.session_state.caption = caption
        
        st.subheader("Generated Caption:")
        st.markdown(f"#### *\"{st.session_state.caption}\"*")

        with st.spinner("Analyzing text..."):
            # Sentiment Analysis
            sentiment = analysis_functions.analyze_sentiment(st.session_state.caption, models)
            st.subheader("Sentiment Analysis")
            st.metric("Predicted Sentiment", sentiment['label'], f"{sentiment['score']*100:.2f}% Confidence")

            # NER
            entities = analysis_functions.perform_ner(st.session_state.caption, models)
            st.subheader("Named Entity Recognition (NER)")
            if entities:
                for ent in entities:
                    st.markdown(f"- **{ent.text}** (`{ent.label_}`)")
            else:
                st.write("No named entities found.")
            
            # Zero-Shot Classification
            st.subheader("Zero-Shot Classification")
            custom_labels_str = st.text_input("Enter custom labels, separated by commas", "outdoors, indoors, portrait, event")
            if custom_labels_str:
                labels = [label.strip() for label in custom_labels_str.split(',')]
                result = analysis_functions.classify_zero_shot(st.session_state.caption, labels, models)
                st.write(result)