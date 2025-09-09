# import streamlit as st
# from PIL import Image
# import torch
# from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
# import spacy
# import time

# # App Configuration
# st.set_page_config(page_title="Multi-Modal Image Analysis (PyTorch)", layout="wide")

# # Model Loading

# # Load pre-trained models from Hugging Face and spaCy (cached for performance)
# @st.cache_resource
# def load_models():
#     # 1. Load the new, state-of-the-art Image Captioning Model (Salesforce BLIP)
#     caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
#     caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
#     # 2. Load the Sentiment Analysis and NER models (these are framework-agnostic)
#     sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
#     nlp_ner = spacy.load("en_core_web_sm")
    
#     return caption_processor, caption_model, sentiment_analyzer, nlp_ner

# st.markdown("<h4>Loading AI Models...</h4>", unsafe_allow_html=True)
# caption_processor, caption_model, sentiment_analyzer, nlp_ner = load_models()
# st.success("AI Models Loaded Successfully!")

# # --- Core Function for Caption Generation ---
# def generate_caption(image):
#     """
#     Generates a caption for a given image using the BLIP model.
#     """
#     # Prepare the image for the model using the processor
#     inputs = caption_processor(images=image, return_tensors="pt")
    
#     # Generate the caption (IDs)
#     pixel_values = inputs.pixel_values
#     output_ids = caption_model.generate(pixel_values, max_length=50, num_beams=4)
    
#     # Decode the IDs to a human-readable string
#     caption = caption_processor.decode(output_ids[0], skip_special_tokens=True)
#     return caption.capitalize()

# # Streamlit UI
# st.title("📸 Multi-Modal Image Analysis Dashboard (PyTorch Edition)")
# st.markdown("Upload an image to generate a descriptive caption, analyze its sentiment, and identify named entities.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption='Uploaded Image.', use_column_width=True)

#     # Generate analysis on button click
#     if st.button("Analyze Image", type="primary"):
#         with st.spinner('Analyzing... This may take a moment.'):
#             # Caption Generation ---
#             with st.expander("Step 1: Image Captioning", expanded=True):
#                 generated_caption = generate_caption(image)
#                 st.subheader("Generated Caption:")
#                 st.markdown(f"#### *\"{generated_caption}\"*")
#                 time.sleep(1) 

#             # Sentiment Analysis
#             with st.expander("Step 2: Sentiment Analysis of the Caption", expanded=True):
#                 if generated_caption:
#                     sentiment = sentiment_analyzer(generated_caption)[0]
#                     st.subheader("Sentiment:")
#                     label = sentiment['label']
#                     score = sentiment['score']
#                     if label == "POSITIVE":
#                         st.success(f"**{label}** (Confidence: {score:.2f})")
#                     else:
#                         st.error(f"**{label}** (Confidence: {score:.2f})")
#                 else:
#                     st.warning("Could not analyze sentiment without a caption.")
#                 time.sleep(1)

#             # Named Entity Recognition
#             with st.expander("Step 3: Named Entity Recognition (NER) in the Caption", expanded=True):
#                 if generated_caption:
#                     doc = nlp_ner(generated_caption)
#                     st.subheader("Identified Entities:")
#                     if doc.ents:
#                         for ent in doc.ents:
#                             st.markdown(f"- **{ent.text}** (`{ent.label_}`)")
#                     else:
#                         st.info("No specific named entities (like people, places, or organizations) were found in the caption.")
#                 else:
#                     st.warning("Could not perform NER without a caption.")


# app.py
import streamlit as st
from src import model_loader

st.set_page_config(
    page_title="AI Playground for Vision & Language",
    page_icon="🤖",
    layout="wide"
)

# Load models and store them in the session state
if 'models' not in st.session_state:
    with st.spinner("Loading AI Models... This is a one-time setup and may take a few minutes."):
        st.session_state.models = model_loader.load_all_models()
    st.success("All AI Models Loaded Successfully!")

st.title("Welcome to the AI Playground!")
st.markdown(
    """
    This is an interactive application showcasing a variety of state-of-the-art AI models for vision and language tasks. 
    The code has been structured into a professional, easy-to-understand package.
    
    ### **Select an analysis page from the sidebar to begin!**
    
    Each page allows you to upload an image and perform a different kind of AI-powered analysis.
    """
)

st.sidebar.success("Select an analysis page above.")