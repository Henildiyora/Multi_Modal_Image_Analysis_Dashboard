import streamlit as st
from transformers import pipeline,BlipProcessor, BlipForConditionalGeneration, ViltProcessor, ViltForQuestionAnswering
import spacy

@st.cache_resource
def load_all_models():
    """Loads and caches all AI models needed for the application."""
    # Captioning Model (BLIP)
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    # Sentiment Analysis Model
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # NER Model
    nlp_ner = spacy.load("en_core_web_sm")
    
    # Object Detection Model
    object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    
    # Visual Question Answering Model
    vqa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    vqa_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # Zero-Shot Classification Model
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    models = {
        "caption": {"processor": caption_processor, "model": caption_model},
        "sentiment": sentiment_analyzer,
        "ner": nlp_ner,
        "detection": object_detector,
        "vqa": {"processor": vqa_processor, "model": vqa_model},
        "zero_shot": zero_shot_classifier
    }
    return models