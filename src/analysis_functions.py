from PIL import Image

def generate_caption(image, model_dict):
    processor = model_dict["caption"]["processor"]
    model = model_dict["caption"]["model"]
    
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values
    output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.capitalize()

def detect_objects(image, model_dict):
    detector = model_dict["detection"]
    return detector(image)

def answer_question(image, question, model_dict):
    processor = model_dict["vqa"]["processor"]
    model = model_dict["vqa"]["model"]
    
    encoding = processor(image, question, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    answer = model.config.id2label[idx]
    return answer.capitalize()

def analyze_sentiment(text, model_dict):
    analyzer = model_dict["sentiment"]
    return analyzer(text)[0]

def perform_ner(text, model_dict):
    nlp = model_dict["ner"]
    return nlp(text).ents

def classify_zero_shot(text, labels, model_dict):
    classifier = model_dict["zero_shot"]
    return classifier(text, candidate_labels=labels)