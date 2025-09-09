from PIL import ImageDraw, ImageFont

def draw_object_boxes(image, objects):
    """Draws bounding boxes and labels on the image for detected objects."""
    draw = ImageDraw.Draw(image)
    try:
        # Use the downloaded font file
        font = ImageFont.truetype("assets/arial.ttf", 25)
    except IOError:
        font = ImageFont.load_default()
        
    for obj in objects:
        box = obj['box']
        label = obj['label']
        score = obj['score']
        
        xmin, ymin, xmax, ymax = box.values()
        
        # Draw rectangle
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="cyan", width=3)
        
        # Draw label background and text
        text = f"{label} ({score:.2f})"
        text_bbox = draw.textbbox((xmin, ymin - 30), text, font=font)
        draw.rectangle(text_bbox, fill="cyan")
        draw.text((xmin, ymin - 30), text, fill="black", font=font)
        
    return image