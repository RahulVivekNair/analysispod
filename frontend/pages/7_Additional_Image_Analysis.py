import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO
import torch

# Ensure the user is authenticated
if not st.session_state.get('authentication_status', False):
    st.info('Please Login and try again!')
    st.stop()

# Title and Information
st.title("Image Analysis")
st.write("Upload an image to detect fire and weapons.")
st.divider()
torch.cuda.empty_cache()

# Checkbox for user to select detection options
fire = st.checkbox("Detect Fire")
weapon = st.checkbox("Detect Weapons")

# Load specific models based on user selection
model_fire = None
model_knife = None
model_pistol = None

if fire:
    model_fire = YOLO('yolomodels/fire.pt')
if weapon:
    model_knife = YOLO('yolomodels/knife.pt')
    model_pistol = YOLO('yolomodels/pistol.pt')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to draw bounding boxes and class names on image
def draw_boxes(image, results):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    detected_classes = set()
    
    for result in results:
        # Iterate through each detection
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Extract bounding box coordinates
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            detected_classes.add(class_name)
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), class_name, fill="red", font=font)
    
    return image, detected_classes

# Process the uploaded file
if uploaded_file is not None:
    # Open the image file
    image = Image.open(uploaded_file)
    
    # Convert to numpy array
    img_np = np.array(image)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image.')
    
    # Set image size and confidence threshold
    img_size = 640
    conf_threshold = 0.5
    
    detections = []
    
    # Run detection models if selected
    if fire and model_fire:
        results_fire = model_fire.predict(source=image, imgsz=img_size, conf=conf_threshold)
        st.write("Fire Detection Results:")
        # Draw and display results
        image_with_boxes, detected_classes = draw_boxes(image.copy(), results_fire)
        st.image(image_with_boxes, caption='Detected Fire')
        detections.extend(detected_classes)

    if weapon:
        if model_knife:
            results_knife = model_knife.predict(source=image, imgsz=img_size, conf=conf_threshold)
            st.write("Knife Detection Results:")
            image_with_boxes, detected_classes = draw_boxes(image.copy(), results_knife)
            st.image(image_with_boxes, caption='Detected Knives')
            detections.extend(detected_classes)
        
        if model_pistol:
            results_pistol = model_pistol.predict(source=image, imgsz=img_size, conf=conf_threshold)
            st.write("Pistol Detection Results:")
            image_with_boxes, detected_classes = draw_boxes(image.copy(), results_pistol)
            st.image(image_with_boxes, caption='Detected Pistols')
            detections.extend(detected_classes)
    
    # Display detection results
    if detections:
        st.write("Detected: " + ", ".join(set(detections)))
    else:
        st.write("No objects detected.")
