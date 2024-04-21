import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
# Title and Information
st.title("Image Analysis")
st.write("Upload an image to detect fire and weapons.")
st.divider()
torch.cuda.empty_cache()
# Checkbox for user to select detection options
fire = st.checkbox("Detect Fire")
weapon = st.checkbox("Detect Weapons")

# Load specific models based on user selection
if fire:
    model_fire = YOLO('yolomodels/fire.pt')
if weapon:
    model_knife = YOLO('yolomodels/knife.pt')
    model_pistol = YOLO('yolomodels/pistol.pt')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Perform detection when the button is clicked
    if st.button("Analyze Image"):
        if fire:
            results_fire = model_fire(uploaded_file)
            if len(results_fire.xyxy[0]) > 0:
                st.write("Fire Detected")
                # Render and display image with bounding boxes
                image_fire = results_fire.render()[0]
                st.image(image_fire, caption="Fire Detection", use_column_width=True)
            else:
                st.write("No Fire Detected")
        
        if weapon:
            results_knife = model_knife(image_array)
            results_pistol = model_pistol(image_array)
            knife_detected = len(results_knife.xyxy[0]) > 0
            pistol_detected = len(results_pistol.xyxy[0]) > 0
            
            if knife_detected or pistol_detected:
                st.write("Weapon Detected")
                if knife_detected:
                    st.write("Knife Detected")
                    # Render and display image with bounding boxes for knife
                    image_knife = results_knife.render()[0]
                    st.image(image_knife, caption="Knife Detection", use_column_width=True)
                if pistol_detected:
                    st.write("Pistol Detected")
                    # Render and display image with bounding boxes for pistol
                    image_pistol = results_pistol.render()[0]
                    st.image(image_pistol, caption="Pistol Detection", use_column_width=True)
            else:
                st.write("No Weapons Detected")