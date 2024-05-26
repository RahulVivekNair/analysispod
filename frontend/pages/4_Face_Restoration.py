import streamlit as st
import cv2
import numpy as np
from gfpgan import GFPGANer
from streamlit_image_comparison import image_comparison
import torch
if not st.session_state.get('authentication_status', False):
    st.info('Please Login and try again!')
    st.stop()
st.title("GFPGAN Face Restoration")

# Setup the face restoration model
upscale = 4
arch = 'clean'
channel_multiplier = 2
bg_upsampler = None
model = GFPGANer(
    model_path="models/GFPGANv1.4.pth",
    upscale=upscale,
    arch=arch,
    channel_multiplier=channel_multiplier,
    bg_upsampler=bg_upsampler
)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)

    if st.button('Restore Face'):
        with st.spinner('Restoring Face...'):
            cropped_faces, restored_faces, restored_img = model.enhance(
                original_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=0.5  # Adjustable weights
            )

            # Convert the restored image from BGR to RGB
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

            # Render the image comparison
            image_comparison(
                img1=cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
                img2=restored_img,
                label1="Original Image",
                label2="Face Restored Image",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
            )