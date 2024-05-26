import streamlit as st
import torch
from streamlit_image_comparison import image_comparison
import cv2
from PIL import Image
import numpy as np
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
from io import BytesIO
if not st.session_state.get('authentication_status', False):
    st.info('Please Login and try again!')
    st.stop()
st.title("Real-ESRGAN Image Upscaler")

# Setup the upsampler
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
model_path = "models/RealESRGAN_x4plus.pth"
upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)

    if st.button('Upscale'):
        with st.spinner('Upscaling...'):
            output, _ = upsampler.enhance(original_img, outscale=netscale)
            # Convert the upscaled image from BGR to RGB
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        # Render the image comparison
        image_comparison(
            img1=cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
            img2=output,
            label1="Original Image",
            label2="Upscaled Image",
            width=700,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
        )
        