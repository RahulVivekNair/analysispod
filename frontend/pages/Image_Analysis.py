import streamlit as st
from upscalers import upscale

# Title and Information
st.title("Image Upscaling with ESRGAN")
st.write("Upload your image to experience AI-powered upscaling using the ESRGAN model.")


# Image Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if st.button("upscale"):
    scale_factor = 4.0
    result = upscale('R-ESRGAN General 4xV3', uploaded_file, scale_factor)
    st.subheader("Original Image vs. Upscaled Image")
    cols = st.columns(2)
    cols[0].image(uploaded_file, caption='Original Image')
    cols[1].image(result, caption='Upscaled Image') 