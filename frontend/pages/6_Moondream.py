import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

st.title("Moondream Analysis")
st.divider()
torch.cuda.empty_cache()
model_id = "vikhyatk/moondream2"
revision = "2024-04-02"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,torch_dtype=torch.float16
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None and st.button("Describe Image"):
    image = Image.open(uploaded_file)
    enc_image = model.encode_image(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(model.answer_question(enc_image, "Describe this image in great detail.", tokenizer))



