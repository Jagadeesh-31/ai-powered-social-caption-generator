"""
STEP 5: Simple demo UI for client presentations.

Run:
    streamlit run 5_streamlit_demo.py
"""

import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="Image Caption Generator", layout="centered")
st.title("🖼️ Image Caption Generator")
st.write("Upload an image and get an AI-generated caption.")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out[0], skip_special_tokens=True)

    st.success(f"**Caption:** {caption}")
