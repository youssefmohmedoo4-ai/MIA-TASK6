import streamlit as st
from PIL import Image
from src.inference import CaptionGenerator

st.set_page_config(page_title="AI Image Caption Generator", layout="centered")

st.title("AI Image Caption Generator")
st.write("Upload any image to generate a natural-language description using a CNN-LSTM deep learning model.")

@st.cache_resource
def load_caption_engine():
    return CaptionGenerator()

try:
    generator = load_caption_engine()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    model_loaded = False

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Analyzing image features and generating caption..."):
        caption = generator.generate_caption(image)
        
    st.success("### Generated Caption:")
    st.markdown(f"> **{caption.capitalize()}**")