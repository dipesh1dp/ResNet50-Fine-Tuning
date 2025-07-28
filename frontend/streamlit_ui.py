import streamlit as st 
from PIL import Image
import requests 
from io import BytesIO

# # Set page config
st.set_page_config(page_title="Flower Classification", layout="centered")

# Centered header image
header_image = Image.open("frontend/assets/header.png")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(header_image, width=200)

# API endpoint
API_URL = "http://127.0.0.1:8000/predict/"

# Title and description
st.title("🌸 Flower Classification")
st.write("Identify flowers from images.st Upload a photo to find out which species it is!")

st.markdown("---")

# Image uploader
img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if img_file is not None:
    # Buffer the image file into memory
    file_bytes = img_file.read()
    image = Image.open(BytesIO(file_bytes))

    # Show thumbnail preview 
    st.image(image, caption="Preview", width=100)

    if st.button("🔍 Predict"):
        files = {
            "file": (img_file.name, BytesIO(file_bytes), img_file.type)
        }

        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            # Extract and format results
            label = result["label"].capitalize()
            confidence = result["confidence"]
            class_id = result["class_id"]

            # Display results
            st.success(f"🌼 **Prediction:** {label}")
            st.markdown(f"🆔 Class ID: `{class_id}`")
            st.metric("🎯 Confidence", f"{confidence * 100:.2f}%")
            st.progress(confidence)

        else:
            st.error("❌ Failed to get prediction from the server.")
