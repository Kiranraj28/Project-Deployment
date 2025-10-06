# streamlit_app.py

import streamlit as st
import os
import zipfile
import tensorflow as tf
from PIL import Image
import numpy as np
import requests

# ---------------------------
# Title and description
# ---------------------------
st.title("Rice Leaf Disease Classification 🌾")
st.write("""
This app classifies rice leaf images into the following disease categories:
- Bacterial Leaf Blight
- Brown Spot
- Leaf Smut
""")

# ---------------------------
# Download and extract model zip from Git (if not present)
# ---------------------------
MODEL_ZIP = "Rice Leaf Disease/rice_leaf_model.zip"
MODEL_FOLDER = "Rice Leaf Disease/rice_leaf_model"

if not os.path.exists(MODEL_FOLDER):
    if not os.path.exists(MODEL_ZIP):
        # Replace with your raw GitHub URL of the zip file
        ZIP_URL = "https://github.com/Kiranraj28/Project-Deployment/blob/main/Rice%20Leaf%20Disease/rice_leaf_model.zip"
        st.info("Downloading model...")
        r = requests.get(ZIP_URL)
        with open(MODEL_ZIP, "wb") as f:
            f.write(r.content)

    # Extract the zip
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(MODEL_FOLDER)
        st.success("Model extracted successfully!")

# ---------------------------
# Load the model
# ---------------------------
@st.cache_resource
def load_dl_model():
    model = tf.keras.models.load_model(MODEL_FOLDER)
    return model

model = load_dl_model()

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).resize((256,256))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1,256,256,3)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    class_names = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut"]
    st.write(f"### Predicted Disease: **{class_names[predicted_class]}**")
    st.write(f"Prediction Confidence: {predictions[0][predicted_class]*100:.2f}%")

    # Optional: show top 3 predictions
    top3_idx = predictions[0].argsort()[-3:][::-1]
    st.write("### Top 3 Predictions:")
    for i in top3_idx:
        st.write(f"{class_names[i]}: {predictions[0][i]*100:.2f}%")
