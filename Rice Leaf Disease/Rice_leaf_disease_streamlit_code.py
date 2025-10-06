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

import tensorflow as tf

model_path = 'Rice Leaf Disease/best_model1.keras'
model = tf.keras.models.load_model(model_path)
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
