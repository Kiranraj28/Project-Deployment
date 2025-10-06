# streamlit_app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ---------------------------
# Title and Description
# ---------------------------
st.title("Rice Leaf Disease Classification 🌾")
st.write("""
This app uses a deep learning model to classify rice leaf images into the following disease categories:
- Bacterial Leaf Blight
- Brown Spot
- Leaf Smut
""")

# ---------------------------
# Load the trained model
# ---------------------------
# In Rice_leaf_disease_streamlit_code.py, line ~47
MODEL_FOLDER = "Rice Leaf Disease/my_model.h5"
# OR just "my_model.h5" if the script assumes the current directory.

# Then, load the model
model = tf.keras.models.load_model(MODEL_FOLDER)

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)


    # Preprocess the image
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 256, 256, 3)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Map the predicted index to class name
    class_names = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut"]

    # Display the prediction
    st.write(f"### Predicted Disease: **{class_names[predicted_class]}**")
    st.write(f"Prediction Confidence: {predictions[0][predicted_class]*100:.2f}%")
