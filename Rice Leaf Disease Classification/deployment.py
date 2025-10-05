# rice_leaf_disease_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image

# -----------------------------
# 1. App title and description
# -----------------------------
st.set_page_config(page_title="Rice Leaf Disease Classifier", layout="centered")
st.title("Rice Leaf Disease Classification")
st.write(
    """
    Upload an image of a rice leaf and the model will predict if it has:
    - Bacterial leaf blight  
    - Brown spot  
    - Leaf smut
    """
)

# -----------------------------
# 2. Load the trained model
# -----------------------------
@st.cache_resource  # cache the model to avoid reloading every time
def load_model():
    model = tf.keras.models.load_model("rice_leaf_classifier_final.keras")
    return model

model = load_model()
st.success("Model loaded successfully!")

# -----------------------------
# 3. Class names
# -----------------------------
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# -----------------------------
# 4. Image uploader
# -----------------------------
uploaded_file = st.file_uploader("Choose a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image_array = img_to_array(image)
    image_array = tf.image.resize(image_array, [256, 256])
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # shape: (1, 256, 256, 3)

    # Predict
    pred = model.predict(image_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred) * 100

    # Display result
    st.write(f"**Predicted Class:** {class_names[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
