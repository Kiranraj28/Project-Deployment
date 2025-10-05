# ==============================
# Streamlit App for Rice Leaf Disease Prediction
# ==============================

import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# ------------------------------
# 1. Load Model
# ------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("rice_leaf_classifier_final.keras")

model = load_model()

# ------------------------------
# 2. Class Names
# ------------------------------
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']  # same as in your dataset

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.title("🌾 Rice Leaf Disease Classification")
st.write("Upload an image of a rice leaf, and the model will predict its disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded Leaf Image', use_column_width=True)

    # Preprocess image
    image = load_img(uploaded_file, target_size=(256, 256))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    pred = model.predict(image_array)
    pred_class = np.argmax(pred)
    confidence = np.max(pred)

    # Display result
    st.success(f"Predicted Disease: {class_names[pred_class]}")
    st.info(f"Confidence: {confidence*100:.2f}%")
