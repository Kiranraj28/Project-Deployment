# ==============================
# 🌾 Rice Leaf Disease Prediction App
# ==============================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="🌾 Rice Leaf Disease Classifier", layout="centered")

# ------------------------------
# Sidebar Section
# ------------------------------
with st.sidebar:
    st.title("🌾 Project Overview")

    st.markdown("""
    ### 🎯 Objective
    Build a robust image classification model to accurately detect **Rice Leaf Diseases** 
    from leaf images using **Deep Learning**.

    ### 🧠 Approach
    1. Dataset Preparation: Organized labeled images for each disease class.
    2. Preprocessing: Resized images to 256×256 & normalized using MobileNetV2 preprocess_input.
    3. Model Architecture: MobileNetV2 + custom Dense layers for classification.
    4. Training & Evaluation: Monitored validation accuracy & loss.

    ### 🏆 Outcome
    High accuracy in classifying:
    - Bacterial Leaf Blight
    - Brown Spot
    - Leaf Smut
    """)
    st.markdown("---")
    st.markdown("📘 *Developed by [Kiran Raj T](https://github.com/Kiranraj28)*")


# ------------------------------
# Load Model (cached)
# ------------------------------
@st.cache_resource
def load_model(model_path="rice_leaf_classifier_clean.keras"):
    """
    Load the Keras model from a given path.
    If not found, return None (fallback to upload).
    """
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    return None


model = load_model()

# ------------------------------
# Fallback: Model Upload
# ------------------------------
if model is None:
    st.warning("❌ Preloaded model not found. Please upload your model file (.keras).")
    uploaded_model = st.file_uploader("Upload Keras model", type=["keras", "h5"])
    if uploaded_model is not None:
        try:
            model = tf.keras.models.load_model(uploaded_model, compile=False)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load the uploaded model. Error: {e}")
            st.stop()
    else:
        st.stop()


# ------------------------------
# Class Names
# ------------------------------
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']


# ------------------------------
# Image Preprocessing
# ------------------------------
def preprocess_image(image):
    image = image.resize((256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🌾 Rice Leaf Disease Classification")
st.markdown("Upload a rice leaf image to predict its disease type using **MobileNetV2** model.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"✅ Predicted Disease: **{predicted_class}**")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

# ------------------------------
# Footer
# ------------------------------
st.caption("Model: MobileNetV2 | Framework: TensorFlow | Deployment: Streamlit Cloud")
