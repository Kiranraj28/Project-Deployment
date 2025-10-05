# ==============================
# 🌾 Rice Leaf Disease Prediction App
# ==============================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ------------------------------
# 1. Load Model (cached)
# ------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "rice_leaf_classifier_clean2.keras")
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model = load_model()

# ------------------------------
# 2. Class Names (based on your training)
# ------------------------------
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.set_page_config(page_title="🌾 Rice Leaf Disease Classifier", layout="centered")

st.title("🌾 Rice Leaf Disease Classification")
st.markdown("Upload a rice leaf image to predict its disease type using a **MobileNetV2** model.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display result
    st.success(f"✅ Predicted Disease: **{predicted_class}**")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.progress(int(confidence))

st.caption("Model: MobileNetV2 | Framework: TensorFlow | Deployment: Streamlit Cloud")
