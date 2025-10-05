#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Rice Leaf Disease Detection Platform",
    page_icon="🌾",
    layout="centered"
)



st.sidebar.title("About This Project")
st.sidebar.markdown("""
**🌾Rice Leaf Disease Classification**

This project leverages deep learning to automate the identification of common rice leaf diseases from images. Using a Convolutional Neural Network with MobileNetV2 as backbone and custom classification layers, the model distinguishes between Bacterial Leaf Blight, Brown Spot, and Leaf Smut.

**Key Features:**
- **Robust deep learning architecture:** Transfer learning with MobileNetV2 provides strong generalization on small agricultural datasets.
- **Real-time image analysis:** Upload rice leaf images and receive instant disease predictions with confidence scores.
- **Data augmentation:** Comprehensive techniques (flips, rotations, zooms, contrast adjustments) increase model resilience to varied field conditions.
- **Practical application:** Accelerates disease detection for researchers and farmers, helping minimize crop loss and enhance food security.

**Results:**
Achieves high accuracy in classifying major rice leaf diseases, demonstrating AI’s value in precision agriculture and plant pathology.
""")

st.sidebar.markdown("""
**Understanding Confidence Score**
The confidence score expresses how certain the model is about its prediction, ranging from 0 (no certainty) to 1 (maximum certainty).
- Values closer to 1 indicate the prediction is highly reliable.
- Moderate values (e.g., around 0.5-0.7) suggest more uncertainty or similarity between categories.
- The confidence score helps guide user trust and follow-up decisions when reviewing results.
""")


from pathlib import Path
import tensorflow as tf
import streamlit as st

@st.cache_resource
def load_model():
    # 👇 both files are in the same folder, so direct reference is enough
    model_path = Path(__file__).parent / "best_model.h5"

    # optional check for debugging
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
    else:
        st.success(f"✅ Model found at: {model_path}")

    return tf.keras.models.load_model(str(model_path), compile=False)


model = load_model()
CLASS_NAMES = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut'] # adjust names as per your classes

st.title("🌾Rice Leaf Disease Prediction")

uploaded_file = st.file_uploader("Upload a rice leaf image (.jpg)", type="jpg")
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")
    img_resized = image.resize((256,256))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    result_idx = np.argmax(preds)
    st.success(f"Prediction: {CLASS_NAMES[result_idx]}")
    st.info(f"Confidence: {preds[0][result_idx]:.2f}")
    
