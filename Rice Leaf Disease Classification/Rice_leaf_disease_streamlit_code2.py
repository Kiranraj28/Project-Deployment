#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image

# ---------------------------
# 1. App Title
# ---------------------------
st.set_page_config(page_title="Rice Leaf Disease Classifier", layout="centered")
st.title("🌾 Rice Leaf Disease Classification")
st.write("Upload an image of a rice leaf, and the model will predict the disease.")

# ---------------------------
# 2. Load the trained model
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_rice_model():
    model = load_model("best_model1.keras")
    return model

model = load_rice_model()

# Define class names
class_names = ['Bacterial Blight', 'Brown Spot', 'Leaf Blast']

# ---------------------------
# 3. Upload image
# ---------------------------
uploaded_file = st.file_uploader("Choose a rice leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_column_width=True)
    
    # ---------------------------
    # 4. Preprocess image
    # ---------------------------
    img = image.resize((256, 256))            # resize
    img_array = img_to_array(img)             # convert to array
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    img_array = img_array / 255.0             # normalize

    # ---------------------------
    # 5. Prediction
    # ---------------------------
    with st.spinner("Predicting..."):
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds) * 100

    # ---------------------------
    # 6. Show results
    # ---------------------------
    st.success(f"Predicted Disease: {class_names[pred_class]}")
    st.info(f"Confidence: {confidence:.2f}%")

