#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()
CLASS_NAMES = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut'] # adjust names as per your classes

st.title("Rice Leaf Disease Prediction")

uploaded_file = st.file_uploader("Upload a rice leaf image (.jpg)", type="jpg")
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")
    img_resized = image.resize((256,256))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    result_idx = np.argmax(preds)
    st.write(f"Prediction: {CLASS_NAMES[result_idx]}")
    st.write(f"Confidence: {preds[0][result_idx]:.2f}")

