#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(256,256,3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    saved_model = tf.keras.models.load_model("best_model.keras", compile=False, safe_mode=False)
    model.set_weights(saved_model.get_weights())
    return model
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

