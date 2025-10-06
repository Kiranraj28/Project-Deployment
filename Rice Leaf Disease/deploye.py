# streamlit_app.py

import streamlit as st
import tensorflow as tf
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
# Correct import for Keras 3
try:
    from keras.layers.experimental import TFSMLayer
except ImportError:
    st.error("TFSMLayer not found. Make sure you are using Keras 3 or install the correct package.")
    st.stop()

@st.cache_resource
def load_dl_model():
    SAVEDMODEL_DIR = "rice_leaf_model" 
    
    # Wrap the SavedModel for Keras 3
    model = tf.keras.Sequential([
        TFSMLayer(SAVEDMODEL_DIR, call_endpoint='serving_default')
    ])
    
    # Build the model with input shape
    model.build(input_shape=(None, 224, 224, 3)) 
    
    return model

# Load model once
model = load_dl_model()

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # ---------------------------
    # Preprocess the image
    # ---------------------------
    img = img.resize((224, 224))  # match model input
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)

    # ---------------------------
    # Make Prediction
    # ---------------------------
    predictions = model.predict(img_array)
    
    # Ensure predictions shape is (1, 3)
    if predictions.shape[1] != 3:
        st.error(f"Unexpected model output shape: {predictions.shape}")
    else:
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        class_names = ["Bacterial Leaf Blight", "Brown Spot", "Leaf Smut"]

        # Display the prediction
        st.write(f"### Predicted Disease: **{class_names[predicted_class_idx]}**")
        st.write(f"Prediction Confidence: {predictions[0][predicted_class_idx]*100:.2f}%")
