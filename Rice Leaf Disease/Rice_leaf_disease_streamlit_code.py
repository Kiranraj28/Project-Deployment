import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Rice Leaf Disease Classification", layout="centered")

st.title("Rice Leaf Disease Classification")
st.write("Upload an image of a rice leaf and the model will predict the disease class.")

# --- Debug: Show working directory and files to help with file path issues ---
st.write("Working Directory:", os.getcwd())
st.write("Files in working directory:", os.listdir())

if "Rice Leaf Disease" in os.listdir():
    st.write("Files in 'Rice Leaf Disease':", os.listdir("Rice Leaf Disease"))

# --- Model Loading ---
@st.cache_resource
def load_trained_model():
    model_path = "Rice Leaf Disease/best_model.h5"
    st.write("Trying to load model from:", model_path)
    return tf.keras.models.load_model(model_path, compile=False)

model = load_trained_model()

# --- Class Names ---
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload a rice leaf image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # To batch dimension

    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    
    st.success(f"Prediction: {class_names[class_idx]}")
    st.info(f"Confidence: {confidence:.2f}")
else:
    st.info("Please upload an image to get the prediction.")
