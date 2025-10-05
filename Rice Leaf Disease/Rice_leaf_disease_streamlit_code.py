import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.title("Rice Leaf Disease Classification")

# Load model once and cache it
@st.cache_resource
def load_trained_model():
    return load_model('best_model.h5')

model = load_trained_model()

# Update these to your actual class names
class_names = ['Disease_1', 'Disease_2', 'Healthy']

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((256, 256))
    img_array = np.array(image)
    if img_array.shape[2] == 4:
        img_array = img_array[..., :3]  # Convert RGBA to RGB if present
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict(image: Image.Image):
    img = preprocess_image(image)
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    return class_names[class_idx], confidence

uploaded_file = st.file_uploader("Upload a rice leaf image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    pred_class, conf = predict(image)
    st.write(f"Prediction: **{pred_class}** with confidence {conf:.2f}")
else:
    st.write("Please upload an image file to classify.")
