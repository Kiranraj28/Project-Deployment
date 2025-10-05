import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Set page title
st.title("Rice Leaf Disease Classification")

# Load the best model saved as H5 file (make sure best_model.h5 is in the working directory)
@st.cache_resource
def load_trained_model():
    model = load_model('best_model.h5')
    return model

model = load_trained_model()

# Define class names (update as per your dataset classes)
class_names = ['Disease_1', 'Disease_2', 'Healthy']

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize image to 256x256 as used in training
    img = image.resize((256, 256))
    img_array = np.array(img)
    if img_array.shape[2] == 4:
        img_array = img_array[..., :3]  # Convert RGBA to RGB if necessary
    img_array = preprocess_input(img_array)  # Normalize as MobileNetV2 expects
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(image: Image.Image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence

# Image uploader widget
uploaded_file = st.file_uploader("Upload a rice leaf image for disease classification", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class, confidence = predict(image)
    st.write(f"Prediction: **{predicted_class}** with confidence {confidence:.2f}")

else:
    st.write("Please upload an image to get started.")

